import time
import yaml
import logging
import pandas as pd
import networkx as nx
from pathlib import Path
from functools import partial
from requests.exceptions import HTTPError

from snakemake.utils import update_config


REGION_COLS = ["geometry", "name", "x", "y", "country"]


def to_date_period(dt):
    assert dt.tz is not None, 'Datetime must have timezone'
    dt = dt.tz_convert('Europe/London')
    return dt.strftime('%Y-%m-%d'), dt.hour * 2 + dt.minute // 30 + 1


def to_datetime(day, period):
    return pd.Timestamp(day, tz='Europe/London').tz_convert('utc') + pd.Timedelta(minutes=30) * (period - 1)


def to_daterange(*args):
    '''Somewhat redundant, but ensures consistency between datasets'''

    if len(args) == 2:
        assert (start.tz is not None + end.tz is not None) == 2, 'Both start and end must be timezone-naive'
        start, end = args[0], args[1]
    
    elif len(args) == 1:

        if not isinstance(args[0], pd.Timestamp):
            assert len(args[0]) == 10, 'Expects single day or start and end dates'

        start, end = pd.Timestamp(args[0]), pd.Timestamp(args[0]) + pd.Timedelta(days=1) - pd.Timedelta(minutes=30)

    return pd.date_range(start, end, freq='30min', tz='Europe/London')


def get_run_path(fn, dir, rdir, shared_resources, exclude_from_shared):
    """
    Dynamically provide paths based on shared resources and filename.

    Use this function for snakemake rule inputs or outputs that should be
    optionally shared across runs or created individually for each run.

    Parameters
    ----------
    fn : str
        The filename for the path to be generated.
    dir : str
        The base directory.
    rdir : str
        Relative directory for non-shared resources.
    shared_resources : str or bool
        Specifies which resources should be shared.
        - If string is "base", special handling for shared "base" resources (see notes).
        - If random string other than "base", this folder is used instead of the `rdir` keyword.
        - If boolean, directly specifies if the resource is shared.
    exclude_from_shared: list
        List of filenames to exclude from shared resources. Only relevant if shared_resources is "base".

    Returns
    -------
    str
        Full path where the resource should be stored.

    Notes
    -----
    Special case for "base" allows no wildcards other than "technology", "year"
    and "scope" and excludes filenames starting with "networks/elec" or
    "add_electricity". All other resources are shared.
    """
    if shared_resources == "base":
        pattern = r"\{([^{}]+)\}"
        existing_wildcards = set(re.findall(pattern, fn))
        irrelevant_wildcards = {"technology", "year", "scope", "kind"}
        no_relevant_wildcards = not existing_wildcards - irrelevant_wildcards
        not_shared_rule = (
            not fn.startswith("networks/elec")
            and not fn.startswith("add_electricity")
            and not any(fn.startswith(ex) for ex in exclude_from_shared)
        )
        is_shared = no_relevant_wildcards and not_shared_rule
        rdir = "" if is_shared else rdir
    elif isinstance(shared_resources, str):
        rdir = shared_resources + "/"
    elif isinstance(shared_resources, bool):
        rdir = "" if shared_resources else rdir
    else:
        raise ValueError(
            "shared_resources must be a boolean, str, or 'base' for special handling."
        )

    return f"{dir}{rdir}{fn}"


def path_provider(dir, rdir, shared_resources, exclude_from_shared):
    """
    Returns a partial function that dynamically provides paths based on shared
    resources and the filename.

    Returns
    -------
    partial function
        A partial function that takes a filename as input and
        returns the path to the file based on the shared_resources parameter.
    """
    return partial(
        get_run_path,
        dir=dir,
        rdir=rdir,
        shared_resources=shared_resources,
        exclude_from_shared=exclude_from_shared,
    )


def configure_logging(snakemake, skip_handlers=False):
    """
    Configure the basic behaviour for the logging module.

    Note: Must only be called once from the __main__ section of a script.

    The setup includes printing log messages to STDERR and to a log file defined
    by either (in priority order): snakemake.log.python, snakemake.log[0] or "logs/{rulename}.log".
    Additional keywords from logging.basicConfig are accepted via the snakemake configuration
    file under snakemake.config.logging.

    Parameters
    ----------
    snakemake : snakemake object
        Your snakemake object containing a snakemake.config and snakemake.log.
    skip_handlers : True | False (default)
        Do (not) skip the default handlers created for redirecting output to STDERR and file.
    """
    import logging
    import sys

    kwargs = snakemake.config.get("logging", dict()).copy()
    kwargs.setdefault("level", "INFO")

    if skip_handlers is False:
        fallback_path = Path(__file__).parent.joinpath(
            "..", "logs", f"{snakemake.rule}.log"
        )
        logfile = snakemake.log.get(
            "python", snakemake.log[0] if snakemake.log else fallback_path
        )
        kwargs.update(
            {
                "handlers": [
                    # Prefer the 'python' log, otherwise take the first log for each
                    # Snakemake rule
                    logging.FileHandler(logfile),
                    logging.StreamHandler(),
                ]
            }
        )
    logging.basicConfig(**kwargs)

    # Setup a function to handle uncaught exceptions and include them with their stacktrace into logfiles
    def handle_exception(exc_type, exc_value, exc_traceback):
        # Log the exception
        logger = logging.getLogger()
        logger.error(
            "Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback)
        )

    sys.excepthook = handle_exception


def calculate_annuity(n, r):
    """
    Calculate the annuity factor for an asset with lifetime n years and.

    discount rate of r, e.g. annuity(20, 0.05) * 20 = 1.6
    """
    if isinstance(r, pd.Series):
        return pd.Series(1 / n, index=r.index).where(
            r == 0, r / (1.0 - 1.0 / (1.0 + r) ** n)
        )
    elif r > 0:
        return r / (1.0 - 1.0 / (1.0 + r) ** n)
    else:
        return 1 / n


def update_p_nom_max(n):
    # if extendable carriers (solar/onwind/...) have capacity >= 0,
    # e.g. existing assets from the OPSD project are included to the network,
    # the installed capacity might exceed the expansion limit.
    # Hence, we update the assumptions.

    n.generators.p_nom_max = n.generators[["p_nom_min", "p_nom_max"]].max(1)


def load_costs(tech_costs, config, max_hours, Nyears=1.0):
    # set all asset costs and other parameters
    costs = pd.read_csv(tech_costs, index_col=[0, 1]).sort_index()

    # correct units to MW
    costs.loc[costs.unit.str.contains("/kW"), "value"] *= 1e3
    costs.unit = costs.unit.str.replace("/kW", "/MW")

    fill_values = config["fill_values"]
    costs = costs.value.unstack().fillna(fill_values)

    costs["capital_cost"] = (
        (
            calculate_annuity(costs["lifetime"], costs["discount rate"])
            + costs["FOM"] / 100.0
        )
        * costs["investment"]
        * Nyears
    )
    costs.at["OCGT", "fuel"] = costs.at["gas", "fuel"]
    costs.at["CCGT", "fuel"] = costs.at["gas", "fuel"]

    costs["marginal_cost"] = costs["VOM"] + costs["fuel"] / costs["efficiency"]

    costs = costs.rename(columns={"CO2 intensity": "co2_emissions"})

    costs.at["OCGT", "co2_emissions"] = costs.at["gas", "co2_emissions"]
    costs.at["CCGT", "co2_emissions"] = costs.at["gas", "co2_emissions"]

    costs.at["solar", "capital_cost"] = (
        config["rooftop_share"] * costs.at["solar-rooftop", "capital_cost"]
        + (1 - config["rooftop_share"]) * costs.at["solar-utility", "capital_cost"]
    )

    def costs_for_storage(store, link1, link2=None, max_hours=1.0):
        capital_cost = link1["capital_cost"] + max_hours * store["capital_cost"]
        if link2 is not None:
            capital_cost += link2["capital_cost"]
        return pd.Series(
            dict(capital_cost=capital_cost, marginal_cost=0.0, co2_emissions=0.0)
        )

    costs.loc["battery"] = costs_for_storage(
        costs.loc["battery storage"],
        costs.loc["battery inverter"],
        max_hours=6,
    )
    costs.loc["H2"] = costs_for_storage(
        costs.loc["hydrogen storage underground"],
        costs.loc["fuel cell"],
        costs.loc["electrolysis"],
        max_hours=168,
    )

    for attr in ("marginal_cost", "capital_cost"):
        overwrites = config.get(attr)
        if overwrites is not None:
            overwrites = pd.Series(overwrites)
            costs.loc[overwrites.index, attr] = overwrites

    return costs


def set_scenario_config(snakemake):
    scenario = snakemake.config["run"].get("scenarios", {})
    if scenario.get("enable") and "run" in snakemake.wildcards.keys():
        try:
            with open(scenario["file"], "r") as f:
                scenario_config = yaml.safe_load(f)
        except FileNotFoundError:
            # fallback for mock_snakemake
            script_dir = Path(__file__).parent.resolve()
            root_dir = script_dir.parent
            with open(root_dir / scenario["file"], "r") as f:
                scenario_config = yaml.safe_load(f)
        update_config(snakemake.config, scenario_config[snakemake.wildcards.run])


def check_network_consistency(n):
    """Checks if the network is one connected graph. Returns isolated buses"""

    graph = n.graph()
    connected_components = list(nx.connected_components(graph))

    main_system = max(connected_components, key=len)

    return n.buses.loc[~n.buses.index.isin(main_system)].index.tolist()


def set_nested_attr(obj, attr_path, value):
    """
    Sets a nested attribute on an object given an attribute path.

    Parameters:
    - obj: The object on which to set the attribute.
    - attr_path: A string representing the nested attribute path, e.g., 'a.b.c'.
    - value: The value to set for the nested attribute.
    """

    attrs = attr_path.split('.')

    for attr in attrs[:-1]:
        obj = getattr(obj, attr)

    setattr(obj, attrs[-1], value)


def classify_north_south(lon, lat):
    """Splits GB into north and south, where north represents regions 
    with diminished wholesale market prices"""

    lon = float(lon)
    lat = float(lat)

    m = 0.55
    b = 56.4

    if lat > m * lon + b:
        return 'north'
    else:
        return 'south'


def mock_snakemake(
    rulename,
    root_dir=None,
    configfiles=None,
    submodule_dir="workflow/submodules/pypsa-eur",
    **wildcards,
):
    """
    This function is expected to be executed from the 'scripts'-directory of '
    the snakemake project. It returns a snakemake.script.Snakemake object,
    based on the Snakefile.

    If a rule has wildcards, you have to specify them in **wildcards.

    Parameters
    ----------
    rulename: str
        name of the rule for which the snakemake object should be generated
    root_dir: str/path-like
        path to the root directory of the snakemake project
    configfiles: list, str
        list of configfiles to be used to update the config
    submodule_dir: str, Path
        in case PyPSA-Eur is used as a submodule, submodule_dir is
        the path of pypsa-eur relative to the project directory.
    **wildcards:
        keyword arguments fixing the wildcards. Only necessary if wildcards are
        needed.
    """
    import os

    import snakemake as sm
    from pypsa.descriptors import Dict
    from snakemake.api import Workflow
    from snakemake.common import SNAKEFILE_CHOICES
    from snakemake.script import Snakemake
    from snakemake.settings.types import (
        ConfigSettings,
        DAGSettings,
        ResourceSettings,
        StorageSettings,
        WorkflowSettings,
    )

    script_dir = Path(__file__).parent.resolve()
    if root_dir is None:
        root_dir = script_dir.parent
    else:
        root_dir = Path(root_dir).resolve()

    workdir = None
    user_in_script_dir = Path.cwd().resolve() == script_dir
    if str(submodule_dir) in __file__:
        # the submodule_dir path is only need to locate the project dir
        os.chdir(Path(__file__[: __file__.find(str(submodule_dir))]))
    elif user_in_script_dir:
        os.chdir(root_dir)
    elif Path.cwd().resolve() != root_dir:
        logging.info(
            "Not in scripts or root directory, will assume this is a separate workdir"
        )
        workdir = Path.cwd()

    try:
        for p in SNAKEFILE_CHOICES:
            p = root_dir / p
            if os.path.exists(p):
                snakefile = p
                break
        if configfiles is None:
            configfiles = []
        elif isinstance(configfiles, str):
            configfiles = [configfiles]

        resource_settings = ResourceSettings()
        config_settings = ConfigSettings(configfiles=map(Path, configfiles))
        workflow_settings = WorkflowSettings()
        storage_settings = StorageSettings()
        dag_settings = DAGSettings(rerun_triggers=[])
        workflow = Workflow(
            config_settings,
            resource_settings,
            workflow_settings,
            storage_settings,
            dag_settings,
            storage_provider_settings=dict(),
            overwrite_workdir=workdir,
        )
        workflow.include(snakefile)

        if configfiles:
            for f in configfiles:
                if not os.path.exists(f):
                    raise FileNotFoundError(f"Config file {f} does not exist.")
                workflow.configfile(f)

        workflow.global_resources = {}
        rule = workflow.get_rule(rulename)
        dag = sm.dag.DAG(workflow, rules=[rule])
        wc = Dict(wildcards)
        job = sm.jobs.Job(rule, dag, wc)

        def make_accessable(*ios):
            for io in ios:
                for i, _ in enumerate(io):
                    io[i] = os.path.abspath(io[i])

        make_accessable(job.input, job.output, job.log)
        snakemake = Snakemake(
            job.input,
            job.output,
            job.params,
            job.wildcards,
            job.threads,
            job.resources,
            job.log,
            job.dag.workflow.config,
            job.rule.name,
            None,
        )
        # create log and output dir if not existent
        for path in list(snakemake.log) + list(snakemake.output):
            Path(path).parent.mkdir(parents=True, exist_ok=True)

    finally:
        if user_in_script_dir:
            os.chdir(script_dir)
    return snakemake