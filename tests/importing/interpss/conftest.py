import os

import pandas as pd
import pytest

# Paths to ipss-agent resources
IPSS_AGENT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "ipss-agent"))
IEEE14_PATH = os.path.join(IPSS_AGENT_ROOT, "wspace", "data", "ieee", "Ieee14Bus", "ieee14.ieee")
IEEE118_PATH = os.path.join(IPSS_AGENT_ROOT, "wspace", "data", "ieee", "Ieee118Bus", "ieee118.ieee")
IEEE14_REF_DIR = os.path.join(IPSS_AGENT_ROOT, "wspace", "data", "ieee", "Ieee14Bus", "result")

INTERPSS_CONFIG_PATH = os.path.join(IPSS_AGENT_ROOT, "config", "config.json")


def _ipss_agent_available():
    return os.path.isfile(INTERPSS_CONFIG_PATH) and os.path.isfile(IEEE14_PATH)


skip_if_no_ipss = pytest.mark.skipif(
    not _ipss_agent_available(),
    reason="ipss-agent not found or InterPSS test data not available",
)


@pytest.fixture(scope="session")
def jvm_initialized():
    """Start JVM once per test session using ipss-agent config."""
    import jpype

    if jpype.isJVMStarted():
        yield
        return

    from dc_plus.importing.interpss.interpss_import_helpers import initialize_jvm

    initialize_jvm(INTERPSS_CONFIG_PATH)
    yield


@pytest.fixture(scope="session")
def ieee14_net(jvm_initialized):
    """Load IEEE 14-bus AclfNet from CDF file."""
    from dc_plus.importing.interpss.interpss_import_helpers import load_ieee_cdf

    return load_ieee_cdf(IEEE14_PATH)


@pytest.fixture(scope="session")
def ieee14_net_solved(ieee14_net):
    """Run AC loadflow on IEEE 14 and return solved net."""
    from dc_plus.importing.interpss.interpss_import_helpers import run_aclf

    run_aclf(ieee14_net)
    return ieee14_net


@pytest.fixture(scope="session")
def ieee14_dfs(ieee14_net_solved):
    """Return dict of {bus, gen, load, branch} DataFrames from solved net."""
    from dc_plus.importing.interpss.interpss_import_helpers import extract_dataframes

    return extract_dataframes(ieee14_net_solved)


@pytest.fixture(scope="session")
def ieee14_tap_info(ieee14_net_solved):
    """Return branch tap ratio info for IEEE 14."""
    from dc_plus.importing.interpss.interpss_import_helpers import extract_branch_tap_info

    return extract_branch_tap_info(ieee14_net_solved)


@pytest.fixture(scope="session")
def ieee14_ref_bus():
    """Load reference bus CSV from ipss-agent results."""
    path = os.path.join(IEEE14_REF_DIR, "ieee14_DF_bus.csv")
    return pd.read_csv(path)


@pytest.fixture(scope="session")
def ieee14_ref_branch():
    """Load reference branch CSV from ipss-agent results."""
    path = os.path.join(IEEE14_REF_DIR, "ieee14_DF_branch.csv")
    return pd.read_csv(path)


@pytest.fixture(scope="session")
def ieee14_ref_gen():
    """Load reference gen CSV from ipss-agent results."""
    path = os.path.join(IEEE14_REF_DIR, "ieee14_DF_gen.csv")
    return pd.read_csv(path)


@pytest.fixture(scope="session")
def ieee14_ref_load():
    """Load reference load CSV from ipss-agent results."""
    path = os.path.join(IEEE14_REF_DIR, "ieee14_DF_load.csv")
    return pd.read_csv(path)
