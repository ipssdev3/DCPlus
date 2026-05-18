import jpype
import numpy as np
import pytest

from tests.importing.interpss.conftest import (
    IEEE14_PATH,
    INTERPSS_CONFIG_PATH,
    skip_if_no_ipss,
)

EXPECTED_BUS_COLUMNS = [
    "ID",
    "Number",
    "Name",
    "AreaName",
    "AreaNum",
    "ZoneName",
    "ZoneNum",
    "OwnerName",
    "OwnerNum",
    "InService",
    "BusType",
    "NomVolt",
    "VoltMag",
    "VoltAng",
    "GenP",
    "GenQ",
    "GenQMax",
    "GenQMin",
    "LoadP",
    "LoadQ",
    "AdjustableShuntB",
    "AdjustableShuntBMax",
    "AdjustableShuntBMin",
    "BusInjP",
    "BusInjQ",
]

EXPECTED_BRANCH_COLUMNS = [
    "ID",
    "Name",
    "Circuit",
    "Status",
    "FromBusID",
    "FromBusNumber",
    "FromBusName",
    "ToBusID",
    "ToBusNumber",
    "ToBusName",
    "InService",
    "BranchCode",
    "IsXfmr",
    "R",
    "X",
    "B",
    "LimMvaA",
    "LimMvaB",
    "LimMvaC",
    "PFrom2To",
    "QFrom2To",
    "PTo2From",
    "QTo2From",
    "Flow@FromSide",
    "Loading%",
]


@skip_if_no_ipss
class TestJvmLifecycle:
    def test_jvm_starts_successfully(self, jvm_initialized):
        assert jpype.isJVMStarted()

    def test_jvm_already_started_is_noop(self, jvm_initialized):
        from dc_plus.importing.interpss.interpss_import_helpers import initialize_jvm

        # Second call should not raise
        initialize_jvm(INTERPSS_CONFIG_PATH)
        assert jpype.isJVMStarted()


@skip_if_no_ipss
class TestLoadAclfNet:
    def test_load_ieee14_returns_aclf_net(self, jvm_initialized):
        from dc_plus.importing.interpss.interpss_import_helpers import load_ieee_cdf

        net = load_ieee_cdf(IEEE14_PATH)
        assert net is not None
        assert net.getNoActiveBus() == 14

    def test_load_ieee14_bus_count(self, jvm_initialized):
        from dc_plus.importing.interpss.interpss_import_helpers import load_ieee_cdf

        net = load_ieee_cdf(IEEE14_PATH)
        assert net.getNoActiveBus() == 14

    def test_load_ieee14_branch_count(self, jvm_initialized):
        from dc_plus.importing.interpss.interpss_import_helpers import load_ieee_cdf

        net = load_ieee_cdf(IEEE14_PATH)
        assert net.getNoActiveBranch() == 20


@skip_if_no_ipss
class TestExtractDataframes:
    def test_extract_dataframes_returns_all_four(self, ieee14_dfs):
        assert "bus" in ieee14_dfs
        assert "gen" in ieee14_dfs
        assert "load" in ieee14_dfs
        assert "branch" in ieee14_dfs

    def test_bus_df_non_empty(self, ieee14_dfs):
        assert len(ieee14_dfs["bus"]) == 14

    def test_branch_df_non_empty(self, ieee14_dfs):
        assert len(ieee14_dfs["branch"]) == 20

    def test_gen_df_non_empty(self, ieee14_dfs):
        assert len(ieee14_dfs["gen"]) >= 1

    def test_load_df_non_empty(self, ieee14_dfs):
        assert len(ieee14_dfs["load"]) >= 1

    def test_bus_df_columns_match_expected(self, ieee14_dfs):
        assert list(ieee14_dfs["bus"].columns) == EXPECTED_BUS_COLUMNS

    def test_branch_df_columns_match_expected(self, ieee14_dfs):
        assert list(ieee14_dfs["branch"].columns) == EXPECTED_BRANCH_COLUMNS


@skip_if_no_ipss
class TestRunAclf:
    def test_loadflow_converges(self, ieee14_net_solved):
        net = ieee14_net_solved
        assert net.isLfConverged()

    def test_bus_voltages_match_reference(self, ieee14_dfs, ieee14_ref_bus):
        bus_df = ieee14_dfs["bus"]
        # Compare voltage magnitudes for first few buses
        for i in range(min(5, len(bus_df))):
            ref_vmag = ieee14_ref_bus.iloc[i]["VoltMag"]
            actual_vmag = bus_df.iloc[i]["VoltMag"]
            np.testing.assert_allclose(actual_vmag, ref_vmag, atol=1e-6)
