import numpy as np
import pytest

from dc_plus.importing.import_schema import BranchParamSchema
from tests.importing.interpss.conftest import skip_if_no_ipss


@skip_if_no_ipss
class TestBranchExtraction:
    def test_branch_count_ieee14(self, ieee14_dfs):
        from dc_plus.importing.interpss.interpss_import import _get_branches_parameter_interpss
        from dc_plus.importing.interpss.interpss_import_helpers import build_bus_number_to_index

        bus_map = build_bus_number_to_index(ieee14_dfs["bus"])
        branches = _get_branches_parameter_interpss(ieee14_dfs["branch"], bus_map)
        assert len(branches) == 20

    def test_branch_schema_validates(self, ieee14_dfs):
        from dc_plus.importing.interpss.interpss_import import _get_branches_parameter_interpss
        from dc_plus.importing.interpss.interpss_import_helpers import build_bus_number_to_index

        bus_map = build_bus_number_to_index(ieee14_dfs["bus"])
        branches = _get_branches_parameter_interpss(ieee14_dfs["branch"], bus_map)
        BranchParamSchema.validate(branches)

    def test_line_branch_type(self, ieee14_dfs):
        from dc_plus.importing.interpss.interpss_import import _get_branches_parameter_interpss
        from dc_plus.importing.interpss.interpss_import_helpers import build_bus_number_to_index

        bus_map = build_bus_number_to_index(ieee14_dfs["bus"])
        branches = _get_branches_parameter_interpss(ieee14_dfs["branch"], bus_map)
        lines = branches[branches["branch_type"] == "LINE"]
        assert len(lines) > 0

    def test_transformer_branch_type(self, ieee14_dfs):
        from dc_plus.importing.interpss.interpss_import import _get_branches_parameter_interpss
        from dc_plus.importing.interpss.interpss_import_helpers import build_bus_number_to_index

        bus_map = build_bus_number_to_index(ieee14_dfs["bus"])
        branches = _get_branches_parameter_interpss(ieee14_dfs["branch"], bus_map)
        trafos = branches[branches["branch_type"] == "TWO_WINDINGS_TRANSFORMER"]
        assert len(trafos) > 0

    def test_all_branches_connected(self, ieee14_dfs):
        from dc_plus.importing.interpss.interpss_import import _get_branches_parameter_interpss
        from dc_plus.importing.interpss.interpss_import_helpers import build_bus_number_to_index

        bus_map = build_bus_number_to_index(ieee14_dfs["bus"])
        branches = _get_branches_parameter_interpss(ieee14_dfs["branch"], bus_map)
        assert all(branches["connected"])

    def test_line_r_x_values(self, ieee14_dfs):
        from dc_plus.importing.interpss.interpss_import import _get_branches_parameter_interpss
        from dc_plus.importing.interpss.interpss_import_helpers import build_bus_number_to_index

        bus_map = build_bus_number_to_index(ieee14_dfs["bus"])
        branches = _get_branches_parameter_interpss(ieee14_dfs["branch"], bus_map)
        # First branch Bus1->Bus2: r=0.01938, x=0.05917
        first = branches.iloc[0]
        np.testing.assert_allclose(first["r"], 0.01938, atol=1e-5)
        np.testing.assert_allclose(first["x"], 0.05917, atol=1e-5)

    def test_line_charging_split(self, ieee14_dfs):
        from dc_plus.importing.interpss.interpss_import import _get_branches_parameter_interpss
        from dc_plus.importing.interpss.interpss_import_helpers import build_bus_number_to_index

        bus_map = build_bus_number_to_index(ieee14_dfs["bus"])
        branches = _get_branches_parameter_interpss(ieee14_dfs["branch"], bus_map)
        # For lines, g1=g2=0.0, b1=b2=B/2
        lines = branches[branches["branch_type"] == "LINE"]
        for _, row in lines.iterrows():
            np.testing.assert_allclose(row["g1"], 0.0, atol=1e-10)
            np.testing.assert_allclose(row["g2"], 0.0, atol=1e-10)

    def test_line_rho_one(self, ieee14_dfs):
        from dc_plus.importing.interpss.interpss_import import _get_branches_parameter_interpss
        from dc_plus.importing.interpss.interpss_import_helpers import build_bus_number_to_index

        bus_map = build_bus_number_to_index(ieee14_dfs["bus"])
        branches = _get_branches_parameter_interpss(ieee14_dfs["branch"], bus_map)
        lines = branches[branches["branch_type"] == "LINE"]
        np.testing.assert_allclose(lines["rho"].values, 1.0, atol=1e-10)

    def test_line_alpha_zero(self, ieee14_dfs):
        from dc_plus.importing.interpss.interpss_import import _get_branches_parameter_interpss
        from dc_plus.importing.interpss.interpss_import_helpers import build_bus_number_to_index

        bus_map = build_bus_number_to_index(ieee14_dfs["bus"])
        branches = _get_branches_parameter_interpss(ieee14_dfs["branch"], bus_map)
        np.testing.assert_allclose(branches["alpha"].values, 0.0, atol=1e-10)

    def test_from_to_bus_indices_valid(self, ieee14_dfs):
        from dc_plus.importing.interpss.interpss_import import _get_branches_parameter_interpss
        from dc_plus.importing.interpss.interpss_import_helpers import build_bus_number_to_index

        bus_map = build_bus_number_to_index(ieee14_dfs["bus"])
        branches = _get_branches_parameter_interpss(ieee14_dfs["branch"], bus_map)
        n_buses = len(ieee14_dfs["bus"])
        assert all(branches["from_bus_index"] >= 0)
        assert all(branches["from_bus_index"] < n_buses)
        assert all(branches["to_bus_index"] >= 0)
        assert all(branches["to_bus_index"] < n_buses)

    def test_branch_flows_after_loadflow(self, ieee14_dfs, ieee14_ref_branch):
        from dc_plus.importing.interpss.interpss_import import _get_branches_parameter_interpss
        from dc_plus.importing.interpss.interpss_import_helpers import build_bus_number_to_index

        bus_map = build_bus_number_to_index(ieee14_dfs["bus"])
        branches = _get_branches_parameter_interpss(ieee14_dfs["branch"], bus_map)
        # Compare first branch PFrom2To with reference
        ref_p1 = float(ieee14_ref_branch.iloc[0]["PFrom2To"])
        np.testing.assert_allclose(branches.iloc[0]["p1"], ref_p1, atol=1e-4)
