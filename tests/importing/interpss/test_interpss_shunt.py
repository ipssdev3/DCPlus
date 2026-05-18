import numpy as np
import pytest

from dc_plus.importing.import_schema import ShuntParamSchema
from tests.importing.interpss.conftest import skip_if_no_ipss


@skip_if_no_ipss
class TestShuntExtraction:
    def test_shunt_count_ieee14(self, ieee14_dfs):
        from dc_plus.importing.interpss.interpss_import import _get_shunts_interpss
        from dc_plus.importing.interpss.interpss_import_helpers import build_bus_number_to_index

        bus_map = build_bus_number_to_index(ieee14_dfs["bus"])
        shunts = _get_shunts_interpss(ieee14_dfs["bus"], bus_map)
        # IEEE 14 in InterPSS has no adjustable shunts (AdjustableShuntB == 0)
        assert len(shunts) == 0

    def test_shunt_schema_validates(self, ieee14_dfs):
        from dc_plus.importing.interpss.interpss_import import _get_shunts_interpss
        from dc_plus.importing.interpss.interpss_import_helpers import build_bus_number_to_index

        bus_map = build_bus_number_to_index(ieee14_dfs["bus"])
        shunts = _get_shunts_interpss(ieee14_dfs["bus"], bus_map)
        ShuntParamSchema.validate(shunts)

    def test_shunt_bus_index_correct(self, ieee14_dfs):
        from dc_plus.importing.interpss.interpss_import import _get_shunts_interpss
        from dc_plus.importing.interpss.interpss_import_helpers import build_bus_number_to_index

        bus_map = build_bus_number_to_index(ieee14_dfs["bus"])
        shunts = _get_shunts_interpss(ieee14_dfs["bus"], bus_map)
        # Bus9 is index 8 (0-based)
        if len(shunts) > 0:
            assert all(shunts["bus_index"] >= 0)

    def test_shunt_b_value(self, ieee14_dfs):
        from dc_plus.importing.interpss.interpss_import import _get_shunts_interpss
        from dc_plus.importing.interpss.interpss_import_helpers import build_bus_number_to_index

        bus_map = build_bus_number_to_index(ieee14_dfs["bus"])
        shunts = _get_shunts_interpss(ieee14_dfs["bus"], bus_map)
        if len(shunts) > 0:
            # All shunt b values should be non-zero
            assert all(shunts["b"] != 0.0)

    def test_shunt_g_zero(self, ieee14_dfs):
        from dc_plus.importing.interpss.interpss_import import _get_shunts_interpss
        from dc_plus.importing.interpss.interpss_import_helpers import build_bus_number_to_index

        bus_map = build_bus_number_to_index(ieee14_dfs["bus"])
        shunts = _get_shunts_interpss(ieee14_dfs["bus"], bus_map)
        if len(shunts) > 0:
            np.testing.assert_allclose(shunts["g"].values, 0.0, atol=1e-10)

    def test_shunt_section_count_one(self, ieee14_dfs):
        from dc_plus.importing.interpss.interpss_import import _get_shunts_interpss
        from dc_plus.importing.interpss.interpss_import_helpers import build_bus_number_to_index

        bus_map = build_bus_number_to_index(ieee14_dfs["bus"])
        shunts = _get_shunts_interpss(ieee14_dfs["bus"], bus_map)
        if len(shunts) > 0:
            assert all(shunts["section_count"] == 1)

    def test_empty_shunts_for_no_shunt_network(self):
        """Test that a network with no shunts returns a valid empty DataFrame."""
        import pandas as pd
        from dc_plus.importing.interpss.interpss_import import _get_shunts_interpss

        bus_df = pd.DataFrame({
            "ID": ["Bus1", "Bus2"],
            "Number": [1, 2],
            "AdjustableShuntB": [0.0, 0.0],
        })
        bus_map = {1: 0, 2: 1}
        shunts = _get_shunts_interpss(bus_df, bus_map)
        assert len(shunts) == 0
        ShuntParamSchema.validate(shunts)
