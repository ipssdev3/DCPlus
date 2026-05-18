import numpy as np
import pytest

from dc_plus.importing.import_schema import BusParamSchema
from tests.importing.interpss.conftest import skip_if_no_ipss


@skip_if_no_ipss
class TestBusExtraction:
    def test_bus_count_ieee14(self, ieee14_dfs):
        from dc_plus.importing.interpss.interpss_import import _get_buses_interpss

        buses = _get_buses_interpss(ieee14_dfs["bus"], slack_bus_id="Bus1")
        assert len(buses) == 14

    def test_bus_schema_validates(self, ieee14_dfs):
        from dc_plus.importing.interpss.interpss_import import _get_buses_interpss

        buses = _get_buses_interpss(ieee14_dfs["bus"], slack_bus_id="Bus1")
        BusParamSchema.validate(buses)

    def test_bus_id_int_sequential(self, ieee14_dfs):
        from dc_plus.importing.interpss.interpss_import import _get_buses_interpss

        buses = _get_buses_interpss(ieee14_dfs["bus"], slack_bus_id="Bus1")
        np.testing.assert_array_equal(buses["id_int"].values, range(14))

    def test_bus_id_str_format(self, ieee14_dfs):
        from dc_plus.importing.interpss.interpss_import import _get_buses_interpss

        buses = _get_buses_interpss(ieee14_dfs["bus"], slack_bus_id="Bus1")
        expected = [f"Bus{i}" for i in range(1, 15)]
        assert list(buses["id_str"].values) == expected

    def test_slack_bus_type_zero(self, ieee14_dfs):
        from dc_plus.importing.interpss.interpss_import import _get_buses_interpss

        buses = _get_buses_interpss(ieee14_dfs["bus"], slack_bus_id="Bus1")
        slack = buses[buses["id_str"] == "Bus1"]
        assert slack["bus_type"].values[0] == 0

    def test_pv_bus_type_one(self, ieee14_dfs):
        from dc_plus.importing.interpss.interpss_import import _get_buses_interpss

        buses = _get_buses_interpss(ieee14_dfs["bus"], slack_bus_id="Bus1")
        pv_ids = {"Bus2", "Bus3", "Bus6", "Bus8"}
        pv_buses = buses[buses["id_str"].isin(pv_ids)]
        assert all(pv_buses["bus_type"] == 1)

    def test_pq_bus_type_two(self, ieee14_dfs):
        from dc_plus.importing.interpss.interpss_import import _get_buses_interpss

        buses = _get_buses_interpss(ieee14_dfs["bus"], slack_bus_id="Bus1")
        pv_slack = {"Bus1", "Bus2", "Bus3", "Bus6", "Bus8"}
        pq_buses = buses[~buses["id_str"].isin(pv_slack)]
        assert all(pq_buses["bus_type"] == 2)

    def test_slack_voltage_magnitude(self, ieee14_dfs):
        from dc_plus.importing.interpss.interpss_import import _get_buses_interpss

        buses = _get_buses_interpss(ieee14_dfs["bus"], slack_bus_id="Bus1")
        slack = buses[buses["id_str"] == "Bus1"]
        np.testing.assert_allclose(slack["voltage_magnitude"].values[0], 1.06, atol=1e-6)

    def test_slack_voltage_angle_zero(self, ieee14_dfs):
        from dc_plus.importing.interpss.interpss_import import _get_buses_interpss

        buses = _get_buses_interpss(ieee14_dfs["bus"], slack_bus_id="Bus1")
        slack = buses[buses["id_str"] == "Bus1"]
        np.testing.assert_allclose(slack["voltage_angle"].values[0], 0.0, atol=1e-6)

    def test_pv_voltage_angle_nonzero(self, ieee14_dfs):
        from dc_plus.importing.interpss.interpss_import import _get_buses_interpss

        buses = _get_buses_interpss(ieee14_dfs["bus"], slack_bus_id="Bus1")
        bus2 = buses[buses["id_str"] == "Bus2"]
        np.testing.assert_allclose(bus2["voltage_angle"].values[0], -0.087, atol=0.001)

    def test_grid_island_id_all_zero(self, ieee14_dfs):
        from dc_plus.importing.interpss.interpss_import import _get_buses_interpss

        buses = _get_buses_interpss(ieee14_dfs["bus"], slack_bus_id="Bus1")
        assert all(buses["grid_island_id"] == 0)
