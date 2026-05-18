import math

import numpy as np
import pytest

from tests.importing.interpss.conftest import (
    IEEE14_PATH,
    INTERPSS_CONFIG_PATH,
    skip_if_no_ipss,
)


@skip_if_no_ipss
class TestEndToEnd:
    @pytest.fixture(scope="class")
    def network_data(self, jvm_initialized):
        from dc_plus.importing.interpss.interpss_import_helpers import (
            initialize_jvm,
            load_ieee_cdf,
            run_aclf,
            extract_dataframes,
            extract_branch_tap_info,
            extract_bus_shunt_info,
            get_slack_bus_id,
            build_bus_number_to_index,
        )
        from dc_plus.importing.interpss.interpss_import import (
            _get_buses_interpss,
            _get_branches_parameter_interpss,
            _get_injections_interpss,
            _get_shunts_interpss,
            _get_limits_parameter_interpss,
        )
        from dc_plus.preprocess.create_network_data import _create_network_data

        net = load_ieee_cdf(IEEE14_PATH)
        run_aclf(net)
        dfs = extract_dataframes(net)

        slack_id = get_slack_bus_id(net)
        bus_map = build_bus_number_to_index(dfs["bus"])
        tap_info = extract_branch_tap_info(net)
        shunt_info = extract_bus_shunt_info(net)

        buses = _get_buses_interpss(dfs["bus"], slack_id)
        branches = _get_branches_parameter_interpss(dfs["branch"], bus_map, tap_info)
        injections = _get_injections_interpss(dfs["gen"], dfs["load"], dfs["bus"], bus_map)
        shunts = _get_shunts_interpss(dfs["bus"], bus_map, shunt_info)
        limits = _get_limits_parameter_interpss(dfs["branch"])

        static, dynamic, string = _create_network_data(buses, branches, injections, limits, shunts)
        return static, dynamic, string

    def test_returns_tuple(self, network_data):
        static, dynamic, string = network_data
        assert static is not None
        assert dynamic is not None
        assert string is not None

    def test_bus_count_matches(self, network_data):
        _, dynamic, _ = network_data
        assert dynamic.n_buses == 14

    def test_branch_count_matches(self, network_data):
        _, dynamic, _ = network_data
        assert dynamic.n_branches == 20

    def test_voltages_reasonable(self, network_data):
        _, dynamic, _ = network_data
        v = dynamic.bus_voltage_magnitudes
        assert all(v > 0.9)
        assert all(v < 1.2)

    def test_angles_reasonable(self, network_data):
        _, dynamic, _ = network_data
        angles = dynamic.bus_voltage_angles_rad
        assert all(np.abs(angles) < math.pi / 2)

    def test_no_nan_in_admittances(self, network_data):
        _, dynamic, _ = network_data
        for arr in [dynamic.branch_effective_admittance_from_to,
                     dynamic.branch_effective_admittance_from_from,
                     dynamic.branch_effective_admittance_to_to,
                     dynamic.branch_effective_admittance_to_from]:
            assert all(np.isfinite(arr))

    def test_one_slack_bus(self, network_data):
        _, dynamic, _ = network_data
        slack_count = sum(dynamic.bus_type == 0)
        assert slack_count == 1

    def test_branch_types_populated(self, network_data):
        _, _, string = network_data
        assert len(string.branch_types) == 20
        assert "LINE" in string.branch_types
        assert "TWO_WINDINGS_TRANSFORMER" in string.branch_types

    def test_injection_types_populated(self, network_data):
        _, _, string = network_data
        assert "GENERATOR" in string.injection_types
        assert "LOAD" in string.injection_types
