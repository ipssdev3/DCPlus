import numpy as np
import pytest

from tests.importing.interpss.conftest import (
    IEEE14_PATH,
    INTERPSS_CONFIG_PATH,
    skip_if_no_ipss,
)


def _load_interpss_network_data(jvm_initialized):
    """Load IEEE 14 via InterPSS and return (buses, branches, injections) DataFrames."""
    from dc_plus.importing.interpss.interpss_import_helpers import (
        load_ieee_cdf,
        run_aclf,
        extract_dataframes,
        extract_branch_tap_info,
        get_slack_bus_id,
        build_bus_number_to_index,
    )
    from dc_plus.importing.interpss.interpss_import import (
        _get_buses_interpss,
        _get_branches_parameter_interpss,
        _get_injections_interpss,
    )

    net = load_ieee_cdf(IEEE14_PATH)
    run_aclf(net)
    dfs = extract_dataframes(net)
    slack_id = get_slack_bus_id(net)
    bus_map = build_bus_number_to_index(dfs["bus"])
    tap_info = extract_branch_tap_info(net)

    buses = _get_buses_interpss(dfs["bus"], slack_id)
    branches = _get_branches_parameter_interpss(dfs["branch"], bus_map, tap_info)
    injections = _get_injections_interpss(dfs["gen"], dfs["load"], dfs["bus"], bus_map)
    return buses, branches, injections


def _load_pypowsybl_network_data():
    """Load IEEE 14 via pypowsybl and return (buses, branches, injections) DataFrames."""
    import pypowsybl
    from dc_plus.importing.powsybl.powsybl_import import (
        _get_buses_powsybl,
        _get_branches_parameter_powsybl,
        _get_injections_powsybl,
    )

    net = pypowsybl.network.create_ieee14()
    net.per_unit = True
    injections = _get_injections_powsybl(net)
    slack_id = net.get_extensions("slackTerminal")["bus_id"].values[0]
    buses = _get_buses_powsybl(net=net, slack_id=slack_id, injections=injections)
    branches = _get_branches_parameter_powsybl(net, split_trafo_charging=True)
    return buses, branches, injections


@skip_if_no_ipss
class TestCrossConsistency:
    @pytest.fixture(scope="class")
    def interpss_data(self, jvm_initialized):
        return _load_interpss_network_data(jvm_initialized)

    @pytest.fixture(scope="class")
    def pypowsybl_data(self):
        return _load_pypowsybl_network_data()

    def test_bus_count_matches_pypowsybl(self, interpss_data, pypowsybl_data):
        ipss_buses, _, _ = interpss_data
        pow_buses, _, _ = pypowsybl_data
        assert len(ipss_buses) == len(pow_buses)

    def test_branch_count_matches_pypowsybl(self, interpss_data, pypowsybl_data):
        _, ipss_branches, _ = interpss_data
        _, pow_branches, _ = pypowsybl_data
        assert len(ipss_branches) == len(pow_branches)

    def test_slack_bus_exists_both(self, interpss_data, pypowsybl_data):
        ipss_buses, _, _ = interpss_data
        pow_buses, _, _ = pypowsybl_data
        assert sum(ipss_buses["bus_type"] == 0) == 1
        assert sum(pow_buses["bus_type"] == 0) == 1

    def test_voltage_magnitudes_close(self, interpss_data, pypowsybl_data):
        ipss_buses, _, _ = interpss_data
        pow_buses, _, _ = pypowsybl_data
        # Both are in per-unit, should agree within solver tolerance
        ipss_v = np.sort(ipss_buses["voltage_magnitude"].values)
        pow_v = np.sort(pow_buses["voltage_magnitude"].values)
        np.testing.assert_allclose(ipss_v, pow_v, atol=2e-3)

    def test_branch_r_x_close(self, interpss_data, pypowsybl_data):
        _, ipss_branches, _ = interpss_data
        _, pow_branches, _ = pypowsybl_data
        # Sort both by r value for comparison
        ipss_r = np.sort(ipss_branches["r"].values)
        pow_r = np.sort(pow_branches["r"].values)
        np.testing.assert_allclose(ipss_r, pow_r, atol=1e-4)

    def test_injection_count_generators(self, interpss_data, pypowsybl_data):
        _, _, ipss_inj = interpss_data
        _, _, pow_inj = pypowsybl_data
        ipss_gens = ipss_inj[ipss_inj["injection_type"] == "GENERATOR"]
        pow_gens = pow_inj[pow_inj["injection_type"].isin(["GENERATOR", "BATTERY"])]
        assert len(ipss_gens) == len(pow_gens)
