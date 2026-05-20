# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Test LODF JAX with InterPSS-imported networks.

Compares JAX LODF results against InterPSS AC N-1 analysis as ground truth.
Both the solver input data and the reference come from the same InterPSS source,
ensuring consistent unit conventions and data mapping.

Supported test cases:
- IEEE 118 bus (IEEE CDF format)
- PSSE Texas 2000 / ACTIVSg2000 (PSSE RAW format)
- PSSE Eastern Interconnect / OpenEI (PSSE RAW format)
"""

import os
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dc_plus.importing.interpss.interpss_import_helpers import (
    build_bus_number_to_index,
    extract_branch_tap_info,
    extract_bus_shunt_info,
    extract_dataframes,
    extract_interpss_jacobian,
    get_slack_bus_id,
    initialize_jvm,
    interpss_n1_analysis,
    load_ieee_cdf,
    load_psse_raw,
    run_aclf,
)
from dc_plus.importing.interpss.interpss_import import (
    _get_buses_interpss,
    _get_branches_parameter_interpss,
    _get_injections_interpss,
    _get_shunts_interpss,
    _get_limits_parameter_interpss,
)
from dc_plus.jax.lodf_branches import line_outage_post_contingency_monitored
from dc_plus.preprocess.create_network_data import _create_network_data
from dc_plus.preprocess.helper_functions import _find_bridges

# Enable 64-bit floats in JAX
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

# Paths — test data lives in testdata/ at the repo root
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TESTDATA_DIR = os.path.join(_PROJECT_ROOT, "testdata")
INTERPSS_CONFIG_PATH = os.path.join(_PROJECT_ROOT, "config", "config.json")
IEEE118_PATH = os.path.join(TESTDATA_DIR, "ieee", "ieee118.ieee")
TEXAS2K_PATH = os.path.join(TESTDATA_DIR, "psse", "ACTIVSg2000.RAW")
OPENEI_PATH = os.path.join(TESTDATA_DIR, "psse", "Base_Eastern_Interconnect_515GW.RAW")


def _ipss_test_data_available():
    return os.path.isfile(INTERPSS_CONFIG_PATH)


skip_if_no_ipss = pytest.mark.skipif(
    not _ipss_test_data_available(),
    reason="InterPSS config not available",
)


def _load_interpss_test_grid(file_path: str, loader: Callable = load_ieee_cdf):
    """Load a network via InterPSS and return (dynamic_info, string_info, jacobian_data, net, bus_id_to_index).

    Also returns the raw AclfNet and bus ID mapping for N-1 analysis.
    """
    initialize_jvm(INTERPSS_CONFIG_PATH)
    net = loader(file_path)
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

    _, dynamic_info, string_info = _create_network_data(buses, branches, injections, limits, shunts)

    # Extract Jacobian directly from InterPSS to avoid recomputation discrepancies
    jac_sub, jac_inv, pvpq, pq = extract_interpss_jacobian(
        net, list(dynamic_info.bus_type), voltage_magnitudes=np.asarray(dynamic_info.bus_voltage_magnitudes)
    )

    # Build JacobianInterface-compatible object for the LODF kernel
    n_pv = dynamic_info.n_pv_buses
    n_pq = dynamic_info.n_pq_buses
    is_angle_component = np.zeros(jac_sub.shape[0], dtype=bool)
    is_angle_component[: (n_pv + n_pq)] = True
    is_magnitude_component = np.zeros(jac_sub.shape[0], dtype=bool)
    is_magnitude_component[(n_pv + n_pq) :] = True

    class _JacData:
        pass

    jacobian_data = _JacData()
    jacobian_data.jacobian = jac_sub
    jacobian_data.inverse_jacobian = jac_inv
    jacobian_data.is_angle_component = is_angle_component
    jacobian_data.is_magnitude_component = is_magnitude_component
    jacobian_data.pvpq_indices = pvpq
    jacobian_data.pq_indices = pq
    jacobian_data.angle_component_indices = np.full(dynamic_info.n_buses, -1, dtype=np.int32)
    jacobian_data.angle_component_indices[pvpq] = np.flatnonzero(is_angle_component)
    jacobian_data.magnitude_component_indices = np.full(dynamic_info.n_buses, -1, dtype=np.int32)
    jacobian_data.magnitude_component_indices[pq] = np.flatnonzero(is_magnitude_component)

    # Build bus ID to index mapping for N-1 result alignment
    bus_id_to_index = {str(bid): idx for idx, bid in enumerate(string_info.bus_ids)}

    return dynamic_info, string_info, jacobian_data, net, bus_id_to_index


def _run_lodf_comparison(dynamic_info, string_info, jacobian_data, net, bus_id_to_index, pass_threshold=0.95):
    """Run JAX LODF kernel and compare against InterPSS N-1 AC analysis.

    Returns the pass rate (fraction of N-1 cases matching within tolerance).
    """
    j_inverse = jacobian_data.inverse_jacobian
    theta_actual = dynamic_info.bus_voltage_angles_rad
    vm_actual = dynamic_info.bus_voltage_magnitudes

    v_mag_hat = np.asarray(vm_actual, dtype=float).reshape(-1)
    theta_hat = np.asarray(theta_actual, dtype=float).reshape(-1)

    jacobian_inv_np = np.asarray(j_inverse)
    jacobian_inv_transposed_np = jacobian_inv_np.T

    branch_from = np.asarray(dynamic_info.branch_from_bus, dtype=np.int32).reshape(-1)
    branch_to = np.asarray(dynamic_info.branch_to_bus, dtype=np.int32).reshape(-1)
    y_ff = np.asarray(dynamic_info.branch_effective_admittance_from_from, dtype=np.complex128).reshape(-1)
    y_ft = np.asarray(dynamic_info.branch_effective_admittance_from_to, dtype=np.complex128).reshape(-1)
    y_tf = np.asarray(dynamic_info.branch_effective_admittance_to_from, dtype=np.complex128).reshape(-1)
    y_tt = np.asarray(dynamic_info.branch_effective_admittance_to_to, dtype=np.complex128).reshape(-1)

    is_bridge = _find_bridges(dynamic_info)
    outage_candidates = np.flatnonzero(~is_bridge)

    v0 = v_mag_hat * np.exp(1j * theta_hat)
    v_from = v0[branch_from]
    v_to = v0[branch_to]

    i_from = y_ff * v_from + y_ft * v_to
    i_to = y_tf * v_from + y_tt * v_to
    s_from = v_from * np.conj(i_from)
    s_to = v_to * np.conj(i_to)

    branch_connected_base = np.asarray(dynamic_info.branch_connected, dtype=bool).reshape(-1)
    s_from = np.where(branch_connected_base, s_from, 0.0 + 0.0j)
    s_to = np.where(branch_connected_base, s_to, 0.0 + 0.0j)

    branch_pq_base = np.empty((branch_from.size, 4), dtype=jacobian_inv_np.dtype)
    branch_pq_base[:, 0] = s_from.real
    branch_pq_base[:, 1] = s_to.real
    branch_pq_base[:, 2] = s_from.imag
    branch_pq_base[:, 3] = s_to.imag

    angle_component_indices = np.asarray(jacobian_data.angle_component_indices, dtype=np.int32)
    magnitude_component_indices = np.asarray(jacobian_data.magnitude_component_indices, dtype=np.int32)

    lf_res = line_outage_post_contingency_monitored(
        jacobian_inv_transposed=jnp.asarray(jacobian_inv_transposed_np),
        outage_branch_idx=jnp.asarray(outage_candidates, dtype=jnp.int32),
        branch_from=jnp.asarray(branch_from, dtype=jnp.int32),
        branch_to=jnp.asarray(branch_to, dtype=jnp.int32),
        v_mag_hat=jnp.asarray(v_mag_hat),
        theta_hat=jnp.asarray(theta_hat),
        angle_component_indices=jnp.asarray(angle_component_indices),
        magnitude_component_indices=jnp.asarray(magnitude_component_indices),
        y_ff=jnp.asarray(y_ff),
        y_ft=jnp.asarray(y_ft),
        y_tf=jnp.asarray(y_tf),
        y_tt=jnp.asarray(y_tt),
        monitor_bus_indices=jnp.arange(v_mag_hat.size, dtype=jnp.int32),
        branch_pq_base=jnp.asarray(branch_pq_base),
        monitor_branch_indices=jnp.arange(branch_from.size, dtype=jnp.int32),
        bus_to_mon_index=jnp.arange(v_mag_hat.size, dtype=jnp.int32),
    )

    # InterPSS N-1 as ground truth
    outage_ids = [str(string_info.branch_ids[i]) for i in outage_candidates]
    n1_results = interpss_n1_analysis(net, outage_ids)

    # Compare
    n_comparisons = 0
    n_passed = 0
    for pos, outage_idx in enumerate(outage_candidates):
        outage_id = str(string_info.branch_ids[outage_idx])

        if outage_id not in n1_results:
            continue
        n1 = n1_results[outage_id]

        # Build InterPSS N-1 voltage vector in index order
        v_mag_n1 = np.zeros(v_mag_hat.size)
        v_ang_n1 = np.zeros(v_mag_hat.size)
        for bus_id, (vm, va) in n1["voltages"].items():
            if bus_id in bus_id_to_index:
                idx = bus_id_to_index[bus_id]
                v_mag_n1[idx] = vm
                v_ang_n1[idx] = va

        v_ref_n1 = v_mag_n1 * np.exp(1j * v_ang_n1)
        lf_res_voltage = np.asarray(lf_res.n_1_voltage[pos]) * np.exp(
            1j * np.asarray(lf_res.n_1_theta[pos])
        )

        n_comparisons += 1
        # Linearized LODF vs full AC: use relaxed tolerance
        if np.allclose(lf_res_voltage, v_ref_n1, atol=1e-4, rtol=5e-3):
            n_passed += 1

    if n_comparisons > 0:
        pass_rate = n_passed / n_comparisons
        assert pass_rate >= pass_threshold, (
            f"Only {n_passed}/{n_comparisons} ({pass_rate:.1%}) N-1 cases passed. "
            f"Expected at least {pass_threshold:.0%}."
        )


@pytest.fixture(scope="module")
def jvm():
    """Start JVM once for the module."""
    if not _ipss_test_data_available():
        pytest.skip("InterPSS config not available")
    initialize_jvm(INTERPSS_CONFIG_PATH)


@skip_if_no_ipss
@pytest.mark.skipif(not os.path.isfile(IEEE118_PATH), reason="IEEE 118 file not found")
def test_lodf_jax_interpss_ieee118(jvm):
    """LODF JAX test for IEEE 118 bus using InterPSS-imported data."""
    dynamic_info, string_info, jacobian_data, net, bus_id_to_index = _load_interpss_test_grid(
        IEEE118_PATH, loader=load_ieee_cdf
    )
    assert jacobian_data.jacobian.shape[0] == dynamic_info.n_pv_buses + 2 * dynamic_info.n_pq_buses
    _run_lodf_comparison(dynamic_info, string_info, jacobian_data, net, bus_id_to_index)


@skip_if_no_ipss
@pytest.mark.skipif(not os.path.isfile(TEXAS2K_PATH), reason="Texas2k PSSE file not found")
def test_lodf_jax_interpss_texas2k(jvm):
    """LODF JAX test for ACTIVSg2000 (Texas 2000) using InterPSS-imported data."""
    dynamic_info, string_info, jacobian_data, net, bus_id_to_index = _load_interpss_test_grid(
        TEXAS2K_PATH, loader=load_psse_raw
    )
    assert jacobian_data.jacobian.shape[0] == dynamic_info.n_pv_buses + 2 * dynamic_info.n_pq_buses
    _run_lodf_comparison(dynamic_info, string_info, jacobian_data, net, bus_id_to_index)


@skip_if_no_ipss
@pytest.mark.skipif(not os.path.isfile(OPENEI_PATH), reason="OpenEI PSSE file not found")
def test_lodf_jax_interpss_openei(jvm):
    """LODF JAX test for Eastern Interconnect (OpenEI) using InterPSS-imported data."""
    dynamic_info, string_info, jacobian_data, net, bus_id_to_index = _load_interpss_test_grid(
        OPENEI_PATH, loader=load_psse_raw
    )
    assert jacobian_data.jacobian.shape[0] == dynamic_info.n_pv_buses + 2 * dynamic_info.n_pq_buses
    _run_lodf_comparison(dynamic_info, string_info, jacobian_data, net, bus_id_to_index)
