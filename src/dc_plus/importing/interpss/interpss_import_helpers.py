# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""InterPSS import helpers: JVM lifecycle, AclfNet loading, DataFrame extraction."""

import glob
import json
import logging
import os
import re
from pathlib import Path
from typing import Any

import jpype
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import inv as sparse_inv

logger = logging.getLogger(__name__)


def initialize_jvm(config_path: str | None = None) -> None:
    """Initialize the JVM for InterPSS if not already started.

    Parameters
    ----------
    config_path : str or None
        Path to a JSON config file with 'jvm_path', 'jar_path', and optionally
        'log_config_path'. If None, falls back to the IPSS_CONFIG_PATH env var,
        then to ``<ipss-agent>/config/config.json``.
    """
    if jpype.isJVMStarted():
        return

    config = _load_config(config_path)

    jvm_path = config["jvm_path"]
    jar_path = config["jar_path"]
    log_config = config.get("log_config_path")

    jvm_args = [jvm_path, "-ea", f"-Djava.class.path={jar_path}"]
    if log_config:
        jvm_args.append(f"-Dlog4j.configurationFile={log_config}")

    jpype.startJVM(*jvm_args)


def load_ieee_cdf(file_path: str) -> Any:
    """Load an IEEE CDF file into an InterPSS AclfNet.

    Must be called after ``initialize_jvm()``.
    """
    _ensure_jvm()
    ODMAclfParserMapper = jpype.JClass("org.interpss.odm.mapper.ODMAclfParserMapper")
    IeeeCDFAdapter = jpype.JClass("org.ieee.odm.adapter.ieeecdf.IeeeCDFAdapter")
    IEEECDFVersion = jpype.JClass("org.ieee.odm.adapter.ieeecdf.IeeeCDFAdapter$IEEECDFVersion")

    adapter = IeeeCDFAdapter(IEEECDFVersion.Default)
    adapter.parseInputFile(file_path)
    return ODMAclfParserMapper().map2Model(adapter.getModel()).getAclfNet()


def load_psse_raw(file_path: str) -> Any:
    """Load a PSSE RAW file into an InterPSS AclfNet.

    Must be called after ``initialize_jvm()``.
    """
    _ensure_jvm()
    IpssAdapter = jpype.JClass("org.interpss.plugin.pssl.plugin.IpssAdapter")
    psseVersion = IpssAdapter.parsePsseVersion(file_path)
    return (
        IpssAdapter.importAclfNet(file_path)
        .setFormat(IpssAdapter.FileFormat.PSSE)
        .setPsseVersion(psseVersion)
        .load()
        .getImportedObj()
    )


def run_aclf(net: Any, config_path: str | None = None) -> None:
    """Run AC loadflow on an InterPSS AclfNet.

    Parameters
    ----------
    net : AclfNet
        The InterPSS network object (modified in place).
    config_path : str or None
        Path to an ``aclf_run.json`` config. Uses defaults if None.
    """
    _ensure_jvm()
    LoadflowAlgoObjectFactory = jpype.JClass("com.interpss.core.LoadflowAlgoObjectFactory")
    AclfRunConfigRec = jpype.JClass("org.interpss.plugin.aclf.config.AclfRunConfigRec")

    algo = LoadflowAlgoObjectFactory.createLoadflowAlgorithm(net)

    if config_path and os.path.isfile(config_path):
        aclf_config = AclfRunConfigRec.loadAclfRunConfig(config_path)
        aclf_config.configAclfRun(algo, aclf_config.polarCoordinate, aclf_config.includeAdjustments, False)
    else:
        algo.setLfMethod(jpype.JClass("com.interpss.core.algo.AclfMethodType").NR)
        algo.getLfAdjAlgo().setApplyAdjustAlgo(False)

    algo.loadflow()


def extract_dataframes(net: Any) -> dict[str, pd.DataFrame]:
    """Extract bus, gen, load, and branch DataFrames from a solved AclfNet.

    Parameters
    ----------
    net : AclfNet
        A solved InterPSS network.

    Returns
    -------
    dict[str, pd.DataFrame]
        Keys: 'bus', 'gen', 'load', 'branch'.
    """
    _ensure_jvm()
    AclfNetDFrameAdapter = jpype.JClass("org.interpss.plugin.result.dframe.AclfNetDFrameAdapter")

    dfAdapter = AclfNetDFrameAdapter()
    dfAdapter.adapt(net)

    return {
        "bus": _dframe_to_pandas(dfAdapter.getDfBus()),
        "gen": _dframe_to_pandas(dfAdapter.getDfGen()),
        "load": _dframe_to_pandas(dfAdapter.getDfLoad()),
        "branch": _dframe_to_pandas(dfAdapter.getDfBranch()),
    }


def get_base_mva(net: Any) -> float:
    """Get the system base MVA from an AclfNet."""
    return float(net.getBaseKva()) / 1000.0


def get_slack_bus_id(net: Any) -> str:
    """Get the slack bus ID string from an AclfNet."""
    _ensure_jvm()
    bus_iter = net.getBusList().iterator()
    while bus_iter.hasNext():
        bus = bus_iter.next()
        if bus.isSwing():
            return str(bus.getId())
    raise ValueError("No slack bus found in network")


def build_bus_number_to_index(bus_df: pd.DataFrame) -> dict[int, int]:
    """Build a mapping from InterPSS bus Number to 0-based index.

    Assumes bus_df is sorted by Number (ascending).
    """
    return {int(row["Number"]): idx for idx, row in bus_df.iterrows()}


def extract_branch_tap_info(net: Any) -> dict[str, tuple[float, float]]:
    """Extract tap ratio (rho) and phase shift (alpha) for each branch.

    Returns dict mapping branch ID to (rho, alpha).
    For non-transformer branches, returns (1.0, 0.0).
    """
    _ensure_jvm()
    tap_info: dict[str, tuple[float, float]] = {}
    branch_iter = net.getBranchList().iterator()
    while branch_iter.hasNext():
        branch = branch_iter.next()
        branch_id = str(branch.getId())
        if branch.isXfr() or branch.isPSXfr():
            rho = float(branch.getFromTurnRatio())
            alpha = 0.0
            if branch.isPSXfr():
                alpha = float(branch.getFromPSXfrAngle())
            tap_info[branch_id] = (rho, alpha)
        else:
            tap_info[branch_id] = (1.0, 0.0)
    return tap_info


def extract_bus_shunt_info(net: Any) -> dict[str, tuple[float, float]]:
    """Extract fixed shunt admittance (g, b) for each bus.

    InterPSS stores bus shunts via ``bus.getShuntY()``, which includes both
    fixed shunts and any capacitor/reactor data from the CDF file. The
    DataFrame adapter's ``AdjustableShuntB`` column only covers adjustable
    shunts, so this helper is needed to capture all shunt contributions.

    Returns dict mapping bus ID to (g, b).
    """
    _ensure_jvm()
    shunt_info: dict[str, tuple[float, float]] = {}
    bus_iter = net.getBusList().iterator()
    while bus_iter.hasNext():
        bus = bus_iter.next()
        sy = bus.getShuntY()
        g = float(sy.getReal())
        b = float(sy.getImaginary())
        if abs(g) > 1e-12 or abs(b) > 1e-12:
            shunt_info[str(bus.getId())] = (g, b)
    return shunt_info


def extract_interpss_jacobian(
    net: Any,
    bus_types: list[int],
    voltage_magnitudes: np.ndarray | None = None,
) -> tuple:
    """Extract the Jacobian matrix directly from InterPSS.

    Uses ``net.formJMatrix()`` to get the exact Jacobian that InterPSS's
    internal NR solver uses, avoiding discrepancies from recomputing it
    in Python.

    InterPSS uses the normalized Jacobian formulation where voltage magnitude
    columns are scaled by |V| (i.e., V*∂P/∂|V| instead of ∂P/∂|V|). This
    function converts to the un-normalized ∂P/∂|V| form expected by DCPlus.

    Parameters
    ----------
    net : AclfNet
        A solved InterPSS network.
    bus_types : list[int]
        Bus type array (0=SLACK, 1=PV, 2=PQ) for each bus in order.
    voltage_magnitudes : np.ndarray or None
        Bus voltage magnitudes (per-unit). Required for converting the
        normalized Jacobian to the standard form.

    Returns
    -------
    tuple[jacobian, inverse_jacobian, pvpq_indices, pq_indices]
        jacobian: sparse CSR Jacobian in DCPlus ordering
        inverse_jacobian: dense inverse Jacobian
        pvpq_indices: indices of PV+PQ buses
        pq_indices: indices of PQ buses
    """
    _ensure_jvm()

    j_obj = net.formJMatrix()
    dim = j_obj.getDimension()

    # Extract full InterPSS Jacobian (2N x 2N, interleaved P/Q rows, θ/|V| cols)
    jac_full = np.zeros((dim, dim))
    for i in range(dim):
        for j in range(dim):
            jac_full[i, j] = j_obj.getAij(i, j)

    # Map to DCPlus ordering: [P@PV, P@PQ, Q@PQ] x [θ@PV+PQ, |V|@PQ]
    bus_types_arr = np.asarray(bus_types)
    pv_indices = np.where(bus_types_arr == 1)[0]
    pq_indices = np.where(bus_types_arr == 2)[0]
    pvpq = np.concatenate([pv_indices, pq_indices])

    # P equation rows for PV+PQ, Q equation rows for PQ only
    p_rows = 2 * pvpq
    q_rows = 2 * pq_indices + 1
    all_rows = np.concatenate([p_rows, q_rows])

    # θ columns for PV+PQ, |V| columns for PQ only
    theta_cols = 2 * pvpq
    vmag_cols = 2 * pq_indices + 1
    all_cols = np.concatenate([theta_cols, vmag_cols])

    jac_sub = jac_full[np.ix_(all_rows, all_cols)]

    # Convert from InterPSS normalized form (V*∂P/∂|V|) to standard form (∂P/∂|V|)
    # by dividing |V| columns by the corresponding bus voltage magnitudes.
    if voltage_magnitudes is not None:
        v_pq = voltage_magnitudes[pq_indices]
        n_pvpq = len(pvpq)
        for j, v in enumerate(v_pq):
            col_idx = n_pvpq + j
            if abs(v) > 1e-15:
                jac_sub[:, col_idx] /= v

    jac_sub = sparse.csr_array(jac_sub)
    jac_inv = sparse_inv(jac_sub).toarray()

    return jac_sub, jac_inv, pvpq, pq_indices


def interpss_n1_analysis(
    net: Any,
    outage_branch_ids: list[str],
) -> dict[str, dict]:
    """Run N-1 contingency analysis using InterPSS AC loadflow.

    For each outage branch, creates a copy of the network, disconnects
    the branch, runs AC loadflow, and collects bus voltage results.

    Parameters
    ----------
    net : AclfNet
        The solved InterPSS network (base case).
    outage_branch_ids : list[str]
        Branch IDs to outage one at a time.

    Returns
    -------
    dict[str, dict]
        Keys are outage branch IDs. Values have:
        - 'converged': bool
        - 'voltages': dict[str, (float, float)] mapping bus ID to (V_mag, V_ang_rad)
    """
    _ensure_jvm()
    LoadflowAlgoObjectFactory = jpype.JClass("com.interpss.core.LoadflowAlgoObjectFactory")
    AclfMethodType = jpype.JClass("com.interpss.core.algo.AclfMethodType")

    results: dict[str, dict] = {}
    for branch_id in outage_branch_ids:
        net_copy = net.jsonCopy()
        branch = net_copy.getBranch(branch_id)
        if branch is None:
            continue
        branch.setStatus(False)

        algo = LoadflowAlgoObjectFactory.createLoadflowAlgorithm(net_copy)
        algo.setLfMethod(AclfMethodType.NR)
        algo.getLfAdjAlgo().setApplyAdjustAlgo(False);
        algo.setMaxIterations(0)
        algo.loadflow()
        converged = bool(net_copy.isLfConverged())

        voltages: dict[str, tuple[float, float]] = {}
        bus_iter = net_copy.getBusList().iterator()
        while bus_iter.hasNext():
            bus = bus_iter.next()
            bid = str(bus.getId())
            v_mag = float(bus.getVoltageMag())
            v_ang = float(bus.getVoltageAng())
            voltages[bid] = (v_mag, v_ang)

        results[branch_id] = {
            "converged": converged,
            "voltages": voltages,
        }

    return results


def _ensure_jvm() -> None:
    if not jpype.isJVMStarted():
        raise RuntimeError("JVM not started. Call initialize_jvm() first.")


def _load_config(config_path: str | None = None) -> dict:
    if config_path is None:
        config_path = os.environ.get("IPSS_CONFIG_PATH")
    if config_path is None:
        # Try sibling ipss-agent directory
        candidate = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "ipss-agent", "config", "config.json")
        candidate = os.path.abspath(candidate)
        if os.path.isfile(candidate):
            config_path = candidate

    if config_path is None or not os.path.isfile(config_path):
        raise FileNotFoundError(
            "InterPSS config not found. Set IPSS_CONFIG_PATH or pass config_path."
        )

    with open(config_path) as f:
        config = json.load(f)

    project_root = Path(config_path).parent.parent

    # Expand HOME in jvm_path
    if "jvm_path" in config:
        home = os.getenv("HOME", "")
        config["jvm_path"] = config["jvm_path"].replace("{HOME}", home)

    # Resolve jar_path (may contain multiple dirs separated by : or ;)
    if "jar_path" in config:
        parts = [p.strip() for p in re.split(r"[;:]", config["jar_path"]) if p.strip()]
        resolved = []
        for p in parts:
            if not os.path.isabs(p):
                p = str(project_root / p)
            if os.path.isdir(p):
                resolved.extend(sorted(glob.glob(os.path.join(p, "*.jar"))))
            elif glob.has_magic(p):
                resolved.extend(sorted(glob.glob(p)))
            else:
                resolved.append(p)
        config["jar_path"] = os.pathsep.join(resolved)

    if "log_config_path" in config and not os.path.isabs(config["log_config_path"]):
        config["log_config_path"] = str(project_root / config["log_config_path"])

    return config


def _dframe_to_pandas(dframe: Any) -> pd.DataFrame:
    """Convert an InterPSS (dflib) DataFrame to a pandas DataFrame."""
    n_cols = dframe.width()
    n_rows = dframe.height()
    col_index = dframe.getColumnsIndex()
    col_names = [str(col_index.get(i)) for i in range(n_cols)]

    data = {}
    for c in range(n_cols):
        series = dframe.getColumn(c)
        values = [str(series.get(r)) for r in range(n_rows)]
        data[col_names[c]] = values

    df = pd.DataFrame(data)

    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except (ValueError, TypeError):
            pass

    return df
