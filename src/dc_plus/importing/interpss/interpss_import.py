# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""InterPSS import functions for extracting network data into DCPlus schemas."""

import numpy as np
import pandas as pd


def _get_buses_interpss(
    bus_df: pd.DataFrame,
    slack_bus_id: str,
) -> pd.DataFrame:
    """Extract bus data from InterPSS bus DataFrame.

    Parameters
    ----------
    bus_df : pd.DataFrame
        InterPSS bus DataFrame with columns: ID, Number, Name, VoltMag, VoltAng, BusType.
    slack_bus_id : str
        The slack bus ID string (e.g. "Bus1").

    Returns
    -------
    pd.DataFrame validated by BusParamSchema.
    """
    bus_df = bus_df.sort_values("Number").reset_index(drop=True)

    bus_type_map = {"Swing": 0, "PV": 1, "PQ": 2}

    buses = pd.DataFrame(
        {
            "id_int": range(len(bus_df)),
            "id_str": bus_df["ID"].values,
            "name": bus_df["Name"].values,
            "voltage_magnitude": bus_df["VoltMag"].values,
            "voltage_angle": bus_df["VoltAng"].values,
            "bus_type": [bus_type_map.get(str(bt), 2) for bt in bus_df["BusType"].values],
            "grid_island_id": np.zeros(len(bus_df), dtype=int),
        }
    )

    return buses


def _get_branches_parameter_interpss(
    branch_df: pd.DataFrame,
    bus_number_to_index: dict[int, int],
    tap_info: dict[str, tuple[float, float]] | None = None,
) -> pd.DataFrame:
    """Extract branch data from InterPSS branch DataFrame.

    Parameters
    ----------
    branch_df : pd.DataFrame
        InterPSS branch DataFrame.
    bus_number_to_index : dict[int, int]
        Mapping from InterPSS bus Number to 0-based index.

    Returns
    -------
    pd.DataFrame validated by BranchParamSchema.
    """
    records = []
    for i, (_, row) in enumerate(branch_df.iterrows()):
        from_idx = bus_number_to_index[int(row["FromBusNumber"])]
        to_idx = bus_number_to_index[int(row["ToBusNumber"])]
        is_xfmr = str(row["IsXfmr"]).lower() == "true"
        connected = str(row["InService"]).lower() == "true" and str(row["Status"]).lower() == "true"

        b_total = float(row["B"]) if pd.notna(row["B"]) else 0.0
        # Split total charging equally between both ends
        b_half = b_total / 2.0

        branch_type = "TWO_WINDINGS_TRANSFORMER" if is_xfmr else "LINE"
        name = str(row["Name"]) if pd.notna(row["Name"]) and str(row["Name"]).strip() else str(row["ID"])

        branch_id = str(row["ID"])
        if tap_info and branch_id in tap_info:
            rho, alpha = tap_info[branch_id]
        else:
            rho = 1.0
            alpha = 0.0

        records.append(
            {
                "id_int": i,
                "id_str": str(row["ID"]),
                "name": name,
                "connected": connected,
                "r": float(row["R"]) if pd.notna(row["R"]) else 0.0,
                "x": float(row["X"]) if pd.notna(row["X"]) else 0.0,
                "g1": 0.0,
                "b1": b_half,
                "g2": 0.0,
                "b2": b_half,
                "p1": float(row["PFrom2To"]) if pd.notna(row["PFrom2To"]) else np.nan,
                "q1": float(row["QFrom2To"]) if pd.notna(row["QFrom2To"]) else np.nan,
                "i1": np.nan,
                "p2": float(row["PTo2From"]) if pd.notna(row["PTo2From"]) else np.nan,
                "q2": float(row["QTo2From"]) if pd.notna(row["QTo2From"]) else np.nan,
                "i2": np.nan,
                "rho": rho,
                "alpha": alpha,
                "from_bus_index": from_idx,
                "to_bus_index": to_idx,
                "branch_type": branch_type,
            }
        )

    return pd.DataFrame(records)


def _get_injections_interpss(
    gen_df: pd.DataFrame,
    load_df: pd.DataFrame,
    bus_df: pd.DataFrame,
    bus_number_to_index: dict[int, int],
) -> pd.DataFrame:
    """Extract injection data from InterPSS gen and load DataFrames.

    Parameters
    ----------
    gen_df : pd.DataFrame
        InterPSS generator DataFrame.
    load_df : pd.DataFrame
        InterPSS load DataFrame.
    bus_df : pd.DataFrame
        InterPSS bus DataFrame (used for bus type lookup).
    bus_number_to_index : dict[int, int]
        Mapping from bus Number to 0-based index.

    Returns
    -------
    pd.DataFrame validated by InjectionParamSchema.
    """
    # Build bus ID -> bus type mapping
    bus_type_map = {}
    for _, row in bus_df.iterrows():
        bus_type_map[str(row["ID"])] = str(row["BusType"])

    gen_records = []
    for _, row in gen_df.iterrows():
        bus_id = str(row["BusID"])
        bus_type = bus_type_map.get(bus_id, "PQ")
        bus_idx = bus_number_to_index.get(int(row["BusNumber"]), -1)
        is_regulated = bus_type in ("Swing", "PV")

        p_gen = float(row["PGen"]) if pd.notna(row["PGen"]) else 0.0
        q_gen = float(row["QGen"]) if pd.notna(row["QGen"]) else 0.0

        # InterPSS creates gen entries for all buses (GenCode="NonGen").
        # Include generators at regulated buses (Swing/PV) or with non-zero output.
        # After loadflow, some PV buses may switch to PQ when Q limits are violated,
        # but they still have active generation that must be included.
        has_output = abs(p_gen) > 1e-8 or abs(q_gen) > 1e-8
        if not (is_regulated or has_output):
            continue

        gen_records.append(
            {
                "id_int": 0,  # reassigned below
                "id_str": f"{bus_id}-G{row['GenID']}",
                "injection_type": "GENERATOR",
                "p": p_gen,
                "q": q_gen,
                "i": np.nan,
                "setpoint_p": p_gen,
                "setpoint_q": np.nan,
                "min_q": float(row["QMin"]) if pd.notna(row["QMin"]) and float(row["QMin"]) != 0.0 else np.nan,
                "max_q": float(row["QMax"]) if pd.notna(row["QMax"]) and float(row["QMax"]) != 0.0 else np.nan,
                "min_p": float(row["PMin"]) if pd.notna(row["PMin"]) and float(row["PMin"]) != 0.0 else np.nan,
                "max_p": float(row["PMax"]) if pd.notna(row["PMax"]) and float(row["PMax"]) != 0.0 else np.nan,
                "bus_index": bus_idx,
                "connected": str(row["InService"]).lower() == "true",
                "voltage_regulation": is_regulated,
                "regulated_bus_id_str": bus_id if is_regulated else "",
                "regulated_bus_id_int": bus_idx if is_regulated else -1,
            }
        )

    load_records = []
    for _, row in load_df.iterrows():
        bus_idx = bus_number_to_index.get(int(row["BusNumber"]), -1)

        p_load = float(row["PLoadTotal"]) if pd.notna(row["PLoadTotal"]) else 0.0
        q_load = float(row["QLoadTotal"]) if pd.notna(row["QLoadTotal"]) else 0.0

        load_records.append(
            {
                "id_int": 0,
                "id_str": f"{row['BusID']}-L{row['LoadID']}",
                "injection_type": "LOAD",
                "p": -p_load,  # loads are negative injections
                "q": -q_load,
                "i": np.nan,
                "setpoint_p": -p_load,
                "setpoint_q": -q_load,
                "min_q": np.nan,
                "max_q": np.nan,
                "min_p": np.nan,
                "max_p": np.nan,
                "bus_index": bus_idx,
                "connected": str(row["InService"]).lower() == "true",
                "voltage_regulation": False,
                "regulated_bus_id_str": "",
                "regulated_bus_id_int": -1,
            }
        )

    all_records = gen_records + load_records
    for i, rec in enumerate(all_records):
        rec["id_int"] = i

    return pd.DataFrame(all_records)


def _get_shunts_interpss(
    bus_df: pd.DataFrame,
    bus_number_to_index: dict[int, int],
    bus_shunt_info: dict[str, tuple[float, float]] | None = None,
) -> pd.DataFrame:
    """Extract shunt data from InterPSS.

    Uses ``extract_bus_shunt_info`` to capture fixed bus shunts from the
    InterPSS net object. Falls back to ``AdjustableShuntB`` from the DataFrame
    if no shunt info is provided.

    Parameters
    ----------
    bus_df : pd.DataFrame
        InterPSS bus DataFrame.
    bus_number_to_index : dict[int, int]
        Mapping from bus Number to 0-based index.
    bus_shunt_info : dict or None
        Mapping from bus ID to (g, b) from ``extract_bus_shunt_info()``.

    Returns
    -------
    pd.DataFrame validated by ShuntParamSchema.
    """
    if bus_shunt_info:
        shunt_data = bus_shunt_info
    else:
        shunt_data: dict[str, tuple[float, float]] = {}
        for _, row in bus_df.iterrows():
            shunt_b = float(row["AdjustableShuntB"]) if pd.notna(row["AdjustableShuntB"]) else 0.0
            if shunt_b != 0.0:
                shunt_data[str(row["ID"])] = (0.0, shunt_b)

    records = []
    for bus_id, (g, b) in shunt_data.items():
        bus_row = bus_df[bus_df["ID"] == bus_id]
        if bus_row.empty:
            continue
        bus_row = bus_row.iloc[0]
        bus_idx = bus_number_to_index.get(int(bus_row["Number"]), -1)
        v_mag = float(bus_row["VoltMag"]) if pd.notna(bus_row["VoltMag"]) else 1.0

        records.append(
            {
                "id_int": len(records),
                "id_str": f"{bus_id}-shunt",
                "name": f"{bus_id}-shunt",
                "connected": str(bus_row["InService"]).lower() == "true",
                "g": g,
                "b": b,
                "p": g * v_mag**2,
                "q": -(b * v_mag**2),
                "i": np.nan,
                "bus_index": bus_idx,
                "section_count": 1,
                "max_section_count": 1,
                "voltage_regulation": False,
                "regulated_bus_id_str": "",
                "regulated_bus_id_int": -1,
            }
        )

    if not records:
        return _empty_shunt_df()

    return pd.DataFrame(records)


def _get_limits_parameter_interpss(
    branch_df: pd.DataFrame,
    base_mva: float = 100.0,
) -> pd.DataFrame:
    """Extract limit data from InterPSS branch DataFrame.

    Parameters
    ----------
    branch_df : pd.DataFrame
        InterPSS branch DataFrame with LimMvaA, LimMvaB, LimMvaC columns.
    base_mva : float
        System base MVA for per-unit conversion.

    Returns
    -------
    pd.DataFrame validated by LimitParamSchema.
    """
    records = []
    limit_id = 0

    for _, row in branch_df.iterrows():
        for rating_col, side in [("LimMvaA", "ONE"), ("LimMvaB", "TWO")]:
            if rating_col not in row.index:
                continue
            val = float(row[rating_col]) if pd.notna(row[rating_col]) else 0.0
            if val == 0.0:
                continue

            branch_id = str(row["ID"])
            is_xfmr = bool(row["IsXfmr"])
            element_type = "TWO_WINDINGS_TRANSFORMER" if is_xfmr else "LINE"

            records.append(
                {
                    "id_int": limit_id,
                    "element_id_str": branch_id,
                    "limit_type": "APPARENT_POWER",
                    "element_type": element_type,
                    "acceptable_duration": float("inf"),
                    "side": side,
                    "name": f"{branch_id}_rating_{side.lower()}",
                    "value": val / base_mva,
                }
            )
            limit_id += 1

    if not records:
        return _empty_limit_df()

    return pd.DataFrame(records)


def _empty_shunt_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id_int": pd.Series([], dtype=int),
            "id_str": pd.Series([], dtype=str),
            "name": pd.Series([], dtype=str),
            "connected": pd.Series([], dtype=bool),
            "g": pd.Series([], dtype=float),
            "b": pd.Series([], dtype=float),
            "p": pd.Series([], dtype=float),
            "q": pd.Series([], dtype=float),
            "i": pd.Series([], dtype=float),
            "bus_index": pd.Series([], dtype=int),
            "section_count": pd.Series([], dtype=int),
            "max_section_count": pd.Series([], dtype=int),
            "voltage_regulation": pd.Series([], dtype=bool),
            "regulated_bus_id_str": pd.Series([], dtype=str),
            "regulated_bus_id_int": pd.Series([], dtype=int),
        }
    )


def _empty_limit_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id_int": pd.Series([], dtype=int),
            "element_id_str": pd.Series([], dtype=str),
            "limit_type": pd.Series([], dtype=str),
            "element_type": pd.Series([], dtype=str),
            "acceptable_duration": pd.Series([], dtype=float),
            "side": pd.Series([], dtype=str),
            "name": pd.Series([], dtype=str),
            "value": pd.Series([], dtype=float),
        }
    )
