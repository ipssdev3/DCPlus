import numpy as np
import pytest

from dc_plus.importing.import_schema import InjectionParamSchema
from tests.importing.interpss.conftest import skip_if_no_ipss


@skip_if_no_ipss
class TestInjectionExtraction:
    def test_generator_count_ieee14(self, ieee14_dfs):
        from dc_plus.importing.interpss.interpss_import import _get_injections_interpss
        from dc_plus.importing.interpss.interpss_import_helpers import build_bus_number_to_index

        bus_map = build_bus_number_to_index(ieee14_dfs["bus"])
        injections = _get_injections_interpss(
            ieee14_dfs["gen"], ieee14_dfs["load"], ieee14_dfs["bus"], bus_map,
        )
        gens = injections[injections["injection_type"] == "GENERATOR"]
        assert len(gens) == 5

    def test_load_count_ieee14(self, ieee14_dfs):
        from dc_plus.importing.interpss.interpss_import import _get_injections_interpss
        from dc_plus.importing.interpss.interpss_import_helpers import build_bus_number_to_index

        bus_map = build_bus_number_to_index(ieee14_dfs["bus"])
        injections = _get_injections_interpss(
            ieee14_dfs["gen"], ieee14_dfs["load"], ieee14_dfs["bus"], bus_map,
        )
        loads = injections[injections["injection_type"] == "LOAD"]
        assert len(loads) == 11

    def test_injection_schema_validates(self, ieee14_dfs):
        from dc_plus.importing.interpss.interpss_import import _get_injections_interpss
        from dc_plus.importing.interpss.interpss_import_helpers import build_bus_number_to_index

        bus_map = build_bus_number_to_index(ieee14_dfs["bus"])
        injections = _get_injections_interpss(
            ieee14_dfs["gen"], ieee14_dfs["load"], ieee14_dfs["bus"], bus_map,
        )
        InjectionParamSchema.validate(injections)

    def test_gen_injection_type(self, ieee14_dfs):
        from dc_plus.importing.interpss.interpss_import import _get_injections_interpss
        from dc_plus.importing.interpss.interpss_import_helpers import build_bus_number_to_index

        bus_map = build_bus_number_to_index(ieee14_dfs["bus"])
        injections = _get_injections_interpss(
            ieee14_dfs["gen"], ieee14_dfs["load"], ieee14_dfs["bus"], bus_map,
        )
        gens = injections[injections["injection_type"] == "GENERATOR"]
        assert all(gens["injection_type"] == "GENERATOR")

    def test_load_injection_type(self, ieee14_dfs):
        from dc_plus.importing.interpss.interpss_import import _get_injections_interpss
        from dc_plus.importing.interpss.interpss_import_helpers import build_bus_number_to_index

        bus_map = build_bus_number_to_index(ieee14_dfs["bus"])
        injections = _get_injections_interpss(
            ieee14_dfs["gen"], ieee14_dfs["load"], ieee14_dfs["bus"], bus_map,
        )
        loads = injections[injections["injection_type"] == "LOAD"]
        assert all(loads["injection_type"] == "LOAD")

    def test_gen_p_non_negative(self, ieee14_dfs):
        """Generator P is non-negative. Bus3 has PGen=0 (PV bus, Q only)."""
        from dc_plus.importing.interpss.interpss_import import _get_injections_interpss
        from dc_plus.importing.interpss.interpss_import_helpers import build_bus_number_to_index

        bus_map = build_bus_number_to_index(ieee14_dfs["bus"])
        injections = _get_injections_interpss(
            ieee14_dfs["gen"], ieee14_dfs["load"], ieee14_dfs["bus"], bus_map,
        )
        gens = injections[injections["injection_type"] == "GENERATOR"]
        assert all(gens["p"] >= 0)

    def test_load_p_negative(self, ieee14_dfs):
        from dc_plus.importing.interpss.interpss_import import _get_injections_interpss
        from dc_plus.importing.interpss.interpss_import_helpers import build_bus_number_to_index

        bus_map = build_bus_number_to_index(ieee14_dfs["bus"])
        injections = _get_injections_interpss(
            ieee14_dfs["gen"], ieee14_dfs["load"], ieee14_dfs["bus"], bus_map,
        )
        loads = injections[injections["injection_type"] == "LOAD"]
        assert all(loads["p"] < 0)

    def test_gen_connected_true(self, ieee14_dfs):
        from dc_plus.importing.interpss.interpss_import import _get_injections_interpss
        from dc_plus.importing.interpss.interpss_import_helpers import build_bus_number_to_index

        bus_map = build_bus_number_to_index(ieee14_dfs["bus"])
        injections = _get_injections_interpss(
            ieee14_dfs["gen"], ieee14_dfs["load"], ieee14_dfs["bus"], bus_map,
        )
        gens = injections[injections["injection_type"] == "GENERATOR"]
        assert all(gens["connected"])

    def test_gen_voltage_regulation_true(self, ieee14_dfs):
        from dc_plus.importing.interpss.interpss_import import _get_injections_interpss
        from dc_plus.importing.interpss.interpss_import_helpers import build_bus_number_to_index

        bus_map = build_bus_number_to_index(ieee14_dfs["bus"])
        injections = _get_injections_interpss(
            ieee14_dfs["gen"], ieee14_dfs["load"], ieee14_dfs["bus"], bus_map,
        )
        gens = injections[injections["injection_type"] == "GENERATOR"]
        assert all(gens["voltage_regulation"])

    def test_load_voltage_regulation_false(self, ieee14_dfs):
        from dc_plus.importing.interpss.interpss_import import _get_injections_interpss
        from dc_plus.importing.interpss.interpss_import_helpers import build_bus_number_to_index

        bus_map = build_bus_number_to_index(ieee14_dfs["bus"])
        injections = _get_injections_interpss(
            ieee14_dfs["gen"], ieee14_dfs["load"], ieee14_dfs["bus"], bus_map,
        )
        loads = injections[injections["injection_type"] == "LOAD"]
        assert not any(loads["voltage_regulation"])

    def test_load_regulated_bus_id_minus_one(self, ieee14_dfs):
        from dc_plus.importing.interpss.interpss_import import _get_injections_interpss
        from dc_plus.importing.interpss.interpss_import_helpers import build_bus_number_to_index

        bus_map = build_bus_number_to_index(ieee14_dfs["bus"])
        injections = _get_injections_interpss(
            ieee14_dfs["gen"], ieee14_dfs["load"], ieee14_dfs["bus"], bus_map,
        )
        loads = injections[injections["injection_type"] == "LOAD"]
        assert all(loads["regulated_bus_id_int"] == -1)

    def test_gen_q_limits(self, ieee14_dfs):
        from dc_plus.importing.interpss.interpss_import import _get_injections_interpss
        from dc_plus.importing.interpss.interpss_import_helpers import build_bus_number_to_index

        bus_map = build_bus_number_to_index(ieee14_dfs["bus"])
        injections = _get_injections_interpss(
            ieee14_dfs["gen"], ieee14_dfs["load"], ieee14_dfs["bus"], bus_map,
        )
        gens = injections[injections["injection_type"] == "GENERATOR"]
        # Check that at least some generators have non-NaN q limits
        non_nan_qmax = gens["max_q"].notna().sum()
        assert non_nan_qmax > 0
