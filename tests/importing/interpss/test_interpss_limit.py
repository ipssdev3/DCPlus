import pytest

from dc_plus.importing.import_schema import LimitParamSchema
from tests.importing.interpss.conftest import skip_if_no_ipss


@skip_if_no_ipss
class TestLimitExtraction:
    def test_no_limits_for_zero_ratings(self, ieee14_dfs):
        """IEEE 14 has LimMvaA == 0 for all branches, so limits should be empty."""
        from dc_plus.importing.interpss.interpss_import import _get_limits_parameter_interpss

        limits = _get_limits_parameter_interpss(ieee14_dfs["branch"])
        assert len(limits) == 0
        LimitParamSchema.validate(limits)

    def test_limit_schema_validates_empty(self):
        import pandas as pd
        from dc_plus.importing.interpss.interpss_import import _get_limits_parameter_interpss

        branch_df = pd.DataFrame({
            "ID": ["B1->B2(1)"],
            "IsXfmr": [False],
            "LimMvaA": [0.0],
        })
        limits = _get_limits_parameter_interpss(branch_df)
        assert len(limits) == 0
        LimitParamSchema.validate(limits)

    def test_limits_from_branch_ratings(self):
        """When ratings are non-zero, create APPARENT_POWER limits."""
        import pandas as pd
        from dc_plus.importing.interpss.interpss_import import _get_limits_parameter_interpss

        branch_df = pd.DataFrame({
            "ID": ["B1->B2(1)", "B3->B4(1)"],
            "IsXfmr": [False, True],
            "LimMvaA": [100.0, 0.0],
        })
        limits = _get_limits_parameter_interpss(branch_df, base_mva=100.0)
        assert len(limits) == 1
        assert limits.iloc[0]["limit_type"] == "APPARENT_POWER"
        assert limits.iloc[0]["value"] == 1.0  # 100 MVA / 100 base = 1.0 pu
        LimitParamSchema.validate(limits)
