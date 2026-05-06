import policyengine_us_data
import policyengine_us_data.datasets.cps as cps_package


def test_cps_package_preserves_legacy_lazy_export_names():
    assert {
        "CPS_CLONE_FEATURE_VARIABLES",
        "CPS_ONLY_IMPUTED_VARIABLES",
        "CPS_STAGE2_DEMOGRAPHIC_PREDICTORS",
        "CPS_STAGE2_INCOME_PREDICTORS",
        "CURRENT_HEALTH_COVERAGE_REPORTED_VAR_MAP",
        "ESI_POLICYHOLDER_VARIABLE",
    } <= set(cps_package.__all__)


def test_package_root_exports_are_unique():
    assert len(policyengine_us_data.__all__) == len(set(policyengine_us_data.__all__))
