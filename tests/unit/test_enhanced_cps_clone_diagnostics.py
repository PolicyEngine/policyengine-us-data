import pytest

from policyengine_us_data.datasets.cps.enhanced_cps import (
    compute_clone_diagnostics_summary,
)


def test_compute_clone_diagnostics_summary():
    diagnostics = compute_clone_diagnostics_summary(
        household_is_puf_clone=[False, True],
        household_weight=[9.0, 1.0],
        person_is_puf_clone=[False, True, True],
        person_weight=[4.0, 3.0, 3.0],
        person_in_poverty=[False, True, True],
        person_reported_in_poverty=[False, False, True],
        spm_unit_is_puf_clone=[False, True, True],
        spm_unit_weight=[2.0, 3.0, 5.0],
        spm_unit_capped_work_childcare_expenses=[0.0, 6000.0, 7000.0],
        spm_unit_pre_subsidy_childcare_expenses=[0.0, 5000.0, 8000.0],
        spm_unit_taxes=[100.0, 9000.0, 200.0],
        spm_unit_market_income=[1000.0, 8000.0, 1000.0],
    )

    assert diagnostics["clone_household_weight_share_pct"] == pytest.approx(10.0)
    assert diagnostics["clone_poor_modeled_only_person_weight_share_pct"] == pytest.approx(
        30.0
    )
    assert diagnostics["poor_modeled_only_within_clone_person_weight_share_pct"] == pytest.approx(
        50.0
    )
    assert diagnostics["clone_childcare_exceeds_pre_subsidy_share_pct"] == pytest.approx(
        37.5
    )
    assert diagnostics["clone_childcare_above_5000_share_pct"] == pytest.approx(100.0)
    assert diagnostics["clone_taxes_exceed_market_income_share_pct"] == pytest.approx(
        37.5
    )
