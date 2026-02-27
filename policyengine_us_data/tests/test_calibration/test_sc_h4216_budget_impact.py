from policyengine_core.reforms import Reform
from policyengine_us import Microsimulation
from policyengine_us.reforms.states.sc.h4216.sc_h4216 import (
    create_sc_h4216,
)

SC_DATASET = "hf://policyengine/policyengine-us-data/staging/states/SC.h5"

PARAM_OVERRIDES = {
    "gov.contrib.states.sc.h4216.in_effect": {
        "2026-01-01.2100-12-31": True,
    },
    "gov.contrib.states.sc.h4216.rates[1].rate": {
        "2026-01-01.2100-12-31": 0.0539,
    },
}


def test_sc_h4216_budget_impact():
    structural_reform = create_sc_h4216()
    param_reform = Reform.from_dict(PARAM_OVERRIDES, country_id="us")
    full_reform = (structural_reform, param_reform)

    baseline = Microsimulation(dataset=SC_DATASET)
    reformed = Microsimulation(reform=full_reform, dataset=SC_DATASET)

    baseline_tax = baseline.calculate("sc_income_tax", 2026, map_to="tax_unit")
    reform_tax = reformed.calculate("sc_income_tax", 2026, map_to="tax_unit")
    weight = baseline.calculate("tax_unit_weight", 2026)

    # BUG (double-weighting): MicroSeries.sum() already multiplies
    # by the attached .weights, so this computes sum(diff * w * w):
    # budget_impact = ((reform_tax - baseline_tax) * weight).sum()
    budget_impact = (
        (reform_tax.values - baseline_tax.values) * weight.values
    ).sum()
    budget_impact_m = budget_impact / 1e6

    print(f"\nSC H.4216 budget impact: ${budget_impact_m:,.1f}M")
    assert -130 < budget_impact_m < -110, (
        f"Budget impact ${budget_impact_m:,.1f}M outside calibration target "
        f"[-$130M, -$110M] (RFA fiscal note: -$119.1M)"
    )
