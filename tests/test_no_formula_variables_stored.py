"""Test dataset variable integrity.

1. Variables with formulas in policyengine-us will be recomputed by the
   simulation engine, ignoring any stored value. Storing them wastes
   space and can mislead during validation.

2. Stored input values should match what the simulation actually uses,
   to catch issues like set_input_divide_by_period silently altering
   values, or sub-components not summing to their parent total.
"""

import h5py
import numpy as np
import pytest
from policyengine_us_data.datasets.cps.extended_cps import ExtendedCPS_2024


KNOWN_FORMULA_EXCEPTIONS = {
    # person_id is stored for identity tracking even though it has a
    # trivial formula (arange). Safe to keep.
    "person_id",
}


@pytest.fixture(scope="module")
def tax_benefit_system():
    from policyengine_us import CountryTaxBenefitSystem

    return CountryTaxBenefitSystem()


@pytest.fixture(scope="module")
def formula_variables(tax_benefit_system):
    """Return set of variable names that have formulas."""
    return {
        name
        for name, var in tax_benefit_system.variables.items()
        if hasattr(var, "formulas") and len(var.formulas) > 0
    }


@pytest.fixture(scope="module")
def dataset_path():
    path = ExtendedCPS_2024.file_path
    if not path.exists():
        pytest.skip("Extended CPS 2024 not built locally")
    return path


@pytest.fixture(scope="module")
def stored_variables(dataset_path):
    """Return set of variable names stored in the extended CPS."""
    with h5py.File(dataset_path, "r") as f:
        return set(f.keys())


@pytest.fixture(scope="module")
def sim():
    from policyengine_us import Microsimulation

    return Microsimulation(dataset=ExtendedCPS_2024)


def test_no_formula_variables_stored(formula_variables, stored_variables):
    """Variables with formulas should not be stored in the dataset."""
    overlap = (stored_variables & formula_variables) - KNOWN_FORMULA_EXCEPTIONS
    if overlap:
        msg = (
            f"These {len(overlap)} variables have formulas in "
            f"policyengine-us but are stored in the dataset "
            f"(stored values will be IGNORED by the simulation):\n"
        )
        for v in sorted(overlap):
            msg += f"  - {v}\n"
        pytest.fail(msg)


def test_stored_values_match_computed(
    sim, stored_variables, formula_variables, dataset_path
):
    """For input-only variables, stored values should match what the
    simulation returns (catches set_input issues, rounding, etc.)."""
    input_vars = stored_variables - formula_variables
    mismatches = []

    with h5py.File(dataset_path, "r") as f:
        for var_name in sorted(input_vars):
            if var_name not in f or "2024" not in f[var_name]:
                continue
            stored = f[var_name]["2024"][...]
            if stored.dtype.kind not in ("f", "i", "u"):
                continue

            stored_f = stored.astype(float)
            stored_total = np.sum(stored_f)
            if abs(stored_total) < 1:
                continue

            try:
                computed = np.array(sim.calculate(var_name, 2024))
            except Exception:
                continue

            computed_total = np.sum(computed.astype(float))
            if abs(stored_total) > 0:
                pct_diff = (
                    abs(stored_total - computed_total)
                    / abs(stored_total)
                    * 100
                )
            else:
                pct_diff = 0

            if pct_diff > 1:
                mismatches.append(
                    f"  {var_name}: stored=${stored_total:,.0f}, "
                    f"computed=${computed_total:,.0f}, "
                    f"diff={pct_diff:.1f}%"
                )

    if mismatches:
        msg = (
            f"These {len(mismatches)} input variables have >1% "
            f"difference between stored and computed values:\n"
        )
        msg += "\n".join(mismatches)
        pytest.fail(msg)


def test_ss_subcomponents_sum_to_total(dataset_path):
    """Social Security sub-components should sum to the total."""
    with h5py.File(dataset_path, "r") as f:
        ss_total = f["social_security"]["2024"][...].astype(float)
        ss_retirement = f["social_security_retirement"]["2024"][...].astype(
            float
        )
        ss_disability = f["social_security_disability"]["2024"][...].astype(
            float
        )
        ss_survivors = f["social_security_survivors"]["2024"][...].astype(
            float
        )
        ss_dependents = f["social_security_dependents"]["2024"][...].astype(
            float
        )

    sub_sum = ss_retirement + ss_disability + ss_survivors + ss_dependents

    # Only check records that have any SS income
    has_ss = ss_total > 0
    if not np.any(has_ss):
        pytest.skip("No SS recipients in dataset")

    total_ss = np.sum(ss_total[has_ss])
    total_sub = np.sum(sub_sum[has_ss])
    pct_diff = abs(total_ss - total_sub) / total_ss * 100

    assert pct_diff < 1, (
        f"SS sub-components don't sum to total: "
        f"total=${total_ss:,.0f}, sub_sum=${total_sub:,.0f}, "
        f"diff={pct_diff:.1f}%"
    )

    # Per-person check: no person's sub-components should exceed total
    excess = sub_sum[has_ss] - ss_total[has_ss]
    n_excess = np.sum(excess > 1)
    assert n_excess == 0, (
        f"{n_excess} people have SS sub-components exceeding "
        f"their total SS income"
    )
