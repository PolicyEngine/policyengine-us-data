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
    """Return set of variable names computed by policyengine-us.

    Includes variables with explicit formulas as well as those using
    ``adds`` or ``subtracts`` (which the engine auto-sums at runtime).
    """
    return {
        name
        for name, var in tax_benefit_system.variables.items()
        if (hasattr(var, "formulas") and len(var.formulas) > 0)
        or getattr(var, "adds", None)
        or getattr(var, "subtracts", None)
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
    """Computed variables should not be stored in the dataset."""
    overlap = (stored_variables & formula_variables) - KNOWN_FORMULA_EXCEPTIONS
    if overlap:
        msg = (
            f"These {len(overlap)} variables are computed by "
            f"policyengine-us (formulas/adds/subtracts) but are "
            f"stored in the dataset "
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


def test_ss_subcomponents_sum_to_computed_total(sim, dataset_path):
    """Social Security sub-components should sum to the computed total.

    ``social_security`` is computed via ``adds`` in policyengine-us and
    is NOT stored in the dataset.  We verify that the sub-components
    stored in the dataset sum to the simulation's computed total.
    """
    with h5py.File(dataset_path, "r") as f:
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
    computed_total = np.array(sim.calculate("social_security", 2024)).astype(
        float
    )

    # Only check records that have any SS income
    has_ss = computed_total > 0
    if not np.any(has_ss):
        pytest.skip("No SS recipients in dataset")

    total_computed = np.sum(computed_total[has_ss])
    total_sub = np.sum(sub_sum[has_ss])
    pct_diff = abs(total_computed - total_sub) / total_computed * 100

    assert pct_diff < 1, (
        f"SS sub-components don't sum to computed total: "
        f"computed=${total_computed:,.0f}, sub_sum=${total_sub:,.0f}, "
        f"diff={pct_diff:.1f}%"
    )

    # Per-person check: no person's sub-components should exceed total
    excess = sub_sum[has_ss] - computed_total[has_ss]
    n_excess = np.sum(excess > 1)
    assert n_excess == 0, (
        f"{n_excess} people have SS sub-components exceeding "
        f"their computed total SS income"
    )
