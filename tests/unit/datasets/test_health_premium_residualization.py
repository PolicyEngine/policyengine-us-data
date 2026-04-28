import numpy as np

from policyengine_us_data.datasets.cps.cps import (
    _premium_values_to_person,
    compute_premium_residual,
)


def test_premium_residual_subtracts_computed_premiums() -> None:
    reported = np.array([500.0, 200.0, 50.0])
    computed = np.array([125.0, 250.0, 0.0])

    result = compute_premium_residual(
        reported_premium=reported,
        baseline_computed_premium=computed,
    )

    np.testing.assert_allclose(result, [375.0, -50.0, 50.0])


def test_premium_residual_preserves_reported_input() -> None:
    reported = np.array([500.0, 200.0])
    computed = np.array([125.0, 250.0])

    _ = compute_premium_residual(
        reported_premium=reported,
        baseline_computed_premium=computed,
    )

    np.testing.assert_allclose(reported, [500.0, 200.0])


def test_tax_unit_premiums_allocate_to_first_person_only() -> None:
    data = {
        "person_id": np.array([1, 2, 3, 4]),
        "tax_unit_id": np.array([10, 20]),
        "person_tax_unit_id": np.array([10, 10, 20, 20]),
    }

    result = _premium_values_to_person(
        data=data,
        source_entity="tax_unit",
        values=np.array([300.0, 800.0]),
    )

    np.testing.assert_allclose(result, [300.0, 0.0, 800.0, 0.0])


def test_person_premiums_pass_through_to_person_rows() -> None:
    data = {"person_id": np.array([1, 2, 3])}
    values = np.array([100.0, 200.0, 300.0])

    result = _premium_values_to_person(
        data=data,
        source_entity="person",
        values=values,
    )

    np.testing.assert_allclose(result, values)
