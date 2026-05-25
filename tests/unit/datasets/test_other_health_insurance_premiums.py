from types import SimpleNamespace

import numpy as np

from policyengine_us_data.datasets.cps.cps import (
    _premium_values_to_person,
    compute_other_health_insurance_premiums,
    derive_other_health_insurance_premiums,
)


def test_other_health_insurance_premiums_subtracts_computed_premiums() -> None:
    reported = np.array([500.0, 200.0, 50.0])
    computed = np.array([125.0, 250.0, 0.0])

    result = compute_other_health_insurance_premiums(
        reported_premium=reported,
        baseline_computed_premium=computed,
    )

    np.testing.assert_allclose(result, [375.0, 0.0, 50.0])


def test_other_health_insurance_premiums_preserves_reported_input() -> None:
    reported = np.array([500.0, 200.0])
    computed = np.array([125.0, 250.0])

    _ = compute_other_health_insurance_premiums(
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


def test_derive_other_health_insurance_premiums_emits_output(
    monkeypatch,
) -> None:
    class FakeDataset:
        time_period = 2024

        def __init__(self):
            self.saved_data = None
            self.data = {
                "person_id": np.array([1, 2]),
                "health_insurance_premiums_without_medicare_part_b": np.array(
                    [500.0, 200.0]
                ),
            }

        def load_dataset(self):
            return self.data.copy()

        def save_dataset(self, data):
            self.saved_data = data

    class FakeMicrosimulation:
        tax_benefit_system = SimpleNamespace(
            variables={
                "chip_premium": SimpleNamespace(entity=SimpleNamespace(key="person")),
                "marketplace_net_premium": SimpleNamespace(
                    entity=SimpleNamespace(key="person")
                ),
                "medicaid_premium": SimpleNamespace(
                    entity=SimpleNamespace(key="person")
                ),
            }
        )

        def __init__(self, dataset):
            pass

        def calculate(self, variable, period):
            values = {
                "chip_premium": np.array([50.0, 75.0]),
                "marketplace_net_premium": np.array([25.0, 0.0]),
                "medicaid_premium": np.array([0.0, 10.0]),
            }
            return SimpleNamespace(values=values[variable])

    monkeypatch.setattr("policyengine_us.Microsimulation", FakeMicrosimulation)

    dataset = FakeDataset()
    derive_other_health_insurance_premiums(dataset)

    assert dataset.saved_data is not None
    np.testing.assert_allclose(
        dataset.saved_data["other_health_insurance_premiums"],
        [425.0, 115.0],
    )
