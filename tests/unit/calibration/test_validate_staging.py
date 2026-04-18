import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from policyengine_us_data.calibration.validate_staging import (
    CSV_COLUMNS,
    _get_reform_income_tax_delta,
    validate_area,
)


class _FakeArrayResult:
    def __init__(self, values):
        self.values = values


class _FakeMicrosimulation:
    def __init__(self, dataset=None, reform=None):
        self.dataset = dataset
        self.reform = reform

    def calculate(self, variable, map_to=None, period=None):
        assert variable == "income_tax"
        assert map_to == "household"
        assert period == 2024
        return _FakeArrayResult(np.array([150.0, 260.0], dtype=np.float32))


@patch.dict(
    sys.modules,
    {"policyengine_us": SimpleNamespace(Microsimulation=_FakeMicrosimulation)},
)
def test_get_reform_income_tax_delta_caches_delta():
    baseline_income_tax = np.array([100.0, 200.0], dtype=np.float32)
    cache = {}

    delta = _get_reform_income_tax_delta(
        dataset_path="fake.h5",
        period=2024,
        variable="salt_deduction",
        baseline_income_tax=baseline_income_tax,
        reform_delta_cache=cache,
    )

    np.testing.assert_array_equal(delta, np.array([50.0, 60.0], dtype=np.float32))
    np.testing.assert_array_equal(cache["salt_deduction"], delta)

    # The cached value should remain the delta, not the raw reform income tax.
    cached = _get_reform_income_tax_delta(
        dataset_path="fake.h5",
        period=2024,
        variable="salt_deduction",
        baseline_income_tax=np.array([0.0, 0.0], dtype=np.float32),
        reform_delta_cache=cache,
    )
    np.testing.assert_array_equal(cached, np.array([50.0, 60.0], dtype=np.float32))


class _FakeValidationSim:
    def calculate(self, variable, map_to=None, period=None):
        if variable == "household_id" and map_to == "household":
            return _FakeArrayResult(np.array([10, 20], dtype=np.int64))
        if variable == "household_weight" and map_to == "household":
            return _FakeArrayResult(np.array([1.0, 2.0], dtype=np.float64))
        raise AssertionError(f"Unexpected calculate call: {variable=} {map_to=}")


def test_validate_area_emits_distinct_area_id_and_display_name(monkeypatch):
    monkeypatch.setattr(
        "policyengine_us_data.calibration.validate_staging._build_entity_rel",
        lambda sim: object(),
    )
    monkeypatch.setattr(
        "policyengine_us_data.calibration.validate_staging._calculate_target_values_standalone",
        lambda **kwargs: np.array([3.0, 4.0], dtype=np.float64),
    )
    monkeypatch.setattr(
        "policyengine_us_data.calibration.validate_staging.UnifiedMatrixBuilder._make_target_name",
        lambda *args, **kwargs: "target-name",
    )

    results = validate_area(
        sim=_FakeValidationSim(),
        targets_df=pd.DataFrame(
            [
                {
                    "variable": "household_count",
                    "value": 11.0,
                    "stratum_id": 7,
                    "period": 2024,
                    "reform_id": 0,
                }
            ]
        ),
        engine=None,
        area_type="states",
        area_id="37",
        display_id="NC",
        dataset_path="fake.h5",
        period=2024,
        training_mask=np.array([True], dtype=bool),
        variable_entity_map={},
        constraints_map={7: []},
    )

    assert CSV_COLUMNS[:3] == ["area_type", "area_id", "display_name"]
    assert results[0]["area_id"] == "37"
    assert results[0]["display_name"] == "NC"
