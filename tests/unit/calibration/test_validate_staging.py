import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from policyengine_us_data.calibration.validate_staging import (
    _get_reform_income_tax_delta,
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
