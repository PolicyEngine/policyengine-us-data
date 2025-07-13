import numpy as np
import importlib.util
import typing
import re

spec = importlib.util.spec_from_file_location(
    'td',
    'policyengine_us_data/utils/target_diagnostics.py'
)
mod = importlib.util.module_from_spec(spec)
mod.re = re
mod.List = typing.List
mod.Callable = typing.Callable
mod.Iterable = typing.Iterable
mod.Tuple = typing.Tuple
mod.Sequence = typing.Sequence

# pandas and numpy are required; if not installed, skip the test
try:
    import pandas as pd
    mod.pd = pd
    mod.np = np
    spec.loader.exec_module(mod)
except ModuleNotFoundError:
    import pytest
    pytest.skip("Required dependencies not available", allow_module_level=True)


def test_repair_inconsistencies_runs():
    names = [
        "nation/irs/aca_enrollment",
        "US06/irs/aca_enrollment",
        "state/CA/irs/aca_enrollment/district1",
        "state/CA/irs/aca_enrollment/district2",
    ]
    targets = np.array([100.0, 40.0, 20.0, 15.0])
    result = mod.repair_inconsistencies(targets, names)
    assert result.shape == targets.shape
