"""Regression test ensuring the initial weight jitter in EnhancedCPS is seeded.

Previously ``np.random.normal(1, 0.1, ...)`` ran with whatever numpy global
state the process happened to be in. ``reweight()`` re-seeds, but only
afterwards, so the final L0 weights differed run to run even with
``seed=1456`` inside ``reweight``.

Fix: call ``set_seeds(1456)`` right before the jitter in
``EnhancedCPS.generate`` and ``ReweightedCPS_2024.generate``.
"""

import numpy as np

from policyengine_us_data.utils.seed import set_seeds


def _mock_jitter(n: int = 10) -> np.ndarray:
    """Mirror the enhanced_cps perturbation shape."""
    return np.random.normal(1, 0.1, n)


def test_set_seeds_makes_numpy_normal_reproducible():
    set_seeds(1456)
    a = _mock_jitter()
    set_seeds(1456)
    b = _mock_jitter()
    assert np.array_equal(a, b)


def test_unseeded_numpy_normal_is_non_reproducible():
    """Sanity check: without set_seeds in between, two consecutive draws differ."""
    np.random.seed(None)  # reset to fresh entropy
    a = _mock_jitter()
    # Don't reseed — same process draws again, distinct state.
    b = _mock_jitter()
    assert not np.array_equal(a, b)


def test_enhanced_cps_sources_call_set_seeds_before_jitter():
    """The fix places ``set_seeds(1456)`` immediately before
    ``np.random.normal`` in both generate() methods. Verify the file
    preserves that invariant so regressions are caught by lint.
    """
    import policyengine_us_data.datasets.cps.enhanced_cps as ec

    source = open(ec.__file__).read()
    # Split into the two generate() bodies. Both must contain the
    # set_seeds call before the np.random.normal call.
    # The simplest invariant: every occurrence of np.random.normal
    # must be preceded (within the previous 5 non-blank lines) by a
    # set_seeds(...) call.
    lines = source.splitlines()
    normal_indices = [i for i, line in enumerate(lines) if "np.random.normal" in line]
    assert normal_indices, "Expected at least one np.random.normal site"
    for idx in normal_indices:
        window = "\n".join(lines[max(0, idx - 5) : idx])
        assert "set_seeds(" in window, (
            f"np.random.normal on line {idx + 1} is not preceded by set_seeds("
            f") within the previous 5 lines; window was:\n{window}"
        )
