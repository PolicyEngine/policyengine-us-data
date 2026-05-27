"""Regression tests for deterministic EnhancedCPS weight initialization.

Earlier versions used global ``np.random.normal(1, 0.1, ...)`` jitter before
``reweight()`` reseeded the optimizer. Current code routes both dense CPS
weighting paths through ``initialize_weight_priors()``, which preserves positive
survey weight shape and gives zero-weight clone records deterministic uniform
prior mass.
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


def test_enhanced_cps_sources_use_deterministic_weight_priors():
    """Both generate() methods should use deterministic priors, not global RNG."""
    import policyengine_us_data.datasets.cps.enhanced_cps as ec

    source = open(ec.__file__).read()

    assert "np.random.normal" not in source
    assert source.count("initialize_weight_priors(original_weights.values)") == 2


def test_initialize_weight_priors_preserves_source_weight_total():
    from policyengine_us_data.datasets.cps.enhanced_cps import (
        initialize_weight_priors,
    )

    priors = initialize_weight_priors(
        np.array([80.0, 20.0, 0.0, 0.0]),
        zero_weight_total_share=0.5,
    )

    np.testing.assert_allclose(priors.sum(), 100.0)
    np.testing.assert_allclose(priors, np.array([40.0, 10.0, 25.0, 25.0]))
