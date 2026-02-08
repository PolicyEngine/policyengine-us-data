import warnings
import numpy as np


def _stable_string_hash(s: str) -> np.uint64:
    """Deterministic hash consistent across Python processes.

    Python's built-in hash() is not deterministic across processes
    (since 3.3), so we use a polynomial rolling hash with mixing.

    Ported from policyengine_core.commons.formulas._stable_string_hash.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", "overflow encountered", RuntimeWarning
        )
        h = np.uint64(0)
        for byte in s.encode("utf-8"):
            h = h * np.uint64(31) + np.uint64(byte)
        h = h ^ (h >> np.uint64(33))
        h = h * np.uint64(0xFF51AFD7ED558CCD)
        h = h ^ (h >> np.uint64(33))
    return h


def seeded_rng(variable_name: str, salt: str = None) -> np.random.Generator:
    """Create a per-variable RNG seeded by variable name hash."""
    key = variable_name if salt is None else f"{variable_name}:{salt}"
    seed = int(_stable_string_hash(key)) % (2**63)
    return np.random.default_rng(seed=seed)
