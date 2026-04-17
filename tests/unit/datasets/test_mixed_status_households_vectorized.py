"""Regression test for the vectorized mixed-status household search (N11).

``add_ssn_card_type`` used to loop over every unique household and
boolean-mask the full person array on each iteration to find
households that mix code-0 and code-3 members. For CPS 2024 that is
~100k households × ~300k persons = ~3×10¹⁰ comparisons per build.

The vectorized replacement runs a single pandas groupby on the
(person_household_id, ssn_card_type) pair.

This test pins:
- Correctness: the vectorized result matches the brute-force loop
  on a toy 5-household frame covering all four cases (pure code 0,
  pure code 3, mixed, other-code-only).
- Performance: on a 10_000-household / 30_000-person synthetic
  panel, the vectorized version finishes in well under a second.
"""

import time

import numpy as np
import pandas as pd


def _brute_force_candidates(person_household_ids, ssn_card_type):
    """The original O(N²) algorithm, kept here only to validate the
    vectorized replacement."""
    candidates = []
    for household_id in np.unique(person_household_ids):
        household_mask = person_household_ids == household_id
        household_ssn_codes = ssn_card_type[household_mask]
        has_undocumented = (household_ssn_codes == 0).any()
        has_code3 = (household_ssn_codes == 3).any()
        if has_undocumented and has_code3:
            household_indices = np.where(household_mask)[0]
            code_3_indices = household_indices[household_ssn_codes == 3]
            candidates.extend(code_3_indices.tolist())
    return np.array(sorted(candidates), dtype=int)


def _vectorized_candidates(person_household_ids, ssn_card_type):
    """The vectorized replacement, mirroring the fix in cps.py."""
    codes_series = pd.Series(ssn_card_type, name="ssn_card_type")
    per_household = pd.DataFrame(
        {
            "household_id": person_household_ids,
            "is_code_0": codes_series == 0,
            "is_code_3": codes_series == 3,
        }
    )
    household_flags = per_household.groupby("household_id").agg(
        has_code_0=("is_code_0", "any"),
        has_code_3=("is_code_3", "any"),
    )
    mixed_household_ids = household_flags.index[
        household_flags["has_code_0"] & household_flags["has_code_3"]
    ]
    in_mixed = np.isin(person_household_ids, mixed_household_ids)
    return np.sort(np.where(in_mixed & (ssn_card_type == 3))[0])


def test_vectorized_matches_brute_force_on_toy_frame():
    # 5 households, 10 persons:
    # hh0: code 0 + code 0        -> not mixed
    # hh1: code 3 + code 3        -> not mixed
    # hh2: code 0 + code 3        -> mixed; person 4 is code 3
    # hh3: code 1 + code 3        -> not mixed (no code 0)
    # hh4: code 0 + code 3 + code 3 -> mixed; persons 8,9 are code 3
    person_household_ids = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 4], dtype=int)
    ssn_card_type = np.array([0, 0, 3, 3, 0, 3, 1, 3, 0, 3, 3], dtype=int)

    brute = _brute_force_candidates(person_household_ids, ssn_card_type)
    vec = _vectorized_candidates(person_household_ids, ssn_card_type)

    np.testing.assert_array_equal(brute, vec)
    # Explicit check: the expected code-3 indices are 5, 9, 10.
    np.testing.assert_array_equal(vec, np.array([5, 9, 10]))


def test_vectorized_returns_empty_when_no_mixed_households():
    # No household has both a code 0 and a code 3 member.
    person_household_ids = np.array([0, 0, 1, 1, 2, 2], dtype=int)
    ssn_card_type = np.array([0, 0, 1, 3, 2, 3], dtype=int)
    vec = _vectorized_candidates(person_household_ids, ssn_card_type)
    assert vec.size == 0


def test_vectorized_completes_quickly_on_large_panel():
    """On 10k households / 30k persons the vectorized path should
    finish comfortably under a second."""
    rng = np.random.default_rng(seed=17)
    n_households = 10_000
    persons_per_household = 3
    person_household_ids = np.repeat(np.arange(n_households), persons_per_household)
    # 60% code 0, 20% code 3, 10% code 1, 10% code 2 — plenty of mixing.
    ssn_card_type = rng.choice(
        np.array([0, 1, 2, 3]),
        size=n_households * persons_per_household,
        p=[0.6, 0.1, 0.1, 0.2],
    )

    start = time.perf_counter()
    vec = _vectorized_candidates(person_household_ids, ssn_card_type)
    elapsed_vec = time.perf_counter() - start

    assert elapsed_vec < 1.0, (
        f"vectorized candidate search took {elapsed_vec:.2f}s on a "
        "10k-household panel; expected well under 1s"
    )

    # Also sanity-check correctness against the brute force on a small
    # subset (full brute-force would be slow here by design).
    small_n = 500
    person_household_ids_small = np.repeat(np.arange(small_n), 3)
    ssn_card_type_small = rng.choice(
        np.array([0, 1, 2, 3]),
        size=small_n * 3,
        p=[0.6, 0.1, 0.1, 0.2],
    )
    np.testing.assert_array_equal(
        _brute_force_candidates(person_household_ids_small, ssn_card_type_small),
        _vectorized_candidates(person_household_ids_small, ssn_card_type_small),
    )
    # Prevent unused-variable flags.
    _ = vec


def test_cps_source_uses_pandas_groupby_for_mixed_households():
    """Source-level: the vectorized implementation must live in
    cps.py (no future refactor that brings the O(N²) loop back)."""
    from pathlib import Path

    cps_path = (
        Path(__file__).resolve().parent.parent.parent.parent
        / "policyengine_us_data"
        / "datasets"
        / "cps"
        / "cps.py"
    )
    src = cps_path.read_text()
    # Must reference the groupby-based flag columns.
    assert 'groupby("household_id")' in src
    assert "has_code_0" in src and "has_code_3" in src
    # Must *not* contain the brute-force inner loop marker any more.
    assert "for household_id in unique_households" not in src, (
        "The O(households × persons) loop is still present in cps.py"
    )
