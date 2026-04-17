"""Regression test for SIPP tip income column matching.

Previously `str.contains("TXAMT")` matched both `TJB*_TXAMT` (dollar
amounts) and `AJB*_TXAMT` (Census allocation flags). The fix narrows
to `TJB\\d_TXAMT` (dollar-amount columns only).
"""

import pandas as pd


def test_tip_regex_matches_dollar_amounts_only():
    # SIPP column naming: TJB<N>_TXAMT is the dollar amount for job N,
    # AJB<N>_TXAMT is the allocation flag for the same field.
    # Include several distractors that the old `contains("TXAMT")` regex
    # would have caught.
    columns = pd.Index(
        [
            "TJB1_TXAMT",
            "TJB2_TXAMT",
            "AJB1_TXAMT",  # allocation flag — should NOT be summed
            "AJB2_TXAMT",  # allocation flag — should NOT be summed
            "SOME_TXAMT_OTHER",  # unrelated non-numbered column
            "TPTOTINC",  # unrelated
        ]
    )

    matches = columns[columns.str.match(r"TJB\d_TXAMT")]

    assert list(matches) == ["TJB1_TXAMT", "TJB2_TXAMT"]


def test_tip_sum_excludes_allocation_flags():
    df = pd.DataFrame(
        {
            "TJB1_TXAMT": [100.0, 200.0],
            "TJB2_TXAMT": [50.0, 75.0],
            "AJB1_TXAMT": [1, 2],  # allocation flags: small ints
            "AJB2_TXAMT": [0, 1],
        }
    )
    # Mirror the sipp.py computation using the new regex.
    tip_income_monthly = (
        df[df.columns[df.columns.str.match(r"TJB\d_TXAMT")]].fillna(0).sum(axis=1)
    )
    assert list(tip_income_monthly) == [150.0, 275.0]

    # Sanity check: the buggy regex would have included AJB flags.
    buggy_tip_income_monthly = (
        df[df.columns[df.columns.str.contains("TXAMT")]].fillna(0).sum(axis=1)
    )
    assert list(buggy_tip_income_monthly) == [151.0, 278.0]
