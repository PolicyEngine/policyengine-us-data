"""Auto loan balance/interest summarization from raw SCF columns."""

import pandas as pd

from policyengine_us_data.datasets.scf.scf import _summarize_auto_loans


def test_summarize_auto_loans_floors_negative_sentinels():
    """Negative SCF "not applicable" sentinels must not produce negative or
    inflated auto loan amounts.

    Row 0: a real $20,000 loan at 5% (rate stored as 500 -> 0.05) plus -1
    sentinels for the absent cars 2-4. Row 1: a real balance whose rate is
    unreported (-1) -- the exact case that previously yielded a spurious
    negative interest (balance * -1 / 10,000).
    """
    df = pd.DataFrame(
        {
            "x2209": [20000.0, 30000.0],  # balance car 1
            "x2309": [-1.0, -1.0],  # balance car 2 (n/a sentinel)
            "x2409": [-1.0, -1.0],
            "x7158": [-1.0, -1.0],
            "x2219": [500.0, -1.0],  # rate car 1 (0.05; row 1 unreported)
            "x2319": [-1.0, -1.0],
            "x2419": [-1.0, -1.0],
            "x7170": [-1.0, -1.0],
        }
    )

    out = _summarize_auto_loans(df)

    assert (out["auto_loan_interest"] >= 0).all()
    assert (out["auto_loan_balance"] >= 0).all()
    # Only the real loan contributes: 20,000 * 0.05 = 1,000.
    assert out["auto_loan_interest"].iloc[0] == 1000.0
    # Real balance but unreported rate -> zero interest, not a negative.
    assert out["auto_loan_interest"].iloc[1] == 0.0
    # Sentinel balances do not inflate the total.
    assert out["auto_loan_balance"].iloc[0] == 20000.0
