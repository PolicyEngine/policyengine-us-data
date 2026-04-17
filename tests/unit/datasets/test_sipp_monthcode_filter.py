"""Regression test for the SIPP tip-model MONTHCODE filter (N9).

SIPP panels have one row per person per month. ``sipp.train_tip_model``
annualizes tip and employment income as ``monthly_value * 12`` on every
row, then samples training rows. Without a MONTHCODE filter the
training frame contains 12 rows per person — each annualized from a
different month — which inflates effective sample size and mixes
seasonal tip values (restaurant, holiday) with the annual figures.

Fix: filter to MONTHCODE == 12 (end of year) before sampling so every
training row represents one person-year.

This is a source-level regression test because train_tip_model itself
downloads files from HF and trains a QRF.
"""

import ast
from pathlib import Path

import pandas as pd

SIPP_SOURCE = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "policyengine_us_data"
    / "datasets"
    / "sipp"
    / "sipp.py"
)


def test_train_tip_model_filters_to_monthcode_12_before_sampling():
    """The train_tip_model function body must filter ``MONTHCODE``
    before the weighted resample so the training frame has one row
    per person."""
    src = SIPP_SOURCE.read_text()
    tree = ast.parse(src)
    fn = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "train_tip_model"
        ),
        None,
    )
    assert fn is not None
    fn_src = ast.get_source_segment(src, fn)
    assert fn_src is not None

    # Must mention the MONTHCODE == 12 filter somewhere in the body.
    assert 'df["MONTHCODE"] == 12' in fn_src or "df.MONTHCODE == 12" in fn_src, (
        "train_tip_model must filter to MONTHCODE == 12 to avoid 12x "
        "duplicate training rows per person; got:\n" + fn_src
    )


def test_monthcode_filter_collapses_to_one_row_per_person():
    """Toy SIPP-shaped frame: one person with 12 monthly rows, the
    MONTHCODE==12 filter must produce exactly 1 row (their December
    values)."""
    df = pd.DataFrame(
        {
            "SSUID": [1] * 12 + [2] * 12,
            "PNUM": [100] * 12 + [200] * 12,
            "MONTHCODE": list(range(1, 13)) * 2,
            "TPTOTINC": [1000.0] * 24,
        }
    )
    filtered = df[df["MONTHCODE"] == 12]
    assert len(filtered) == 2  # one December row per person
    assert set(filtered["SSUID"].tolist()) == {1, 2}


def test_without_filter_there_are_12x_duplicate_rows_per_person():
    """Pin the bug premise: without the filter the frame has 12
    rows per person after the annualization."""
    df = pd.DataFrame(
        {
            "SSUID": [1] * 12,
            "PNUM": [100] * 12,
            "MONTHCODE": list(range(1, 13)),
            "TPTOTINC": [1000.0] * 12,
        }
    )
    df["employment_income"] = df["TPTOTINC"] * 12
    # Before the fix, every row in the training frame keyed on the
    # (SSUID, PNUM) pair appeared 12 times. The filter is what brings
    # it down to one row per person.
    assert len(df) == 12
    assert len(df[df["MONTHCODE"] == 12]) == 1
