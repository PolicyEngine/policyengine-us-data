"""Helpers for validating JCT calibration diagnostics."""

JCT_REL_ABS_ERROR_LIMIT = 0.5

# These rows are kept in the dense calibration log as diagnostics, but
# current model/data support does not make them reliable release gates.
KNOWN_HIGH_ERROR_JCT_DIAGNOSTICS = {
    "nation/jct/charitable_deduction_expenditure",
    "nation/jct/interest_deduction_expenditure",
    "nation/jct/qualified_business_income_deduction_expenditure",
    "nation/jct/salt_deduction_expenditure",
}


def assert_no_unexpected_high_error_jct_diagnostics(calibration_log):
    final_epoch = calibration_log["epoch"].max()
    jct_rows = calibration_log[
        (calibration_log["target_name"].str.contains("jct/"))
        & (calibration_log["epoch"] == final_epoch)
    ]

    assert not jct_rows.empty, "No final-epoch JCT diagnostics found."

    high_error_rows = jct_rows[jct_rows["rel_abs_error"] >= JCT_REL_ABS_ERROR_LIMIT]
    unexpected = high_error_rows[
        ~high_error_rows["target_name"].isin(KNOWN_HIGH_ERROR_JCT_DIAGNOSTICS)
    ]

    assert unexpected.empty, (
        "Unexpected JCT tax expenditure diagnostics exceeded "
        f"{JCT_REL_ABS_ERROR_LIMIT:.0%} relative absolute error:\n"
        + unexpected[["target_name", "target", "estimate", "rel_abs_error"]].to_string(
            index=False
        )
    )
