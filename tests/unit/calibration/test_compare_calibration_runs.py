import json

import pandas as pd

from policyengine_us_data.calibration.compare_calibration_runs import (
    DEFAULT_VARIABLES,
    RunComparisonPaths,
    build_h5_comparison_rows,
    build_markdown_report,
    determine_overall_status,
    parse_variables,
    summarize_diagnostics,
    summarize_h5_rows,
    write_csv,
)


def test_run_comparison_paths_are_run_scoped():
    paths = RunComparisonPaths("usdata-gha123-a1", version="1.73.0")

    assert (
        paths.regional_diagnostics
        == "hf://policyengine/policyengine-us-data/calibration/runs/"
        "usdata-gha123-a1/diagnostics/unified_diagnostics.csv"
    )
    assert (
        paths.national_diagnostics
        == "hf://policyengine/policyengine-us-data/calibration/runs/"
        "usdata-gha123-a1/diagnostics/national_unified_diagnostics.csv"
    )
    assert (
        paths.candidate_h5 == "hf://policyengine/policyengine-us-data/staging/"
        "1.73.0-usdata-gha123-a1/national/US.h5"
    )
    assert (
        paths.legacy_h5 == "hf://policyengine/policyengine-us-data/staging/"
        "1.73.0-usdata-gha123-a1/enhanced_cps_2024.h5"
    )


def test_parse_variables_preserves_requested_order():
    assert parse_variables("snap, eitc, ctc") == ["snap", "eitc", "ctc"]


def test_default_h5_comparison_uses_raw_ssi():
    assert "ssi" in DEFAULT_VARIABLES
    assert "ssi_federal_fiscal_year_outlays" not in DEFAULT_VARIABLES


def test_summarize_diagnostics_uses_achievable_target_tail():
    diagnostics = pd.DataFrame(
        {
            "target": ["a", "b", "c", "impossible"],
            "abs_rel_error": [0.01, 0.02, 0.20, 9.0],
            "achievable": [True, True, True, False],
        }
    )

    summary = summarize_diagnostics(
        diagnostics,
        label="regional",
        p95_threshold=0.10,
        p99_threshold=0.50,
    )

    assert summary["status"] == "WARN"
    assert summary["target_count"] == 4
    assert summary["achievable_target_count"] == 3
    assert summary["unachievable_target_count"] == 1
    assert summary["worst_target"] == "c"
    assert summary["worst_abs_rel_error"] == 0.20


def test_build_h5_comparison_rows_marks_large_deltas():
    rows = build_h5_comparison_rows(
        candidate_totals={
            "snap": 132.0,
            "person_count": 110.0,
            "missing": {"error": "not defined"},
        },
        legacy_totals={
            "snap": 100.0,
            "person_count": 100.0,
            "missing": 1.0,
        },
        reference_values={"snap": (120.0, "reference")},
        max_delta_vs_legacy=0.20,
    )
    by_variable = {row["variable"]: row for row in rows}

    assert by_variable["snap"]["status"] == "WARN"
    assert by_variable["snap"]["pct_delta_vs_legacy"] == 0.32
    assert by_variable["snap"]["candidate_pct_error_vs_reference"] == 0.10
    assert by_variable["person_count"]["display_type"] == "count"
    assert by_variable["missing"]["status"] == "ERROR"

    summary = summarize_h5_rows(rows)
    assert summary["status"] == "ERROR"
    assert summary["worst_variable"] == "snap"


def test_determine_overall_status_prefers_error_then_warn():
    assert (
        determine_overall_status([{"status": "PASS"}, {"status": "WARN"}], []) == "WARN"
    )
    assert (
        determine_overall_status(
            [{"status": "PASS"}],
            [{"status": "ERROR", "pct_delta_vs_legacy": None}],
        )
        == "ERROR"
    )


def test_write_report_artifacts(tmp_path):
    summaries = [
        {
            "label": "regional",
            "status": "PASS",
            "achievable_target_count": 2,
            "p95_abs_rel_error": 0.01,
            "p99_abs_rel_error": 0.02,
            "worst_target": "target_b",
            "worst_abs_rel_error": 0.02,
        }
    ]
    h5_rows = [
        {
            "variable": "snap",
            "status": "PASS",
            "candidate_value": 105.0,
            "legacy_value": 100.0,
            "pct_delta_vs_legacy": 0.05,
            "candidate_pct_error_vs_reference": None,
            "legacy_pct_error_vs_reference": None,
            "display_type": "amount",
        }
    ]

    markdown = build_markdown_report(
        run_id="run-1",
        summaries=summaries,
        h5_rows=h5_rows,
        overall_status="PASS",
        fail_on_threshold=False,
    )
    assert "Calibration comparison report" in markdown
    assert "target_b" in markdown

    csv_path = tmp_path / "diagnostics_summary.csv"
    write_csv(csv_path, summaries)
    assert csv_path.read_text().splitlines()[0]

    json_path = tmp_path / "summary.json"
    json_path.write_text(json.dumps({"overall_status": "PASS"}), encoding="utf-8")
    assert json.loads(json_path.read_text())["overall_status"] == "PASS"
