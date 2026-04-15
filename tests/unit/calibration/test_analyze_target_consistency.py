"""Unit tests for analyze_target_consistency."""

import pandas as pd
import pytest

from policyengine_us_data.calibration.analyze_target_consistency import (
    bucket_coverage_check,
    buckets_form_complete_partition,
    classify_geo_level,
    classify_severity,
    cross_level_check,
    extract_agi_bounds,
    has_agi_bucket_clause,
    load_diagnostics,
    parse_target,
    strip_agi_bucket_clauses,
)


class TestParsing:
    def test_parse_target_three_parts(self):
        geo, var, filt = parse_target("cd_615/foo/[bar==1]")
        assert geo == "cd_615"
        assert var == "foo"
        assert filt == "[bar==1]"

    def test_parse_target_two_parts(self):
        geo, var, filt = parse_target("national/foo")
        assert geo == "national"
        assert var == "foo"
        assert filt == ""

    def test_parse_target_preserves_inner_slashes(self):
        geo, var, filt = parse_target("state_06/foo/[a/b==1]")
        assert geo == "state_06"
        assert var == "foo"
        assert filt == "[a/b==1]"

    def test_classify_geo_level(self):
        assert classify_geo_level("national") == "national"
        assert classify_geo_level("state_06") == "state"
        assert classify_geo_level("cd_0601") == "district"
        assert classify_geo_level("weird_x") == "other"


class TestAgiStripping:
    def test_strip_removes_both_sides(self):
        base = strip_agi_bucket_clauses(
            "[adjusted_gross_income<1000000,"
            "adjusted_gross_income>=500000,tax_unit_is_filer==1]"
        )
        assert base == "[tax_unit_is_filer==1]"

    def test_strip_leaves_unbucketed_alone(self):
        assert (
            strip_agi_bucket_clauses("[tax_unit_is_filer==1,total_se>0]")
            == "[tax_unit_is_filer==1,total_se>0]"
        )

    def test_strip_empty_filter(self):
        assert strip_agi_bucket_clauses("") == ""

    def test_has_agi_bucket_clause(self):
        assert has_agi_bucket_clause(
            "[adjusted_gross_income<1.0,adjusted_gross_income>=-inf,tax_unit_is_filer==1]"
        )
        assert not has_agi_bucket_clause("[tax_unit_is_filer==1]")

    def test_strip_handles_inf_and_negative(self):
        base = strip_agi_bucket_clauses(
            "[adjusted_gross_income<inf,adjusted_gross_income>=-inf,filer==1]"
        )
        assert base == "[filer==1]"


class TestSeverity:
    def test_thresholds(self):
        assert classify_severity(0.5) == "ok"
        assert classify_severity(1.0) == "flag"
        assert classify_severity(4.99) == "flag"
        assert classify_severity(5.0) == "significant"
        assert classify_severity(14.99) == "significant"
        assert classify_severity(15.0) == "critical"
        assert classify_severity(21.0) == "critical"

    def test_nan_is_ok(self):
        import numpy as np

        assert classify_severity(np.nan) == "ok"


def _make_diagnostics_dataframe(rows):
    """Helper: build a diagnostics-shaped dataframe from (target, true_value) pairs."""
    df = pd.DataFrame(rows, columns=["target", "true_value"])
    # Mimic the other diagnostic columns so load_diagnostics's column check passes
    df["estimate"] = df["true_value"]
    df["rel_error"] = 0.0
    df["abs_rel_error"] = 0.0
    df["achievable"] = True
    return df


class TestCrossLevelCheck:
    def test_flags_5_percent_gap(self, tmp_path):
        # national $1050, sum of 2 districts $1000 → +5% gap
        rows = [
            ("national/foo/[bar>0]", 1_050.0),
            ("cd_0101/foo/[bar>0]", 400.0),
            ("cd_0102/foo/[bar>0]", 600.0),
        ]
        df = _make_diagnostics_dataframe(rows)
        path = tmp_path / "diag.csv"
        df.to_csv(path, index=False)
        loaded = load_diagnostics(path)
        report = cross_level_check(loaded)
        assert len(report) == 1
        row = report.iloc[0]
        assert row["variable"] == "foo"
        assert row["national"] == pytest.approx(1050.0)
        assert row["sum_districts"] == pytest.approx(1000.0)
        assert row["rel_nat_vs_dist_pct"] == pytest.approx(5.0)
        assert row["severity"] == "significant"

    def test_does_not_flag_clean_group(self, tmp_path):
        # national exactly equals sum of districts
        rows = [
            ("national/foo/[bar>0]", 1_000.0),
            ("cd_0101/foo/[bar>0]", 400.0),
            ("cd_0102/foo/[bar>0]", 600.0),
        ]
        df = _make_diagnostics_dataframe(rows)
        path = tmp_path / "diag.csv"
        df.to_csv(path, index=False)
        loaded = load_diagnostics(path)
        report = cross_level_check(loaded)
        assert len(report) == 1
        assert report.iloc[0]["severity"] == "ok"

    def test_ignores_single_level_groups(self, tmp_path):
        # only district → not comparable
        rows = [
            ("cd_0101/foo/[bar>0]", 400.0),
            ("cd_0102/foo/[bar>0]", 600.0),
        ]
        df = _make_diagnostics_dataframe(rows)
        path = tmp_path / "diag.csv"
        df.to_csv(path, index=False)
        loaded = load_diagnostics(path)
        report = cross_level_check(loaded)
        assert report.empty

    def test_three_level_critical(self, tmp_path):
        # national 2000, state sum 1000, district sum 1000 → national vs others +100%
        rows = [
            ("national/foo/[bar>0]", 2_000.0),
            ("state_01/foo/[bar>0]", 500.0),
            ("state_02/foo/[bar>0]", 500.0),
            ("cd_0101/foo/[bar>0]", 400.0),
            ("cd_0201/foo/[bar>0]", 600.0),
        ]
        df = _make_diagnostics_dataframe(rows)
        path = tmp_path / "diag.csv"
        df.to_csv(path, index=False)
        loaded = load_diagnostics(path)
        report = cross_level_check(loaded)
        assert len(report) == 1
        row = report.iloc[0]
        assert row["national"] == pytest.approx(2000.0)
        assert row["sum_states"] == pytest.approx(1000.0)
        assert row["sum_districts"] == pytest.approx(1000.0)
        assert row["rel_nat_vs_state_pct"] == pytest.approx(100.0)
        assert row["rel_state_vs_dist_pct"] == pytest.approx(0.0)
        assert row["severity"] == "critical"


class TestBucketCoverageCheck:
    def test_complete_coverage_not_flagged(self, tmp_path):
        # 3 buckets forming a complete partition (-inf, inf) + 1 aggregate,
        # buckets sum to aggregate
        rows = [
            ("national/foo/[bar>0]", 1_000.0),
            (
                "national/foo/[adjusted_gross_income<100,adjusted_gross_income>=-inf,bar>0]",
                200.0,
            ),
            (
                "national/foo/[adjusted_gross_income<500,adjusted_gross_income>=100,bar>0]",
                500.0,
            ),
            (
                "national/foo/[adjusted_gross_income<inf,adjusted_gross_income>=500,bar>0]",
                300.0,
            ),
        ]
        df = _make_diagnostics_dataframe(rows)
        path = tmp_path / "diag.csv"
        df.to_csv(path, index=False)
        loaded = load_diagnostics(path)
        report = bucket_coverage_check(loaded)
        assert len(report) == 1
        row = report.iloc[0]
        assert row["aggregate"] == pytest.approx(1000.0)
        assert row["sum_buckets"] == pytest.approx(1000.0)
        assert row["n_buckets"] == 3
        assert row["severity"] == "ok"

    def test_incomplete_bucket_values_flagged(self, tmp_path):
        # Complete partition (-inf, inf), but bucket values sum to $500 while
        # aggregate is $1000 → −50% gap (data error, not partition gap)
        rows = [
            ("national/foo/[bar>0]", 1_000.0),
            (
                "national/foo/[adjusted_gross_income<100,adjusted_gross_income>=-inf,bar>0]",
                200.0,
            ),
            (
                "national/foo/[adjusted_gross_income<inf,adjusted_gross_income>=100,bar>0]",
                300.0,
            ),
        ]
        df = _make_diagnostics_dataframe(rows)
        path = tmp_path / "diag.csv"
        df.to_csv(path, index=False)
        loaded = load_diagnostics(path)
        report = bucket_coverage_check(loaded)
        assert len(report) == 1
        row = report.iloc[0]
        assert row["aggregate"] == pytest.approx(1000.0)
        assert row["sum_buckets"] == pytest.approx(500.0)
        assert row["sum_vs_agg_pct"] == pytest.approx(-50.0)
        assert row["severity"] == "critical"

    def test_skips_when_no_aggregate(self, tmp_path):
        # only buckets, no aggregate → nothing to compare against
        rows = [
            (
                "national/foo/[adjusted_gross_income<100,adjusted_gross_income>=0,bar>0]",
                200.0,
            ),
            (
                "national/foo/[adjusted_gross_income<500,adjusted_gross_income>=100,bar>0]",
                300.0,
            ),
        ]
        df = _make_diagnostics_dataframe(rows)
        path = tmp_path / "diag.csv"
        df.to_csv(path, index=False)
        loaded = load_diagnostics(path)
        report = bucket_coverage_check(loaded)
        assert report.empty

    def test_skips_partial_range_buckets(self, tmp_path):
        # buckets only cover $500K+ of AGI, aggregate is the full filer total.
        # These must NOT be compared (partial partition).
        rows = [
            ("national/foo/[bar>0]", 15_000.0),
            (
                "national/foo/[adjusted_gross_income<1000,adjusted_gross_income>=500,bar>0]",
                1_000.0,
            ),
            (
                "national/foo/[adjusted_gross_income<inf,adjusted_gross_income>=1000,bar>0]",
                500.0,
            ),
        ]
        df = _make_diagnostics_dataframe(rows)
        path = tmp_path / "diag.csv"
        df.to_csv(path, index=False)
        loaded = load_diagnostics(path)
        report = bucket_coverage_check(loaded)
        # partition is not complete ($-inf to $500 missing) → skip
        assert report.empty


class TestPartitionDetection:
    def test_complete_partition(self):
        bounds = [
            (float("-inf"), 100.0),
            (100.0, 500.0),
            (500.0, float("inf")),
        ]
        assert buckets_form_complete_partition(bounds)

    def test_missing_bottom(self):
        bounds = [
            (100.0, 500.0),
            (500.0, float("inf")),
        ]
        assert not buckets_form_complete_partition(bounds)

    def test_missing_top(self):
        bounds = [
            (float("-inf"), 100.0),
            (100.0, 500.0),
        ]
        assert not buckets_form_complete_partition(bounds)

    def test_gap_in_middle(self):
        bounds = [
            (float("-inf"), 100.0),
            (200.0, float("inf")),  # gap 100..200
        ]
        assert not buckets_form_complete_partition(bounds)

    def test_extract_bounds(self):
        assert extract_agi_bounds(
            "[adjusted_gross_income<1000000,adjusted_gross_income>=500000,foo]"
        ) == (500000.0, 1000000.0)
        assert extract_agi_bounds(
            "[adjusted_gross_income<inf,adjusted_gross_income>=-inf,foo]"
        ) == (float("-inf"), float("inf"))
        assert extract_agi_bounds("[foo==1]") is None
