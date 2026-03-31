from __future__ import annotations

import json
import numpy as np
import pytest

from policyengine_us_data.datasets.cps.long_term import calibration as calibration_module
from policyengine_us_data.datasets.cps.long_term.calibration import (
    assess_nonnegative_feasibility,
    build_calibration_audit,
    calibrate_entropy,
    calibrate_weights,
)
from policyengine_us_data.datasets.cps.long_term.calibration_artifacts import (
    rebuild_dataset_manifest,
    update_dataset_manifest,
    write_year_metadata,
)
from policyengine_us_data.datasets.cps.long_term.calibration_profiles import (
    build_profile_from_flags,
    classify_calibration_quality,
    get_profile,
    validate_calibration_audit,
)
from policyengine_us_data.datasets.cps.long_term.ssa_data import (
    available_long_term_target_sources,
    describe_long_term_target_source,
    load_taxable_payroll_projections,
)


class ExplodingCalibrator:
    def calibrate(self, **kwargs):
        raise RuntimeError("boom")


def test_named_profile_lookup():
    profile = get_profile("ss-payroll-tob")
    assert profile.calibration_method == "entropy"
    assert profile.use_greg is False
    assert profile.use_ss is True
    assert profile.use_payroll is True
    assert profile.use_tob is True
    assert profile.use_h6_reform is False
    assert profile.max_negative_weight_pct == 0.0


def test_legacy_flags_map_to_named_profile():
    profile = build_profile_from_flags(
        use_greg=False,
        use_ss=True,
        use_payroll=True,
        use_h6_reform=False,
        use_tob=True,
    )
    assert profile.name == "ss-payroll-tob"
    assert profile.calibration_method == "entropy"


def test_strict_greg_failure_raises():
    X = np.array([[1.0, 0.0], [0.0, 1.0]])
    y_target = np.array([1.0, 1.0])
    baseline_weights = np.array([1.0, 1.0])

    with pytest.raises(RuntimeError, match="fallback was disabled"):
        calibrate_weights(
            X=X,
            y_target=y_target,
            baseline_weights=baseline_weights,
            method="greg",
            calibrator=ExplodingCalibrator(),
            allow_fallback_to_ipf=False,
        )


def test_build_calibration_audit_reports_constraint_error():
    X = np.array([[1.0, 0.0], [0.0, 1.0]])
    y_target = np.array([1.0, 1.0])
    baseline_weights = np.array([1.0, 1.0])
    weights = np.array([1.0, 1.0])
    audit = build_calibration_audit(
        X=X,
        y_target=y_target,
        weights=weights,
        baseline_weights=baseline_weights,
        calibration_event={
            "method_requested": "greg",
            "method_used": "greg",
            "greg_attempted": True,
            "greg_error": None,
            "fell_back_to_ipf": False,
        },
        payroll_values=np.array([10.0, 0.0]),
        payroll_target=20.0,
    )

    assert audit["constraints"]["payroll_total"]["achieved"] == 10.0
    assert audit["constraints"]["payroll_total"]["pct_error"] == -50.0


def test_profile_validation_rejects_fallback_and_large_error():
    profile = build_profile_from_flags(
        use_greg=True,
        use_ss=True,
        use_payroll=True,
        use_h6_reform=False,
        use_tob=False,
    )
    audit = {
        "fell_back_to_ipf": True,
        "age_max_pct_error": 0.0,
        "negative_weight_pct": 0.0,
        "constraints": {
            "payroll_total": {"pct_error": 0.2},
        },
    }

    issues = validate_calibration_audit(audit, profile)
    assert "GREG calibration fell back to IPF" in issues
    assert any("payroll_total error" in issue for issue in issues)


def test_classify_calibration_quality_marks_invalid_audit_approximate():
    profile = get_profile("ss-payroll-tob")
    quality = classify_calibration_quality(
        {
            "fell_back_to_ipf": False,
            "age_max_pct_error": 0.0,
            "negative_weight_pct": 0.0,
            "constraints": {
                "ss_total": {"pct_error": 0.0},
                "payroll_total": {"pct_error": 0.0},
                "oasdi_tob": {"pct_error": 0.5},
                "hi_tob": {"pct_error": 0.0},
            },
        },
        profile,
        year=2078,
    )
    assert quality == "approximate"


def test_entropy_profile_rejects_negative_weights():
    profile = get_profile("ss-payroll-tob")
    issues = validate_calibration_audit(
        {
            "fell_back_to_ipf": False,
            "age_max_pct_error": 0.0,
            "negative_weight_pct": 0.01,
            "constraints": {
                "ss_total": {"pct_error": 0.0},
                "payroll_total": {"pct_error": 0.0},
                "oasdi_tob": {"pct_error": 0.0},
                "hi_tob": {"pct_error": 0.0},
            },
        },
        profile,
    )
    assert any("Negative weight share" in issue for issue in issues)


def test_approximate_window_is_year_bounded():
    profile = get_profile("ss-payroll-tob")
    quality = classify_calibration_quality(
        {
            "fell_back_to_ipf": False,
            "age_max_pct_error": 3.0,
            "negative_weight_pct": 0.0,
            "constraints": {
                "ss_total": {"pct_error": 0.0},
                "payroll_total": {"pct_error": 3.0},
                "oasdi_tob": {"pct_error": 3.0},
                "hi_tob": {"pct_error": 3.0},
            },
        },
        profile,
        year=2080,
    )
    assert quality == "approximate"

    quality = classify_calibration_quality(
        {
            "fell_back_to_ipf": False,
            "age_max_pct_error": 3.0,
            "negative_weight_pct": 0.0,
            "constraints": {
                "ss_total": {"pct_error": 0.0},
                "payroll_total": {"pct_error": 3.0},
                "oasdi_tob": {"pct_error": 3.0},
                "hi_tob": {"pct_error": 3.0},
            },
        },
        profile,
        year=2035,
    )
    assert quality == "aggregate"


def test_manifest_updates_and_rejects_profile_mismatch(tmp_path):
    profile = get_profile("ss-payroll-tob")
    audit = {
        "method_used": "greg",
        "fell_back_to_ipf": False,
        "negative_weight_pct": 1.5,
    }

    year_2026 = tmp_path / "2026.h5"
    year_2026.write_text("", encoding="utf-8")
    metadata_2026 = write_year_metadata(
        year_2026,
        year=2026,
        base_dataset_path="hf://policyengine/policyengine-us-data/enhanced_cps_2024.h5",
        profile=profile.to_dict(),
        calibration_audit=audit,
    )
    manifest_path = update_dataset_manifest(
        tmp_path,
        year=2026,
        h5_path=year_2026,
        metadata_path=metadata_2026,
        base_dataset_path="hf://policyengine/policyengine-us-data/enhanced_cps_2024.h5",
        profile=profile.to_dict(),
        calibration_audit=audit,
    )

    year_2027 = tmp_path / "2027.h5"
    year_2027.write_text("", encoding="utf-8")
    metadata_2027 = write_year_metadata(
        year_2027,
        year=2027,
        base_dataset_path="hf://policyengine/policyengine-us-data/enhanced_cps_2024.h5",
        profile=profile.to_dict(),
        calibration_audit=audit,
    )
    update_dataset_manifest(
        tmp_path,
        year=2027,
        h5_path=year_2027,
        metadata_path=metadata_2027,
        base_dataset_path="hf://policyengine/policyengine-us-data/enhanced_cps_2024.h5",
        profile=profile.to_dict(),
        calibration_audit=audit,
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["profile"]["name"] == "ss-payroll-tob"
    assert manifest["years"] == [2026, 2027]
    assert manifest["datasets"]["2026"]["metadata"] == "2026.h5.metadata.json"

    with pytest.raises(ValueError, match="different calibration profile"):
        update_dataset_manifest(
            tmp_path,
            year=2028,
            h5_path=tmp_path / "2028.h5",
            metadata_path=tmp_path / "2028.h5.metadata.json",
            base_dataset_path="hf://policyengine/policyengine-us-data/enhanced_cps_2024.h5",
            profile=get_profile("ss").to_dict(),
            calibration_audit=audit,
        )

    manifest_path.unlink()
    rebuilt_path = rebuild_dataset_manifest(tmp_path)
    rebuilt = json.loads(rebuilt_path.read_text(encoding="utf-8"))
    assert rebuilt["years"] == [2026, 2027]


def test_entropy_calibration_produces_nonnegative_weights_and_hits_targets():
    X = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]
    )
    y_target = np.array([2.0, 3.0])
    baseline_weights = np.array([1.0, 1.0, 1.0])
    payroll_values = np.array([1.0, 0.0, 2.0])
    payroll_target = 3.5

    weights, _ = calibrate_entropy(
        X=X,
        y_target=y_target,
        baseline_weights=baseline_weights,
        payroll_values=payroll_values,
        payroll_target=payroll_target,
        n_ages=2,
    )

    assert np.all(weights > 0)
    np.testing.assert_allclose(X.T @ weights, y_target, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(np.dot(payroll_values, weights), payroll_target, rtol=1e-8, atol=1e-8)


def test_entropy_calibration_can_fall_back_to_lp_approximate_solution():
    X = np.array(
        [
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
        ]
    )
    y_target = np.array([1.0, 1.0, 3.0])
    baseline_weights = np.array([1.0, 1.0])

    weights, _, audit = calibrate_weights(
        X=X,
        y_target=y_target,
        baseline_weights=baseline_weights,
        method="entropy",
        n_ages=3,
        allow_approximate_entropy=True,
        approximate_max_error_pct=40.0,
    )

    assert audit["approximate_solution_used"] is True
    assert audit["approximation_method"] == "lp_minimax"
    assert audit["approximate_solution_error_pct"] > 10.0
    assert np.all(weights >= 0)


def test_entropy_calibration_uses_lp_exact_fallback_even_before_approximate_window(
    monkeypatch,
):
    monkeypatch.setattr(
        calibration_module,
        "calibrate_entropy",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("entropy stalled")),
    )
    monkeypatch.setattr(
        calibration_module,
        "calibrate_entropy_approximate",
        lambda *args, **kwargs: (
            np.array([1.0, 2.0]),
            1,
            {"best_case_max_pct_error": 0.0},
        ),
    )

    weights, _, audit = calibrate_weights(
        X=np.array([[1.0], [0.0]]),
        y_target=np.array([1.0]),
        baseline_weights=np.array([1.0, 1.0]),
        method="entropy",
        n_ages=1,
        allow_approximate_entropy=False,
    )

    np.testing.assert_allclose(weights, np.array([1.0, 2.0]))
    assert audit["lp_fallback_used"] is True
    assert audit["approximate_solution_used"] is False
    assert audit["approximation_method"] == "lp_minimax_exact"


def test_nonnegative_feasibility_diagnostic_distinguishes_feasible_and_infeasible():
    feasible_A = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]
    )
    feasible_targets = np.array([1.0, 2.0, 3.0])
    feasible = assess_nonnegative_feasibility(feasible_A, feasible_targets)
    assert feasible["success"] is True
    assert feasible["best_case_max_pct_error"] < 1e-6

    infeasible_A = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]
    )
    infeasible_targets = np.array([1.0, 1.0, 3.0])
    infeasible = assess_nonnegative_feasibility(infeasible_A, infeasible_targets)
    assert infeasible["success"] is True
    assert infeasible["best_case_max_pct_error"] > 10.0


def test_long_term_target_sources_are_available_and_distinct():
    sources = available_long_term_target_sources()
    assert "trustees_2025_current_law" in sources

    trustees = describe_long_term_target_source("trustees_2025_current_law")
    assert trustees["file"] == "trustees_2025_current_law.csv"

    payroll_2026 = load_taxable_payroll_projections(
        2026,
        source_name="trustees_2025_current_law",
    )
    assert payroll_2026 == pytest.approx(11_129_000_000_000.0)
