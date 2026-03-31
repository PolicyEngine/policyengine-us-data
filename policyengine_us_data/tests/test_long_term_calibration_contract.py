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
    normalize_metadata,
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
from policyengine_us_data.datasets.cps.long_term.projection_utils import (
    aggregate_age_targets,
    aggregate_household_age_matrix,
    build_age_bins,
)
from policyengine_us_data.datasets.cps.long_term.ssa_data import (
    available_long_term_target_sources,
    describe_long_term_target_source,
    load_taxable_payroll_projections,
)
from policyengine_us_data.datasets.cps.long_term.support_augmentation import (
    AgeShiftCloneRule,
    CompositePayrollRule,
    SinglePersonSyntheticGridRule,
    SupportAugmentationProfile,
    augment_input_dataframe,
    household_support_summary,
    select_donor_households,
)


class ExplodingCalibrator:
    def calibrate(self, **kwargs):
        raise RuntimeError("boom")


def _toy_support_dataframe():
    return json.loads(
        json.dumps(
            {
                "person_id__2024": [101, 102, 201, 202, 301],
                "household_id__2024": [1, 1, 2, 2, 3],
                "person_household_id__2024": [1, 1, 2, 2, 3],
                "family_id__2024": [11.0, 11.0, 21.0, 21.0, 31.0],
                "person_family_id__2024": [11, 11, 21, 21, 31],
                "tax_unit_id__2024": [101, 101, 201, 202, 301],
                "person_tax_unit_id__2024": [101, 101, 201, 202, 301],
                "spm_unit_id__2024": [1001, 1001, 2001, 2001, 3001],
                "person_spm_unit_id__2024": [1001, 1001, 2001, 2001, 3001],
                "marital_unit_id__2024": [501, 501, 601, 602, 701],
                "person_marital_unit_id__2024": [501, 501, 601, 602, 701],
                "age__2024": [70.0, 68.0, 80.0, 77.0, 60.0],
                "household_weight__2024": [10.0, 10.0, 8.0, 8.0, 5.0],
                "person_weight__2024": [10.0, 10.0, 8.0, 8.0, 5.0],
                "social_security_retirement__2024": [20_000.0, 0.0, 30_000.0, 0.0, 0.0],
                "social_security_disability__2024": [0.0, 0.0, 0.0, 0.0, 0.0],
                "social_security_survivors__2024": [0.0, 0.0, 0.0, 0.0, 0.0],
                "social_security_dependents__2024": [0.0, 0.0, 0.0, 0.0, 0.0],
                "employment_income_before_lsr__2024": [5_000.0, 0.0, 12_000.0, 0.0, 50_000.0],
                "self_employment_income_before_lsr__2024": [0.0, 0.0, 0.0, 0.0, 0.0],
            }
        )
    )


def test_named_profile_lookup():
    profile = get_profile("ss-payroll-tob")
    assert profile.calibration_method == "entropy"
    assert profile.use_greg is False
    assert profile.use_ss is True
    assert profile.use_payroll is True
    assert profile.use_tob is False
    assert profile.benchmark_tob is True
    assert profile.use_h6_reform is False
    assert profile.max_negative_weight_pct == 0.0
    assert profile.approximate_windows[0].age_bucket_size == 5
    assert profile.min_positive_household_count == 1000
    assert profile.min_effective_sample_size == 75.0
    assert profile.max_top_10_weight_share_pct == 25.0
    assert profile.max_top_100_weight_share_pct == 95.0


def test_support_augmentation_selects_expected_donors():
    import pandas as pd

    df = pd.DataFrame(_toy_support_dataframe())
    summary = household_support_summary(df, base_year=2024)
    rule = AgeShiftCloneRule(
        name="older_ss_pay",
        min_max_age=65,
        max_max_age=74,
        age_shift=10,
        ss_state="positive",
        payroll_state="positive",
    )
    donors = select_donor_households(summary, rule)
    assert list(donors) == [1]


def test_support_augmentation_clones_households_with_new_ids():
    import pandas as pd

    df = pd.DataFrame(_toy_support_dataframe())
    profile = SupportAugmentationProfile(
        name="test-profile",
        description="Toy support augmentation profile.",
        rules=(
            AgeShiftCloneRule(
                name="older_ss_pay",
                min_max_age=65,
                max_max_age=74,
                age_shift=10,
                ss_state="positive",
                payroll_state="positive",
                clone_weight_scale=0.5,
            ),
        ),
    )
    augmented_df, report = augment_input_dataframe(
        df,
        base_year=2024,
        profile=profile,
    )
    assert report["base_household_count"] == 3
    assert report["augmented_household_count"] == 4
    cloned_household_ids = set(augmented_df["household_id__2024"].unique()) - {1, 2, 3}
    assert len(cloned_household_ids) == 1
    cloned_rows = augmented_df[augmented_df["household_id__2024"].isin(cloned_household_ids)]
    assert cloned_rows["age__2024"].max() == pytest.approx(80.0)
    assert cloned_rows["household_weight__2024"].iloc[0] == pytest.approx(5.0)
    assert cloned_rows["person_id__2024"].min() > df["person_id__2024"].max()


def test_support_augmentation_synthesizes_composite_payroll_household():
    import pandas as pd

    df = pd.DataFrame(_toy_support_dataframe())
    profile = SupportAugmentationProfile(
        name="composite-profile",
        description="Toy composite support augmentation profile.",
        rules=(
            CompositePayrollRule(
                name="older_ss_only_plus_payroll",
                recipient_min_max_age=75,
                recipient_max_max_age=84,
                donor_min_max_age=55,
                donor_max_max_age=64,
                recipient_ss_state="positive",
                recipient_payroll_state="positive",
                donor_ss_state="nonpositive",
                donor_payroll_state="positive",
                payroll_transfer_scale=0.5,
                clone_weight_scale=0.25,
            ),
        ),
    )
    augmented_df, report = augment_input_dataframe(
        df,
        base_year=2024,
        profile=profile,
    )
    assert report["base_household_count"] == 3
    assert report["augmented_household_count"] == 4
    cloned_household_ids = set(augmented_df["household_id__2024"].unique()) - {1, 2, 3}
    assert len(cloned_household_ids) == 1
    cloned_rows = augmented_df[augmented_df["household_id__2024"].isin(cloned_household_ids)]
    assert cloned_rows["age__2024"].max() == pytest.approx(80.0)
    assert cloned_rows["social_security_retirement__2024"].sum() == pytest.approx(30_000.0)
    assert cloned_rows["employment_income_before_lsr__2024"].sum() == pytest.approx(37_000.0)


def test_support_augmentation_appends_single_person_synthetic_grid_households():
    import pandas as pd

    df = pd.DataFrame(
        {
            "person_id__2024": [101, 201, 301],
            "household_id__2024": [1, 2, 3],
            "person_household_id__2024": [1, 2, 3],
            "family_id__2024": [11.0, 21.0, 31.0],
            "person_family_id__2024": [11, 21, 31],
            "tax_unit_id__2024": [101, 201, 301],
            "person_tax_unit_id__2024": [101, 201, 301],
            "spm_unit_id__2024": [1001, 2001, 3001],
            "person_spm_unit_id__2024": [1001, 2001, 3001],
            "marital_unit_id__2024": [501, 601, 701],
            "person_marital_unit_id__2024": [501, 601, 701],
            "age__2024": [78.0, 86.0, 60.0],
            "household_weight__2024": [10.0, 8.0, 5.0],
            "person_weight__2024": [10.0, 8.0, 5.0],
            "social_security_retirement__2024": [20_000.0, 24_000.0, 0.0],
            "social_security_disability__2024": [0.0, 0.0, 0.0],
            "social_security_survivors__2024": [0.0, 0.0, 0.0],
            "social_security_dependents__2024": [0.0, 0.0, 0.0],
            "employment_income_before_lsr__2024": [0.0, 0.0, 50_000.0],
            "self_employment_income_before_lsr__2024": [0.0, 0.0, 0.0],
            "w2_wages_from_qualified_business__2024": [0.0, 0.0, 0.0],
        }
    )
    profile = SupportAugmentationProfile(
        name="grid-profile",
        description="Toy single-person synthetic grid.",
        rules=(
                SinglePersonSyntheticGridRule(
                    name="older_grid",
                    template_min_max_age=75,
                    template_max_max_age=86,
                    target_ages=(77, 85),
                    ss_quantiles=(0.5,),
                    payroll_quantiles=(0.5,),
                template_ss_state="positive",
                template_payroll_state="any",
                payroll_donor_min_max_age=55,
                payroll_donor_max_max_age=64,
                clone_weight_scale=0.2,
            ),
        ),
    )
    augmented_df, report = augment_input_dataframe(
        df,
        base_year=2024,
        profile=profile,
    )
    assert report["base_household_count"] == 3
    assert report["augmented_household_count"] == 5
    synthetic_household_ids = set(augmented_df["household_id__2024"].unique()) - {1, 2, 3}
    assert len(synthetic_household_ids) == 2
    synthetic_rows = augmented_df[augmented_df["household_id__2024"].isin(synthetic_household_ids)]
    assert set(synthetic_rows["age__2024"].tolist()) == {77.0, 85.0}
    assert set(synthetic_rows["social_security_retirement__2024"].tolist()) == {22_000.0}
    assert set(synthetic_rows["employment_income_before_lsr__2024"].tolist()) == {50_000.0}


def test_age_bin_helpers_preserve_population_totals():
    bins = build_age_bins(n_ages=86, bucket_size=5)
    assert bins[0] == (0, 5)
    assert bins[-1] == (85, 86)

    X = np.eye(86)
    y = np.arange(86, dtype=float)
    X_coarse = aggregate_household_age_matrix(X, bins)
    y_coarse = aggregate_age_targets(y, bins)

    assert X_coarse.shape == (86, 18)
    assert y_coarse.shape == (18,)
    assert X_coarse.sum() == pytest.approx(X.sum())
    assert y_coarse.sum() == pytest.approx(y.sum())

    target_matrix = np.column_stack([y, y * 2])
    aggregated_target_matrix = aggregate_age_targets(target_matrix, bins)
    assert aggregated_target_matrix.shape == (18, 2)
    assert aggregated_target_matrix[:, 0].sum() == pytest.approx(y.sum())
    assert aggregated_target_matrix[:, 1].sum() == pytest.approx((y * 2).sum())


def test_legacy_flags_map_to_named_profile():
    profile = build_profile_from_flags(
        use_greg=False,
        use_ss=True,
        use_payroll=True,
        use_h6_reform=False,
        use_tob=True,
    )
    assert profile.name == "custom-ss-payroll-tob"
    assert profile.calibration_method == "ipf"


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
    assert audit["positive_weight_count"] == 2
    assert audit["positive_weight_pct"] == 100.0
    assert audit["effective_sample_size"] == pytest.approx(2.0)
    assert audit["top_10_weight_share_pct"] == pytest.approx(100.0)
    assert audit["top_100_weight_share_pct"] == pytest.approx(100.0)


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
                "payroll_total": {"pct_error": 0.5},
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
            },
        },
        profile,
    )
    assert any("Negative weight share" in issue for issue in issues)


def test_support_thresholds_reject_concentrated_weights():
    profile = get_profile("ss-payroll-tob")
    issues = validate_calibration_audit(
        {
            "fell_back_to_ipf": False,
            "age_max_pct_error": 0.0,
            "negative_weight_pct": 0.0,
            "positive_weight_count": 90,
            "effective_sample_size": 57.6,
            "top_10_weight_share_pct": 26.6,
            "top_100_weight_share_pct": 100.0,
            "constraints": {
                "ss_total": {"pct_error": 0.0},
                "payroll_total": {"pct_error": 0.0},
            },
        },
        profile,
        year=2075,
        quality="exact",
    )
    assert any("Positive household count" in issue for issue in issues)
    assert any("Top-10 weight share" in issue for issue in issues)
    assert any("Top-100 weight share" in issue for issue in issues)


def test_classify_calibration_quality_marks_support_collapse_aggregate():
    profile = get_profile("ss-payroll-tob")
    quality = classify_calibration_quality(
        {
            "fell_back_to_ipf": False,
            "age_max_pct_error": 0.0,
            "negative_weight_pct": 0.0,
            "positive_weight_count": 6840,
            "effective_sample_size": 24.98,
            "top_10_weight_share_pct": 54.8,
            "top_100_weight_share_pct": 97.4,
            "constraints": {
                "ss_total": {"pct_error": 0.0},
                "payroll_total": {"pct_error": 0.0},
            },
        },
        profile,
        year=2075,
    )
    assert quality == "aggregate"


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
            },
        },
        profile,
        year=2035,
    )
    assert quality == "aggregate"


def test_normalize_metadata_harmonizes_lp_fallback_labels():
    profile = get_profile("ss-payroll-tob")
    metadata = normalize_metadata(
        {
            "year": 2075,
            "profile": profile.to_dict(),
            "calibration_audit": {
                "lp_fallback_used": True,
                "approximation_method": "lp_minimax_exact",
                "approximate_solution_error_pct": 0.0,
                "max_constraint_pct_error": 0.368,
                "age_max_pct_error": 0.0,
                "negative_weight_pct": 0.0,
                "constraints": {
                    "ss_total": {"pct_error": 0.368},
                    "payroll_total": {"pct_error": 0.0},
                },
            },
        }
    )

    audit = metadata["calibration_audit"]
    assert audit["calibration_quality"] == "approximate"
    assert audit["approximation_method"] == "lp_minimax"
    assert audit["approximate_solution_used"] is True
    assert audit["approximate_solution_error_pct"] == pytest.approx(0.368)


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


def test_benchmarked_tob_does_not_affect_quality_classification():
    profile = get_profile("ss-payroll-tob")
    quality = classify_calibration_quality(
        {
            "fell_back_to_ipf": False,
            "age_max_pct_error": 0.0,
            "negative_weight_pct": 0.0,
            "constraints": {
                "ss_total": {"pct_error": 0.0},
                "payroll_total": {"pct_error": 0.0},
            },
            "benchmarks": {
                "oasdi_tob": {"pct_error": 12.0},
                "hi_tob": {"pct_error": -9.0},
            },
        },
        profile,
        year=2035,
    )
    assert quality == "exact"


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
