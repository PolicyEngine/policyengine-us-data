from __future__ import annotations

import hashlib
import json
import subprocess
from argparse import Namespace
from types import SimpleNamespace
import numpy as np
import pytest
from policyengine_core.data.dataset import Dataset

from policyengine_us_data.datasets.cps.long_term import (
    build_long_term_target_sources as target_source_builder,
    calibration as calibration_module,
)
from policyengine_us_data.datasets.cps.long_term import (
    prototype_synthetic_2100_support as synthetic_support_module,
)
from policyengine_us_data.datasets.cps.long_term.calibration import (
    assess_nonnegative_feasibility,
    build_calibration_audit,
    calibrate_entropy,
    calibrate_entropy_bounded,
    calibrate_weights,
)
from policyengine_us_data.datasets.cps.long_term.calibration_artifacts import (
    capture_base_dataset_snapshot,
    capture_policyengine_us_provenance,
    normalize_metadata,
    rebuild_dataset_manifest,
    update_dataset_manifest,
    write_support_augmentation_report,
    write_year_metadata,
)
from policyengine_us_data.datasets.cps.long_term.calibration_profiles import (
    approximate_window_for_year,
    build_profile_from_flags,
    classify_calibration_quality,
    get_profile,
    validate_calibration_audit,
)
from policyengine_us_data.datasets.cps.long_term.projection_utils import (
    aggregate_age_targets,
    aggregate_household_age_matrix,
    build_age_bins,
    project_input_variable_values_to_person_rows,
    validate_projected_social_security_cap,
)
from policyengine_us_data.utils.policyengine import PolicyEngineUSBuildInfo
from policyengine_us_data.datasets.cps.long_term.ssa_data import (
    available_long_term_target_sources,
    describe_long_term_target_source,
    load_oasdi_tob_projections,
    load_taxable_payroll_projections,
    validate_long_term_target_source,
)
from policyengine_us_data.datasets.cps.long_term.support_augmentation import (
    AgeShiftCloneRule,
    CompositePayrollRule,
    MixedAgeAppendRule,
    SinglePersonSyntheticGridRule,
    SupportAugmentationProfile,
    augment_input_dataframe,
    build_targeted_donor_augmented_dataset,
    household_support_summary,
    is_targeted_donor_support_augmentation_profile,
    select_donor_households,
    valid_support_augmentation_profile_names,
)
from policyengine_us_data.datasets.cps.long_term.tax_assumptions import (
    TRUSTEES_CORE_THRESHOLD_ASSUMPTION,
    create_trustees_core_thresholds_reform,
)
from policyengine_us_data.datasets.cps.long_term.prototype_synthetic_2100_support import (
    SyntheticCandidate,
    _compose_role_donor_rows_to_target,
    build_role_composite_calibration_blueprint,
    build_role_donor_composites,
    summarize_realized_clone_translation,
)
from policyengine_us_data.datasets.cps.long_term.run_household_projection_parallel import (
    forwarded_args_for_year,
    merge_outputs,
    parse_years,
    run_year,
    validate_forwarded_args,
    year_artifacts_complete,
    year_output_dir,
)
from policyengine_us_data.datasets.cps.long_term.run_long_term_production import (
    build_projection_command,
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
                "social_security_retirement__2024": [
                    20_000.0,
                    0.0,
                    30_000.0,
                    0.0,
                    0.0,
                ],
                "social_security_disability__2024": [0.0, 0.0, 0.0, 0.0, 0.0],
                "social_security_survivors__2024": [0.0, 0.0, 0.0, 0.0, 0.0],
                "social_security_dependents__2024": [0.0, 0.0, 0.0, 0.0, 0.0],
                "employment_income_before_lsr__2024": [
                    5_000.0,
                    0.0,
                    12_000.0,
                    0.0,
                    50_000.0,
                ],
                "self_employment_income_before_lsr__2024": [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
            }
        )
    )


def test_named_profile_lookup():
    profile = get_profile("ss-payroll-tob")
    assert profile.calibration_method == "entropy"
    assert profile.use_greg is False
    assert profile.use_ss is True
    assert profile.use_payroll is True
    assert profile.use_tob is True
    assert profile.benchmark_tob is False
    assert profile.use_h6_reform is False
    assert profile.max_negative_weight_pct == 0.0
    assert profile.approximate_windows[0].age_bucket_size == 5
    assert profile.min_positive_household_count == 1000
    assert profile.min_effective_sample_size == 75.0
    assert profile.max_top_10_weight_share_pct == 25.0
    assert profile.max_top_100_weight_share_pct == 95.0


def test_project_input_variable_values_aligns_tax_unit_outputs_to_person_rows():
    class FakeSim:
        def __init__(self):
            self.tax_benefit_system = SimpleNamespace(
                variables={
                    "interest_deduction": SimpleNamespace(
                        entity=SimpleNamespace(key="tax_unit")
                    )
                }
            )

        def calculate(self, variable, *, period=None, map_to=None):
            assert period == 2073 or variable == "tax_unit_id"
            if variable == "interest_deduction":
                return SimpleNamespace(values=np.array([100.0, 250.0]))
            if variable == "tax_unit_id":
                assert map_to == "tax_unit"
                return SimpleNamespace(values=np.array([11, 22]))
            raise AssertionError(variable)

    import pandas as pd

    df = pd.DataFrame(
        {
            "person_id__2024": [1, 2, 3],
            "person_tax_unit_id__2024": [11, 11, 22],
        }
    )

    values = project_input_variable_values_to_person_rows(
        FakeSim(),
        df,
        var_name="interest_deduction",
        year=2073,
        base_period=2024,
    )

    assert np.array_equal(values, np.array([100.0, 100.0, 250.0]))


def test_project_input_variable_values_falls_back_to_entity_id_column():
    class FakeSim:
        def __init__(self):
            self.tax_benefit_system = SimpleNamespace(
                variables={
                    "net_worth": SimpleNamespace(
                        entity=SimpleNamespace(key="household")
                    )
                }
            )

        def calculate(self, variable, *, period=None, map_to=None):
            assert period == 2073 or variable == "household_id"
            if variable == "net_worth":
                return SimpleNamespace(values=np.array([10.0, 20.0]))
            if variable == "household_id":
                assert map_to == "household"
                return SimpleNamespace(values=np.array([101, 202]))
            raise AssertionError(variable)

    import pandas as pd

    df = pd.DataFrame(
        {
            "person_id__2024": [1, 2, 3],
            "household_id__2024": [101, 101, 202],
        }
    )

    values = project_input_variable_values_to_person_rows(
        FakeSim(),
        df,
        var_name="net_worth",
        year=2073,
        base_period=2024,
    )

    assert np.array_equal(values, np.array([10.0, 10.0, 20.0]))


def test_project_input_variable_values_uses_year_renamed_membership_column():
    class FakeSim:
        def __init__(self):
            self.tax_benefit_system = SimpleNamespace(
                variables={
                    "interest_deduction": SimpleNamespace(
                        entity=SimpleNamespace(key="tax_unit")
                    )
                }
            )

        def calculate(self, variable, *, period=None, map_to=None):
            assert period == 2073 or variable == "tax_unit_id"
            if variable == "interest_deduction":
                return SimpleNamespace(values=np.array([100.0, 250.0]))
            if variable == "tax_unit_id":
                assert map_to == "tax_unit"
                return SimpleNamespace(values=np.array([11, 22]))
            raise AssertionError(variable)

    import pandas as pd

    df = pd.DataFrame(
        {
            "person_id__2024": [1, 2, 3],
            "person_tax_unit_id__2073": [11, 11, 22],
        }
    )

    values = project_input_variable_values_to_person_rows(
        FakeSim(),
        df,
        var_name="interest_deduction",
        year=2073,
        base_period=2024,
    )

    assert np.array_equal(values, np.array([100.0, 100.0, 250.0]))


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


def test_support_augmentation_profile_registry_includes_runner_profiles():
    profiles = valid_support_augmentation_profile_names()
    assert "late-clone-v1" in profiles
    assert "donor-backed-composite-v1" in profiles
    assert is_targeted_donor_support_augmentation_profile("donor-backed-composite-v1")
    assert not is_targeted_donor_support_augmentation_profile("late-clone-v1")


def test_targeted_donor_support_builder_rejects_unknown_profile():
    with pytest.raises(ValueError, match="Unknown targeted donor support profile"):
        build_targeted_donor_augmented_dataset(
            base_dataset="unused.h5",
            base_year=2024,
            target_year=2100,
            profile="late-clone-v1",
        )


@pytest.mark.parametrize(
    ("profile", "builder_name"),
    [
        ("donor-backed-synthetic-v1", "build_donor_backed_augmented_dataset"),
        ("donor-backed-composite-v1", "build_role_composite_augmented_dataset"),
    ],
)
def test_targeted_donor_support_builder_forwards_reform(
    monkeypatch,
    profile,
    builder_name,
):
    from policyengine_us_data.datasets.cps.long_term import (
        prototype_synthetic_2100_support as prototype_module,
    )

    reform = {"gov.irs.uprating": {"2035-01-01.2100-12-31": "gov.ssa.nawi"}}
    calls = {}

    def fake_builder(**kwargs):
        calls["reform"] = kwargs.get("reform")
        return object(), {}

    monkeypatch.setattr(prototype_module, builder_name, fake_builder)

    _, report = build_targeted_donor_augmented_dataset(
        base_dataset="unused.h5",
        base_year=2024,
        target_year=2075,
        profile=profile,
        reform=reform,
    )

    assert calls["reform"] is reform
    assert report["profile"] == profile


def test_role_composite_dataset_passes_reform_to_input_builder(monkeypatch):
    reform = object()
    captured = {}

    def build_input_dataframe(**kwargs):
        captured["reform"] = kwargs.get("reform")
        return "input-frame", {"ok": True}

    monkeypatch.setattr(
        synthetic_support_module,
        "build_role_composite_augmented_input_dataframe",
        build_input_dataframe,
    )
    monkeypatch.setattr(
        synthetic_support_module.Dataset,
        "from_dataframe",
        staticmethod(lambda df, year: {"df": df, "year": year}),
    )

    dataset, report = synthetic_support_module.build_role_composite_augmented_dataset(
        base_dataset="base.h5",
        base_year=2024,
        target_year=2100,
        reform=reform,
    )

    assert captured["reform"] is reform
    assert dataset == {"df": "input-frame", "year": 2024}
    assert report == {"ok": True}


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
    cloned_household_ids = set(augmented_df["household_id__2024"].unique()) - {
        1,
        2,
        3,
    }
    assert len(cloned_household_ids) == 1
    cloned_rows = augmented_df[
        augmented_df["household_id__2024"].isin(cloned_household_ids)
    ]
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
    cloned_household_ids = set(augmented_df["household_id__2024"].unique()) - {
        1,
        2,
        3,
    }
    assert len(cloned_household_ids) == 1
    cloned_rows = augmented_df[
        augmented_df["household_id__2024"].isin(cloned_household_ids)
    ]
    assert cloned_rows["age__2024"].max() == pytest.approx(80.0)
    assert cloned_rows["social_security_retirement__2024"].sum() == pytest.approx(
        30_000.0
    )
    assert cloned_rows["employment_income_before_lsr__2024"].sum() == pytest.approx(
        37_000.0
    )


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
    synthetic_household_ids = set(augmented_df["household_id__2024"].unique()) - {
        1,
        2,
        3,
    }
    assert len(synthetic_household_ids) == 2
    synthetic_rows = augmented_df[
        augmented_df["household_id__2024"].isin(synthetic_household_ids)
    ]
    assert set(synthetic_rows["age__2024"].tolist()) == {77.0, 85.0}
    assert set(synthetic_rows["social_security_retirement__2024"].tolist()) == {
        22_000.0
    }
    assert set(synthetic_rows["employment_income_before_lsr__2024"].tolist()) == {
        50_000.0
    }


def test_support_augmentation_appends_mixed_age_household():
    import pandas as pd

    df = pd.DataFrame(_toy_support_dataframe())
    profile = SupportAugmentationProfile(
        name="mixed-age-profile",
        description="Toy mixed-age household support augmentation profile.",
        rules=(
            MixedAgeAppendRule(
                name="older_plus_younger_earner",
                recipient_min_max_age=75,
                recipient_max_max_age=84,
                donor_min_max_age=55,
                donor_max_max_age=64,
                recipient_ss_state="positive",
                recipient_payroll_state="any",
                donor_ss_state="nonpositive",
                donor_payroll_state="positive",
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
    assert report["augmented_household_count"] == 4
    synthetic_household_ids = set(augmented_df["household_id__2024"].unique()) - {
        1,
        2,
        3,
    }
    assert len(synthetic_household_ids) == 1
    synthetic_rows = augmented_df[
        augmented_df["household_id__2024"].isin(synthetic_household_ids)
    ]
    assert sorted(synthetic_rows["age__2024"].tolist()) == [60.0, 77.0, 80.0]
    assert synthetic_rows["social_security_retirement__2024"].sum() == pytest.approx(
        30_000.0
    )
    assert synthetic_rows["employment_income_before_lsr__2024"].sum() == pytest.approx(
        62_000.0
    )
    assert synthetic_rows["tax_unit_id__2024"].nunique() == 3


def test_role_donor_composites_build_structural_candidate_from_role_donors():
    import pandas as pd

    candidates = [
        SyntheticCandidate(
            archetype="older_plus_prime_worker_family",
            head_age=67,
            spouse_age=42,
            dependent_ages=(10,),
            head_wages=0.0,
            spouse_wages=100_000.0,
            head_ss=40_000.0,
            spouse_ss=0.0,
            pension_income=0.0,
            dividend_income=0.0,
        )
    ]
    actual_summary = pd.DataFrame(
        [
            {
                "tax_unit_id": 1,
                "head_age": 70.0,
                "spouse_age": None,
                "adult_count": 1,
                "dependent_count": 0,
                "dependent_ages": (),
                "head_payroll": 0.0,
                "spouse_payroll": 0.0,
                "head_ss": 40_000.0,
                "spouse_ss": 0.0,
                "payroll_total": 0.0,
                "ss_total": 40_000.0,
                "dividend_income": 2_000.0,
                "pension_income": 8_000.0,
                "support_count_weight": 1.0,
                "person_weight_proxy": 1.0,
                "archetype": "older_beneficiary_single",
            },
            {
                "tax_unit_id": 2,
                "head_age": 41.0,
                "spouse_age": 39.0,
                "adult_count": 2,
                "dependent_count": 1,
                "dependent_ages": (10,),
                "head_payroll": 60_000.0,
                "spouse_payroll": 40_000.0,
                "head_ss": 0.0,
                "spouse_ss": 0.0,
                "payroll_total": 100_000.0,
                "ss_total": 0.0,
                "dividend_income": 0.0,
                "pension_income": 0.0,
                "support_count_weight": 1.0,
                "person_weight_proxy": 1.0,
                "archetype": "prime_worker_family",
            },
        ]
    )

    composite_candidates, prior_weights, report = build_role_donor_composites(
        candidates,
        np.array([1.0]),
        actual_summary,
        ss_scale=1.0,
        earnings_scale=1.0,
        top_n_targets=1,
        older_donors_per_target=1,
        worker_donors_per_target=1,
    )

    assert len(composite_candidates) == 1
    assert composite_candidates[0].archetype.endswith("_role_donor")
    assert composite_candidates[0].spouse_wages == pytest.approx(100_000.0)
    assert composite_candidates[0].head_ss == pytest.approx(40_000.0)
    assert prior_weights.tolist() == pytest.approx([1.0])
    assert report["skipped_targets"] == []


def test_role_donor_composites_preserve_taxable_payroll_under_cap():
    import pandas as pd

    payroll_cap = 100_000.0
    candidates = [
        SyntheticCandidate(
            archetype="prime_worker_couple",
            head_age=40,
            spouse_age=38,
            dependent_ages=(),
            head_wages=75_000.0,
            spouse_wages=75_000.0,
            head_ss=0.0,
            spouse_ss=0.0,
            pension_income=0.0,
            dividend_income=0.0,
        )
    ]
    actual_summary = pd.DataFrame(
        [
            {
                "tax_unit_id": 2,
                "head_age": 41.0,
                "spouse_age": 39.0,
                "adult_count": 2,
                "dependent_count": 0,
                "dependent_ages": (),
                "head_payroll": 180_000.0,
                "spouse_payroll": 20_000.0,
                "head_ss": 0.0,
                "spouse_ss": 0.0,
                "payroll_total": 200_000.0,
                "ss_total": 0.0,
                "dividend_income": 0.0,
                "pension_income": 0.0,
                "support_count_weight": 1.0,
                "person_weight_proxy": 1.0,
                "archetype": "prime_worker_couple",
            },
        ]
    )

    composite_candidates, _, report = build_role_donor_composites(
        candidates,
        np.array([1.0]),
        actual_summary,
        ss_scale=1.0,
        earnings_scale=1.0,
        top_n_targets=1,
        older_donors_per_target=1,
        worker_donors_per_target=1,
        payroll_cap=payroll_cap,
    )

    assert report["skipped_targets"] == []
    assert len(composite_candidates) == 1
    composite = composite_candidates[0]
    assert composite.head_wages == pytest.approx(payroll_cap)
    assert composite.spouse_wages == pytest.approx(50_000.0)
    assert composite.taxable_payroll_total(payroll_cap) == pytest.approx(150_000.0)


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


def test_validate_projected_social_security_cap_rejects_flat_tail():
    from types import SimpleNamespace

    def accessor(year: int):
        cap = 254_400.0 if year >= 2035 else 186_000.0
        return SimpleNamespace(
            gov=SimpleNamespace(
                irs=SimpleNamespace(
                    payroll=SimpleNamespace(social_security=SimpleNamespace(cap=cap))
                )
            )
        )

    with pytest.raises(RuntimeError, match="flat after 2035"):
        validate_projected_social_security_cap(accessor, 2100)


def test_role_composite_calibration_blueprint_reweights_clone_priors():
    report = {
        "target_year": 2100,
        "clone_household_reports": [
            {
                "clone_household_id": 1001,
                "target_head_age": 70,
                "target_spouse_age": 68,
                "target_dependent_ages": [12],
                "target_ss_total": 20_000.0,
                "target_payroll_total": 50_000.0,
                "target_taxable_payroll_total": 45_000.0,
                "per_clone_weight_share_pct": 60.0,
            },
            {
                "clone_household_id": 1002,
                "target_head_age": 80,
                "target_spouse_age": None,
                "target_dependent_ages": [],
                "target_ss_total": 30_000.0,
                "target_payroll_total": 10_000.0,
                "per_clone_weight_share_pct": 40.0,
            },
        ],
    }
    baseline_weights = np.array([10.0, 20.0, 30.0], dtype=float)
    blueprint = build_role_composite_calibration_blueprint(
        report,
        year=2100,
        age_bins=build_age_bins(n_ages=86, bucket_size=5),
        hh_id_to_idx={999: 0, 1001: 1, 1002: 2},
        baseline_weights=baseline_weights,
        base_weight_scale=0.5,
    )

    assert blueprint is not None
    assert blueprint["baseline_weights"].tolist() == pytest.approx([5.0, 36.0, 24.0])
    assert blueprint["ss_overrides"] == {1: 20_000.0, 2: 30_000.0}
    assert blueprint["payroll_overrides"] == {1: 45_000.0, 2: 10_000.0}
    assert blueprint["age_overrides"][1].sum() == pytest.approx(3.0)
    assert blueprint["age_overrides"][2].sum() == pytest.approx(1.0)
    assert blueprint["summary"]["clone_household_count"] == 2
    assert blueprint["summary"]["base_weight_scale"] == pytest.approx(0.5)
    assert blueprint["summary"]["include_value_overrides"] is True

    actual_value_blueprint = build_role_composite_calibration_blueprint(
        report,
        year=2100,
        age_bins=build_age_bins(n_ages=86, bucket_size=5),
        hh_id_to_idx={999: 0, 1001: 1, 1002: 2},
        baseline_weights=baseline_weights,
        base_weight_scale=0.5,
        include_value_overrides=False,
    )

    assert actual_value_blueprint is not None
    assert actual_value_blueprint["ss_overrides"] == {}
    assert actual_value_blueprint["payroll_overrides"] == {}
    assert actual_value_blueprint["summary"]["include_value_overrides"] is False


def test_role_composite_blueprint_prefers_realized_support_values():
    report = {
        "target_year": 2100,
        "clone_household_reports": [
            {
                "clone_household_id": 1001,
                "target_head_age": 70,
                "target_spouse_age": None,
                "target_dependent_ages": [],
                "target_ss_total": 20_000.0,
                "target_payroll_total": 50_000.0,
                "per_clone_weight_share_pct": 10.0,
            }
        ],
    }
    blueprint = build_role_composite_calibration_blueprint(
        report,
        year=2100,
        age_bins=build_age_bins(n_ages=86, bucket_size=5),
        hh_id_to_idx={1001: 1},
        baseline_weights=np.array([100.0, 200.0]),
        ss_values_actual=np.array([1_000.0, 21_000.0]),
        payroll_values_actual=np.array([2_000.0, 55_000.0]),
    )

    assert blueprint is not None
    assert blueprint["ss_overrides"] == {1: 21_000.0}
    assert blueprint["payroll_overrides"] == {1: 55_000.0}


def test_legacy_flags_map_to_named_profile():
    profile = build_profile_from_flags(
        use_greg=False,
        use_ss=True,
        use_payroll=True,
        use_h6_reform=False,
        use_tob=True,
    )
    assert profile.name == "custom-greg-ss-payroll-tob"
    assert profile.calibration_method == "greg"
    assert profile.use_greg is True


def test_approximate_window_none_selects_open_ended_tail():
    profile = get_profile("ss-payroll-tob")
    window = approximate_window_for_year(profile, None)
    assert window is not None
    assert window.start_year == 2096
    assert window.end_year is None


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
    assert audit["negative_weight_household_pct"] == 0.0
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


def test_year_metadata_and_manifest_stamp_policyengine_us_build(tmp_path):
    profile = get_profile("ss-payroll-tob")
    audit = {
        "method_used": "greg",
        "fell_back_to_ipf": False,
        "negative_weight_pct": 0.0,
    }
    build_info = PolicyEngineUSBuildInfo(
        version="1.700.0",
        locked_version="1.700.0",
        git_commit="abc123",
        git_dirty=False,
        package_file_sha256="f" * 64,
        package_tree_sha256="t" * 64,
    )
    h5_path = tmp_path / "2100.h5"
    h5_path.write_text("", encoding="utf-8")

    metadata_path = write_year_metadata(
        h5_path,
        year=2100,
        base_dataset_path="test.h5",
        profile=profile.to_dict(),
        calibration_audit=audit,
        policyengine_us=build_info,
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert metadata["policyengine_us"]["version"] == "1.700.0"
    assert metadata["policyengine_us"]["locked_version"] == "1.700.0"
    assert metadata["policyengine_us"]["commit_id"] == "abc123"
    assert metadata["policyengine_us"]["git_dirty"] is False
    assert metadata["policyengine_us"]["package_tree_sha256"] == "t" * 64

    manifest_path = update_dataset_manifest(
        tmp_path,
        year=2100,
        h5_path=h5_path,
        metadata_path=metadata_path,
        base_dataset_path="test.h5",
        profile=profile.to_dict(),
        calibration_audit=audit,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["policyengine_us"] == metadata["policyengine_us"]
    assert manifest["datasets"]["2100"]["policyengine_us_version"] == "1.700.0"


def test_hard_target_tob_affects_quality_classification():
    profile = get_profile("ss-payroll-tob")
    quality = classify_calibration_quality(
        {
            "fell_back_to_ipf": False,
            "age_max_pct_error": 0.0,
            "negative_weight_pct": 0.0,
            "constraints": {
                "ss_total": {"pct_error": 0.0},
                "payroll_total": {"pct_error": 0.0},
                "oasdi_tob": {"pct_error": 12.0},
                "hi_tob": {"pct_error": -9.0},
            },
        },
        profile,
        year=2035,
    )
    assert quality == "aggregate"


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
    np.testing.assert_allclose(
        np.dot(payroll_values, weights), payroll_target, rtol=1e-8, atol=1e-8
    )


def test_bounded_entropy_calibration_returns_positive_approximate_weights():
    X = np.array(
        [
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
        ]
    )
    y_target = np.array([1.0, 1.0, 3.0])
    baseline_weights = np.array([1.0, 1.0])

    weights, _, info = calibrate_entropy_bounded(
        X=X,
        y_target=y_target,
        baseline_weights=baseline_weights,
        n_ages=3,
        max_constraint_error_pct=40.0,
    )

    assert info["best_case_max_pct_error"] <= 40.0
    assert np.all(weights > 0)


def test_entropy_calibration_prefers_bounded_entropy_over_lp_approximate_solution():
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
    assert audit["approximation_method"] == "bounded_entropy"
    assert audit["approximate_solution_error_pct"] > 10.0
    assert np.all(weights > 0)


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
        "calibrate_lp_minimax",
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


def test_entropy_calibration_rejects_large_constraint_error_without_exception(
    monkeypatch,
):
    monkeypatch.setattr(
        calibration_module,
        "calibrate_entropy",
        lambda *args, **kwargs: (np.array([1.0, 1.0]), 3),
    )
    monkeypatch.setattr(
        calibration_module,
        "calibrate_lp_minimax",
        lambda *args, **kwargs: (
            np.array([10.0, 2.0]),
            1,
            {"best_case_max_pct_error": 0.0},
        ),
    )

    weights, _, audit = calibrate_weights(
        X=np.array([[1.0], [0.0]]),
        y_target=np.array([1.0]),
        baseline_weights=np.array([1.0, 1.0]),
        method="entropy",
        payroll_values=np.array([1.0, 0.0]),
        payroll_target=10.0,
        n_ages=1,
        allow_approximate_entropy=False,
    )

    np.testing.assert_allclose(weights, np.array([10.0, 2.0]))
    assert audit["lp_fallback_used"] is True
    assert audit["approximation_method"] == "lp_minimax_exact"
    assert "above allowable" in audit["entropy_error"]


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
    assert "oact_2025_08_05_provisional" in sources

    trustees = describe_long_term_target_source("trustees_2025_current_law")
    assert trustees["file"] == "trustees_2025_current_law.csv"

    payroll_2026 = load_taxable_payroll_projections(
        2026,
        source_name="trustees_2025_current_law",
    )
    assert payroll_2026 == pytest.approx(11_129_000_000_000.0)

    trustees_oasdi_2026 = load_oasdi_tob_projections(
        2026,
        source_name="trustees_2025_current_law",
    )
    oact_oasdi_2026 = load_oasdi_tob_projections(
        2026,
        source_name="oact_2025_08_05_provisional",
    )
    assert oact_oasdi_2026 < trustees_oasdi_2026


def test_post_obbba_target_source_carries_reproducibility_contract():
    source = describe_long_term_target_source("oact_2025_08_05_provisional")
    assert source["scenario_id"] == "crfb_post_obbba_tob_75y"
    assert source["baseline_kind"] == "calibration_target"
    assert source["not_law"] is True
    assert source["law_mode"] == "trustees-2025-core-thresholds-v1"
    assert source["artifact_contract"] == {
        "must_consume_baseline_sha256": source["sha256"],
        "must_expose_scenario_id": "crfb_post_obbba_tob_75y",
        "reject_raw_current_law_substitution": True,
    }

    validation = validate_long_term_target_source("oact_2025_08_05_provisional")
    assert validation["sha256"] == source["sha256"]
    assert validation["rows"] == 76
    assert validation["years"] == [2025, 2100]


def test_long_term_target_source_rebuild_manifest_preserves_contract(
    tmp_path, monkeypatch
):
    sources_dir = tmp_path / "long_term_target_sources"
    sources_dir.mkdir()
    trustees_path = sources_dir / "trustees_2025_current_law.csv"
    oact_path = sources_dir / "oact_2025_08_05_provisional.csv"
    manifest_path = sources_dir / "sources.json"
    trustees_path.write_text("year,value\n2025,1\n", encoding="utf-8")
    oact_path.write_text("year,value\n2025,2\n", encoding="utf-8")

    monkeypatch.setattr(target_source_builder, "SOURCES_DIR", sources_dir)
    monkeypatch.setattr(
        target_source_builder,
        "TRUSTEES_OUTPUT_PATH",
        trustees_path,
    )
    monkeypatch.setattr(target_source_builder, "OACT_OUTPUT_PATH", oact_path)
    monkeypatch.setattr(target_source_builder, "MANIFEST_PATH", manifest_path)

    target_source_builder.write_manifest()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    oact_sha256 = hashlib.sha256(oact_path.read_bytes()).hexdigest()
    trustees_sha256 = hashlib.sha256(trustees_path.read_bytes()).hexdigest()
    trustees = manifest["sources"]["trustees_2025_current_law"]
    oact = manifest["sources"]["oact_2025_08_05_provisional"]
    assert trustees["sha256"] == trustees_sha256
    assert trustees["baseline_kind"] == "current_law_comparator"
    assert trustees["not_law"] is False
    assert oact["sha256"] == oact_sha256
    assert oact["scenario_id"] == "crfb_post_obbba_tob_75y"
    assert oact["baseline_kind"] == "calibration_target"
    assert oact["calibration_target_id"] == "post_obbba_calibrated_tob_75y"
    assert oact["law_mode"] == "trustees-2025-core-thresholds-v1"
    assert oact["not_law"] is True
    assert oact["artifact_contract"] == {
        "must_consume_baseline_sha256": oact_sha256,
        "must_expose_scenario_id": "crfb_post_obbba_tob_75y",
        "reject_raw_current_law_substitution": True,
    }


def test_trustees_core_threshold_assumption_preserves_pre_2035_baseline():
    from policyengine_us import CountryTaxBenefitSystem

    baseline = CountryTaxBenefitSystem().parameters
    reformed = CountryTaxBenefitSystem(
        reform=(create_trustees_core_thresholds_reform(),)
    ).parameters

    baseline_threshold = baseline.gov.irs.income.bracket.thresholds.children["1"].SINGLE
    reformed_threshold = reformed.gov.irs.income.bracket.thresholds.children["1"].SINGLE
    ss_threshold = (
        baseline.gov.irs.social_security.taxability.threshold.base.main.SINGLE
    )
    reformed_ss_threshold = (
        reformed.gov.irs.social_security.taxability.threshold.base.main.SINGLE
    )

    assert TRUSTEES_CORE_THRESHOLD_ASSUMPTION["not_default_current_law"] is True
    assert reformed_threshold("2034-01-01") == baseline_threshold("2034-01-01")
    assert reformed_threshold("2035-01-01") != baseline_threshold("2035-01-01")
    assert reformed_ss_threshold("2100-01-01") == ss_threshold("2100-01-01")


def test_normalize_metadata_backfills_validation_passed():
    metadata = normalize_metadata(
        {
            "year": 2091,
            "profile": {"name": "ss-payroll-tob"},
            "calibration_audit": {
                "lp_fallback_used": True,
                "approximation_method": "lp_blend",
                "approximate_solution_error_pct": 16.0,
                "max_constraint_pct_error": 16.0,
                "age_max_pct_error": 14.5,
                "negative_weight_pct": 0.0,
                "positive_weight_count": 6840,
                "effective_sample_size": 12.0,
                "top_10_weight_share_pct": 80.0,
                "top_100_weight_share_pct": 99.0,
                "constraints": {
                    "ss_total": {"pct_error": 14.5},
                    "payroll_total": {"pct_error": 16.0},
                },
            },
        }
    )

    audit = metadata["calibration_audit"]
    assert audit["validation_passed"] is False
    assert isinstance(audit["validation_issues"], list)
    assert len(audit["validation_issues"]) > 0


def test_manifest_contains_invalid_artifacts_flag(tmp_path):
    profile = get_profile("ss-payroll-tob")

    valid_audit = {
        "method_used": "entropy",
        "fell_back_to_ipf": False,
        "age_max_pct_error": 0.0,
        "negative_weight_pct": 0.0,
        "positive_weight_count": 70000,
        "effective_sample_size": 5000.0,
        "top_10_weight_share_pct": 1.5,
        "top_100_weight_share_pct": 10.0,
        "max_constraint_pct_error": 0.0,
        "constraints": {},
        "validation_passed": True,
        "validation_issues": [],
    }

    invalid_audit = {
        "method_used": "entropy",
        "fell_back_to_ipf": False,
        "age_max_pct_error": 14.0,
        "negative_weight_pct": 0.0,
        "positive_weight_count": 6840,
        "effective_sample_size": 12.0,
        "top_10_weight_share_pct": 80.0,
        "top_100_weight_share_pct": 99.0,
        "max_constraint_pct_error": 16.0,
        "constraints": {"payroll_total": {"pct_error": 16.0}},
        "validation_passed": False,
        "validation_issues": ["ESS too low"],
    }

    # First year: valid
    year_2030 = tmp_path / "2030.h5"
    year_2030.write_text("", encoding="utf-8")
    metadata_2030 = write_year_metadata(
        year_2030,
        year=2030,
        base_dataset_path="test.h5",
        profile=profile.to_dict(),
        calibration_audit=valid_audit,
    )
    manifest_path = update_dataset_manifest(
        tmp_path,
        year=2030,
        h5_path=year_2030,
        metadata_path=metadata_2030,
        base_dataset_path="test.h5",
        profile=profile.to_dict(),
        calibration_audit=valid_audit,
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["contains_invalid_artifacts"] is False

    # Second year: invalid
    year_2091 = tmp_path / "2091.h5"
    year_2091.write_text("", encoding="utf-8")
    metadata_2091 = write_year_metadata(
        year_2091,
        year=2091,
        base_dataset_path="test.h5",
        profile=profile.to_dict(),
        calibration_audit=invalid_audit,
    )
    update_dataset_manifest(
        tmp_path,
        year=2091,
        h5_path=year_2091,
        metadata_path=metadata_2091,
        base_dataset_path="test.h5",
        profile=profile.to_dict(),
        calibration_audit=invalid_audit,
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["contains_invalid_artifacts"] is True
    assert manifest["datasets"]["2030"]["validation_passed"] is True
    assert manifest["datasets"]["2091"]["validation_passed"] is False


def test_manifest_persists_support_augmentation_metadata(tmp_path):
    profile = get_profile("ss-payroll-tob")
    audit = {
        "method_used": "entropy",
        "fell_back_to_ipf": False,
        "age_max_pct_error": 0.0,
        "negative_weight_pct": 0.0,
        "positive_weight_count": 70000,
        "effective_sample_size": 5000.0,
        "top_10_weight_share_pct": 1.5,
        "top_100_weight_share_pct": 10.0,
        "max_constraint_pct_error": 0.0,
        "constraints": {},
        "validation_passed": True,
        "validation_issues": [],
    }
    support_augmentation = {
        "name": "donor-backed-synthetic-v1",
        "activation_start_year": 2075,
        "target_year": 2100,
        "report_file": "support_augmentation_report.json",
        "report_summary": {
            "base_household_count": 41314,
            "augmented_household_count": 41326,
        },
    }

    year_2100 = tmp_path / "2100.h5"
    year_2100.write_text("", encoding="utf-8")
    metadata_path = write_year_metadata(
        year_2100,
        year=2100,
        base_dataset_path="test.h5",
        profile=profile.to_dict(),
        calibration_audit=audit,
        support_augmentation=support_augmentation,
    )
    manifest_path = update_dataset_manifest(
        tmp_path,
        year=2100,
        h5_path=year_2100,
        metadata_path=metadata_path,
        base_dataset_path="test.h5",
        profile=profile.to_dict(),
        calibration_audit=audit,
        support_augmentation=support_augmentation,
    )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert metadata["support_augmentation"]["name"] == "donor-backed-synthetic-v1"
    assert (
        metadata["support_augmentation"]["report_file"]
        == "support_augmentation_report.json"
    )
    assert (
        manifest["support_augmentation"]["report_summary"]["augmented_household_count"]
        == 41326
    )


def test_update_dataset_manifest_ignores_support_augmentation_run_year_fields(tmp_path):
    profile = get_profile("ss-payroll-tob")
    audit = {
        "method_used": "entropy",
        "fell_back_to_ipf": False,
        "age_max_pct_error": 0.0,
        "negative_weight_pct": 0.0,
        "positive_weight_count": 70000,
        "effective_sample_size": 5000.0,
        "top_10_weight_share_pct": 1.5,
        "top_100_weight_share_pct": 10.0,
        "max_constraint_pct_error": 0.0,
        "constraints": {},
        "validation_passed": True,
        "validation_issues": [],
    }

    year_2075 = tmp_path / "2075.h5"
    year_2075.write_text("", encoding="utf-8")
    metadata_2075 = write_year_metadata(
        year_2075,
        year=2075,
        base_dataset_path="test.h5",
        profile=profile.to_dict(),
        calibration_audit=audit,
        support_augmentation={
            "name": "donor-backed-composite-v1",
            "activation_start_year": 2075,
            "target_year": 2075,
            "report_file": "support_augmentation_report_2075.json",
            "report_summary": {"augmented_household_count": 41674},
        },
    )
    update_dataset_manifest(
        tmp_path,
        year=2075,
        h5_path=year_2075,
        metadata_path=metadata_2075,
        base_dataset_path="test.h5",
        profile=profile.to_dict(),
        calibration_audit=audit,
        support_augmentation={
            "name": "donor-backed-composite-v1",
            "activation_start_year": 2075,
            "target_year": 2075,
            "report_file": "support_augmentation_report_2075.json",
            "report_summary": {"augmented_household_count": 41674},
        },
    )

    year_2100 = tmp_path / "2100.h5"
    year_2100.write_text("", encoding="utf-8")
    metadata_2100 = write_year_metadata(
        year_2100,
        year=2100,
        base_dataset_path="test.h5",
        profile=profile.to_dict(),
        calibration_audit=audit,
        support_augmentation={
            "name": "donor-backed-composite-v1",
            "activation_start_year": 2075,
            "target_year": 2100,
            "report_file": "support_augmentation_report_2100.json",
            "report_summary": {"augmented_household_count": 41950},
        },
    )
    manifest_path = update_dataset_manifest(
        tmp_path,
        year=2100,
        h5_path=year_2100,
        metadata_path=metadata_2100,
        base_dataset_path="test.h5",
        profile=profile.to_dict(),
        calibration_audit=audit,
        support_augmentation={
            "name": "donor-backed-composite-v1",
            "activation_start_year": 2075,
            "target_year": 2100,
            "report_file": "support_augmentation_report_2100.json",
            "report_summary": {"augmented_household_count": 41950},
        },
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["years"] == [2075, 2100]
    assert manifest["support_augmentation"]["name"] == "donor-backed-composite-v1"
    assert manifest["support_augmentation"]["target_year"] == 2100
    assert (
        manifest["support_augmentation"]["report_file"]
        == "support_augmentation_report_2100.json"
    )


def test_manifest_persists_tax_assumption_metadata(tmp_path):
    profile = get_profile("ss-payroll-tob")
    audit = {
        "method_used": "entropy",
        "fell_back_to_ipf": False,
        "age_max_pct_error": 0.0,
        "negative_weight_pct": 0.0,
        "positive_weight_count": 70000,
        "effective_sample_size": 5000.0,
        "top_10_weight_share_pct": 1.5,
        "top_100_weight_share_pct": 10.0,
        "max_constraint_pct_error": 0.0,
        "constraints": {},
        "validation_passed": True,
        "validation_issues": [],
    }
    tax_assumption = {
        "name": "trustees-2025-core-thresholds-v1",
        "start_year": 2035,
        "end_year": 2100,
    }

    year_2100 = tmp_path / "2100.h5"
    year_2100.write_text("", encoding="utf-8")
    metadata_path = write_year_metadata(
        year_2100,
        year=2100,
        base_dataset_path="test.h5",
        profile=profile.to_dict(),
        calibration_audit=audit,
        tax_assumption=tax_assumption,
    )
    manifest_path = update_dataset_manifest(
        tmp_path,
        year=2100,
        h5_path=year_2100,
        metadata_path=metadata_path,
        base_dataset_path="test.h5",
        profile=profile.to_dict(),
        calibration_audit=audit,
        tax_assumption=tax_assumption,
    )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert metadata["tax_assumption"]["name"] == "trustees-2025-core-thresholds-v1"
    assert manifest["tax_assumption"]["end_year"] == 2100


def test_write_year_metadata_persists_runtime_and_snapshot_provenance(tmp_path):
    profile = get_profile("ss-payroll-tob")
    audit = {
        "method_used": "entropy",
        "fell_back_to_ipf": False,
        "age_max_pct_error": 0.0,
        "negative_weight_pct": 0.0,
        "positive_weight_count": 70000,
        "effective_sample_size": 5000.0,
        "top_10_weight_share_pct": 1.5,
        "top_100_weight_share_pct": 10.0,
        "max_constraint_pct_error": 0.0,
        "constraints": {},
        "validation_passed": True,
        "validation_issues": [],
    }

    year_h5 = tmp_path / "2100.h5"
    year_h5.write_text("", encoding="utf-8")
    base_dataset = tmp_path / "enhanced_cps_2024.h5"
    base_dataset.write_text("dataset", encoding="utf-8")

    metadata_path = write_year_metadata(
        year_h5,
        year=2100,
        base_dataset_path=str(base_dataset),
        profile=profile.to_dict(),
        calibration_audit=audit,
        policyengine_us={
            "version": "1.2.3",
            "git_head": "abc123",
        },
        base_dataset_snapshot={
            "requested_path": str(base_dataset),
            "resolved_path": str(base_dataset.resolve()),
            "resolved_file_sha256": "deadbeef",
        },
    )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["policyengine_us"]["git_head"] == "abc123"
    assert metadata["policyengine_us"]["commit_id"] == "abc123"
    assert (
        metadata["policyengine_us"]["direct_url"]["vcs_info"]["commit_id"] == "abc123"
    )
    assert metadata["base_dataset_snapshot"]["resolved_file_sha256"] == "deadbeef"


def test_update_dataset_manifest_persists_runtime_and_snapshot_provenance(tmp_path):
    profile = get_profile("ss-payroll-tob")
    audit = {
        "method_used": "entropy",
        "fell_back_to_ipf": False,
        "age_max_pct_error": 0.0,
        "negative_weight_pct": 0.0,
        "positive_weight_count": 70000,
        "effective_sample_size": 5000.0,
        "top_10_weight_share_pct": 1.5,
        "top_100_weight_share_pct": 10.0,
        "max_constraint_pct_error": 0.0,
        "constraints": {},
        "validation_passed": True,
        "validation_issues": [],
    }

    year_h5 = tmp_path / "2100.h5"
    year_h5.write_text("", encoding="utf-8")
    metadata_path = write_year_metadata(
        year_h5,
        year=2100,
        base_dataset_path="test.h5",
        profile=profile.to_dict(),
        calibration_audit=audit,
        policyengine_us={
            "version": "1.2.3",
            "git_head": "abc123",
        },
        base_dataset_snapshot={
            "requested_path": "hf://policyengine/policyengine-us-data/enhanced_cps_2024.h5",
            "resolved_path": "/cache/enhanced_cps_2024.h5",
            "resolved_file_sha256": "deadbeef",
        },
    )
    manifest_path = update_dataset_manifest(
        tmp_path,
        year=2100,
        h5_path=year_h5,
        metadata_path=metadata_path,
        base_dataset_path="test.h5",
        profile=profile.to_dict(),
        calibration_audit=audit,
        policyengine_us={
            "version": "1.2.3",
            "git_head": "abc123",
        },
        base_dataset_snapshot={
            "requested_path": "hf://policyengine/policyengine-us-data/enhanced_cps_2024.h5",
            "resolved_path": "/cache/enhanced_cps_2024.h5",
            "resolved_file_sha256": "deadbeef",
        },
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["policyengine_us"]["git_head"] == "abc123"
    assert manifest["policyengine_us"]["commit_id"] == "abc123"
    assert (
        manifest["policyengine_us"]["direct_url"]["vcs_info"]["commit_id"] == "abc123"
    )
    assert manifest["base_dataset_snapshot"]["resolved_file_sha256"] == "deadbeef"


def test_capture_base_dataset_snapshot_fingerprints_local_file(tmp_path):
    dataset_file = tmp_path / "enhanced_cps_2024.h5"
    dataset_file.write_text("dataset", encoding="utf-8")

    snapshot = capture_base_dataset_snapshot(str(dataset_file))

    assert snapshot["requested_path"] == str(dataset_file)
    assert snapshot["resolved_path"] == str(dataset_file.resolve())
    assert snapshot["resolved_file_sha256"]
    assert snapshot["resolved_size"] == dataset_file.stat().st_size


def test_capture_policyengine_us_provenance_uses_managed_build_schema():
    provenance = capture_policyengine_us_provenance()

    assert provenance["package_file_sha256"]
    assert provenance["package_tree_sha256"]
    assert provenance["version"]
    assert "source_path" not in provenance
    assert "package_file" not in provenance


def test_update_dataset_manifest_ignores_tax_assumption_end_year(tmp_path):
    profile = get_profile("ss-payroll-tob")
    audit = {
        "method_used": "entropy",
        "fell_back_to_ipf": False,
        "age_max_pct_error": 0.0,
        "negative_weight_pct": 0.0,
        "positive_weight_count": 70000,
        "effective_sample_size": 5000.0,
        "top_10_weight_share_pct": 1.5,
        "top_100_weight_share_pct": 10.0,
        "max_constraint_pct_error": 0.0,
        "constraints": {},
        "validation_passed": True,
        "validation_issues": [],
    }

    year_2075 = tmp_path / "2075.h5"
    year_2075.write_text("", encoding="utf-8")
    metadata_2075 = write_year_metadata(
        year_2075,
        year=2075,
        base_dataset_path="test.h5",
        profile=profile.to_dict(),
        calibration_audit=audit,
        tax_assumption={
            "name": "trustees-core-thresholds-v1",
            "start_year": 2035,
            "end_year": 2075,
        },
    )
    update_dataset_manifest(
        tmp_path,
        year=2075,
        h5_path=year_2075,
        metadata_path=metadata_2075,
        base_dataset_path="test.h5",
        profile=profile.to_dict(),
        calibration_audit=audit,
        tax_assumption={
            "name": "trustees-core-thresholds-v1",
            "start_year": 2035,
            "end_year": 2075,
        },
    )

    year_2100 = tmp_path / "2100.h5"
    year_2100.write_text("", encoding="utf-8")
    metadata_2100 = write_year_metadata(
        year_2100,
        year=2100,
        base_dataset_path="test.h5",
        profile=profile.to_dict(),
        calibration_audit=audit,
        tax_assumption={
            "name": "trustees-core-thresholds-v1",
            "start_year": 2035,
            "end_year": 2100,
        },
    )
    manifest_path = update_dataset_manifest(
        tmp_path,
        year=2100,
        h5_path=year_2100,
        metadata_path=metadata_2100,
        base_dataset_path="test.h5",
        profile=profile.to_dict(),
        calibration_audit=audit,
        tax_assumption={
            "name": "trustees-core-thresholds-v1",
            "start_year": 2035,
            "end_year": 2100,
        },
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["years"] == [2075, 2100]
    assert manifest["tax_assumption"]["name"] == "trustees-core-thresholds-v1"
    assert manifest["tax_assumption"]["end_year"] == 2100


def test_write_support_augmentation_report(tmp_path):
    report = {
        "name": "donor-backed-composite-v1",
        "clone_household_count": 2,
        "clone_household_reports": [{"clone_household_id": 1001}],
    }
    report_path = write_support_augmentation_report(tmp_path, report)
    assert report_path == tmp_path / "support_augmentation_report.json"
    loaded = json.loads(report_path.read_text(encoding="utf-8"))
    assert loaded["clone_household_count"] == 2
    assert loaded["clone_household_reports"][0]["clone_household_id"] == 1001


def test_write_support_augmentation_report_custom_filename(tmp_path):
    report = {"name": "dynamic-augmentation", "target_year": 2090}
    report_path = write_support_augmentation_report(
        tmp_path,
        report,
        filename="support_augmentation_report_2090.json",
    )
    assert report_path == tmp_path / "support_augmentation_report_2090.json"
    loaded = json.loads(report_path.read_text(encoding="utf-8"))
    assert loaded["target_year"] == 2090


def test_parallel_projection_parse_years_supports_ranges_and_sorting():
    assert parse_years("2030,2028-2029,2030,2027") == [2027, 2028, 2029, 2030]


def test_long_term_production_command_carries_2100_contract(tmp_path):
    args = Namespace(
        years="2100",
        jobs=1,
        profile="ss-payroll-tob",
        target_source="oact_2025_08_05_provisional",
        tax_assumption=TRUSTEES_CORE_THRESHOLD_ASSUMPTION["name"],
        keep_temp=False,
        base_dataset="",
        support_augmentation_profile="donor-backed-composite-v1",
        support_augmentation_target_year=2100,
        support_augmentation_align_to_run_year=False,
        support_augmentation_start_year=None,
        support_augmentation_top_n_targets=None,
        support_augmentation_donors_per_target=None,
        support_augmentation_max_distance=None,
        support_augmentation_clone_weight_scale=None,
        support_augmentation_blueprint_base_weight_scale=0.5,
        support_augmentation_sanitize_worker_non_target_income=False,
        support_augmentation_sanitize_clone_non_target_income=True,
        allow_validation_failures=True,
    )

    command = build_projection_command(args, tmp_path)

    assert "run_household_projection_parallel.py" in command[1]
    assert command[command.index("--years") + 1] == "2100"
    assert command[command.index("--profile") + 1] == "ss-payroll-tob"
    assert (
        command[command.index("--target-source") + 1] == "oact_2025_08_05_provisional"
    )
    assert command[command.index("--tax-assumption") + 1] == (
        "trustees-2025-core-thresholds-v1"
    )
    assert command[command.index("--support-augmentation-target-year") + 1] == "2100"
    assert command[command.index("--support-augmentation-profile") + 1] == (
        "donor-backed-composite-v1"
    )
    assert (
        command[command.index("--support-augmentation-blueprint-base-weight-scale") + 1]
        == "0.5"
    )
    assert "--support-augmentation-sanitize-worker-non-target-income" not in command
    assert "--support-augmentation-sanitize-clone-non-target-income" in command
    assert "--allow-validation-failures" in command


def test_parallel_projection_validate_forwarded_args_rejects_wrapper_flags():
    with pytest.raises(ValueError, match="--output-dir"):
        validate_forwarded_args(["--output-dir", "/tmp/out"])
    with pytest.raises(ValueError, match="--save-h5"):
        validate_forwarded_args(["--save-h5"])


def test_parallel_projection_strips_support_augmentation_before_activation_year():
    forwarded_args = [
        "--profile",
        "ss-payroll-tob",
        "--target-source",
        "trustees_2025_current_law",
        "--support-augmentation-profile",
        "donor-backed-composite-v1",
        "--support-augmentation-target-year",
        "2100",
        "--support-augmentation-start-year",
        "2075",
        "--support-augmentation-blueprint-base-weight-scale",
        "5.0",
        "--support-augmentation-sanitize-clone-non-target-income",
    ]

    early_args = forwarded_args_for_year(2026, forwarded_args)
    late_args = forwarded_args_for_year(2075, forwarded_args)

    assert "--profile" in early_args
    assert "--target-source" in early_args
    assert "--support-augmentation-profile" not in early_args
    assert "--support-augmentation-target-year" not in early_args
    assert "--support-augmentation-sanitize-clone-non-target-income" not in early_args
    assert late_args == forwarded_args


def _write_parallel_temp_year(
    *,
    root,
    year,
    profile,
    audit,
    target_source=None,
    tax_assumption=None,
    support_augmentation=None,
):
    temp_output_dir = year_output_dir(root, year)
    temp_output_dir.mkdir(parents=True, exist_ok=True)
    year_h5 = temp_output_dir / f"{year}.h5"
    year_h5.write_text("", encoding="utf-8")
    metadata_path = write_year_metadata(
        year_h5,
        year=year,
        base_dataset_path="test.h5",
        profile=profile,
        calibration_audit=audit,
        target_source=target_source,
        tax_assumption=tax_assumption,
        support_augmentation=support_augmentation,
    )
    update_dataset_manifest(
        temp_output_dir,
        year=year,
        h5_path=year_h5,
        metadata_path=metadata_path,
        base_dataset_path="test.h5",
        profile=profile,
        calibration_audit=audit,
        target_source=target_source,
        tax_assumption=tax_assumption,
        support_augmentation=support_augmentation,
    )


def test_parallel_projection_merge_outputs_rebuilds_manifest(tmp_path):
    profile = get_profile("ss-payroll-tob").to_dict()
    audit = {
        "method_used": "entropy",
        "fell_back_to_ipf": False,
        "age_max_pct_error": 0.0,
        "negative_weight_pct": 0.0,
        "positive_weight_count": 70000,
        "effective_sample_size": 5000.0,
        "top_10_weight_share_pct": 1.5,
        "top_100_weight_share_pct": 10.0,
        "max_constraint_pct_error": 0.0,
        "constraints": {},
        "validation_passed": True,
        "validation_issues": [],
        "calibration_quality": "exact",
    }
    target_source = {
        "name": "oact_2025_08_05_provisional",
        "source_type": "oact_note",
    }
    tax_assumption = {
        "name": "trustees-2025-core-thresholds-v1",
        "start_year": 2035,
        "end_year": 2100,
    }

    _write_parallel_temp_year(
        root=tmp_path,
        year=2045,
        profile=profile,
        audit=audit,
        target_source=target_source,
        tax_assumption=tax_assumption,
    )
    _write_parallel_temp_year(
        root=tmp_path,
        year=2049,
        profile=profile,
        audit=audit,
        target_source=target_source,
        tax_assumption=tax_assumption,
    )

    manifest_path = merge_outputs(
        years=[2045, 2049],
        output_root=tmp_path,
        keep_temp=False,
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["years"] == [2045, 2049]
    assert manifest["target_source"]["name"] == "oact_2025_08_05_provisional"
    assert manifest["tax_assumption"]["name"] == "trustees-2025-core-thresholds-v1"
    assert (tmp_path / "2045.h5").exists()
    assert (tmp_path / "2049.h5.metadata.json").exists()
    assert not (tmp_path / ".parallel_tmp").exists()


def test_parallel_projection_merge_outputs_allows_support_activation_window(tmp_path):
    profile = get_profile("ss-payroll-tob").to_dict()
    audit = {
        "method_used": "entropy",
        "fell_back_to_ipf": False,
        "age_max_pct_error": 0.0,
        "negative_weight_pct": 0.0,
        "positive_weight_count": 70000,
        "effective_sample_size": 5000.0,
        "top_10_weight_share_pct": 1.5,
        "top_100_weight_share_pct": 10.0,
        "max_constraint_pct_error": 0.0,
        "constraints": {},
        "validation_passed": True,
        "validation_issues": [],
        "calibration_quality": "exact",
    }
    tax_assumption = {
        "name": "trustees-2025-core-thresholds-v1",
        "start_year": 2035,
        "end_year": 2100,
    }
    support_augmentation = {
        "name": "donor-backed-composite-v1",
        "family": "targeted_donor",
        "activation_start_year": 2075,
        "target_year": 2100,
        "target_year_strategy": "fixed",
        "report_file": "support_augmentation_report.json",
        "report_summary": {"clone_household_count": 10},
    }

    _write_parallel_temp_year(
        root=tmp_path,
        year=2026,
        profile=profile,
        audit=audit,
        tax_assumption=tax_assumption,
    )
    _write_parallel_temp_year(
        root=tmp_path,
        year=2075,
        profile=profile,
        audit=audit,
        tax_assumption=tax_assumption,
        support_augmentation=support_augmentation,
    )

    manifest_path = merge_outputs(
        years=[2026, 2075],
        output_root=tmp_path,
        keep_temp=True,
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["years"] == [2026, 2075]
    assert manifest["support_augmentation"]["name"] == "donor-backed-composite-v1"
    early_metadata = json.loads(
        (tmp_path / "2026.h5.metadata.json").read_text(encoding="utf-8")
    )
    late_metadata = json.loads(
        (tmp_path / "2075.h5.metadata.json").read_text(encoding="utf-8")
    )
    assert "support_augmentation" not in early_metadata
    assert late_metadata["support_augmentation"]["activation_start_year"] == 2075


def test_parallel_projection_merge_outputs_rejects_different_fixed_support_targets(
    tmp_path,
):
    profile = get_profile("ss-payroll-tob").to_dict()
    audit = {
        "method_used": "entropy",
        "fell_back_to_ipf": False,
        "age_max_pct_error": 0.0,
        "negative_weight_pct": 0.0,
        "positive_weight_count": 70000,
        "effective_sample_size": 5000.0,
        "top_10_weight_share_pct": 1.5,
        "top_100_weight_share_pct": 10.0,
        "max_constraint_pct_error": 0.0,
        "constraints": {},
        "validation_passed": True,
        "validation_issues": [],
        "calibration_quality": "exact",
    }
    base_support = {
        "name": "donor-backed-composite-v1",
        "family": "targeted_donor",
        "activation_start_year": 2075,
        "target_year": 2100,
        "target_year_strategy": "fixed",
        "report_file": "support_augmentation_report.json",
        "report_summary": {"clone_household_count": 10},
    }
    different_support = dict(base_support, target_year=2090)

    _write_parallel_temp_year(
        root=tmp_path,
        year=2075,
        profile=profile,
        audit=audit,
        support_augmentation=base_support,
    )
    _write_parallel_temp_year(
        root=tmp_path,
        year=2080,
        profile=profile,
        audit=audit,
        support_augmentation=different_support,
    )

    with pytest.raises(
        ValueError,
        match="Temp manifest mismatch for support_augmentation",
    ):
        merge_outputs(
            years=[2075, 2080],
            output_root=tmp_path,
            keep_temp=True,
        )


def test_parallel_projection_merge_outputs_allows_dynamic_support_target_years(
    tmp_path,
):
    profile = get_profile("ss-payroll-tob").to_dict()
    audit = {
        "method_used": "entropy",
        "fell_back_to_ipf": False,
        "age_max_pct_error": 0.0,
        "negative_weight_pct": 0.0,
        "positive_weight_count": 70000,
        "effective_sample_size": 5000.0,
        "top_10_weight_share_pct": 1.5,
        "top_100_weight_share_pct": 10.0,
        "max_constraint_pct_error": 0.0,
        "constraints": {},
        "validation_passed": True,
        "validation_issues": [],
        "calibration_quality": "exact",
    }
    support_2075 = {
        "name": "donor-backed-composite-v1",
        "family": "targeted_donor",
        "activation_start_year": 2075,
        "target_year": 2075,
        "target_year_strategy": "run_year",
        "report_file": "support_augmentation_report_2075.json",
        "report_summary": {"clone_household_count": 10},
    }
    support_2080 = dict(
        support_2075,
        target_year=2080,
        report_file="support_augmentation_report_2080.json",
        report_summary={"clone_household_count": 12},
    )

    _write_parallel_temp_year(
        root=tmp_path,
        year=2075,
        profile=profile,
        audit=audit,
        support_augmentation=support_2075,
    )
    _write_parallel_temp_year(
        root=tmp_path,
        year=2080,
        profile=profile,
        audit=audit,
        support_augmentation=support_2080,
    )

    manifest_path = merge_outputs(
        years=[2075, 2080],
        output_root=tmp_path,
        keep_temp=True,
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["years"] == [2075, 2080]
    assert manifest["support_augmentation"]["target_year_strategy"] == "run_year"


def test_parallel_projection_run_year_skips_existing_complete_temp_output(
    tmp_path, monkeypatch
):
    profile = get_profile("ss-payroll-tob").to_dict()
    audit = {
        "method_used": "entropy",
        "fell_back_to_ipf": False,
        "age_max_pct_error": 0.0,
        "negative_weight_pct": 0.0,
        "positive_weight_count": 70000,
        "effective_sample_size": 5000.0,
        "top_10_weight_share_pct": 1.5,
        "top_100_weight_share_pct": 10.0,
        "max_constraint_pct_error": 0.0,
        "constraints": {},
        "validation_passed": True,
        "validation_issues": [],
        "calibration_quality": "exact",
    }
    _write_parallel_temp_year(
        root=tmp_path,
        year=2048,
        profile=profile,
        audit=audit,
        tax_assumption={"name": "trustees-core-thresholds-v1"},
    )

    called = False

    def fake_run(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("subprocess.run should not be called for a complete year")

    monkeypatch.setattr(subprocess, "run", fake_run)

    year, output_dir, skipped = run_year(
        year=2048,
        output_root=tmp_path,
        forwarded_args=[],
    )

    assert year == 2048
    assert output_dir == year_output_dir(tmp_path, 2048)
    assert skipped is True
    assert called is False
    assert year_artifacts_complete(tmp_path, 2048) is True


def test_parallel_projection_run_year_replaces_partial_temp_output(
    tmp_path, monkeypatch
):
    year = 2049
    output_dir = year_output_dir(tmp_path, year)
    output_dir.mkdir(parents=True, exist_ok=True)
    stale_h5 = output_dir / f"{year}.h5"
    stale_h5.write_bytes(b"stale")

    def fake_run(command, cwd, stdout, stderr, check):
        del command, cwd, stderr, check
        assert not stale_h5.exists()
        (output_dir / f"{year}.h5").write_bytes(b"fresh")
        metadata_path = output_dir / f"{year}.h5.metadata.json"
        metadata_path.write_text(
            json.dumps(
                {"year": year, "calibration_audit": {"calibration_quality": "exact"}}
            ),
            encoding="utf-8",
        )
        (output_dir / "calibration_manifest.json").write_text(
            json.dumps({"ok": True}),
            encoding="utf-8",
        )
        stdout.write("ok\n")
        stdout.flush()
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    result_year, result_dir, skipped = run_year(
        year=year,
        output_root=tmp_path,
        forwarded_args=[],
    )

    assert result_year == year
    assert result_dir == output_dir
    assert skipped is False
    assert (output_dir / f"{year}.h5").read_bytes() == b"fresh"
    assert year_artifacts_complete(tmp_path, year) is True


def test_parallel_projection_merge_outputs_rejects_mismatched_contract(tmp_path):
    profile = get_profile("ss-payroll-tob").to_dict()
    audit = {
        "method_used": "entropy",
        "fell_back_to_ipf": False,
        "age_max_pct_error": 0.0,
        "negative_weight_pct": 0.0,
        "positive_weight_count": 70000,
        "effective_sample_size": 5000.0,
        "top_10_weight_share_pct": 1.5,
        "top_100_weight_share_pct": 10.0,
        "max_constraint_pct_error": 0.0,
        "constraints": {},
        "validation_passed": True,
        "validation_issues": [],
        "calibration_quality": "exact",
    }

    _write_parallel_temp_year(
        root=tmp_path,
        year=2062,
        profile=profile,
        audit=audit,
        tax_assumption={"name": "trustees-2025-core-thresholds-v1"},
    )
    _write_parallel_temp_year(
        root=tmp_path,
        year=2063,
        profile=profile,
        audit=audit,
        tax_assumption={"name": "different-tax-assumption"},
    )

    with pytest.raises(ValueError, match="Temp manifest mismatch for tax_assumption"):
        merge_outputs(
            years=[2062, 2063],
            output_root=tmp_path,
            keep_temp=True,
        )


def test_parallel_projection_merge_outputs_ignores_tax_assumption_end_year(tmp_path):
    profile = get_profile("ss-payroll-tob").to_dict()
    audit = {
        "method_used": "entropy",
        "fell_back_to_ipf": False,
        "age_max_pct_error": 0.0,
        "negative_weight_pct": 0.0,
        "positive_weight_count": 70000,
        "effective_sample_size": 5000.0,
        "top_10_weight_share_pct": 1.5,
        "top_100_weight_share_pct": 10.0,
        "max_constraint_pct_error": 0.0,
        "constraints": {},
        "validation_passed": True,
        "validation_issues": [],
        "calibration_quality": "exact",
    }

    _write_parallel_temp_year(
        root=tmp_path,
        year=2026,
        profile=profile,
        audit=audit,
        tax_assumption={
            "name": "trustees-core-thresholds-v1",
            "start_year": 2035,
            "end_year": 2026,
        },
    )
    _write_parallel_temp_year(
        root=tmp_path,
        year=2027,
        profile=profile,
        audit=audit,
        tax_assumption={
            "name": "trustees-core-thresholds-v1",
            "start_year": 2035,
            "end_year": 2027,
        },
    )

    manifest_path = merge_outputs(
        years=[2026, 2027],
        output_root=tmp_path,
        keep_temp=True,
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["years"] == [2026, 2027]
    assert manifest["tax_assumption"]["name"] == "trustees-core-thresholds-v1"


def test_summarize_realized_clone_translation_matches_toy_clone():
    import pandas as pd

    dataset = Dataset.from_dataframe(pd.DataFrame(_toy_support_dataframe()), 2024)
    augmentation_report = {
        "clone_household_reports": [
            {
                "candidate_idx": 0,
                "archetype": "older_worker_couple",
                "clone_household_id": 1,
                "clone_tax_unit_id": 101,
                "target_head_age": 70,
                "target_spouse_age": 68,
                "target_dependent_ages": [],
                "target_payroll_total": 5_000.0,
                "target_ss_total": 20_000.0,
            }
        ]
    }
    summary = summarize_realized_clone_translation(
        dataset,
        period=2024,
        augmentation_report=augmentation_report,
        age_bucket_size=5,
    )
    assert summary["matched_clone_household_count"] == 1
    assert summary["aggregate_ss_pct_error"] == pytest.approx(0.0)
    assert summary["aggregate_payroll_pct_error"] == pytest.approx(0.0)
    assert summary["per_clone"][0]["realized_ages"] == [70, 68]


def test_compose_role_donor_rows_falls_back_for_missing_dependents():
    import pandas as pd

    df = pd.DataFrame(_toy_support_dataframe())
    enriched = df.copy()
    enriched["__pe_payroll_uprating_factor"] = 2.0
    enriched["__pe_ss_uprating_factor"] = 3.0
    enriched["partnership_s_corp_income__2024"] = 1_000_000.0
    enriched["taxable_interest_income__2024"] = 500_000.0

    older_rows = enriched[enriched["person_tax_unit_id__2024"] == 201].copy()
    worker_rows = enriched[enriched["person_tax_unit_id__2024"] == 301].copy()
    candidate = SyntheticCandidate(
        archetype="older_plus_prime_worker_family_role_donor",
        head_age=80,
        spouse_age=60,
        dependent_ages=(12,),
        head_wages=0.0,
        spouse_wages=100_000.0,
        head_ss=60_000.0,
        spouse_ss=0.0,
        pension_income=0.0,
        dividend_income=0.0,
    )
    clone_df, _ = _compose_role_donor_rows_to_target(
        older_rows,
        worker_rows,
        base_year=2024,
        target_candidate=candidate,
        ss_scale=3.0,
        earnings_scale=2.0,
        id_counters={
            "household": 100,
            "family": 200,
            "tax_unit": 300,
            "spm_unit": 400,
            "marital_unit": 500,
            "person": 600,
        },
        clone_weight_scale=0.1,
        clone_weight_divisor=1,
    )
    assert clone_df is not None
    assert sorted(clone_df["age__2024"].astype(int).tolist()) == [12, 60, 80]


def test_compose_role_donor_rows_can_sanitize_worker_non_target_income():
    import pandas as pd

    df = pd.DataFrame(_toy_support_dataframe())
    enriched = df.copy()
    enriched["__pe_payroll_uprating_factor"] = 2.0
    enriched["__pe_ss_uprating_factor"] = 3.0
    enriched["taxable_interest_income__2024"] = [0.0, 0.0, 7_000.0, 0.0, 12_000.0]
    enriched["qualified_dividend_income__2024"] = [0.0, 0.0, 900.0, 0.0, 2_000.0]
    enriched["long_term_capital_gains_before_response__2024"] = [
        0.0,
        0.0,
        500.0,
        0.0,
        3_000.0,
    ]
    enriched["taxable_private_pension_income__2024"] = [
        0.0,
        0.0,
        4_000.0,
        0.0,
        5_000.0,
    ]
    enriched["partnership_s_corp_income__2024"] = [
        0.0,
        0.0,
        6_000.0,
        0.0,
        8_000.0,
    ]

    older_rows = enriched[enriched["person_tax_unit_id__2024"] == 201].copy()
    worker_rows = enriched[enriched["person_tax_unit_id__2024"] == 301].copy()
    candidate = SyntheticCandidate(
        archetype="older_plus_prime_worker_role_donor",
        head_age=80,
        spouse_age=60,
        dependent_ages=(),
        head_wages=0.0,
        spouse_wages=100_000.0,
        head_ss=60_000.0,
        spouse_ss=0.0,
        pension_income=0.0,
        dividend_income=0.0,
    )
    clone_df, _ = _compose_role_donor_rows_to_target(
        older_rows,
        worker_rows,
        base_year=2024,
        target_candidate=candidate,
        ss_scale=3.0,
        earnings_scale=2.0,
        id_counters={
            "household": 100,
            "family": 200,
            "tax_unit": 300,
            "spm_unit": 400,
            "marital_unit": 500,
            "person": 600,
        },
        clone_weight_scale=0.1,
        clone_weight_divisor=1,
        sanitize_worker_non_target_income=True,
    )

    assert clone_df is not None
    older_clone = clone_df[clone_df["age__2024"] == 80].iloc[0]
    worker_clone = clone_df[clone_df["age__2024"] == 60].iloc[0]
    assert older_clone["taxable_interest_income__2024"] == pytest.approx(7_000.0)
    assert worker_clone["taxable_interest_income__2024"] == pytest.approx(0.0)
    assert worker_clone["qualified_dividend_income__2024"] == pytest.approx(0.0)
    assert worker_clone["long_term_capital_gains_before_response__2024"] == (
        pytest.approx(0.0)
    )
    assert worker_clone["taxable_private_pension_income__2024"] == pytest.approx(0.0)
    assert worker_clone["partnership_s_corp_income__2024"] == pytest.approx(0.0)
    assert worker_clone["employment_income_before_lsr__2024"] == pytest.approx(50_000.0)
    assert (
        "long_term_capital_gains_before_response__2024"
        in clone_df.attrs["sanitized_worker_non_target_income_columns"]
    )


def test_compose_role_donor_rows_can_sanitize_all_clone_non_target_income():
    import pandas as pd

    df = pd.DataFrame(_toy_support_dataframe())
    enriched = df.copy()
    enriched["__pe_payroll_uprating_factor"] = 2.0
    enriched["__pe_ss_uprating_factor"] = 3.0
    enriched["taxable_interest_income__2024"] = [0.0, 0.0, 7_000.0, 0.0, 12_000.0]
    enriched["qualified_dividend_income__2024"] = [0.0, 0.0, 900.0, 0.0, 2_000.0]
    enriched["long_term_capital_gains_before_response__2024"] = [
        0.0,
        0.0,
        500.0,
        0.0,
        3_000.0,
    ]
    enriched["taxable_private_pension_income__2024"] = [
        0.0,
        0.0,
        4_000.0,
        0.0,
        5_000.0,
    ]
    enriched["partnership_s_corp_income__2024"] = [
        0.0,
        0.0,
        6_000.0,
        0.0,
        8_000.0,
    ]

    older_rows = enriched[enriched["person_tax_unit_id__2024"] == 201].copy()
    worker_rows = enriched[enriched["person_tax_unit_id__2024"] == 301].copy()
    candidate = SyntheticCandidate(
        archetype="older_plus_prime_worker_role_donor",
        head_age=80,
        spouse_age=60,
        dependent_ages=(),
        head_wages=0.0,
        spouse_wages=100_000.0,
        head_ss=60_000.0,
        spouse_ss=0.0,
        pension_income=0.0,
        dividend_income=0.0,
    )
    clone_df, _ = _compose_role_donor_rows_to_target(
        older_rows,
        worker_rows,
        base_year=2024,
        target_candidate=candidate,
        ss_scale=3.0,
        earnings_scale=2.0,
        id_counters={
            "household": 100,
            "family": 200,
            "tax_unit": 300,
            "spm_unit": 400,
            "marital_unit": 500,
            "person": 600,
        },
        clone_weight_scale=0.1,
        clone_weight_divisor=1,
        sanitize_worker_non_target_income=True,
        sanitize_clone_non_target_income=True,
    )

    assert clone_df is not None
    older_clone = clone_df[clone_df["age__2024"] == 80].iloc[0]
    worker_clone = clone_df[clone_df["age__2024"] == 60].iloc[0]
    for clone in (older_clone, worker_clone):
        assert clone["taxable_interest_income__2024"] == pytest.approx(0.0)
        assert clone["qualified_dividend_income__2024"] == pytest.approx(0.0)
        assert clone["long_term_capital_gains_before_response__2024"] == (
            pytest.approx(0.0)
        )
        assert clone["taxable_private_pension_income__2024"] == pytest.approx(0.0)
        assert clone["partnership_s_corp_income__2024"] == pytest.approx(0.0)
    assert older_clone["social_security_retirement__2024"] == pytest.approx(20_000.0)
    assert worker_clone["employment_income_before_lsr__2024"] == pytest.approx(50_000.0)
    assert (
        "long_term_capital_gains_before_response__2024"
        in clone_df.attrs["sanitized_clone_non_target_income_columns"]
    )
    assert "sanitized_worker_non_target_income_columns" not in clone_df.attrs
