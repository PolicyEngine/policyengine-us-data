"""Tests for extended CPS QRF imputation functions.

Uses synthetic data to verify that:
1. Sequential QRF preserves covariance between imputed variables
2. CPS-only imputation uses PUF-imputed income (not CPS originals)
3. Variable lists don't overlap (no double-imputation)
4. Post-processing constraints enforce IRS caps and SS normalization
"""

from contextlib import contextmanager

import numpy as np
import pandas as pd
import pytest

from policyengine_us_data.calibration.puf_impute import (
    IMPUTED_VARIABLES,
    OVERRIDDEN_IMPUTED_VARIABLES,
)
from policyengine_us_data.datasets.cps import extended_cps as extended_cps_module
from policyengine_us_data.datasets.cps.cps import ESI_POLICYHOLDER_VARIABLE
from policyengine_us_data.datasets.cps.extended_cps import (
    CPS_CLONE_FEATURE_VARIABLES,
    CPS_ONLY_IMPUTED_VARIABLES,
    CPS_CLONE_FEATURE_PREDICTORS,
    CPS_STAGE2_DEMOGRAPHIC_PREDICTORS,
    CPS_STAGE2_INCOME_PREDICTORS,
    ExtendedCPS,
    _load_raw_spm_capped_housing_subsidy,
    _apply_post_processing,
    _build_clone_test_frame,
    _derive_overtime_occupation_inputs,
    _impute_clone_cps_features,
    apply_retirement_constraints,
    reconcile_ss_subcomponents,
)
from policyengine_us_data.datasets.cps.tipped_occupation import (
    derive_treasury_tipped_occupation_code,
)
from policyengine_us_data.datasets.org import ORG_IMPUTED_VARIABLES
from policyengine_us_data.utils.aotc import (
    get_american_opportunity_credit_amount_scale,
    qualifying_expenses_from_american_opportunity_credit,
)
from policyengine_us_data.utils.dataset_validation import DatasetContractError


class _FakeHousingMicrosimulation:
    outputs = {
        "housing_assistance": np.array([5_000.0, 0.0]),
        "spm_unit_capped_housing_subsidy": np.array([4_000.0, 0.0]),
        "spm_unit_weight": np.array([2.0, 3.0]),
    }
    seen_data = None

    def __init__(self, dataset):
        self.dataset = dataset
        type(self).seen_data = dataset.load_dataset()

    def calculate(self, variable, period, **kwargs):
        return type(self).outputs[variable]


class _FakeCPSDataset:
    raw_cps = object()


class TestVariableListConsistency:
    """Variable lists should not overlap — no variable should be
    imputed by two different mechanisms."""

    def test_no_overlap_imputed_and_cps_only(self):
        overlap = set(IMPUTED_VARIABLES) & set(CPS_ONLY_IMPUTED_VARIABLES)
        assert overlap == set(), f"Variables in both IMPUTED and CPS_ONLY: {overlap}"

    def test_load_raw_spm_capped_housing_subsidy_aligns_to_spm_unit_ids(
        self, monkeypatch
    ):
        raw_spm_unit = pd.DataFrame(
            {
                "SPM_ID": [10, 20, 30],
                "SPM_CAPHOUSESUB": [100.0, 200.0, 300.0],
            }
        )

        @contextmanager
        def fake_open_dataset_read_only(dataset_source):
            yield {"spm_unit": raw_spm_unit}

        monkeypatch.setattr(
            extended_cps_module,
            "_open_dataset_read_only",
            fake_open_dataset_read_only,
        )

        result = _load_raw_spm_capped_housing_subsidy(
            _FakeCPSDataset,
            2024,
            target_spm_unit_ids=np.array([30, 10]),
        )

        assert result[2024].tolist() == [300.0, 100.0]

    def test_no_overlap_overridden_and_cps_only(self):
        overlap = set(OVERRIDDEN_IMPUTED_VARIABLES) & set(CPS_ONLY_IMPUTED_VARIABLES)
        assert overlap == set(), f"Variables in both OVERRIDDEN and CPS_ONLY: {overlap}"

    def test_no_overlap_clone_features_and_cps_only(self):
        overlap = set(CPS_CLONE_FEATURE_VARIABLES) & set(CPS_ONLY_IMPUTED_VARIABLES)
        assert overlap == set(), (
            f"Variables in both clone-feature and CPS_ONLY lists: {overlap}"
        )

    def test_overridden_is_subset_of_imputed(self):
        not_in_imputed = set(OVERRIDDEN_IMPUTED_VARIABLES) - set(IMPUTED_VARIABLES)
        assert not_in_imputed == set(), (
            f"OVERRIDDEN vars not in IMPUTED: {not_in_imputed}"
        )

    def test_stage2_income_predictors_in_imputed(self):
        """Stage-2 income predictors must come from stage-1 imputation."""
        for var in CPS_STAGE2_INCOME_PREDICTORS:
            assert var in IMPUTED_VARIABLES, (
                f"Stage-2 income predictor '{var}' not in "
                f"IMPUTED_VARIABLES — won't have PUF-imputed values"
            )

    def test_stage2_uses_esi_coverage_predictor(self):
        assert "has_esi" in CPS_STAGE2_DEMOGRAPHIC_PREDICTORS

    def test_cps_only_vars_mostly_exist_in_tbs(self):
        """Most CPS-only variables should exist in policyengine-us."""
        from policyengine_us import CountryTaxBenefitSystem

        tbs = CountryTaxBenefitSystem()
        valid = [v for v in CPS_ONLY_IMPUTED_VARIABLES if v in tbs.variables]
        assert len(valid) >= len(CPS_ONLY_IMPUTED_VARIABLES) * 0.9, (
            f"Only {len(valid)}/{len(CPS_ONLY_IMPUTED_VARIABLES)} "
            f"CPS-only vars exist in tax-benefit system"
        )

    def test_retirement_contributions_in_cps_only(self):
        """All 5 retirement contribution vars should be in CPS_ONLY."""
        expected = {
            "traditional_401k_contributions",
            "roth_401k_contributions",
            "traditional_ira_contributions",
            "roth_ira_contributions",
            "self_employed_pension_contributions",
        }
        missing = expected - set(CPS_ONLY_IMPUTED_VARIABLES)
        assert missing == set(), (
            f"Retirement contribution vars missing from CPS_ONLY: {missing}"
        )

    def test_ss_subcomponents_in_cps_only(self):
        """All 4 SS sub-component vars should be in CPS_ONLY."""
        expected = {
            "social_security_retirement",
            "social_security_disability",
            "social_security_dependents",
            "social_security_survivors",
        }
        missing = expected - set(CPS_ONLY_IMPUTED_VARIABLES)
        assert missing == set(), (
            f"SS sub-component vars missing from CPS_ONLY: {missing}"
        )

    def test_org_variables_in_cps_only(self):
        """ORG labor-market inputs should be re-imputed for PUF clones."""
        missing = set(ORG_IMPUTED_VARIABLES) - set(CPS_ONLY_IMPUTED_VARIABLES)
        assert missing == set(), f"ORG vars missing from CPS_ONLY: {missing}"

    def test_nonexistent_vars_not_in_cps_only(self):
        """Variables that don't exist in policyengine-us should not be
        in CPS_ONLY_IMPUTED_VARIABLES."""
        should_not_exist = {
            "roth_ira_distributions",
            "regular_ira_distributions",
            "other_type_retirement_account_distributions",
        }
        present = should_not_exist & set(CPS_ONLY_IMPUTED_VARIABLES)
        assert present == set(), f"Non-existent variables still in CPS_ONLY: {present}"

    def test_pension_income_not_in_cps_only(self):
        """Pension income vars are handled by Stage 1 rename, not
        Stage 2 QRF."""
        should_not_be_here = {
            "taxable_private_pension_income",
            "tax_exempt_private_pension_income",
        }
        present = should_not_be_here & set(CPS_ONLY_IMPUTED_VARIABLES)
        assert present == set(), (
            f"Pension income vars should not be in CPS_ONLY: {present}"
        )

    def test_capped_childcare_not_in_cps_only(self):
        """Capped childcare should not be independently QRF-imputed."""
        assert "spm_unit_capped_work_childcare_expenses" not in set(
            CPS_ONLY_IMPUTED_VARIABLES
        )

    def test_weeks_worked_is_cps_only_imputed_for_clone_records(self):
        assert "weeks_worked" in set(CPS_ONLY_IMPUTED_VARIABLES)

    def test_spm_threshold_is_formula_output_not_qrf_imputed(self):
        assert "spm_unit_spm_threshold" not in set(CPS_ONLY_IMPUTED_VARIABLES)
        data = {
            "spm_unit_spm_threshold": {2024: np.array([20_000.0])},
            "spm_unit_geographic_adjustment": {2024: np.array([1.0])},
            "person_in_poverty": {2024: np.array([False])},
        }

        with pytest.raises(
            DatasetContractError,
            match="spm_unit_geographic_adjustment",
        ):
            ExtendedCPS._assert_no_computed_variables_exported(data, 2024)

    def test_spm_resource_aggregates_are_not_qrf_imputed(self):
        assert "spm_unit_total_income_reported" not in set(CPS_ONLY_IMPUTED_VARIABLES)
        assert "spm_unit_net_income_reported" not in set(CPS_ONLY_IMPUTED_VARIABLES)
        assert "spm_unit_capped_housing_subsidy" not in set(CPS_ONLY_IMPUTED_VARIABLES)
        assert "housing_assistance" not in set(CPS_ONLY_IMPUTED_VARIABLES)
        assert "receives_housing_assistance" in set(CPS_ONLY_IMPUTED_VARIABLES)

    def test_weeks_worked_is_preserved_for_future_year_formulas(self):
        data = {"weeks_worked": {2024: np.array([52])}}

        ExtendedCPS._assert_no_computed_variables_exported(data, 2024)
        with pytest.raises(DatasetContractError, match="weeks_worked"):
            ExtendedCPS._assert_no_computed_variables_exported(data, 2025)

    def test_final_export_contract_allows_leaf_ss_retirement_input(self):
        data = {"social_security_retirement": {2024: np.array([12_000.0])}}

        ExtendedCPS._assert_no_computed_variables_exported(data, 2024)

    def test_final_export_contract_rejects_housing_assistance_formula_output(self):
        data = {"housing_assistance": {2024: np.array([3_000.0])}}

        with pytest.raises(DatasetContractError, match="housing_assistance"):
            ExtendedCPS._assert_no_computed_variables_exported(data, 2024)

    def test_final_export_contract_rejects_computed_ss_total(self):
        data = {"social_security": {2024: np.array([12_000.0])}}

        with pytest.raises(DatasetContractError, match="social_security"):
            ExtendedCPS._assert_no_computed_variables_exported(data, 2024)

    def test_final_export_contract_allows_structural_cache_variables(self):
        data = {
            "person_id": {2024: np.array([1])},
            "has_tin": {2024: np.array([True])},
            "has_itin": {2024: np.array([True])},
            "in_nyc": {2024: np.array([False])},
        }

        ExtendedCPS._assert_no_computed_variables_exported(data, 2024)

    def test_drop_final_computed_outputs_keeps_leaf_inputs(self):
        data = {
            "interest_income": {2024: np.array([100.0])},
            "taxable_interest_income": {2024: np.array([80.0])},
            "tax_exempt_interest_income": {2024: np.array([20.0])},
            "dividend_income": {2024: np.array([50.0])},
            "qualified_dividend_income": {2024: np.array([30.0])},
            "non_qualified_dividend_income": {2024: np.array([20.0])},
            "rent": {2024: np.array([1_000.0])},
            "pre_subsidy_rent": {2024: np.array([1_000.0])},
            "spm_unit_capped_work_childcare_expenses": {2024: np.array([500.0])},
            "spm_unit_pre_subsidy_childcare_expenses": {2024: np.array([600.0])},
            "spm_unit_spm_threshold": {2024: np.array([25_000.0])},
            "spm_unit_geographic_adjustment": {2024: np.array([1.1])},
            "person_in_poverty": {2024: np.array([False])},
            "has_tin": {2024: np.array([True])},
            "has_itin": {2024: np.array([True])},
            "in_nyc": {2024: np.array([False])},
        }

        result = ExtendedCPS._drop_final_computed_outputs(data)

        for variable in (
            "interest_income",
            "dividend_income",
            "rent",
            "spm_unit_capped_work_childcare_expenses",
            "spm_unit_spm_threshold",
            "spm_unit_geographic_adjustment",
            "person_in_poverty",
        ):
            assert variable not in result
        for variable in (
            "taxable_interest_income",
            "tax_exempt_interest_income",
            "qualified_dividend_income",
            "non_qualified_dividend_income",
            "pre_subsidy_rent",
            "spm_unit_pre_subsidy_childcare_expenses",
            "has_tin",
            "has_itin",
            "in_nyc",
        ):
            assert variable in result

        ExtendedCPS._assert_no_computed_variables_exported(result, 2024)

    def test_drop_puf_computed_intermediates_after_clone(self):
        data = {
            "cdcc_relevant_expenses": {2024: np.array([1_000.0])},
            "pre_tax_contributions": {2024: np.array([500.0])},
            "self_employed_health_insurance_ald": {2024: np.array([2_000.0])},
            "self_employed_pension_contribution_ald": {2024: np.array([3_000.0])},
            "employment_income": {2024: np.array([50_000.0])},
        }

        result = ExtendedCPS._drop_puf_computed_intermediates(data)

        assert "employment_income" in result
        assert "cdcc_relevant_expenses" not in result
        assert "pre_tax_contributions" not in result
        assert "self_employed_health_insurance_ald" not in result
        assert "self_employed_pension_contribution_ald" not in result

    def test_finalize_stage2_computed_variables_renames_and_drops(self):
        data = {
            "employment_income": {2024: np.array([50_000.0])},
            "weekly_hours_worked": {2024: np.array([40.0])},
            "social_security": {2024: np.array([12_000.0])},
            "social_security_retirement": {2024: np.array([12_000.0])},
            "social_security_disability": {2024: np.array([0.0])},
            "social_security_dependents": {2024: np.array([0.0])},
            "social_security_survivors": {2024: np.array([0.0])},
            "tax_unit_is_joint": {2024: np.array([True])},
            "employment_income_last_year": {2024: np.array([45_000.0])},
        }

        result = ExtendedCPS._finalize_stage2_computed_variables(data)

        assert "employment_income" not in result
        assert "employment_income_before_lsr" in result
        assert "weekly_hours_worked" not in result
        assert "weekly_hours_worked_before_lsr" in result
        assert "social_security" not in result
        assert "tax_unit_is_joint" not in result
        assert "employment_income_last_year" not in result

    def test_housing_assistance_validation_removes_formula_outputs_for_microsim(self):
        data = {
            "receives_housing_assistance": {2024: np.array([True, False])},
            "takes_up_housing_assistance_if_eligible": {2024: np.array([True, False])},
            "housing_assistance": {2024: np.array([99_000.0, 99_000.0])},
            "spm_unit_capped_housing_subsidy": {2024: np.array([3_000.0, 0.0])},
        }
        _FakeHousingMicrosimulation.outputs = {
            "housing_assistance": np.array([5_000.0, 0.0]),
            "spm_unit_capped_housing_subsidy": np.array([4_000.0, 0.0]),
            "spm_unit_weight": np.array([2.0, 3.0]),
        }

        result = ExtendedCPS._validate_housing_assistance_microsimulation(
            data,
            2024,
            microsimulation_cls=_FakeHousingMicrosimulation,
        )

        assert result is data
        assert "housing_assistance" not in _FakeHousingMicrosimulation.seen_data
        assert (
            "spm_unit_capped_housing_subsidy"
            not in _FakeHousingMicrosimulation.seen_data
        )

    def test_housing_assistance_validation_rejects_zero_modeled_benefits(self):
        data = {
            "receives_housing_assistance": {2024: np.array([True])},
            "takes_up_housing_assistance_if_eligible": {2024: np.array([True])},
        }
        _FakeHousingMicrosimulation.outputs = {
            "housing_assistance": np.array([0.0]),
            "spm_unit_capped_housing_subsidy": np.array([0.0]),
            "spm_unit_weight": np.array([1.0]),
        }

        with pytest.raises(RuntimeError, match="do not reconstruct modeled benefits"):
            ExtendedCPS._validate_housing_assistance_microsimulation(
                data,
                2024,
                microsimulation_cls=_FakeHousingMicrosimulation,
            )

    def test_housing_assistance_validation_rejects_tiny_reported_match(self):
        data = {
            "receives_housing_assistance": {2024: np.array([True])},
            "takes_up_housing_assistance_if_eligible": {2024: np.array([True])},
            "spm_unit_capped_housing_subsidy": {2024: np.array([10_000.0])},
        }
        _FakeHousingMicrosimulation.outputs = {
            "housing_assistance": np.array([100.0]),
            "spm_unit_capped_housing_subsidy": np.array([50.0]),
            "spm_unit_weight": np.array([1.0]),
        }

        with pytest.raises(RuntimeError, match="implausibly small"):
            ExtendedCPS._validate_housing_assistance_microsimulation(
                data,
                2024,
                microsimulation_cls=_FakeHousingMicrosimulation,
            )

    def test_housing_assistance_validation_allows_observed_formula_gap(self):
        data = {
            "receives_housing_assistance": {2024: np.array([True])},
            "takes_up_housing_assistance_if_eligible": {2024: np.array([True])},
            "spm_unit_capped_housing_subsidy": {2024: np.array([100.0])},
        }
        _FakeHousingMicrosimulation.outputs = {
            "housing_assistance": np.array([100.0]),
            "spm_unit_capped_housing_subsidy": np.array([58.0]),
            "spm_unit_weight": np.array([1.0]),
        }

        result = ExtendedCPS._validate_housing_assistance_microsimulation(
            data,
            2024,
            microsimulation_cls=_FakeHousingMicrosimulation,
        )

        assert result is data

    def test_housing_assistance_validation_rejects_half_reported_match(self):
        data = {
            "receives_housing_assistance": {2024: np.array([True])},
            "takes_up_housing_assistance_if_eligible": {2024: np.array([True])},
            "spm_unit_capped_housing_subsidy": {2024: np.array([100.0])},
        }
        _FakeHousingMicrosimulation.outputs = {
            "housing_assistance": np.array([100.0]),
            "spm_unit_capped_housing_subsidy": np.array([54.0]),
            "spm_unit_weight": np.array([1.0]),
        }

        with pytest.raises(RuntimeError, match="implausibly small"):
            ExtendedCPS._validate_housing_assistance_microsimulation(
                data,
                2024,
                microsimulation_cls=_FakeHousingMicrosimulation,
            )

    def test_reassign_housing_assistance_takeup_uses_geographic_eligibility(self):
        data = {
            "county_fips": {2024: np.array([1001, 1003, 1005, 1007])},
            "receives_housing_assistance": {
                2024: np.array([True, False, False, False])
            },
            "takes_up_housing_assistance_if_eligible": {
                2024: np.array([True, False, False, False])
            },
            "housing_assistance": {2024: np.array([99_000.0] * 4)},
            "spm_unit_capped_housing_subsidy": {2024: np.array([99_000.0] * 4)},
        }
        _FakeHousingMicrosimulation.outputs = {
            "is_eligible_for_housing_assistance": np.array([True, True, True, False]),
            "spm_unit_weight": np.array([1.0, 1.0, 1.0, 1.0]),
        }

        result = ExtendedCPS._reassign_housing_assistance_takeup_with_geography(
            data,
            2024,
            microsimulation_cls=_FakeHousingMicrosimulation,
            take_up_rate=0.75,
            draws=np.array([0.0, 0.9, 0.0, 0.0]),
        )

        assert result["takes_up_housing_assistance_if_eligible"][2024].tolist() == [
            True,
            False,
            True,
            False,
        ]
        assert "housing_assistance" not in _FakeHousingMicrosimulation.seen_data
        assert (
            "spm_unit_capped_housing_subsidy"
            not in _FakeHousingMicrosimulation.seen_data
        )

    def test_reassign_housing_assistance_takeup_separates_zero_weight_clones(self):
        data = {
            "county_fips": {2024: np.arange(6)},
            "receives_housing_assistance": {
                2024: np.array([True, False, False, True, False, False])
            },
            "takes_up_housing_assistance_if_eligible": {
                2024: np.array([True, False, False, True, False, False])
            },
        }
        _FakeHousingMicrosimulation.outputs = {
            "is_eligible_for_housing_assistance": np.array(
                [True, True, True, True, True, True]
            ),
            "spm_unit_weight": np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]),
        }

        result = ExtendedCPS._reassign_housing_assistance_takeup_with_geography(
            data,
            2024,
            microsimulation_cls=_FakeHousingMicrosimulation,
            take_up_rate=2 / 3,
            draws=np.array([0.0, 0.0, 0.9, 0.0, 0.0, 0.9]),
        )

        assert result["takes_up_housing_assistance_if_eligible"][2024].tolist() == [
            True,
            True,
            False,
            True,
            True,
            False,
        ]

    def test_drop_housing_assistance_formula_outputs_after_validation(self):
        data = {
            "housing_assistance": {2024: np.array([1_000.0])},
            "spm_unit_capped_housing_subsidy": {2024: np.array([800.0])},
            "receives_housing_assistance": {2024: np.array([True])},
        }

        result = ExtendedCPS._drop_housing_assistance_formula_outputs(data)

        assert result is data
        assert "housing_assistance" not in result
        assert "spm_unit_capped_housing_subsidy" not in result
        assert "receives_housing_assistance" in result


class TestStructuralMortgageValidation:
    def test_positive_mortgage_input_ignores_non_mortgage_interest_deduction(self):
        data = {
            "deductible_mortgage_interest": {2024: np.array([0.0, 0.0])},
            "interest_deduction": {2024: np.array([2_500.0])},
        }

        assert ExtendedCPS._has_positive_mortgage_input(data, 2024) is False

    def test_positive_mortgage_input_detects_positive_deductible_interest(self):
        data = {
            "deductible_mortgage_interest": {2024: np.array([0.0, 2_500.0])},
            "interest_deduction": {2024: np.array([2_500.0])},
        }

        assert ExtendedCPS._has_positive_mortgage_input(data, 2024) is True


class TestAOTCEligibilityInputImputation:
    @pytest.fixture
    def pe_us_supports_aotc_inputs(self, monkeypatch):
        monkeypatch.setattr(
            extended_cps_module,
            "_supports_aotc_eligibility_inputs",
            lambda: True,
        )

    def test_aotc_expense_fill_uses_policyengine_us_amount_scale(self):
        amount_scale = get_american_opportunity_credit_amount_scale(2024)
        max_credit = amount_scale.calc(
            np.array([amount_scale.thresholds[-1]], dtype=float)
        )[0]

        expenses = qualifying_expenses_from_american_opportunity_credit(
            max_credit,
            2024,
        )

        np.testing.assert_allclose(
            amount_scale.calc(np.array([expenses], dtype=float))[0],
            max_credit,
        )

    def test_leaves_data_unchanged_without_positive_aotc_signal(
        self,
        pe_us_supports_aotc_inputs,
    ):
        data = {
            "american_opportunity_credit": {2024: np.array([0.0])},
            "tax_unit_id": {2024: np.array([1])},
            "person_tax_unit_id": {2024: np.array([1])},
            "qualified_tuition_expenses": {2024: np.array([1_200.0])},
        }

        result = ExtendedCPS._impute_aotc_eligibility_inputs(data, 2024)

        assert "is_pursuing_credential_for_american_opportunity_credit" not in result
        np.testing.assert_array_equal(
            result["qualified_tuition_expenses"][2024],
            np.array([1_200.0]),
        )

    def test_uses_tuition_signal_when_aotc_output_is_absent(
        self,
        pe_us_supports_aotc_inputs,
    ):
        data = {
            "tax_unit_id": {2024: np.array([1, 2])},
            "person_tax_unit_id": {2024: np.array([1, 1, 2])},
            "qualified_tuition_expenses": {2024: np.array([1_200.0, 0.0, 900.0])},
        }

        result = ExtendedCPS._impute_aotc_eligibility_inputs(data, 2024)

        expected = np.array([True, False, True])
        for variable in (
            "is_pursuing_credential_for_american_opportunity_credit",
            "attends_eligible_educational_institution_for_american_opportunity_credit",
            "is_enrolled_at_least_half_time_for_american_opportunity_credit",
            "has_american_opportunity_credit_1098_t_or_exception",
            "has_american_opportunity_credit_institution_ein",
        ):
            np.testing.assert_array_equal(result[variable][2024], expected)
        np.testing.assert_array_equal(
            result["qualified_tuition_expenses"][2024],
            np.array([1_200.0, 0.0, 900.0]),
        )

    def test_marks_tuition_members_in_positive_aotc_tax_units(
        self,
        pe_us_supports_aotc_inputs,
    ):
        target_credit = 1_000.0
        data = {
            "american_opportunity_credit": {2024: np.array([target_credit, 0.0])},
            "tax_unit_id": {2024: np.array([1, 2])},
            "person_tax_unit_id": {2024: np.array([1, 1, 2])},
            "qualified_tuition_expenses": {2024: np.array([1_200.0, 0.0, 1_200.0])},
            "is_full_time_college_student": {2024: np.array([False, True, True])},
        }

        result = ExtendedCPS._impute_aotc_eligibility_inputs(data, 2024)

        expected = np.array([True, False, False])
        for variable in (
            "is_pursuing_credential_for_american_opportunity_credit",
            "attends_eligible_educational_institution_for_american_opportunity_credit",
            "is_enrolled_at_least_half_time_for_american_opportunity_credit",
            "has_american_opportunity_credit_1098_t_or_exception",
            "has_american_opportunity_credit_institution_ein",
        ):
            np.testing.assert_array_equal(result[variable][2024], expected)
        for variable in (
            "has_completed_first_four_years_of_postsecondary_education",
            "has_felony_drug_conviction",
        ):
            np.testing.assert_array_equal(result[variable][2024], np.zeros(3, bool))
        np.testing.assert_array_equal(
            result["american_opportunity_credit_claimed_prior_years"][2024],
            np.zeros(3, dtype=np.int8),
        )
        np.testing.assert_array_equal(
            result["qualified_tuition_expenses"][2024],
            np.array(
                [
                    qualifying_expenses_from_american_opportunity_credit(
                        target_credit,
                        2024,
                    ),
                    0.0,
                    1_200.0,
                ]
            ),
        )

    def test_fills_tuition_when_positive_aotc_unit_has_no_tuition(
        self,
        pe_us_supports_aotc_inputs,
    ):
        amount_scale = get_american_opportunity_credit_amount_scale(2024)
        max_credit = amount_scale.calc(
            np.array([amount_scale.thresholds[-1]], dtype=float)
        )[0]
        data = {
            "american_opportunity_credit": {2024: np.array([max_credit])},
            "tax_unit_id": {2024: np.array([1])},
            "person_tax_unit_id": {2024: np.array([1, 1])},
            "qualified_tuition_expenses": {2024: np.array([0.0, 0.0])},
            "is_full_time_college_student": {2024: np.array([False, True])},
        }

        result = ExtendedCPS._impute_aotc_eligibility_inputs(data, 2024)

        expected = np.array([False, True])
        for variable in (
            "is_pursuing_credential_for_american_opportunity_credit",
            "attends_eligible_educational_institution_for_american_opportunity_credit",
            "is_enrolled_at_least_half_time_for_american_opportunity_credit",
            "has_american_opportunity_credit_1098_t_or_exception",
            "has_american_opportunity_credit_institution_ein",
        ):
            np.testing.assert_array_equal(result[variable][2024], expected)
        expected_expenses = qualifying_expenses_from_american_opportunity_credit(
            data["american_opportunity_credit"][2024][0],
            2024,
        )
        np.testing.assert_array_equal(
            result["qualified_tuition_expenses"][2024],
            np.array([0.0, expected_expenses]),
        )

    def test_splits_multi_student_credit_across_multiple_candidates(
        self,
        pe_us_supports_aotc_inputs,
    ):
        amount_scale = get_american_opportunity_credit_amount_scale(2024)
        max_credit = amount_scale.calc(
            np.array([amount_scale.thresholds[-1]], dtype=float)
        )[0]
        data = {
            "american_opportunity_credit": {2024: np.array([max_credit * 2])},
            "tax_unit_id": {2024: np.array([1])},
            "person_tax_unit_id": {2024: np.array([1, 1])},
            "qualified_tuition_expenses": {2024: np.array([0.0, 0.0])},
            "is_full_time_college_student": {2024: np.array([True, True])},
        }

        result = ExtendedCPS._impute_aotc_eligibility_inputs(data, 2024)

        expected = np.array([True, True])
        np.testing.assert_array_equal(
            result["is_pursuing_credential_for_american_opportunity_credit"][2024],
            expected,
        )
        expected_expenses = qualifying_expenses_from_american_opportunity_credit(
            max_credit,
            2024,
        )
        np.testing.assert_array_equal(
            result["qualified_tuition_expenses"][2024],
            np.array([expected_expenses, expected_expenses]),
        )

    def test_uses_legacy_eligibility_input_when_pe_us_lacks_new_inputs(
        self,
        monkeypatch,
    ):
        monkeypatch.setattr(
            extended_cps_module,
            "_supports_aotc_eligibility_inputs",
            lambda: False,
        )
        data = {
            "american_opportunity_credit": {2024: np.array([1_000.0])},
            "tax_unit_id": {2024: np.array([1])},
            "person_tax_unit_id": {2024: np.array([1])},
            "qualified_tuition_expenses": {2024: np.array([0.0])},
        }

        result = ExtendedCPS._impute_aotc_eligibility_inputs(data, 2024)

        np.testing.assert_array_equal(
            result["is_eligible_for_american_opportunity_credit"][2024],
            np.array([True]),
        )
        assert "is_pursuing_credential_for_american_opportunity_credit" not in result


class TestLLCEligibilityInputImputation:
    @pytest.fixture
    def pe_us_supports_llc_inputs(self, monkeypatch):
        monkeypatch.setattr(
            extended_cps_module,
            "_supports_llc_eligibility_inputs",
            lambda: True,
        )

    def test_marks_non_aotc_tuition_people_as_llc_eligible(
        self,
        pe_us_supports_llc_inputs,
    ):
        data = {
            "person_tax_unit_id": {2024: np.array([1, 1, 2])},
            "qualified_tuition_expenses": {2024: np.array([1_000.0, 2_000.0, 0.0])},
            "is_pursuing_credential_for_american_opportunity_credit": {
                2024: np.array([True, False, False])
            },
        }

        result = ExtendedCPS._impute_llc_eligibility_inputs(data, 2024)

        expected = np.array([False, True, False])
        for variable in (
            "attends_eligible_educational_institution_for_lifetime_learning_credit",
            "has_lifetime_learning_credit_1098_t_or_exception",
        ):
            np.testing.assert_array_equal(result[variable][2024], expected)

    def test_leaves_data_unchanged_when_pe_us_lacks_llc_inputs(self, monkeypatch):
        monkeypatch.setattr(
            extended_cps_module,
            "_supports_llc_eligibility_inputs",
            lambda: False,
        )
        data = {
            "person_tax_unit_id": {2024: np.array([1])},
            "qualified_tuition_expenses": {2024: np.array([1_000.0])},
        }

        result = ExtendedCPS._impute_llc_eligibility_inputs(data, 2024)

        assert (
            "attends_eligible_educational_institution_for_lifetime_learning_credit"
            not in result
        )


class TestStage2PostProcessing:
    def test_zeroes_esi_premiums_for_non_policyholder_clone_records(self):
        predictions = pd.DataFrame(
            {"employer_sponsored_insurance_premiums": [6_000.0, 4_000.0]}
        )
        x_test = pd.DataFrame({"has_esi": [True, True]})

        result = _apply_post_processing(
            predictions=predictions,
            X_test=x_test,
            time_period=2024,
            data={
                "person_id": {2024: np.array([1, 2, 3, 4])},
                ESI_POLICYHOLDER_VARIABLE: {2024: np.array([True, False, True, False])},
            },
        )

        np.testing.assert_allclose(
            result["employer_sponsored_insurance_premiums"].to_numpy(),
            np.array([6_000.0, 0.0]),
        )


class TestRetirementConstraints:
    """Post-processing retirement constraints enforce IRS caps."""

    @pytest.fixture
    def sample_predictions(self):
        return pd.DataFrame(
            {
                "traditional_401k_contributions": [25000, -500, 5000, 10000, 3000],
                "roth_401k_contributions": [30000, 2000, 0, 50000, 1000],
                "traditional_ira_contributions": [8000, -100, 3000, 15000, 500],
                "roth_ira_contributions": [10000, 1000, 0, 20000, 200],
                "self_employed_pension_contributions": [80000, -200, 5000, 0, 100000],
            }
        )

    @pytest.fixture
    def sample_features(self):
        return pd.DataFrame(
            {
                "age": [55, 30, 45, 60, 25],
                "employment_income": [100000, 50000, 0, 80000, 60000],
                "self_employment_income": [0, 0, 20000, 50000, 200000],
            }
        )

    def test_non_negativity(self, sample_predictions, sample_features):
        result = apply_retirement_constraints(sample_predictions, sample_features, 2024)
        for var in result.columns:
            assert (result[var] >= 0).all(), f"{var} has negative values"

    def test_401k_capped_at_limit(self, sample_predictions, sample_features):
        result = apply_retirement_constraints(sample_predictions, sample_features, 2024)
        from policyengine_us_data.utils.retirement_limits import get_retirement_limits

        limits = get_retirement_limits(2024)
        age = sample_features["age"].values
        catch_up = age >= 50
        cap = limits["401k"] + catch_up * limits["401k_catch_up"]
        for var in ["traditional_401k_contributions", "roth_401k_contributions"]:
            assert (result[var].values <= cap).all(), f"{var} exceeds 401k cap"

    def test_ira_capped_at_limit(self, sample_predictions, sample_features):
        result = apply_retirement_constraints(sample_predictions, sample_features, 2024)
        from policyengine_us_data.utils.retirement_limits import get_retirement_limits

        limits = get_retirement_limits(2024)
        age = sample_features["age"].values
        catch_up = age >= 50
        cap = limits["ira"] + catch_up * limits["ira_catch_up"]
        for var in ["traditional_ira_contributions", "roth_ira_contributions"]:
            assert (result[var].values <= cap).all(), f"{var} exceeds IRA cap"

    def test_401k_zeroed_without_employment_income(
        self, sample_predictions, sample_features
    ):
        result = apply_retirement_constraints(sample_predictions, sample_features, 2024)
        no_emp = sample_features["employment_income"] == 0
        for var in ["traditional_401k_contributions", "roth_401k_contributions"]:
            assert (result[var].values[no_emp] == 0).all(), (
                f"{var} should be zero without employment income"
            )

    def test_se_pension_capped(self, sample_predictions, sample_features):
        result = apply_retirement_constraints(sample_predictions, sample_features, 2024)
        se_income = sample_features["self_employment_income"].values
        se_vals = result["self_employed_pension_contributions"].values
        rate_cap = se_income * 0.25
        assert (se_vals <= rate_cap + 1).all(), "SE pension exceeds 25% of SE income"

    def test_se_pension_zeroed_without_se_income(
        self, sample_predictions, sample_features
    ):
        result = apply_retirement_constraints(sample_predictions, sample_features, 2024)
        no_se = sample_features["self_employment_income"] == 0
        assert (
            result["self_employed_pension_contributions"].values[no_se] == 0
        ).all(), "SE pension should be zero without SE income"


class TestTreasuryTippedOccupationCode:
    def test_derive_treasury_tipped_occupation_code(self):
        derived = derive_treasury_tipped_occupation_code(
            np.array([4040, 4110, 4230, 2770, -1, 9999])
        )

        assert derived.tolist() == [101, 102, 304, 208, 0, 0]


class TestSSReconciliation:
    """Post-processing SS normalization ensures sub-components sum to total."""

    def test_subcomponents_sum_to_total(self):
        predictions = pd.DataFrame(
            {
                "social_security_retirement": [0.6, 0.0, 0.8, 0.3],
                "social_security_disability": [0.3, 0.0, 0.1, 0.5],
                "social_security_dependents": [0.05, 0.0, 0.05, 0.1],
                "social_security_survivors": [0.05, 0.0, 0.05, 0.1],
            }
        )
        total_ss = np.array([20000, 0, 15000, 10000])
        result = reconcile_ss_subcomponents(predictions, total_ss)
        sums = sum(result[col].values for col in result.columns)
        np.testing.assert_allclose(sums, total_ss, atol=0.01)

    def test_zero_ss_zeroes_all_subcomponents(self):
        predictions = pd.DataFrame(
            {
                "social_security_retirement": [0.5, 0.7],
                "social_security_disability": [0.3, 0.2],
                "social_security_dependents": [0.1, 0.05],
                "social_security_survivors": [0.1, 0.05],
            }
        )
        total_ss = np.array([0, 0])
        result = reconcile_ss_subcomponents(predictions, total_ss)
        for col in result.columns:
            assert (result[col].values == 0).all(), f"{col} should be zero"

    def test_shares_are_non_negative(self):
        predictions = pd.DataFrame(
            {
                "social_security_retirement": [-0.5, 0.8],
                "social_security_disability": [1.2, 0.2],
                "social_security_dependents": [0.1, 0.0],
                "social_security_survivors": [0.2, 0.0],
            }
        )
        total_ss = np.array([10000, 5000])
        result = reconcile_ss_subcomponents(predictions, total_ss)
        for col in result.columns:
            assert (result[col].values >= 0).all(), f"{col} has negative values"

    def test_single_component_gets_full_total(self):
        predictions = pd.DataFrame(
            {
                "social_security_retirement": [1.0],
                "social_security_disability": [0.0],
                "social_security_dependents": [0.0],
                "social_security_survivors": [0.0],
            }
        )
        total_ss = np.array([25000])
        result = reconcile_ss_subcomponents(predictions, total_ss)
        assert result["social_security_retirement"].values[0] == pytest.approx(
            25000, abs=0.01
        )


class TestSequentialQRF:
    """Verify that sequential QRF produces correlated outputs."""

    @pytest.fixture
    def correlated_training_data(self):
        rng = np.random.default_rng(42)
        n = 2000
        x = rng.normal(50, 15, n)
        y1 = 0.8 * x + rng.normal(0, 5, n)
        y2 = 0.3 * x + 0.5 * y1 + rng.normal(0, 3, n)
        return pd.DataFrame({"x": x, "y1": y1, "y2": y2})

    def test_sequential_qrf_preserves_correlation(self, correlated_training_data):
        from microimpute.models.qrf import QRF

        df = correlated_training_data
        train = df.sample(1500, random_state=0)
        test_x = df.drop(train.index)[["x"]]

        qrf = QRF(log_level="ERROR", memory_efficient=True)
        result = qrf.fit_predict(
            X_train=train,
            X_test=test_x,
            predictors=["x"],
            imputed_variables=["y1", "y2"],
            n_jobs=1,
        )
        corr = result["y1"].corr(result["y2"])
        assert corr > 0.5, (
            f"Sequential QRF y1-y2 correlation = {corr:.3f}, expected > 0.5"
        )

    def test_single_call_vs_separate_calls_differ(self, correlated_training_data):
        from microimpute.models.qrf import QRF

        df = correlated_training_data
        train = df.sample(1500, random_state=0)
        test_x = df.drop(train.index)[["x"]]

        qrf_seq = QRF(log_level="ERROR", memory_efficient=True)
        result_seq = qrf_seq.fit_predict(
            X_train=train,
            X_test=test_x,
            predictors=["x"],
            imputed_variables=["y1", "y2"],
            n_jobs=1,
        )

        qrf_y1 = QRF(log_level="ERROR", memory_efficient=True)
        result_y1 = qrf_y1.fit_predict(
            X_train=train[["x", "y1"]],
            X_test=test_x,
            predictors=["x"],
            imputed_variables=["y1"],
            n_jobs=1,
        )

        qrf_y2 = QRF(log_level="ERROR", memory_efficient=True)
        result_y2 = qrf_y2.fit_predict(
            X_train=train[["x", "y2"]],
            X_test=test_x,
            predictors=["x"],
            imputed_variables=["y2"],
            n_jobs=1,
        )

        corr_seq = result_seq["y1"].corr(result_seq["y2"])
        corr_indep = result_y1["y1"].corr(result_y2["y2"])
        assert corr_seq > corr_indep, (
            f"Sequential corr ({corr_seq:.3f}) should exceed independent corr ({corr_indep:.3f})"
        )


class TestCloneFeatureImputation:
    def test_build_clone_test_frame_overrides_person_and_household_features(self):
        class FakeMicrosimulation:
            def calculate_dataframe(self, columns):
                base = pd.DataFrame(
                    {
                        "age": [30, 40],
                        "state_fips": [6, 36],
                        "tax_unit_is_joint": [0, 1],
                        "tax_unit_count_dependents": [0, 2],
                        "is_tax_unit_head": [1, 1],
                        "is_tax_unit_spouse": [0, 0],
                        "is_tax_unit_dependent": [0, 0],
                        "employment_income": [20_000, 35_000],
                        "self_employment_income": [0, 0],
                        "social_security": [0, 0],
                    }
                )
                return base[columns]

        tp = 2024
        data = {
            "person_id": {tp: np.array([1, 2, 101, 102])},
            "household_id": {tp: np.array([1, 2, 101, 102])},
            "person_household_id": {tp: np.array([1, 2, 101, 102])},
            "age": {tp: np.array([30, 40, 50, 60], dtype=np.float32)},
            "employment_income": {
                tp: np.array([20_000, 35_000, 90_000, 150_000], dtype=np.float32)
            },
            "self_employment_income": {tp: np.zeros(4, dtype=np.float32)},
            "social_security": {tp: np.zeros(4, dtype=np.float32)},
            "is_tax_unit_head": {tp: np.ones(4, dtype=bool)},
            "is_tax_unit_spouse": {tp: np.zeros(4, dtype=bool)},
            "is_tax_unit_dependent": {tp: np.zeros(4, dtype=bool)},
            "state_fips": {tp: np.array([6, 36, 12, 48], dtype=np.int16)},
        }

        result = _build_clone_test_frame(
            FakeMicrosimulation(),
            data,
            tp,
            CPS_CLONE_FEATURE_PREDICTORS,
        )

        assert result["age"].tolist() == [50, 60]
        assert result["employment_income"].tolist() == [90_000, 150_000]
        assert result["state_fips"].tolist() == [12, 48]
        assert result["tax_unit_is_joint"].tolist() == [0, 1]

    def test_derive_overtime_occupation_inputs(self):
        derived = _derive_overtime_occupation_inputs(np.array([53, 52, 8, 41, 1, 99]))

        assert derived["has_never_worked"].tolist() == [
            True,
            False,
            False,
            False,
            False,
            False,
        ]
        assert derived["is_military"].tolist() == [
            False,
            True,
            False,
            False,
            False,
            False,
        ]
        assert derived["is_computer_scientist"].tolist() == [
            False,
            False,
            True,
            False,
            False,
            False,
        ]
        assert derived["is_farmer_fisher"].tolist() == [
            False,
            False,
            False,
            True,
            False,
            False,
        ]
        assert derived["is_executive_administrative_professional"].tolist() == [
            False,
            False,
            False,
            False,
            True,
            False,
        ]

    def test_clone_feature_imputation_rematches_outputs_and_derives_flags(
        self, monkeypatch
    ):
        import policyengine_us

        train = pd.DataFrame(
            {
                "age": [45, 17],
                "state_fips": [1, 1],
                "tax_unit_is_joint": [0, 0],
                "tax_unit_count_dependents": [0, 1],
                "is_tax_unit_head": [1, 0],
                "is_tax_unit_spouse": [0, 0],
                "is_tax_unit_dependent": [0, 1],
                "employment_income": [95_000, 0],
                "self_employment_income": [0, 0],
                "social_security": [0, 0],
                "is_male": [1, 0],
                "cps_race": [2, 1],
                "is_hispanic": [0, 1],
                "detailed_occupation_recode": [8, 41],
                "treasury_tipped_occupation_code": [101, 304],
            }
        )

        class FakeMicrosimulation:
            def __init__(self, dataset):
                self.dataset = dataset

            def calculate_dataframe(self, columns):
                return train[columns]

        monkeypatch.setattr(policyengine_us, "Microsimulation", FakeMicrosimulation)

        tp = 2024
        data = {
            "person_id": {tp: np.array([1, 2, 101, 102])},
            "household_id": {tp: np.array([1, 2, 101, 102])},
            "person_household_id": {tp: np.array([1, 2, 101, 102])},
            "age": {tp: np.array([45, 17, 46, 17], dtype=np.float32)},
            "state_fips": {tp: np.array([1, 1, 1, 1], dtype=np.int16)},
            "tax_unit_is_joint": {tp: np.zeros(4, dtype=np.float32)},
            "tax_unit_count_dependents": {tp: np.array([0, 1, 0, 1], dtype=np.float32)},
            "is_tax_unit_head": {tp: np.array([1, 0, 1, 0], dtype=bool)},
            "is_tax_unit_spouse": {tp: np.zeros(4, dtype=bool)},
            "is_tax_unit_dependent": {tp: np.array([0, 1, 0, 1], dtype=bool)},
            "employment_income": {
                tp: np.array([95_000, 0, 97_000, 0], dtype=np.float32)
            },
            "self_employment_income": {tp: np.zeros(4, dtype=np.float32)},
            "social_security": {tp: np.zeros(4, dtype=np.float32)},
        }

        result = _impute_clone_cps_features(data, tp, "unused")

        assert result["detailed_occupation_recode"].tolist() == [8, 41]
        assert result["is_male"].tolist() == [1, 0]
        assert result["cps_race"].tolist() == [2, 1]
        assert result["is_hispanic"].tolist() == [0, 1]
        if "treasury_tipped_occupation_code" in result.columns:
            assert result["treasury_tipped_occupation_code"].tolist() == [101, 304]
        assert result["is_computer_scientist"].tolist() == [True, False]
        assert result["is_farmer_fisher"].tolist() == [False, True]
