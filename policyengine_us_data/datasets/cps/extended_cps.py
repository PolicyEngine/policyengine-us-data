import logging
import time
from typing import Type

import numpy as np
import pandas as pd
from policyengine_core.data import Dataset

from policyengine_us_data.datasets.cps.cps import *  # noqa: F403
from policyengine_us_data.datasets.puf import *  # noqa: F403
from policyengine_us_data.storage import STORAGE_FOLDER
from policyengine_us_data.utils.retirement_limits import (
    get_retirement_limits,
    get_se_pension_limits,
)

logger = logging.getLogger(__name__)

# Minimum AGI for PUF records to be injected into the ExtendedCPS
AGI_INJECTION_THRESHOLD = 1_000_000

# CPS-only variables that should be QRF-imputed for the PUF clone half
# instead of naively duplicated from the CPS donor. These are
# income-correlated variables that exist only in the CPS; demographics,
# IDs, weights, and random seeds are fine to duplicate.
CPS_ONLY_IMPUTED_VARIABLES = [
    # Retirement distributions
    "taxable_401k_distributions",
    "tax_exempt_401k_distributions",
    "taxable_403b_distributions",
    "tax_exempt_403b_distributions",
    "keogh_distributions",
    "taxable_sep_distributions",
    "tax_exempt_sep_distributions",
    # Retirement contributions
    "traditional_401k_contributions",
    "roth_401k_contributions",
    "traditional_ira_contributions",
    "roth_ira_contributions",
    "self_employed_pension_contributions",
    # Social Security sub-components
    "social_security_retirement",
    "social_security_disability",
    "social_security_dependents",
    "social_security_survivors",
    # Transfer income
    "unemployment_compensation",
    "tanf_reported",
    "ssi_reported",
    "child_support_received",
    "veterans_benefits",
    "workers_compensation",
    "disability_benefits",
    "strike_benefits",
    "receives_wic",
    # SPM variables
    "spm_unit_total_income_reported",
    "snap_reported",
    "spm_unit_capped_housing_subsidy_reported",
    "free_school_meals_reported",
    "spm_unit_energy_subsidy_reported",
    "spm_unit_wic_reported",
    "spm_unit_broadband_subsidy_reported",
    "spm_unit_payroll_tax_reported",
    "spm_unit_federal_tax_reported",
    "spm_unit_state_tax_reported",
    "spm_unit_capped_work_childcare_expenses",
    "spm_unit_spm_threshold",
    "spm_unit_net_income_reported",
    "spm_unit_pre_subsidy_childcare_expenses",
    # Medical expenses
    "health_insurance_premiums_without_medicare_part_b",
    "over_the_counter_health_expenses",
    "other_medical_expenses",
    "medicare_part_b_premiums",
    "child_support_expense",
    # Hours/employment
    "weekly_hours_worked",
    "hours_worked_last_week",
    # Previous year income
    "employment_income_last_year",
    "self_employment_income_last_year",
]

# Set for O(1) lookup in the splice loop.
_CPS_ONLY_SET = set(CPS_ONLY_IMPUTED_VARIABLES)

# Predictors used for the second-stage CPS-only imputation: demographics
# plus key income variables that were already imputed from PUF data.
CPS_STAGE2_DEMOGRAPHIC_PREDICTORS = [
    "age",
    "is_male",
    "tax_unit_is_joint",
    "tax_unit_count_dependents",
]

CPS_STAGE2_INCOME_PREDICTORS = [
    "employment_income",
    "self_employment_income",
    "social_security",
]


def _impute_cps_only_variables(
    data: dict,
    time_period: int,
    dataset_path: str,
) -> pd.DataFrame:
    """Second-stage QRF: train on CPS, predict for PUF clones.

    For the PUF clone half of the extended CPS we need plausible values
    of CPS-only variables (retirement distributions, transfers, hours,
    SPM components, etc.) that are consistent with the clone's
    PUF-imputed income -- not just naively copied from the CPS donor.

    We train a QRF on CPS person-level data where:
      * predictors = demographics + key income variables
      * outputs    = CPS-only variables listed in
                     ``CPS_ONLY_IMPUTED_VARIABLES``

    For PUF clone prediction we use the PUF-imputed income values
    from the second half of ``data`` (the clone half, which already
    has PUF-imputed income from stage 1).

    Uses ``fit_predict()`` with ``max_train_samples`` instead of
    manual sampling + separate fit/predict.

    Args:
        data: Extended dataset dict after ``puf_clone_dataset()`` --
            already doubled, with PUF-imputed income in the second half.
        time_period: Tax year.
        dataset_path: Path to the CPS h5 file for Microsimulation.

    Returns:
        DataFrame with one column per CPS-only variable, containing
        predicted values for the PUF clone half (person-level).
    """
    from microimpute.models.qrf import QRF
    from policyengine_us import CountryTaxBenefitSystem, Microsimulation

    all_predictors = CPS_STAGE2_DEMOGRAPHIC_PREDICTORS + CPS_STAGE2_INCOME_PREDICTORS

    # Filter to variables that exist in the current policyengine-us.
    tbs = CountryTaxBenefitSystem()
    valid_outputs = [v for v in CPS_ONLY_IMPUTED_VARIABLES if v in tbs.variables]
    skipped = set(CPS_ONLY_IMPUTED_VARIABLES) - set(valid_outputs)
    if skipped:
        logger.warning(
            "CPS-only imputation: %d variables not in tax-benefit system: %s",
            len(skipped),
            sorted(skipped),
        )

    # Load original (non-doubled) CPS for training data.
    cps_sim = Microsimulation(dataset=dataset_path)
    X_train = cps_sim.calculate_dataframe(all_predictors + valid_outputs)

    available_outputs = [col for col in valid_outputs if col in X_train.columns]
    missing_outputs = [col for col in valid_outputs if col not in X_train.columns]
    if missing_outputs:
        logger.warning(
            "CPS-only imputation: %d variables not found in CPS: %s",
            len(missing_outputs),
            missing_outputs,
        )

    # Build PUF clone test data: demographics from CPS sim (PUF clones
    # share demographics with their CPS donors), income from the
    # PUF-imputed values in the second half of the doubled data.
    n_persons_half = len(data["person_id"][time_period]) // 2
    X_test = cps_sim.calculate_dataframe(CPS_STAGE2_DEMOGRAPHIC_PREDICTORS)
    del cps_sim

    for var in CPS_STAGE2_INCOME_PREDICTORS:
        # Income comes from PUF imputation in the second half.
        X_test[var] = data[var][time_period][n_persons_half:]

    logger.info(
        "Stage-2 CPS-only imputation: %d outputs, "
        "training on %d CPS persons, predicting for %d PUF clones",
        len(available_outputs),
        len(X_train),
        len(X_test),
    )
    total_start = time.time()

    qrf = QRF(
        log_level="INFO",
        memory_efficient=True,
        max_train_samples=5000,
    )
    predictions = qrf.fit_predict(
        X_train=X_train[all_predictors + available_outputs],
        X_test=X_test[all_predictors],
        predictors=all_predictors,
        imputed_variables=available_outputs,
        n_jobs=1,
    )

    # Add zeros for variables that weren't available in CPS.
    for var in missing_outputs:
        predictions[var] = 0

    # Apply domain constraints to retirement and SS variables.
    predictions = _apply_post_processing(predictions, X_test, time_period, data)

    logger.info(
        "Stage-2 CPS-only imputation took %.2fs total",
        time.time() - total_start,
    )
    return predictions


def apply_retirement_constraints(predictions, X_test, time_period):
    """Enforce IRS contribution limits on retirement variable predictions.

    Args:
        predictions: DataFrame of QRF predictions for retirement
            contribution variables.
        X_test: DataFrame with at least ``age``,
            ``employment_income``, and ``self_employment_income``.
        time_period: Tax year (int) for IRS limit look-up.

    Returns:
        DataFrame with constrained values (same columns).
    """
    limits = get_retirement_limits(time_period)
    se_limits = get_se_pension_limits(time_period)

    age = X_test["age"].values
    catch_up = age >= 50
    emp_income = X_test["employment_income"].values
    se_income = X_test["self_employment_income"].values

    limit_401k = limits["401k"] + catch_up * limits["401k_catch_up"]
    limit_ira = limits["ira"] + catch_up * limits["ira_catch_up"]
    se_pension_cap = np.minimum(
        se_income * se_limits["se_pension_rate"],
        se_limits["se_pension_dollar_limit"],
    )

    # Explicit mapping: variable -> (cap array, zero_mask or None).
    _CONSTRAINT_MAP = {
        "traditional_401k_contributions": (limit_401k, emp_income == 0),
        "roth_401k_contributions": (limit_401k, emp_income == 0),
        "traditional_ira_contributions": (limit_ira, None),
        "roth_ira_contributions": (limit_ira, None),
        "self_employed_pension_contributions": (
            se_pension_cap,
            se_income == 0,
        ),
    }

    result = predictions.clip(lower=0)
    for var in result.columns:
        cap, zero_mask = _CONSTRAINT_MAP.get(var, (None, None))
        if cap is not None:
            result[var] = np.minimum(result[var].values, cap)
        if zero_mask is not None:
            result.loc[zero_mask, var] = 0

    return result


def reconcile_ss_subcomponents(predictions, total_ss):
    """Normalize Social Security sub-components to sum to total.

    Args:
        predictions: DataFrame with columns for each SS
            sub-component (retirement, disability, dependents,
            survivors).
        total_ss: numpy array of total social_security per record.

    Returns:
        DataFrame with reconciled dollar values.
    """
    values = np.maximum(predictions.values, 0)
    row_sums = values.sum(axis=1)
    positive_mask = total_ss > 0

    shares = np.zeros_like(values)
    nonzero_rows = row_sums > 0
    both = positive_mask & nonzero_rows
    shares[both] = values[both] / row_sums[both, np.newaxis]
    # If row_sum == 0 but total_ss > 0, distribute equally.
    equal_rows = positive_mask & ~nonzero_rows
    shares[equal_rows] = 1.0 / values.shape[1]

    out = np.where(
        positive_mask[:, np.newaxis],
        shares * total_ss[:, np.newaxis],
        0.0,
    )
    return pd.DataFrame(out, columns=predictions.columns)


_RETIREMENT_VARS = {
    "traditional_401k_contributions",
    "roth_401k_contributions",
    "traditional_ira_contributions",
    "roth_ira_contributions",
    "self_employed_pension_contributions",
}

_SS_SUBCOMPONENT_VARS = {
    "social_security_retirement",
    "social_security_disability",
    "social_security_dependents",
    "social_security_survivors",
}


def _apply_post_processing(predictions, X_test, time_period, data):
    """Apply retirement constraints and SS reconciliation."""
    ret_cols = [c for c in predictions.columns if c in _RETIREMENT_VARS]
    if ret_cols:
        constrained = apply_retirement_constraints(
            predictions[ret_cols], X_test, time_period
        )
        for col in ret_cols:
            predictions[col] = constrained[col]

    ss_cols = [c for c in predictions.columns if c in _SS_SUBCOMPONENT_VARS]
    if ss_cols:
        n_half = len(data["person_id"][time_period]) // 2
        total_ss = data["social_security"][time_period][n_half:]
        reconciled = reconcile_ss_subcomponents(predictions[ss_cols], total_ss)
        for col in ss_cols:
            predictions[col] = reconciled[col]

    return predictions


def _splice_cps_only_predictions(
    data: dict,
    predictions: pd.DataFrame,
    time_period: int,
    dataset_path: str,
) -> dict:
    """Replace PUF clone half of CPS-only variables with QRF predictions.

    After ``puf_clone_dataset()`` the CPS-only variables in the second
    half are naive copies of the CPS donor values. This function
    replaces them with the second-stage QRF predictions that are
    consistent with the clone's PUF-imputed income.

    Args:
        data: Extended dataset dict (already doubled).
        predictions: DataFrame from ``_impute_cps_only_variables()``.
        time_period: Tax year.
        dataset_path: Path to CPS h5 file for entity mapping.

    Returns:
        Modified data dict with CPS-only variables spliced in.
    """
    from policyengine_us import Microsimulation

    cps_sim = Microsimulation(dataset=dataset_path)
    tbs = cps_sim.tax_benefit_system

    # Pre-compute half-lengths per entity so we split each
    # variable's array at the correct midpoint.
    entity_half_lengths = {}
    for entity_key in ["person", "tax_unit", "spm_unit", "family", "household"]:
        id_var = f"{entity_key}_id"
        if id_var in data:
            entity_half_lengths[entity_key] = len(data[id_var][time_period]) // 2

    for var in CPS_ONLY_IMPUTED_VARIABLES:
        if var not in data or var not in predictions.columns:
            continue

        pred_values = predictions[var].values
        var_meta = tbs.variables.get(var)
        entity_key = var_meta.entity.key if var_meta is not None else "person"

        if entity_key != "person":
            pred_values = cps_sim.populations[entity_key].value_from_first_person(
                pred_values
            )

        n_half = entity_half_lengths.get(entity_key, len(data[var][time_period]) // 2)
        values = data[var][time_period]
        # First half: keep original CPS values.
        # Second half: replace with QRF predictions.
        cps_half = values[:n_half]
        new_values = np.concatenate([cps_half, pred_values])
        data[var] = {time_period: new_values}

    del cps_sim
    return data


class ExtendedCPS(Dataset):
    cps: Type[CPS]
    puf: Type[PUF]
    data_format = Dataset.TIME_PERIOD_ARRAYS

    def generate(self):
        from policyengine_us import Microsimulation

        from policyengine_us_data.calibration.clone_and_assign import (
            load_global_block_distribution,
        )
        from policyengine_us_data.calibration.puf_impute import (
            puf_clone_dataset,
        )

        logger.info("Loading CPS dataset: %s", self.cps)
        cps_sim = Microsimulation(dataset=self.cps)
        data = cps_sim.dataset.load_dataset()
        del cps_sim

        data_dict = {}
        for var in data:
            data_dict[var] = {self.time_period: data[var][...]}

        n_hh = len(data_dict["household_id"][self.time_period])
        _, _, block_states, block_probs = load_global_block_distribution()
        rng = np.random.default_rng(seed=42)
        indices = rng.choice(len(block_states), size=n_hh, p=block_probs)
        state_fips = block_states[indices]

        logger.info("PUF clone with dataset: %s", self.puf)
        new_data = puf_clone_dataset(
            data=data_dict,
            state_fips=state_fips,
            time_period=self.time_period,
            puf_dataset=self.puf,
            dataset_path=str(self.cps.file_path),
        )

        # Stage 2: QRF-impute CPS-only variables for PUF clones.
        # Train on CPS data using demographics + PUF-imputed income
        # as predictors, so the PUF clone half gets values consistent
        # with its imputed income rather than naive donor duplication.
        logger.info("Stage-2: imputing CPS-only variables for PUF clones")
        cps_only_predictions = _impute_cps_only_variables(
            data=new_data,
            time_period=self.time_period,
            dataset_path=str(self.cps.file_path),
        )
        new_data = _splice_cps_only_predictions(
            data=new_data,
            predictions=cps_only_predictions,
            time_period=self.time_period,
            dataset_path=str(self.cps.file_path),
        )

        new_data = self._rename_imputed_to_inputs(new_data)
        new_data = self._drop_formula_variables(new_data)
        new_data = self._inject_high_income_puf_records(new_data)
        self.save_dataset(new_data)

    def _inject_high_income_puf_records(self, new_data):
        """Inject ultra-high-income PUF records into the extended CPS.

        The CPS severely under-represents the top of the income
        distribution ($5M+ AGI brackets have -95% to -99% calibration
        errors). This directly appends PUF records with AGI above
        AGI_INJECTION_THRESHOLD as additional households, giving the
        reweighter actual high-income observations.

        Uses raw PUF arrays for speed — only creates a Microsimulation
        briefly to identify high-AGI households, then reads stored
        arrays directly instead of calling calculate() per variable.
        """
        from policyengine_us import Microsimulation

        logger.info("Loading PUF for high-income record injection...")
        puf_sim = Microsimulation(dataset=self.puf)

        # Identify high-AGI households (only calculation needed)
        hh_agi = puf_sim.calculate("adjusted_gross_income", map_to="household").values
        tbs = puf_sim.tax_benefit_system
        del puf_sim  # Free simulation memory immediately

        # Load raw PUF arrays (fast, no simulation)
        puf_data = self.puf(require=True).load_dataset()
        all_hh_ids = puf_data["household_id"]
        high_mask_hh = hh_agi > AGI_INJECTION_THRESHOLD
        high_hh_id_set = set(all_hh_ids[high_mask_hh])

        n_high_hh = int(high_mask_hh.sum())
        if n_high_hh == 0:
            logger.info("No high-income PUF households found")
            return new_data

        logger.info(
            "Injecting %d PUF households with AGI > $%d",
            n_high_hh,
            AGI_INJECTION_THRESHOLD,
        )

        # Build entity-level masks from raw arrays
        high_hh_id_arr = np.fromiter(high_hh_id_set, dtype=float)
        person_mask = np.isin(
            puf_data["person_household_id"],
            high_hh_id_arr,
        )
        # In PUF, household = tax_unit = spm_unit = family
        group_mask = np.isin(
            puf_data["household_id"],
            high_hh_id_arr,
        )
        # Marital units: find which belong to high-income persons
        high_marital_ids = np.unique(puf_data["person_marital_unit_id"][person_mask])
        marital_mask = np.isin(
            puf_data["marital_unit_id"],
            high_marital_ids,
        )

        entity_masks = {
            "person": person_mask,
            "household": group_mask,
            "tax_unit": group_mask,
            "spm_unit": group_mask,
            "family": group_mask,
            "marital_unit": marital_mask,
        }

        logger.info(
            "High-income record counts: persons=%d, households=%d, marital_units=%d",
            person_mask.sum(),
            group_mask.sum(),
            marital_mask.sum(),
        )

        # Compute ID offset to avoid collisions with existing data
        id_offset = 0
        for key in new_data:
            if key.endswith("_id"):
                vals = new_data[key][self.time_period]
                if vals.dtype.kind in ("f", "i", "u"):
                    id_offset = max(id_offset, int(vals.max()))
        id_offset += 1_000_000

        # For each variable in new_data, read from raw PUF arrays
        # (no calculate() calls — uses stored values directly)
        n_appended = 0
        for variable in list(new_data.keys()):
            var_meta = tbs.variables.get(variable)
            if var_meta is None:
                continue

            entity = var_meta.entity.key
            if entity not in entity_masks:
                continue

            mask = entity_masks[entity]

            # Read from raw PUF arrays if available; pad with
            # defaults for variables not in PUF (CPS-only variables)
            existing = new_data[variable][self.time_period]
            n_inject = int(mask.sum())

            # For string/enum vars, use the most common existing value
            # as default (empty string is not a valid enum value)
            if existing.dtype.kind in ("S", "U"):
                default_val = np.full(n_inject, existing[0], dtype=existing.dtype)
            else:
                default_val = np.zeros(n_inject, dtype=existing.dtype)

            use_puf = (
                variable in puf_data
                and len(np.asarray(puf_data[variable])) == len(mask)
                and np.asarray(puf_data[variable]).dtype.kind == existing.dtype.kind
            )

            if use_puf:
                puf_values = np.asarray(puf_data[variable])
                puf_subset = puf_values[mask]
            else:
                puf_subset = default_val

            # Offset IDs to avoid collisions
            if variable.endswith("_id") and puf_subset.dtype.kind in (
                "f",
                "i",
                "u",
            ):
                puf_subset = puf_subset + id_offset

            # Ensure dtype matches existing array
            if puf_subset.dtype != existing.dtype:
                try:
                    puf_subset = puf_subset.astype(existing.dtype)
                except (ValueError, TypeError):
                    puf_subset = default_val

            new_data[variable][self.time_period] = np.concatenate(
                [existing, puf_subset]
            )
            n_appended += 1

        logger.info("Appended PUF values for %d variables", n_appended)
        return new_data

    @classmethod
    def _rename_imputed_to_inputs(cls, data):
        """Rename QRF-imputed formula vars to their leaf inputs.

        The QRF imputes formula-level aggregates (e.g.
        taxable_pension_income) but the engine needs leaf inputs
        (e.g. taxable_private_pension_income) so formulas work.
        """
        for formula_var, input_var in cls._IMPUTED_TO_INPUT.items():
            if formula_var in data:
                logger.info(
                    "Renaming %s -> %s (leaf input)",
                    formula_var,
                    input_var,
                )
                data[input_var] = data.pop(formula_var)
        return data

    # Variables with formulas/adds that must still be stored.
    # Includes IDs needed before formulas run and tax-unit-level
    # QRF-imputed vars that can't be renamed to person-level leaves
    # due to entity shape mismatch.
    _KEEP_FORMULA_VARS = {
        "person_id",
        "interest_deduction",
        "self_employed_pension_contribution_ald",
        "self_employed_health_insurance_ald",
    }

    # QRF imputes formula-level variables (e.g. taxable_pension_income)
    # but we must store them under leaf input names so
    # _drop_formula_variables doesn't discard them. The engine then
    # recomputes the formula var from its adds.
    # NOTE: only same-entity renames here; cross-entity vars
    # (tax_unit -> person) go in _KEEP_FORMULA_VARS instead.
    _IMPUTED_TO_INPUT = {
        "taxable_pension_income": "taxable_private_pension_income",
        "tax_exempt_pension_income": "tax_exempt_private_pension_income",
    }

    @classmethod
    def _drop_formula_variables(cls, data):
        """Remove variables that are computed by policyengine-us.

        Variables with formulas, ``adds``, or ``subtracts`` are
        recomputed by the simulation engine, so storing them wastes
        space and can mislead validation.

        Aggregate variables whose ``adds`` include a behavioral-
        response input (e.g. ``employment_income_before_lsr``) are
        renamed to that input before dropping so the raw data is
        preserved under the correct input-variable name.
        """
        from policyengine_us import CountryTaxBenefitSystem

        tbs = CountryTaxBenefitSystem()

        _RESPONSE_SUFFIXES = ("_before_lsr", "_before_response")
        for name, var in tbs.variables.items():
            if name not in data:
                continue
            for add_var in getattr(var, "adds", None) or []:
                if any(add_var.endswith(s) for s in _RESPONSE_SUFFIXES):
                    if add_var not in data:
                        logger.info(
                            "Renaming %s -> %s before drop",
                            name,
                            add_var,
                        )
                        data[add_var] = data.pop(name)
                    break

        formula_vars = {
            name
            for name, var in tbs.variables.items()
            if (hasattr(var, "formulas") and len(var.formulas) > 0)
            or getattr(var, "adds", None)
            or getattr(var, "subtracts", None)
        } - cls._KEEP_FORMULA_VARS
        dropped = sorted(set(data.keys()) & formula_vars)
        if dropped:
            logger.info(
                "Dropping %d formula variables: %s",
                len(dropped),
                dropped,
            )
            for var in dropped:
                del data[var]
        return data


class ExtendedCPS_2024(ExtendedCPS):
    cps = CPS_2024_Full
    puf = PUF_2024
    name = "extended_cps_2024"
    label = "Extended CPS (2024)"
    file_path = STORAGE_FOLDER / "extended_cps_2024.h5"
    time_period = 2024


class ExtendedCPS_2024_Half(ExtendedCPS):
    cps = CPS_2024
    puf = PUF_2024
    name = "extended_cps_2024_half"
    label = "Extended CPS 2024 (half sample)"
    file_path = STORAGE_FOLDER / "extended_cps_2024_half.h5"
    time_period = 2024


if __name__ == "__main__":
    ExtendedCPS_2024().generate()
    ExtendedCPS_2024_Half().generate()
