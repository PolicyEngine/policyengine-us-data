from policyengine_core.data import Dataset
from policyengine_us_data.storage import STORAGE_FOLDER
from typing import Type
from policyengine_us_data.datasets.cps.cps import *
from policyengine_us_data.datasets.puf import *
import pandas as pd
import os
from microimpute.models.qrf import QRF
import time
import logging
import gc

# These are sorted by magnitude.
# First 15 contain 90%.
# First 7 contain 75%.
# If you're trying to debug this part of the code and don't want to wait ages
# to see if something breaks, try limiting to those.
IMPUTED_VARIABLES = [
    "employment_income",
    "partnership_s_corp_income",
    "social_security",
    "taxable_pension_income",
    "interest_deduction",
    "tax_exempt_pension_income",
    "long_term_capital_gains",
    "unreimbursed_business_employee_expenses",
    "pre_tax_contributions",
    "taxable_ira_distributions",
    "self_employment_income",
    "w2_wages_from_qualified_business",
    "unadjusted_basis_qualified_property",
    "business_is_sstb",  # bool
    "short_term_capital_gains",
    "qualified_dividend_income",
    "charitable_cash_donations",
    "self_employed_pension_contribution_ald",
    "unrecaptured_section_1250_gain",
    "taxable_unemployment_compensation",
    "taxable_interest_income",
    "domestic_production_ald",
    "self_employed_health_insurance_ald",
    "rental_income",
    "non_qualified_dividend_income",
    "cdcc_relevant_expenses",
    "tax_exempt_interest_income",
    "salt_refund_income",
    "foreign_tax_credit",
    "estate_income",
    "charitable_non_cash_donations",
    "american_opportunity_credit",
    "miscellaneous_income",
    "alimony_expense",
    "farm_income",
    "alimony_income",
    "health_savings_account_ald",
    "non_sch_d_capital_gains",
    "general_business_credit",
    "energy_efficient_home_improvement_credit",
    "traditional_ira_contributions",
    "amt_foreign_tax_credit",
    "excess_withheld_payroll_tax",
    "savers_credit",
    "student_loan_interest",
    "investment_income_elected_form_4952",
    "early_withdrawal_penalty",
    "prior_year_minimum_tax_credit",
    "farm_rent_income",
    "qualified_tuition_expenses",
    "educator_expense",
    "long_term_capital_gains_on_collectibles",
    "other_credits",
    "casualty_loss",
    "unreported_payroll_tax",
    "recapture_of_investment_credit",
    "deductible_mortgage_interest",
    "qualified_reit_and_ptp_income",
    "qualified_bdc_income",
    "farm_operations_income",
    "estate_income_would_be_qualified",
    "farm_operations_income_would_be_qualified",
    "farm_rent_income_would_be_qualified",
    "partnership_s_corp_income_would_be_qualified",
    "rental_income_would_be_qualified",
    "self_employment_income_would_be_qualified",
]

OVERRIDDEN_IMPUTED_VARIABLES = [
    "partnership_s_corp_income",
    "interest_deduction",
    "unreimbursed_business_employee_expenses",
    "pre_tax_contributions",
    "w2_wages_from_qualified_business",
    "unadjusted_basis_qualified_property",
    "business_is_sstb",
    "charitable_cash_donations",
    "self_employed_pension_contribution_ald",
    "unrecaptured_section_1250_gain",
    "taxable_unemployment_compensation",
    "domestic_production_ald",
    "self_employed_health_insurance_ald",
    "cdcc_relevant_expenses",
    "salt_refund_income",
    "foreign_tax_credit",
    "estate_income",
    "charitable_non_cash_donations",
    "american_opportunity_credit",
    "miscellaneous_income",
    "alimony_expense",
    "health_savings_account_ald",
    "non_sch_d_capital_gains",
    "general_business_credit",
    "energy_efficient_home_improvement_credit",
    "amt_foreign_tax_credit",
    "excess_withheld_payroll_tax",
    "savers_credit",
    "student_loan_interest",
    "investment_income_elected_form_4952",
    "early_withdrawal_penalty",
    "prior_year_minimum_tax_credit",
    "farm_rent_income",
    "qualified_tuition_expenses",
    "educator_expense",
    "long_term_capital_gains_on_collectibles",
    "other_credits",
    "casualty_loss",
    "unreported_payroll_tax",
    "recapture_of_investment_credit",
    "deductible_mortgage_interest",
    "qualified_reit_and_ptp_income",
    "qualified_bdc_income",
    "farm_operations_income",
    "estate_income_would_be_qualified",
    "farm_operations_income_would_be_qualified",
    "farm_rent_income_would_be_qualified",
    "partnership_s_corp_income_would_be_qualified",
    "rental_income_would_be_qualified",
]


class ExtendedCPS(Dataset):
    cps: Type[CPS]
    puf: Type[PUF]
    data_format = Dataset.TIME_PERIOD_ARRAYS

    def generate(self):
        from policyengine_us import Microsimulation

        cps_sim = Microsimulation(dataset=self.cps)
        puf_sim = Microsimulation(dataset=self.puf)

        puf_sim.subsample(10_000)

        INPUTS = [
            "age",
            "is_male",
            "tax_unit_is_joint",
            "tax_unit_count_dependents",
            "is_tax_unit_head",
            "is_tax_unit_spouse",
            "is_tax_unit_dependent",
        ]

        y_full_imputations = impute_income_variables(
            cps_sim,
            puf_sim,
            predictors=INPUTS,
            outputs=IMPUTED_VARIABLES,
        )
        y_cps_imputations = impute_income_variables(
            cps_sim,
            puf_sim,
            predictors=INPUTS,
            outputs=OVERRIDDEN_IMPUTED_VARIABLES,
        )
        cps_sim = Microsimulation(dataset=self.cps)
        data = cps_sim.dataset.load_dataset()
        new_data = {}

        for variable in list(data) + IMPUTED_VARIABLES:
            variable_metadata = cps_sim.tax_benefit_system.variables.get(
                variable
            )
            if variable in data:
                values = data[variable][...]
            else:
                values = cps_sim.calculate(variable).values
            if variable in OVERRIDDEN_IMPUTED_VARIABLES:
                pred_values = y_cps_imputations[variable].values
                entity = variable_metadata.entity.key
                if entity != "person":
                    pred_values = cps_sim.populations[
                        entity
                    ].value_from_first_person(pred_values)
                values = np.concatenate([pred_values, pred_values])
            elif variable in IMPUTED_VARIABLES:
                pred_values = y_full_imputations[variable].values
                entity = variable_metadata.entity.key
                if entity != "person":
                    pred_values = cps_sim.populations[
                        entity
                    ].value_from_first_person(pred_values)
                values = np.concatenate([values, pred_values])
            elif variable == "person_id":
                values = np.concatenate([values, values + values.max()])
            elif "_id" in variable:
                values = np.concatenate([values, values + values.max()])
            elif "_weight" in variable:
                values = np.concatenate([values, values * 0])
            else:
                values = np.concatenate([values, values])
            new_data[variable] = {
                self.time_period: values,
            }

        self.save_dataset(new_data)


def impute_income_variables(
    cps_sim,
    puf_sim,
    predictors: list[str] = None,
    outputs: list[str] = None,
):

    # Calculate all variables together to preserve dependencies
    X_train = puf_sim.calculate_dataframe(predictors + outputs)

    # Check which outputs are actually in the result
    available_outputs = [col for col in outputs if col in X_train.columns]
    missing_outputs = [col for col in outputs if col not in X_train.columns]

    if missing_outputs:
        logging.warning(
            f"The following {len(missing_outputs)} variables were not calculated: {missing_outputs}"
        )
        # Log the specific missing variable that's causing issues
        if "recapture_of_investment_credit" in missing_outputs:
            logging.error(
                "recapture_of_investment_credit is missing from PUF calculation!"
            )

    logging.info(
        f"X_train shape: {X_train.shape}, columns: {len(X_train.columns)}"
    )

    X_test = cps_sim.calculate_dataframe(predictors)

    logging.info(
        f"Imputing {len(available_outputs)} variables using batched sequential QRF"
    )
    total_start = time.time()

    # Batch variables to avoid memory issues with sequential imputation
    batch_size = 10  # Reduce to 10 variables at a time
    result = pd.DataFrame(index=X_test.index)

    # Sample training data more aggressively upfront
    sample_size = min(5000, len(X_train))  # Reduced from 5000
    if len(X_train) > sample_size:
        logging.info(
            f"Sampling training data from {len(X_train)} to {sample_size} rows"
        )
        X_train_sampled = X_train.sample(n=sample_size, random_state=42)
    else:
        X_train_sampled = X_train

    for batch_start in range(0, len(available_outputs), batch_size):
        batch_end = min(batch_start + batch_size, len(available_outputs))
        batch_vars = available_outputs[batch_start:batch_end]

        logging.info(
            f"Processing batch {batch_start//batch_size + 1}: variables {batch_start+1}-{batch_end} ({batch_vars})"
        )

        # Force garbage collection before each batch
        gc.collect()

        # Create a fresh QRF for each batch
        qrf = QRF(
            log_level="INFO",
            memory_efficient=True,
            batch_size=10,
            cleanup_interval=5,
        )

        # Use pre-sampled data for this batch
        batch_X_train = X_train_sampled[predictors + batch_vars].copy()

        # Fit model for this batch with sequential imputation within the batch
        fitted_model = qrf.fit(
            X_train=batch_X_train,
            predictors=predictors,
            imputed_variables=batch_vars,
            n_jobs=1,  # Single thread to reduce memory overhead
        )

        # Predict for this batch
        batch_predictions = fitted_model.predict(X_test=X_test)

        # Extract median predictions and add to result
        for var in batch_vars:
            result[var] = batch_predictions[var]

        # Clean up batch objects
        del fitted_model
        del batch_predictions
        del batch_X_train
        gc.collect()

        logging.info(f"Completed batch {batch_start//batch_size + 1}")

    # Add zeros for missing variables
    for var in missing_outputs:
        result[var] = 0

    logging.info(
        f"Imputing {len(available_outputs)} variables took {time.time() - total_start:.2f} seconds total"
    )

    return result


class ExtendedCPS_2023(ExtendedCPS):
    cps = CPS_2023_Full
    puf = PUF_2023
    name = "extended_cps_2023"
    label = "Extended CPS (2023)"
    file_path = STORAGE_FOLDER / "extended_cps_2023.h5"
    time_period = 2023


class ExtendedCPS_2024(ExtendedCPS):
    cps = CPS_2024
    puf = PUF_2024
    name = "extended_cps_2024"
    label = "Extended CPS (2024)"
    file_path = STORAGE_FOLDER / "extended_cps_2024.h5"
    time_period = 2024


if __name__ == "__main__":
    geo_stacking_mode = os.environ.get("GEO_STACKING_MODE", "").lower() == "true"
    
    if geo_stacking_mode:
        print("Running in GEO_STACKING_MODE")
        print("Generating ExtendedCPS_2023 for geo-stacking pipeline...")
        ExtendedCPS_2023().generate()
        print("Also generating ExtendedCPS_2024 to satisfy downstream dependencies...")
    
    ExtendedCPS_2024().generate()
