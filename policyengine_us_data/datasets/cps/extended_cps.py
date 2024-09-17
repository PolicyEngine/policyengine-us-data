from policyengine_core.data import Dataset
from policyengine_us_data.storage import STORAGE_FOLDER
from typing import Type
from .cps import *
from ..puf import *
import pandas as pd
import os

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
    "misc_deduction",
    "pre_tax_contributions",
    "taxable_ira_distributions",
    "self_employment_income",
    "w2_wages_from_qualified_business",
    "short_term_capital_gains",
    "qualified_dividend_income",
    "charitable_cash_donations",
    "self_employed_pension_contribution_ald",
    "real_estate_taxes",
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
]

if os.environ.get("TEST_LITE"):
    IMPUTED_VARIABLES = IMPUTED_VARIABLES[:7]


class ExtendedCPS(Dataset):
    cps: Type[CPS]
    puf: Type[PUF]
    data_format = Dataset.TIME_PERIOD_ARRAYS

    def generate(self):
        from policyengine_us import Microsimulation
        from survey_enhance import Imputation

        cps_sim = Microsimulation(dataset=self.cps)
        puf_sim = Microsimulation(dataset=self.puf)

        INPUTS = [
            "age",
            "is_male",
            "tax_unit_is_joint",
            "tax_unit_count_dependents",
            "is_tax_unit_head",
            "is_tax_unit_spouse",
            "is_tax_unit_dependent",
        ]

        X_train = puf_sim.calculate_dataframe(INPUTS)
        y_train = puf_sim.calculate_dataframe(IMPUTED_VARIABLES)
        X = cps_sim.calculate_dataframe(INPUTS)
        y = pd.DataFrame(columns=IMPUTED_VARIABLES, index=X.index)

        model = Imputation()
        model.train(
            X_train,
            y_train,
            verbose=True,
            sample_weight=puf_sim.calculate(
                "household_weight", map_to="person"
            ).values,
        )
        y = model.predict(X, verbose=True)

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
            if variable in IMPUTED_VARIABLES:
                pred_values = y[variable].values
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


class ExtendedCPS_2024(ExtendedCPS):
    cps = CPS_2024
    puf = PUF_2024
    name = "extended_cps_2024"
    label = "Extended CPS (2024)"
    file_path = STORAGE_FOLDER / "extended_cps_2024.h5"
    time_period = 2024
