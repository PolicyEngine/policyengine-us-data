from policyengine_core.data import Dataset
from policyengine_us_data.data_storage import STORAGE_FOLDER
from typing import Type
from .cps import *
from ..puf import *
import pandas as pd

IMPUTED_VARIABLES = [
    "alimony_expense",
    "alimony_income",
    "american_opportunity_credit",
    "amt_foreign_tax_credit",
    "casualty_loss",
    "cdcc_relevant_expenses",
    "charitable_cash_donations",
    "charitable_non_cash_donations",
    "domestic_production_ald",
    "early_withdrawal_penalty",
    "educator_expense",
    "employment_income",
    "energy_efficient_home_improvement_credit",
    "estate_income",
    "excess_withheld_payroll_tax",
    "farm_income",
    "farm_rent_income",
    "foreign_tax_credit",
    "general_business_credit",
    "health_savings_account_ald",
    "interest_deduction",
    "investment_income_elected_form_4952",
    "long_term_capital_gains",
    "long_term_capital_gains_on_collectibles",
    "medical_expense",
    "misc_deduction",
    "miscellaneous_income",
    "non_qualified_dividend_income",
    "non_sch_d_capital_gains",
    "other_credits",
    "partnership_s_corp_income",
    "pre_tax_contributions",
    "prior_year_minimum_tax_credit",
    "qualified_dividend_income",
    "qualified_tuition_expenses",
    "real_estate_taxes",
    "recapture_of_investment_credit",
    "rental_income",
    "salt_refund_income",
    "savers_credit",
    "self_employed_health_insurance_ald",
    "self_employed_pension_contribution_ald",
    "self_employment_income",
    "short_term_capital_gains",
    "social_security",
    "student_loan_interest",
    "tax_exempt_interest_income",
    "tax_exempt_pension_income",
    "taxable_interest_income",
    "taxable_ira_distributions",
    "taxable_pension_income",
    "taxable_unemployment_compensation",
    "traditional_ira_contributions",
    "unrecaptured_section_1250_gain",
    "unreported_payroll_tax",
    "w2_wages_from_qualified_business",
]


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

        model = Imputation()

        model.train(X_train, y_train, verbose=True)

        X = cps_sim.calculate_dataframe(INPUTS)
        y = model.predict(X, verbose=True)

        data = cps_sim.dataset.load_dataset()

        new_data = {}

        for variable in list(data) + IMPUTED_VARIABLES:
            variable_metadata = cps_sim.tax_benefit_system.variables.get(variable)
            if variable in data:
                values = data[variable][...]
            else:
                values = cps_sim.calculate(variable).values
            if variable in IMPUTED_VARIABLES:
                pred_values = y[variable].values
                entity = variable_metadata.entity.key
                if entity != "person":
                    pred_values = cps_sim.populations[entity].value_from_first_person(pred_values)
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
