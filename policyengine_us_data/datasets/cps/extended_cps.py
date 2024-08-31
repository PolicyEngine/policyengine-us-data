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
    "state_and_local_sales_or_income_tax",
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
    data_format = Dataset.ARRAYS

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

        model.train(X_train, y_train, verbose=True, num_trees=10)

        X = cps_sim.calculate_dataframe(INPUTS)
        y = model.predict(X, verbose=True)

        original_dataset = cps_sim.to_input_dataframe()
        renames = {
            f"employment_income_before_lsr__{self.time_period}": f"employment_income__{self.time_period}",
            f"self_employment_income_before_lsr__{self.time_period}": f"self_employment_income__{self.time_period}",
        }
        for a, b in renames.items():
            original_dataset[b] = original_dataset[a]
            del original_dataset[a]
        imputed_dataset = original_dataset.copy().reset_index()

        for variable in IMPUTED_VARIABLES:
            imputed_dataset[f"{variable}__{self.time_period}"] = y[variable]

        ENTITIES = ("person", "tax_unit", "family", "spm_unit", "household")
        for entity in ENTITIES:
            for id_name in [
                f"{entity}_id__{self.time_period}",
                f"person_{entity}_id__{self.time_period}",
            ]:
                if "person_person" in id_name:
                    continue
                original_ids = original_dataset[id_name].values
                new_ids = original_ids + original_ids.max()
                imputed_dataset[id_name] = new_ids

        for variable in imputed_dataset.columns:
            if "_weight" in variable:
                imputed_dataset[variable] = 0
        original_dataset["data_source"] = "cps"
        imputed_dataset["data_source"] = "puf_imputed"
        combined = pd.concat([original_dataset, imputed_dataset]).fillna(0)
        # Sort columns in alphabetical order
        combined = combined.reindex(sorted(combined.columns), axis=1)

        data = {}

        for column in combined:
            variable_name, time_period = column.split("__")
            data[variable_name] = combined[column].values

        self.save_dataset(data)


class ExtendedCPS_2024(ExtendedCPS):
    cps = CPS_2024
    puf = PUF_2024
    name = "extended_cps_2024"
    label = "Extended CPS (2024)"
    file_path = STORAGE_FOLDER / "extended_cps_2024.h5"
    time_period = 2024