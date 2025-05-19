from tqdm import tqdm
import numpy as np
import pandas as pd
from microdf import MicroDataFrame
from policyengine_core.data import Dataset
from policyengine_us_data.storage import STORAGE_FOLDER
from policyengine_us_data.datasets.puf.uprate_puf import uprate_puf
from policyengine_us_data.datasets.puf.irs_puf import IRS_PUF_2015
from policyengine_us_data.utils.uprating import (
    create_policyengine_uprating_factors_table,
)

rng = np.random.default_rng(seed=64)


def impute_pension_contributions_to_puf(puf_df):
    from policyengine_us import Microsimulation
    from policyengine_us_data.datasets.cps import CPS_2021

    cps = Microsimulation(dataset=CPS_2021)
    cps_df = cps.calculate_dataframe(
        ["employment_income", "household_weight", "pre_tax_contributions"]
    )

    from policyengine_us_data.utils import QRF

    pension_contributions = QRF()

    pension_contributions.fit(
        cps_df[["employment_income"]],
        cps_df[["pre_tax_contributions"]],
    )
    return pension_contributions.predict(
        X=puf_df[["employment_income"]],
    )


def impute_missing_demographics(
    puf: pd.DataFrame, demographics: pd.DataFrame
) -> pd.DataFrame:
    from policyengine_us_data.utils import QRF

    puf_with_demographics = (
        puf[puf.RECID.isin(demographics.RECID)]
        .merge(demographics, on="RECID")
        .fillna(0)
    )

    DEMOGRAPHIC_VARIABLES = [
        "AGEDP1",
        "AGEDP2",
        "AGEDP3",
        "AGERANGE",
        "EARNSPLIT",
        "GENDER",
    ]
    NON_DEMOGRAPHIC_VARIABLES = [
        "E00200",
        "MARS",
        "DSI",
        "EIC",
        "XTOT",
    ]

    demographics_from_puf = QRF()

    demographics_from_puf.fit(
        puf_with_demographics[NON_DEMOGRAPHIC_VARIABLES],
        puf_with_demographics[DEMOGRAPHIC_VARIABLES],
    )

    puf_without_demographics = puf[
        ~puf.RECID.isin(puf_with_demographics.RECID)
    ].reset_index()
    predicted_demographics = demographics_from_puf.predict(
        X=puf_without_demographics,
    )
    puf_with_imputed_demographics = pd.concat(
        [puf_without_demographics, predicted_demographics], axis=1
    )

    weighted_puf_with_demographics = MicroDataFrame(
        puf_with_demographics, weights="S006"
    )
    weighted_puf_with_imputed_demographics = MicroDataFrame(
        puf_with_imputed_demographics, weights="S006"
    )

    puf_combined = pd.concat(
        [
            weighted_puf_with_demographics,
            weighted_puf_with_imputed_demographics,
        ]
    )

    return puf_combined


def decode_age_filer(age_range: int) -> int:
    if age_range == 0:
        return 40
    AGERANGE_FILER_DECODE = {
        1: 18,
        2: 26,
        3: 35,
        4: 45,
        5: 55,
        6: 65,
        7: 80,
    }
    lower = AGERANGE_FILER_DECODE[age_range]
    upper = AGERANGE_FILER_DECODE[age_range + 1]
    return rng.integers(low=lower, high=upper, endpoint=False)


def decode_age_dependent(age_range: int) -> int:
    if age_range == 0:
        return 0
    AGERANGE_DEPENDENT_DECODE = {
        0: 0,
        1: 0,
        2: 5,
        3: 13,
        4: 17,
        5: 19,
        6: 25,
        7: 30,
    }
    lower = AGERANGE_DEPENDENT_DECODE[age_range]
    upper = AGERANGE_DEPENDENT_DECODE[age_range + 1]
    return rng.integers(low=lower, high=upper, endpoint=False)


def preprocess_puf(puf: pd.DataFrame) -> pd.DataFrame:
    # Add variable renames
    puf.S006 = puf.S006 / 100
    # puf["adjusted_gross_income"] = puf.E00100
    puf["alimony_expense"] = puf.E03500
    puf["alimony_income"] = puf.E00800
    puf["casualty_loss"] = puf.E20500
    puf["cdcc_relevant_expenses"] = puf.E32800
    puf["charitable_cash_donations"] = puf.E19800
    puf["charitable_non_cash_donations"] = puf.E20100
    puf["domestic_production_ald"] = puf.E03240
    puf["early_withdrawal_penalty"] = puf.E03400
    puf["educator_expense"] = puf.E03220
    puf["employment_income"] = puf.E00200
    puf["estate_income"] = puf.E26390 - puf.E26400
    puf["farm_income"] = puf.T27800
    puf["farm_rent_income"] = puf.E27200
    puf["health_savings_account_ald"] = puf.E03290
    puf["interest_deduction"] = puf.E19200
    puf["long_term_capital_gains"] = puf.P23250
    puf["long_term_capital_gains_on_collectibles"] = puf.E24518
    # Split medical expenses using CPS fractions
    for (
        medical_category,
        fraction,
    ) in MEDICAL_EXPENSE_CATEGORY_BREAKDOWNS.items():
        puf[medical_category] = puf.E17500 * fraction
    # Use unreimbursed business employee expenses as a proxy for all miscellaneous expenses
    # that can be deducted under the miscellaneous deduction.
    puf["unreimbursed_business_employee_expenses"] = puf.E20400
    puf["non_qualified_dividend_income"] = puf.E00600 - puf.E00650
    puf["partnership_s_corp_income"] = puf.E26270
    puf["qualified_dividend_income"] = puf.E00650
    puf["qualified_tuition_expenses"] = puf.E03230
    puf["real_estate_taxes"] = puf.E18500
    puf["rental_income"] = puf.E25850 - puf.E25860
    puf["self_employment_income"] = puf.E00900
    puf["self_employed_health_insurance_ald"] = puf.E03270
    puf["self_employed_pension_contribution_ald"] = puf.E03300
    puf["short_term_capital_gains"] = puf.P22250
    puf["social_security"] = puf.E02400
    puf["state_and_local_sales_or_income_tax"] = puf.E18400
    puf["student_loan_interest"] = puf.E03210
    puf["taxable_interest_income"] = puf.E00300
    puf["taxable_pension_income"] = puf.E01700
    puf["taxable_unemployment_compensation"] = puf.E02300
    puf["taxable_ira_distributions"] = puf.E01400
    puf["tax_exempt_interest_income"] = puf.E00400
    puf["tax_exempt_pension_income"] = puf.E01500 - puf.E01700
    puf["traditional_ira_contributions"] = puf.E03150
    puf["unrecaptured_section_1250_gain"] = puf.E24515

    puf["foreign_tax_credit"] = puf.E07300
    puf["amt_foreign_tax_credit"] = puf.E62900
    puf["miscellaneous_income"] = puf.E01200
    puf["salt_refund_income"] = puf.E00700
    puf["investment_income_elected_form_4952"] = puf.E58990
    puf["general_business_credit"] = puf.E07400
    puf["prior_year_minimum_tax_credit"] = puf.E07600
    puf["excess_withheld_payroll_tax"] = puf.E11200
    puf["non_sch_d_capital_gains"] = puf.E01100
    puf["american_opportunity_credit"] = puf.E87521
    puf["energy_efficient_home_improvement_credit"] = puf.E07260
    puf["early_withdrawal_penalty"] = puf.E09900
    # puf["qualified_tuition_expenses"] = puf.E87530 # PE uses the same variable for qualified tuition (general) and qualified tuition (Lifetime Learning Credit). Revisit here.
    puf["other_credits"] = puf.P08000
    puf["savers_credit"] = puf.E07240
    puf["recapture_of_investment_credit"] = puf.E09700
    puf["unreported_payroll_tax"] = puf.E09800
    # Ignore f2441 (AMT form attached)
    # Ignore cmbtp (estimate of AMT income not in AGI)
    # Ignore k1bx14s and k1bx14p (partner self-employment income included in partnership and S-corp income)
    # --- Qualified Business Income Deduction related variables ---
    # Income sources
    puf["sole_proprietorship_net_income"] = puf.E00900  # Schedule C
    puf["schedule_F_farm_net_income"] = puf.E02100  # Schedule F active farming operations 
    puf["farm_rental_net_income"] = puf.E27200  # Schedule E farm rental income
    puf["rent_royalty_net_income"] = puf.E25850 - puf.E25860  # Schedule E rent and royalty
    puf["estate_trust_net_income"] = puf.E26390 - puf.E26400  # Schedule E estate or trust 
    puf["s_corp_net_income"] = puf.E26190 - puf.E26180  # Schedule E Active S Corp Income
    puf["partnership_net_income"] = puf.E25980 - puf.E25960  # Schedule E Active Partnership Income

    puf["reit_dividends"] = 100 
    puf["bdc_dividends"] = 100
    puf["ptp_income"] = 100  # Publically traded partnership income

    qbi = (
        puf["sole_proprietorship_net_income"]
        + puf["schedule_F_farm_net_income"]
        + puf["farm_rental_net_income"]
        + puf["rent_royalty_net_income"]
        + puf["estate_trust_net_income"]
        + puf["s_corp_net_income"]
        + puf["partnership_net_income"]
    )
    # TODO: We should let it be negative, right?
    qbi_prior = np.maximum(0, puf.E00900 + puf.E26270 + puf.E02100 + puf.E27200)
    # NOTE that the current rule engine variables for income are:
    # - self_employment_income
    # - partnership_s_corp_income
    # - farm_income
    # - farm_rent_income
    # So there's a question as to whether any improvement in naming is worth the hassle.
    # But, for instance, there really is Schedule F farm operations and Schedule E farm rent

    print(f"total QBI in Billion USD, New: {np.dot(qbi, puf.S006) / 1E9:.1f} and Old: {np.dot(qbi_prior, puf.S006) / 1E9:.1f}")


    def simulate_w2_wages_from_qualified_business(qbi, diagnostics=False):
        MIN_MARGIN = .03
        MAX_MARGIN = .15

        MIN_LABOR_RATIO = 0.05
        MAX_LABOR_RATIO = 0.15

        margins = MIN_MARGIN + (MAX_MARGIN - MIN_MARGIN) * np.random.beta(2, 2, size=qbi.shape[0])
        revenues = np.maximum(qbi, 0) / margins
        labor_ratios = (
            MIN_LABOR_RATIO
            + (MAX_LABOR_RATIO - MIN_LABOR_RATIO) * np.random.beta(2, 2, size=revenues.shape[0])
        )
        hypothetical_w2_gross_income = revenues * labor_ratios
        
        pr_has_w2_employees = 1 / (1 + np.exp(-0.5E-5 * (revenues - 4E5)))
        has_w2_employees = np.random.binomial(n=1, p=pr_has_w2_employees)
        w2_wages = hypothetical_w2_gross_income * has_w2_employees

        hypothetical_ubia = (
            np.maximum(0,
            -2.2E4 + 3.0E-1 * revenues
            + 4E4 * np.random.normal(size=qbi.shape[0])
            )
        )

        pr_has_qualified_property = 1 / (1 + np.exp(3 + -(1 / 100000.0) * (w2_wages - 100000.0)))
        pr_has_qualified_property 
        np.median(pr_has_qualified_property[w2_wages>0])

        has_qualified_property = np.random.binomial(n=1, p=pr_has_qualified_property)
        ubia_property = hypothetical_ubia * has_qualified_property
       
        if diagnostics:
            print(f"Proportion of records with positive QBI: {np.mean(qbi >0):.2f}")
            print(f"Within positive QBI, proportion with W2 wages: {np.mean(w2_wages[qbi>0]>0):.2f}")
            print(f"Within positive wages, mean in Millions is {np.mean(w2_wages[w2_wages>0])/1E6:.1f}")
            print(f"Within positive wages, median in Millions is {np.median(w2_wages[w2_wages>0])/1E6:.1f}")
            print(f"Within positive wages, 75th percentile in Millions is {np.percentile(w2_wages[w2_wages>0], 75)/1E6:.1f}")
            print(f"Within positive wages, max in Millions is {np.max(w2_wages[w2_wages>0])/1E6:.1f}")
            print(f"Within positive wages, median ubia property probability is {np.median(pr_has_qualified_property[w2_wages>0]):.3f}")
            print(f"Within positive wages, median ubia property in millions is {np.median(ubia_property[w2_wages>0])/1E6:.1f}")
        return w2_wages, ubia_property

    w2_wages, ubia_property = simulate_w2_wages_from_qualified_business(qbi)
    puf["w2_wages_from_qualified_business"] = w2_wages
    puf["unadjusted_basis_qualified_property"] = ubia_property 

    # Business is SSTB 
    largest_qbi_source = np.argmax(puf[[
        # 0: 20%    1: 0%     2: 15%    3: 0%    4: 0%     5: 0%     6: 10%     7: 10%    0%
        "E00900", "E02100", "E26270", "P25700", "E25850", "E27200", "E26390", "E26400", "E02000"]], axis=1)
    largest_qbi_source = np.where(qbi <= 0, -1, largest_qbi_source) 
   
    pr_sstb = np.where(largest_qbi_source == -1, 0,
          np.where(largest_qbi_source == 0, 0.20,
          np.where(largest_qbi_source == 1, 0.00,
          np.where(largest_qbi_source == 2, 0.15,
          np.where(largest_qbi_source == 3, 0.00,
          np.where(largest_qbi_source == 4, 0.00,
          np.where(largest_qbi_source == 5, 0.00,
          np.where(largest_qbi_source == 6, 0.10,
          np.where(largest_qbi_source == 7, 0.10,
          np.where(largest_qbi_source == 8, 0.00,
                  largest_qbi_source))))))))))

    pr_sstb = np.where(qbi < 1E-3, 0, pr_sstb)
    puf["business_is_sstb"] = np.random.binomial(n=1, p=pr_sstb)
    print(f"{100 * np.mean(puf.loc[qbi > 0]["business_is_sstb"]):.1f}% of pos businesses")

    # -------- End QBID work -------
    puf["filing_status"] = puf.MARS.map(
        {
            1: "SINGLE",
            2: "JOINT",
            3: "SEPARATE",
            4: "HEAD_OF_HOUSEHOLD",
        }
    )
    puf["household_id"] = puf.RECID
    puf["household_weight"] = puf.S006
    puf["exemptions_count"] = puf.XTOT

    return puf


FINANCIAL_SUBSET = [
    # "adjusted_gross_income",
    "alimony_expense",
    "alimony_income",
    "casualty_loss",
    "cdcc_relevant_expenses",
    "charitable_cash_donations",
    "charitable_non_cash_donations",
    "domestic_production_ald",
    "early_withdrawal_penalty",
    "educator_expense",
    "employment_income",
    "estate_income",
    "farm_income",
    "farm_rent_income",
    "health_savings_account_ald",
    "interest_deduction",
    "long_term_capital_gains",
    "long_term_capital_gains_on_collectibles",
    "unreimbursed_business_employee_expenses",
    "non_qualified_dividend_income",
    "non_sch_d_capital_gains",
    "partnership_s_corp_income",
    "qualified_dividend_income",
    "qualified_tuition_expenses",
    "real_estate_taxes",
    "rental_income",
    "self_employment_income",
    "self_employed_health_insurance_ald",
    "self_employed_pension_contribution_ald",
    "short_term_capital_gains",
    "social_security",
    "state_and_local_sales_or_income_tax",
    "student_loan_interest",
    "taxable_interest_income",
    "taxable_pension_income",
    "taxable_unemployment_compensation",
    "taxable_ira_distributions",
    "tax_exempt_interest_income",
    "tax_exempt_pension_income",
    "traditional_ira_contributions",
    "unrecaptured_section_1250_gain",
    "foreign_tax_credit",
    "amt_foreign_tax_credit",
    "miscellaneous_income",
    "salt_refund_income",
    "investment_income_elected_form_4952",
    "general_business_credit",
    "prior_year_minimum_tax_credit",
    "excess_withheld_payroll_tax",
    "american_opportunity_credit",
    "energy_efficient_home_improvement_credit",
    "other_credits",
    "savers_credit",
    "recapture_of_investment_credit",
    "unreported_payroll_tax",
    "pre_tax_contributions",
    "w2_wages_from_qualified_business",
    "unadjusted_basis_qualified_property",
    "business_is_sstb",
    "deductible_mortgage_interest",
]


class PUF(Dataset):
    time_period = None
    data_format = Dataset.ARRAYS

    def generate(self):
        from policyengine_us.system import system

        irs_puf = IRS_PUF_2015(require=True)

        puf = irs_puf.load("puf")
        demographics = irs_puf.load("puf_demographics")

        if self.time_period == 2021:
            puf = uprate_puf(puf, 2015, self.time_period)
        elif self.time_period >= 2021:
            puf_2021 = PUF_2021(require=True)
            uprating = create_policyengine_uprating_factors_table()
            arrays = puf_2021.load_dataset()
            for variable in uprating:
                if variable in arrays:
                    current_index = uprating[uprating.Variable == variable][
                        self.time_period
                    ].values[0]
                    start_index = uprating[uprating.Variable == variable][
                        2021
                    ].values[0]
                    growth = current_index / start_index
                    arrays[variable] = arrays[variable] * growth
            self.save_dataset(arrays)
            return

        puf = puf[puf.MARS != 0]  # Remove aggregate records

        original_recid = puf.RECID.values.copy()
        puf = preprocess_puf(puf)
        puf = impute_missing_demographics(puf, demographics)
        puf["pre_tax_contributions"] = impute_pension_contributions_to_puf(
            puf[["employment_income"]]
        )

        # Sort in original PUF order
        puf = puf.set_index("RECID").loc[original_recid].reset_index()
        puf = puf.fillna(0)
        self.variable_to_entity = {
            variable: system.variables[variable].entity.key
            for variable in system.variables
        }

        VARIABLES = [
            "person_id",
            "tax_unit_id",
            "marital_unit_id",
            "spm_unit_id",
            "family_id",
            "household_id",
            "person_tax_unit_id",
            "person_marital_unit_id",
            "person_spm_unit_id",
            "person_family_id",
            "person_household_id",
            "age",
            "household_weight",
            "is_male",
            "filing_status",
            "is_tax_unit_head",
            "is_tax_unit_spouse",
            "is_tax_unit_dependent",
        ] + FINANCIAL_SUBSET

        self.holder = {variable: [] for variable in VARIABLES}

        i = 0
        self.earn_splits = []
        for _, row in puf.iterrows():
            i += 1
            exemptions = row["exemptions_count"]
            tax_unit_id = row["household_id"]
            self.add_tax_unit(row, tax_unit_id)
            self.add_filer(row, tax_unit_id)
            exemptions -= 1
            if row["filing_status"] == "JOINT":
                self.add_spouse(row, tax_unit_id)
                exemptions -= 1

            for j in range(min(3, exemptions)):
                self.add_dependent(row, tax_unit_id, j)

        groups_assumed_to_be_tax_unit_like = [
            "family",
            "spm_unit",
            "household",
        ]

        for group in groups_assumed_to_be_tax_unit_like:
            self.holder[f"{group}_id"] = self.holder["tax_unit_id"]
            self.holder[f"person_{group}_id"] = self.holder[
                "person_tax_unit_id"
            ]

        for key in self.holder:
            if key == "filing_status":
                self.holder[key] = np.array(self.holder[key]).astype("S")
            else:
                self.holder[key] = np.array(self.holder[key]).astype(float)
                assert not np.isnan(self.holder[key]).any(), f"{key} has NaNs."

        self.save_dataset(self.holder)

    def add_tax_unit(self, row, tax_unit_id):
        self.holder["tax_unit_id"].append(tax_unit_id)

        for key in FINANCIAL_SUBSET:
            if self.variable_to_entity[key] == "tax_unit":
                self.holder[key].append(row[key])

        earnings_split = round(row["EARNSPLIT"])
        if earnings_split > 0:
            SPLIT_DECODES = {
                1: 0.0,
                2: 0.25,
                3: 0.75,
                4: 1.0,
            }
            lower = SPLIT_DECODES[earnings_split]
            upper = SPLIT_DECODES[earnings_split + 1]
            frac = (upper - lower) * rng.random() + lower
            self.earn_splits.append(1.0 - frac)
        else:
            self.earn_splits.append(1.0)

        self.holder["filing_status"].append(row["filing_status"])

    def add_filer(self, row, tax_unit_id):
        person_id = int(tax_unit_id * 1e2 + 1)
        self.holder["person_id"].append(person_id)
        self.holder["person_tax_unit_id"].append(tax_unit_id)
        self.holder["person_marital_unit_id"].append(person_id)
        self.holder["marital_unit_id"].append(person_id)
        self.holder["is_tax_unit_head"].append(True)
        self.holder["is_tax_unit_spouse"].append(False)
        self.holder["is_tax_unit_dependent"].append(False)

        self.holder["age"].append(decode_age_filer(round(row["AGERANGE"])))

        self.holder["household_weight"].append(row["household_weight"])
        self.holder["is_male"].append(row["GENDER"] == 1)

        # Assume all of the interest deduction is the filer's deductible mortgage interest

        self.holder["deductible_mortgage_interest"].append(
            row["interest_deduction"]
        )

        for key in FINANCIAL_SUBSET:
            if key == "deductible_mortgage_interest":
                # Skip this one- we are adding it artificially at the filer level.
                continue
            if self.variable_to_entity[key] == "person":
                self.holder[key].append(row[key] * self.earn_splits[-1])

    def add_spouse(self, row, tax_unit_id):
        person_id = int(tax_unit_id * 1e2 + 2)
        self.holder["person_id"].append(person_id)
        self.holder["person_tax_unit_id"].append(tax_unit_id)
        self.holder["person_marital_unit_id"].append(person_id - 1)
        self.holder["is_tax_unit_head"].append(False)
        self.holder["is_tax_unit_spouse"].append(True)
        self.holder["is_tax_unit_dependent"].append(False)

        self.holder["age"].append(
            decode_age_filer(round(row["AGERANGE"]))
        )  # Assume same age as filer for now

        # 96% of joint filers are opposite-gender

        is_opposite_gender = rng.random() < 0.96
        opposite_gender_code = 0 if row["GENDER"] == 1 else 1
        same_gender_code = 1 - opposite_gender_code
        self.holder["is_male"].append(
            opposite_gender_code if is_opposite_gender else same_gender_code
        )

        # Assume all of the interest deduction is the filer's deductible mortgage interest

        self.holder["deductible_mortgage_interest"].append(0)

        for key in FINANCIAL_SUBSET:
            if key == "deductible_mortgage_interest":
                # Skip this one- we are adding it artificially at the filer level.
                continue
            if self.variable_to_entity[key] == "person":
                self.holder[key].append(row[key] * (1 - self.earn_splits[-1]))

    def add_dependent(self, row, tax_unit_id, dependent_id):
        person_id = int(tax_unit_id * 1e2 + 3 + dependent_id)
        self.holder["person_id"].append(person_id)
        self.holder["person_tax_unit_id"].append(tax_unit_id)
        self.holder["person_marital_unit_id"].append(person_id)
        self.holder["marital_unit_id"].append(person_id)
        self.holder["is_tax_unit_head"].append(False)
        self.holder["is_tax_unit_spouse"].append(False)
        self.holder["is_tax_unit_dependent"].append(True)

        age = decode_age_dependent(round(row[f"AGEDP{dependent_id + 1}"]))
        self.holder["age"].append(age)

        # Assume all of the interest deduction is the filer's deductible mortgage interest

        self.holder["deductible_mortgage_interest"].append(0)

        for key in FINANCIAL_SUBSET:
            if key == "deductible_mortgage_interest":
                # Skip this one- we are adding it artificially at the filer level.
                continue
            if self.variable_to_entity[key] == "person":
                self.holder[key].append(0)

        self.holder["is_male"].append(rng.choice([0, 1]))


class PUF_2015(PUF):
    label = "PUF 2015"
    name = "puf_2015"
    time_period = 2015
    file_path = STORAGE_FOLDER / "puf_2015.h5"


class PUF_2021(PUF):
    label = "PUF 2021"
    name = "puf_2021"
    time_period = 2021
    file_path = STORAGE_FOLDER / "puf_2021.h5"
    url = "release://policyengine/irs-soi-puf/1.8.0/puf_2021.h5"


class PUF_2024(PUF):
    label = "PUF 2024 (2015-based)"
    name = "puf_2024"
    time_period = 2024
    file_path = STORAGE_FOLDER / "puf_2024.h5"
    url = "release://policyengine/irs-soi-puf/1.8.0/puf_2024.h5"


MEDICAL_EXPENSE_CATEGORY_BREAKDOWNS = {
    "health_insurance_premiums_without_medicare_part_b": 0.453,
    "other_medical_expenses": 0.325,
    "medicare_part_b_premiums": 0.137,
    "over_the_counter_health_expenses": 0.085,
}

if __name__ == "__main__":
    PUF_2015().generate()
    PUF_2021().generate()
    PUF_2024().generate()
