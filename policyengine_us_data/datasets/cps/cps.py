from importlib.resources import files
from policyengine_core.data import Dataset
from policyengine_us_data.storage import STORAGE_FOLDER, DOCS_FOLDER
import h5py
from policyengine_us_data.datasets.cps.census_cps import *
from pandas import DataFrame, Series
import numpy as np
import pandas as pd
import os
import yaml
from typing import Type
from policyengine_us_data.utils.uprating import (
    create_policyengine_uprating_factors_table,
)
from microimpute.models.qrf import QRF
import logging


test_lite = os.environ.get("TEST_LITE") == "true"
print(f"TEST_LITE == {test_lite}")


class CPS(Dataset):
    name = "cps"
    label = "CPS"
    raw_cps: Type[CensusCPS] = None
    previous_year_raw_cps: Type[CensusCPS] = None
    data_format = Dataset.ARRAYS
    frac: float | None = 1

    def generate(self):
        """Generates the Current Population Survey dataset for PolicyEngine US microsimulations.
        Technical documentation and codebook here: https://www2.census.gov/programs-surveys/cps/techdocs/cpsmar21.pdf

        Args:
            frac (float, optional): Fraction of the dataset to keep. Defaults to 1. Example: To downsample to 25% of dataset,
                set frac=0.25.
        """

        if self.raw_cps is None:
            # Extrapolate from previous year
            if self.time_period == 2025:
                cps_2024 = CPS_2024(require=True)
                arrays = cps_2024.load_dataset()
                arrays = uprate_cps_data(arrays, 2024, self.time_period)
            else:
                # Default to CPS 2023 for backward compatibility
                cps_2023 = CPS_2023(require=True)
                arrays = cps_2023.load_dataset()
                arrays = uprate_cps_data(arrays, 2023, self.time_period)
            self.save_dataset(arrays)
            return

        raw_data = self.raw_cps(require=True).load()
        cps = {}

        ENTITIES = ("person", "tax_unit", "family", "spm_unit", "household")
        person, tax_unit, family, spm_unit, household = [
            raw_data[entity] for entity in ENTITIES
        ]

        logging.info("Adding ID variables")
        add_id_variables(cps, person, tax_unit, family, spm_unit, household)
        logging.info("Adding personal variables")
        add_personal_variables(cps, person)
        logging.info("Adding personal income variables")
        add_personal_income_variables(cps, person, self.raw_cps.time_period)
        logging.info("Adding previous year income variables")
        add_previous_year_income(self, cps)
        logging.info("Adding SSN card type")
        add_ssn_card_type(
            cps,
            person,
            spm_unit,
            self.time_period,
            undocumented_target=13e6,
            undocumented_workers_target=8.3e6,
            undocumented_students_target=0.21 * 1.9e6,
        )
        logging.info("Adding family variables")
        add_spm_variables(cps, spm_unit)
        logging.info("Adding household variables")
        add_household_variables(cps, household)
        logging.info("Adding rent")
        add_rent(self, cps, person, household)
        logging.info("Adding tips")
        add_tips(self, cps)
        logging.info("Adding auto loan balance, interest and wealth")
        add_auto_loan_interest_and_net_worth(self, cps)
        logging.info("Added all variables")

        raw_data.close()
        self.save_dataset(cps)
        logging.info("Adding takeup")
        add_takeup(self)
        logging.info("Downsampling")

        # Downsample
        if self.frac is not None and self.frac < 1.0:
            self.downsample(frac=self.frac)

    def downsample(self, frac: float):
        from policyengine_us import Microsimulation

        # Store original dtypes before modifying
        original_data: dict = self.load_dataset()
        original_dtypes = {
            key: original_data[key].dtype for key in original_data
        }
        sim = Microsimulation(dataset=self)
        sim.subsample(frac=frac)

        for key in original_data:
            if key not in sim.tax_benefit_system.variables:
                logging.warning(
                    f"Attempting to downsample the variable {key} but failing because it is not in the given country package."
                )
                continue
            values = sim.calculate(key).values

            # Preserve the original dtype if possible
            if (
                key in original_dtypes
                and hasattr(values, "dtype")
                and values.dtype != original_dtypes[key]
            ):
                try:
                    original_data[key] = values.astype(original_dtypes[key])
                except:
                    # If conversion fails, log it but continue
                    logging.warning(
                        f"Could not convert {key} back to {original_dtypes[key]}"
                    )
                    original_data[key] = values
            else:
                original_data[key] = values

        self.save_dataset(original_data)


def add_rent(self, cps: h5py.File, person: DataFrame, household: DataFrame):
    cps["tenure_type"] = household.H_TENURE.map(
        {
            0: "NONE",
            1: "OWNED_WITH_MORTGAGE",
            2: "RENTED",
            3: "NONE",
        }
    ).astype("S")
    self.save_dataset(cps)

    from policyengine_us_data.datasets.acs.acs import ACS_2022
    from policyengine_us import Microsimulation

    acs = Microsimulation(dataset=ACS_2022)
    cps_sim = Microsimulation(dataset=self)

    PREDICTORS = [
        "is_household_head",
        "age",
        "is_male",
        "tenure_type",
        "employment_income",
        "self_employment_income",
        "social_security",
        "pension_income",
        "state_code_str",
        "household_size",
    ]
    IMPUTATIONS = ["rent", "real_estate_taxes"]
    train_df = acs.calculate_dataframe(PREDICTORS + IMPUTATIONS)
    train_df.tenure_type = train_df.tenure_type.map(
        {
            "OWNED_OUTRIGHT": "OWNED_WITH_MORTGAGE",
        },
        na_action="ignore",
    ).fillna(train_df.tenure_type)
    train_df = train_df[train_df.is_household_head].sample(10_000)
    inference_df = cps_sim.calculate_dataframe(PREDICTORS)
    mask = inference_df.is_household_head.values
    inference_df = inference_df[mask]

    qrf = QRF()
    logging.info("Training imputation model for rent and real estate taxes.")
    fitted_model = qrf.fit(
        X_train=train_df,
        predictors=PREDICTORS,
        imputed_variables=IMPUTATIONS,
    )
    logging.info("Imputing rent and real estate taxes.")
    imputed_values = fitted_model.predict(X_test=inference_df)
    logging.info("Imputation complete.")
    cps["rent"] = np.zeros_like(cps["age"])
    cps["rent"][mask] = imputed_values["rent"]
    # Assume zero housing assistance since
    cps["pre_subsidy_rent"] = cps["rent"]
    cps["housing_assistance"] = np.zeros_like(
        cps["spm_unit_capped_housing_subsidy_reported"]
    )
    cps["real_estate_taxes"] = np.zeros_like(cps["age"])
    cps["real_estate_taxes"][mask] = imputed_values["real_estate_taxes"]


def add_takeup(self):
    data = self.load_dataset()

    from policyengine_us import system, Microsimulation

    baseline = Microsimulation(dataset=self)
    parameters = baseline.tax_benefit_system.parameters(self.time_period)

    generator = np.random.default_rng(seed=100)

    eitc_takeup_rates = parameters.gov.irs.credits.eitc.takeup
    eitc_child_count = baseline.calculate("eitc_child_count").values
    eitc_takeup_rate = eitc_takeup_rates.calc(eitc_child_count)
    data["takes_up_eitc"] = (
        generator.random(len(data["tax_unit_id"])) < eitc_takeup_rate
    )
    dc_ptc_takeup_rate = parameters.gov.states.dc.tax.income.credits.ptc.takeup
    data["takes_up_dc_ptc"] = (
        generator.random(len(data["tax_unit_id"])) < dc_ptc_takeup_rate
    )
    generator = np.random.default_rng(seed=100)

    data["snap_take_up_seed"] = generator.random(len(data["spm_unit_id"]))
    data["aca_take_up_seed"] = generator.random(len(data["tax_unit_id"]))
    data["medicaid_take_up_seed"] = generator.random(len(data["person_id"]))

    self.save_dataset(data)


def uprate_cps_data(data, from_period, to_period):
    uprating = create_policyengine_uprating_factors_table()
    for variable in uprating.index.unique():
        if variable in data:
            current_index = uprating[uprating.index == variable][
                to_period
            ].values[0]
            start_index = uprating[uprating.index == variable][
                from_period
            ].values[0]
            growth = current_index / start_index
            data[variable] = data[variable] * growth

    return data


def add_id_variables(
    cps: h5py.File,
    person: DataFrame,
    tax_unit: DataFrame,
    family: DataFrame,
    spm_unit: DataFrame,
    household: DataFrame,
) -> None:
    """Add basic ID and weight variables.

    Args:
        cps (h5py.File): The CPS dataset file.
        person (DataFrame): The person table of the ASEC.
        tax_unit (DataFrame): The tax unit table created from the person table
            of the ASEC.
        family (DataFrame): The family table of the ASEC.
        spm_unit (DataFrame): The SPM unit table created from the person table
            of the ASEC.
        household (DataFrame): The household table of the ASEC.
    """
    # Add primary and foreign keys
    cps["person_id"] = person.PH_SEQ * 100 + person.P_SEQ
    cps["family_id"] = family.FH_SEQ * 10 + family.FFPOS
    cps["household_id"] = household.H_SEQ
    cps["person_tax_unit_id"] = person.TAX_ID
    cps["person_spm_unit_id"] = person.SPM_ID
    cps["tax_unit_id"] = tax_unit.TAX_ID
    cps["spm_unit_id"] = spm_unit.SPM_ID
    cps["person_household_id"] = person.PH_SEQ
    cps["person_family_id"] = person.PH_SEQ * 10 + person.PF_SEQ
    cps["is_household_head"] = person.P_SEQ == 1
    cps["household_weight"] = household.HSUP_WGT / 1e2

    # Marital units

    marital_unit_id = person.PH_SEQ * 1e6 + np.maximum(
        person.A_LINENO, person.A_SPOUSE
    )

    # marital_unit_id is not the household ID, zero padded and followed
    # by the index within household (of each person, or their spouse if
    # one exists earlier in the survey).

    marital_unit_id = Series(marital_unit_id).rank(
        method="dense"
    )  # Simplify to a natural number sequence with repetitions [0, 1, 1, 2, 3, ...]

    cps["person_marital_unit_id"] = marital_unit_id.values
    cps["marital_unit_id"] = marital_unit_id.drop_duplicates().values


def add_personal_variables(cps: h5py.File, person: DataFrame) -> None:
    """Add personal demographic variables.

    Args:
        cps (h5py.File): The CPS dataset file.
        person (DataFrame): The CPS person table.
    """

    # The CPS provides age as follows:
    # 00-79 = 0-79 years of age
    # 80 = 80-84 years of age
    # 85 = 85+ years of age
    # We assign the 80 ages randomly between 80 and 84.
    # to avoid unrealistically bunching at 80.
    cps["age"] = np.where(
        person.A_AGE == 80,
        # NB: randint is inclusive of first argument, exclusive of second.
        np.random.randint(80, 85, len(person)),
        person.A_AGE,
    )

    # A_SEX is 1 -> male, 2 -> female.
    cps["is_female"] = person.A_SEX == 2
    # "Is...blind or does...have serious difficulty seeing even when Wearing
    #  glasses?" 1 -> Yes
    cps["is_blind"] = person.PEDISEYE == 1
    DISABILITY_FLAGS = [
        "PEDIS" + i for i in ["DRS", "EAR", "EYE", "OUT", "PHY", "REM"]
    ]
    cps["is_disabled"] = (person[DISABILITY_FLAGS] == 1).any(axis=1)

    def children_per_parent(col: str) -> pd.DataFrame:
        """Calculate number of children in the household using parental
            pointers.

        Args:
            col (str): Either PEPAR1 and PEPAR2, which correspond to A_LINENO
            of the person's first and second parent in the household,
            respectively.
        """
        return (
            person[person[col] > 0]
            .groupby(["PH_SEQ", col])
            .size()
            .reset_index()
            .rename(columns={col: "A_LINENO", 0: "children"})
        )

    # Aggregate to parent.
    res = (
        pd.concat(
            [children_per_parent("PEPAR1"), children_per_parent("PEPAR2")]
        )
        .groupby(["PH_SEQ", "A_LINENO"])
        .children.sum()
        .reset_index()
    )
    tmp = person[["PH_SEQ", "A_LINENO"]].merge(
        res, on=["PH_SEQ", "A_LINENO"], how="left"
    )
    cps["own_children_in_household"] = tmp.children.fillna(0)

    cps["has_marketplace_health_coverage"] = person.NOW_MRK == 1

    cps["has_esi"] = person.NOW_GRP == 1

    cps["cps_race"] = person.PRDTRACE
    cps["is_hispanic"] = person.PRDTHSP != 0

    cps["is_surviving_spouse"] = person.A_MARITL == 4
    cps["is_separated"] = person.A_MARITL == 6
    # High school or college/university enrollment status.
    cps["is_full_time_college_student"] = person.A_HSCOL == 2

    cps["detailed_occupation_recode"] = person.POCCU2
    add_overtime_occupation(cps, person)


def add_personal_income_variables(
    cps: h5py.File, person: DataFrame, year: int
):
    """Add income variables.

    Args:
        cps (h5py.File): The CPS dataset file.
        person (DataFrame): The CPS person table.
        year (int): The CPS year
    """
    # Get income imputation parameters.
    yamlfilename = (
        files("policyengine_us_data")
        / "datasets"
        / "cps"
        / "imputation_parameters.yaml"
    )

    with open(yamlfilename, "r", encoding="utf-8") as yamlfile:
        p = yaml.safe_load(yamlfile)
    assert isinstance(p, dict)

    # Assign CPS variables.
    cps["employment_income"] = person.WSAL_VAL

    cps["weekly_hours_worked"] = person.HRSWK * person.WKSWORK / 52
    cps["hours_worked_last_week"] = person.A_HRS1 * person.WKSWORK / 52

    cps["taxable_interest_income"] = person.INT_VAL * (
        p["taxable_interest_fraction"]
    )
    cps["tax_exempt_interest_income"] = person.INT_VAL * (
        1 - p["taxable_interest_fraction"]
    )
    cps["self_employment_income"] = person.SEMP_VAL
    cps["farm_income"] = person.FRSE_VAL
    cps["qualified_dividend_income"] = person.DIV_VAL * (
        p["qualified_dividend_fraction"]
    )
    cps["non_qualified_dividend_income"] = person.DIV_VAL * (
        1 - p["qualified_dividend_fraction"]
    )
    cps["rental_income"] = person.RNT_VAL
    # Assign Social Security retirement benefits if at least 62.
    MINIMUM_RETIREMENT_AGE = 62
    cps["social_security_retirement"] = np.where(
        person.A_AGE >= MINIMUM_RETIREMENT_AGE, person.SS_VAL, 0
    )
    # Otherwise assign them to Social Security disability benefits.
    cps["social_security_disability"] = (
        person.SS_VAL - cps["social_security_retirement"]
    )
    # Provide placeholders for other Social Security inputs to avoid creating
    # NaNs as they're uprated.
    cps["social_security_dependents"] = np.zeros_like(
        cps["social_security_retirement"]
    )
    cps["social_security_survivors"] = np.zeros_like(
        cps["social_security_retirement"]
    )
    cps["unemployment_compensation"] = person.UC_VAL
    # Add pensions and annuities.
    cps_pensions = person.PNSN_VAL + person.ANN_VAL
    # Assume a constant fraction of pension income is taxable.
    cps["taxable_private_pension_income"] = (
        cps_pensions * p["taxable_pension_fraction"]
    )
    cps["tax_exempt_private_pension_income"] = cps_pensions * (
        1 - p["taxable_pension_fraction"]
    )
    # Retirement account distributions.
    RETIREMENT_CODES = {
        1: "401k",
        2: "403b",
        3: "roth_ira",
        4: "regular_ira",
        5: "keogh",
        6: "sep",  # Simplified Employee Pension plan
        7: "other_type_retirement_account",
    }
    for code, description in RETIREMENT_CODES.items():
        tmp = 0
        # The ASEC splits distributions across four variable pairs.
        for i in ["1", "2", "1_YNG", "2_YNG"]:
            tmp += (person["DST_SC" + i] == code) * person["DST_VAL" + i]
        cps[f"{description}_distributions"] = tmp
    # Allocate retirement distributions by taxability.
    for source_with_taxable_fraction in ["401k", "403b", "sep"]:
        cps[f"taxable_{source_with_taxable_fraction}_distributions"] = (
            cps[f"{source_with_taxable_fraction}_distributions"]
            * p[
                f"taxable_{source_with_taxable_fraction}_distribution_fraction"
            ]
        )
        cps[f"tax_exempt_{source_with_taxable_fraction}_distributions"] = cps[
            f"{source_with_taxable_fraction}_distributions"
        ] * (
            1
            - p[
                f"taxable_{source_with_taxable_fraction}_distribution_fraction"
            ]
        )
        del cps[f"{source_with_taxable_fraction}_distributions"]

    # Assume all regular IRA distributions are taxable,
    # and all Roth IRA distributions are not.
    cps["taxable_ira_distributions"] = cps["regular_ira_distributions"]
    cps["tax_exempt_ira_distributions"] = cps["roth_ira_distributions"]
    # Other income (OI_VAL) is a catch-all for all other income sources.
    # The code for alimony income is 20.
    cps["alimony_income"] = (person.OI_OFF == 20) * person.OI_VAL
    # The code for strike benefits is 12.
    cps["strike_benefits"] = (person.OI_OFF == 12) * person.OI_VAL
    cps["child_support_received"] = person.CSP_VAL
    # Assume all public assistance / welfare dollars (PAW_VAL) are TANF.
    # They could also include General Assistance.
    cps["tanf_reported"] = person.PAW_VAL
    cps["ssi_reported"] = person.SSI_VAL
    # Assume all retirement contributions are traditional 401(k) for now.
    # Procedure for allocating retirement contributions:
    # 1) If they report any self-employment income, allocate entirely to
    #    self-employed pension contributions.
    # 2) If they report any wage and salary income, allocate in this order:
    #    a) Traditional 401(k) contributions up to to limit
    #    b) Roth 401(k) contributions up to the limit
    #    c) IRA contributions up to the limit, split according to administrative fractions
    #    d) Other retirement contributions
    # Disregard reported pension contributions from people who report neither wage and salary
    # nor self-employment income.
    # Assume no 403(b) or 457 contributions for now.
    LIMIT_401K_2022 = 20_500
    LIMIT_401K_CATCH_UP_2022 = 6_500
    LIMIT_IRA_2022 = 6_000
    LIMIT_IRA_CATCH_UP_2022 = 1_000
    CATCH_UP_AGE_2022 = 50
    retirement_contributions = person.RETCB_VAL
    cps["self_employed_pension_contributions"] = np.where(
        person.SEMP_VAL > 0, retirement_contributions, 0
    )
    remaining_retirement_contributions = np.maximum(
        retirement_contributions - cps["self_employed_pension_contributions"],
        0,
    )
    # Compute the 401(k) limit for the person's age.
    catch_up_eligible = person.A_AGE >= CATCH_UP_AGE_2022
    limit_401k = LIMIT_401K_2022 + catch_up_eligible * LIMIT_401K_CATCH_UP_2022
    limit_ira = LIMIT_IRA_2022 + catch_up_eligible * LIMIT_IRA_CATCH_UP_2022
    cps["traditional_401k_contributions"] = np.where(
        person.WSAL_VAL > 0,
        np.minimum(remaining_retirement_contributions, limit_401k),
        0,
    )
    remaining_retirement_contributions = np.maximum(
        remaining_retirement_contributions
        - cps["traditional_401k_contributions"],
        0,
    )
    cps["roth_401k_contributions"] = np.where(
        person.WSAL_VAL > 0,
        np.minimum(remaining_retirement_contributions, limit_401k),
        0,
    )
    remaining_retirement_contributions = np.maximum(
        remaining_retirement_contributions - cps["roth_401k_contributions"],
        0,
    )
    cps["traditional_ira_contributions"] = np.where(
        person.WSAL_VAL > 0,
        np.minimum(remaining_retirement_contributions, limit_ira),
        0,
    )
    remaining_retirement_contributions = np.maximum(
        remaining_retirement_contributions
        - cps["traditional_ira_contributions"],
        0,
    )
    roth_ira_limit = limit_ira - cps["traditional_ira_contributions"]
    cps["roth_ira_contributions"] = np.where(
        person.WSAL_VAL > 0,
        np.minimum(remaining_retirement_contributions, roth_ira_limit),
        0,
    )
    # Allocate capital gains into long-term and short-term based on aggregate split.
    cps["long_term_capital_gains"] = person.CAP_VAL * (
        p["long_term_capgain_fraction"]
    )
    cps["short_term_capital_gains"] = person.CAP_VAL * (
        1 - p["long_term_capgain_fraction"]
    )
    cps["receives_wic"] = person.WICYN == 1
    cps["veterans_benefits"] = person.VET_VAL
    cps["workers_compensation"] = person.WC_VAL
    # Disability income has multiple sources and values split across two pairs
    # of variables. Include everything except for worker's compensation
    # (code 1), which is defined as WC_VAL.
    WORKERS_COMP_DISABILITY_CODE = 1
    disability_benefits_1 = person.DIS_VAL1 * (
        person.DIS_SC1 != WORKERS_COMP_DISABILITY_CODE
    )
    disability_benefits_2 = person.DIS_VAL2 * (
        person.DIS_SC2 != WORKERS_COMP_DISABILITY_CODE
    )
    cps["disability_benefits"] = disability_benefits_1 + disability_benefits_2
    # Expenses.
    # "What is the annual amount of child support paid?"
    cps["child_support_expense"] = person.CHSP_VAL
    cps["health_insurance_premiums_without_medicare_part_b"] = person.PHIP_VAL
    cps["over_the_counter_health_expenses"] = person.POTC_VAL
    cps["other_medical_expenses"] = person.PMED_VAL
    cps["medicare_part_b_premiums"] = person.PEMCPREM

    # Get QBI simulation parameters ---
    yamlfilename = (
        files("policyengine_us_data")
        / "datasets"
        / "puf"
        / "qbi_assumptions.yaml"
    )
    with open(yamlfilename, "r", encoding="utf-8") as yamlfile:
        p = yaml.safe_load(yamlfile)
    assert isinstance(p, dict)

    rng = np.random.default_rng(seed=43)
    for var, prob in p["qbi_qualification_probabilities"].items():
        cps[f"{var}_would_be_qualified"] = rng.random(len(person)) < prob


def add_spm_variables(cps: h5py.File, spm_unit: DataFrame) -> None:
    SPM_RENAMES = dict(
        spm_unit_total_income_reported="SPM_TOTVAL",
        snap_reported="SPM_SNAPSUB",
        spm_unit_capped_housing_subsidy_reported="SPM_CAPHOUSESUB",
        free_school_meals_reported="SPM_SCHLUNCH",
        spm_unit_energy_subsidy_reported="SPM_ENGVAL",
        spm_unit_wic_reported="SPM_WICVAL",
        spm_unit_broadband_subsidy_reported="SPM_BBSUBVAL",
        spm_unit_payroll_tax_reported="SPM_FICA",
        spm_unit_federal_tax_reported="SPM_FEDTAX",
        # State tax includes refundable credits.
        spm_unit_state_tax_reported="SPM_STTAX",
        spm_unit_capped_work_childcare_expenses="SPM_CAPWKCCXPNS",
        spm_unit_spm_threshold="SPM_POVTHRESHOLD",
        spm_unit_net_income_reported="SPM_RESOURCES",
        spm_unit_pre_subsidy_childcare_expenses="SPM_CHILDCAREXPNS",
    )

    for openfisca_variable, asec_variable in SPM_RENAMES.items():
        if asec_variable in spm_unit.columns:
            cps[openfisca_variable] = spm_unit[asec_variable]

    cps["reduced_price_school_meals_reported"] = (
        cps["free_school_meals_reported"] * 0
    )


def add_household_variables(cps: h5py.File, household: DataFrame) -> None:
    cps["state_fips"] = household.GESTFIPS
    cps["county_fips"] = household.GTCO
    state_county_fips = cps["state_fips"] * 1e3 + cps["county_fips"]
    # Assign is_nyc here instead of as a variable formula so that it shows up
    # as toggleable in the webapp.
    # List county FIPS codes for each NYC county/borough.
    NYC_COUNTY_FIPS = [
        5,  # Bronx
        47,  # Kings (Brooklyn)
        61,  # New York (Manhattan)
        81,  # Queens
        85,  # Richmond (Staten Island)
    ]
    # Compute NYC by concatenating NY state FIPS with county FIPS.
    # For example, 36061 is Manhattan.
    NYS_FIPS = 36
    nyc_full_county_fips = [
        NYS_FIPS * 1e3 + county_fips for county_fips in NYC_COUNTY_FIPS
    ]
    cps["in_nyc"] = np.isin(state_county_fips, nyc_full_county_fips)


def add_previous_year_income(self, cps: h5py.File) -> None:
    if self.previous_year_raw_cps is None:
        logging.info(
            "No previous year data available for this dataset, skipping previous year income imputation."
        )
        return

    cps_current_year_data = self.raw_cps(require=True).load()
    cps_previous_year_data = self.previous_year_raw_cps(require=True).load()
    cps_previous_year = cps_previous_year_data.person.set_index(
        cps_previous_year_data.person.PERIDNUM
    )
    cps_current_year = cps_current_year_data.person.set_index(
        cps_current_year_data.person.PERIDNUM
    )

    previous_year_data = cps_previous_year[
        ["WSAL_VAL", "SEMP_VAL", "I_ERNVAL", "I_SEVAL"]
    ].rename(
        {
            "WSAL_VAL": "employment_income_last_year",
            "SEMP_VAL": "self_employment_income_last_year",
        },
        axis=1,
    )

    previous_year_data = previous_year_data[
        (previous_year_data.I_ERNVAL == 0) & (previous_year_data.I_SEVAL == 0)
    ]

    previous_year_data.drop(["I_ERNVAL", "I_SEVAL"], axis=1, inplace=True)

    joined_data = cps_current_year.join(previous_year_data)[
        [
            "employment_income_last_year",
            "self_employment_income_last_year",
            "I_ERNVAL",
            "I_SEVAL",
        ]
    ]
    joined_data["previous_year_income_available"] = (
        ~joined_data.employment_income_last_year.isna()
        & ~joined_data.self_employment_income_last_year.isna()
        & (joined_data.I_ERNVAL == 0)
        & (joined_data.I_SEVAL == 0)
    )
    joined_data = joined_data.fillna(-1).drop(["I_ERNVAL", "I_SEVAL"], axis=1)

    # CPS already ordered by PERIDNUM, so the join wouldn't change the order.
    cps["employment_income_last_year"] = joined_data[
        "employment_income_last_year"
    ].values
    cps["self_employment_income_last_year"] = joined_data[
        "self_employment_income_last_year"
    ].values
    cps["previous_year_income_available"] = joined_data[
        "previous_year_income_available"
    ].values


def add_ssn_card_type(
    cps: h5py.File,
    person: pd.DataFrame,
    spm_unit: pd.DataFrame,
    time_period: int,
    undocumented_target: float = 13e6,
    undocumented_workers_target: float = 8.3e6,
    undocumented_students_target: float = 0.21 * 1.9e6,
) -> None:
    """
    Assign SSN card type using PRCITSHP, employment status, and ASEC-UA conditions.
    Codes:
    - 0: "NONE" - Likely undocumented immigrants
    - 1: "CITIZEN" - US citizens (born or naturalized)
    - 2: "NON_CITIZEN_VALID_EAD" - Non-citizens with work/study authorization
    - 3: "OTHER_NON_CITIZEN" - Non-citizens with indicators of legal status
    """

    # Initialize CSV logging for population tracking
    population_log = []

    def select_random_subset_to_target(
        eligible_ids, current_weighted, target_weighted, random_seed=None
    ):
        """
        Randomly select subset to move current weighted population to target.

        Args:
            eligible_ids: Array of person indices eligible for selection
            current_weighted: Current weighted total
            target_weighted: Target weighted total
            random_seed: Random seed for reproducibility

        Returns:
            Array of selected person indices
        """
        if len(eligible_ids) == 0:
            return np.array([], dtype=int)

        # Calculate how much weighted population needs to be moved
        if current_weighted > target_weighted:
            excess_weighted = current_weighted - target_weighted
            # Calculate fraction to move randomly
            total_reassignable_weight = np.sum(person_weights[eligible_ids])
            share_to_move = excess_weighted / total_reassignable_weight
            share_to_move = min(share_to_move, 1.0)  # Cap at 100%
        else:
            # Calculate how much to move to reach target (for EAD case)
            needed_weighted = (
                current_weighted - target_weighted
            )  # Will be negative
            total_weight = np.sum(person_weights[eligible_ids])
            share_to_move = abs(needed_weighted) / total_weight
            share_to_move = min(share_to_move, 1.0)  # Cap at 100%

        if share_to_move > 0:
            if random_seed is not None:
                if current_weighted > target_weighted:
                    # Use new rng for refinement
                    rng = np.random.default_rng(seed=random_seed)
                    random_draw = rng.random(len(eligible_ids))
                    assign_mask = random_draw < share_to_move
                    selected = eligible_ids[assign_mask]
                else:
                    # Use old np.random for EAD to maintain compatibility
                    np.random.seed(random_seed)
                    n_to_move = int(len(eligible_ids) * share_to_move)
                    selected = np.random.choice(
                        eligible_ids, size=n_to_move, replace=False
                    )
            else:
                selected = np.array([], dtype=int)
        else:
            selected = np.array([], dtype=int)

        return selected

    # Get household weights for population calculations
    household_ids = cps["household_id"]
    household_weights = cps["household_weight"]
    person_household_ids = cps["person_household_id"]
    household_to_weight = dict(zip(household_ids, household_weights))
    person_weights = np.array(
        [household_to_weight.get(hh_id, 0) for hh_id in person_household_ids]
    )

    # Initialize all persons as code 0
    ssn_card_type = np.full(len(person), 0)
    initial_population = np.sum(person_weights[ssn_card_type == 0])
    print(f"Step 0 - Initial: Code 0 people: {initial_population:,.0f}")
    population_log.append(
        {
            "step": "Step 0 - Initial",
            "description": "Code 0 people",
            "population": initial_population,
        }
    )

    # ============================================================================
    # PRIMARY CLASSIFICATIONS
    # ============================================================================

    # Code 1: All US Citizens (naturalized and born)
    citizens_mask = np.isin(person.PRCITSHP, [1, 2, 3, 4])
    ssn_card_type[citizens_mask] = 1
    noncitizens = person.PRCITSHP == 5
    citizens_moved = np.sum(person_weights[citizens_mask])
    print(f"Step 1 - Citizens: Moved {citizens_moved:,.0f} people to Code 1")
    population_log.append(
        {
            "step": "Step 1 - Citizens",
            "description": "Moved to Code 1",
            "population": citizens_moved,
        }
    )

    # ============================================================================
    # ASEC UNDOCUMENTED ALGORITHM CONDITIONS
    # Remove individuals with indicators of legal status from code 0 pool
    # ============================================================================

    # paper source: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4662801
    # Helper mask: Only apply conditions to non-citizens without clear authorization
    potentially_undocumented = ~np.isin(ssn_card_type, [1, 2])

    current_code_0 = np.sum(person_weights[ssn_card_type == 0])
    print(f"\nASEC Conditions - Current Code 0 people: {current_code_0:,.0f}")
    population_log.append(
        {
            "step": "ASEC Conditions",
            "description": "Current Code 0 people",
            "population": current_code_0,
        }
    )

    # CONDITION 1: Pre-1982 Arrivals (IRCA Amnesty Eligible)
    # PEINUSYR values indicating arrival before 1982:
    # 01 = Before 1950
    # 02 = 1950–1959
    # 03 = 1960–1964
    # 04 = 1965–1969
    # 05 = 1970–1974
    # 06 = 1975–1979
    # 07 = 1980–1981
    arrived_before_1982 = np.isin(person.PEINUSYR, [1, 2, 3, 4, 5, 6, 7])
    condition_1_mask = potentially_undocumented & arrived_before_1982
    condition_1_count = np.sum(person_weights[condition_1_mask])
    print(
        f"Condition 1 - Pre-1982 arrivals: {condition_1_count:,.0f} people qualify for Code 3"
    )
    population_log.append(
        {
            "step": "Condition 1",
            "description": "Pre-1982 arrivals qualify for Code 3",
            "population": condition_1_count,
        }
    )

    # CONDITION 2: Eligible Naturalized Citizens
    is_naturalized = person.PRCITSHP == 4
    is_adult = person.A_AGE >= 18
    # 5+ years in US (codes 8-26: 1982-2019)
    has_five_plus_years = np.isin(person.PEINUSYR, list(range(8, 27)))
    # 3+ years in US + married (codes 8-27: 1982-2021)
    has_three_plus_years = np.isin(person.PEINUSYR, list(range(8, 28)))
    is_married = person.A_MARITL.isin([1, 2]) & (person.A_SPOUSE > 0)
    eligible_naturalized = (
        is_naturalized
        & is_adult
        & (has_five_plus_years | (has_three_plus_years & is_married))
    )
    condition_2_mask = potentially_undocumented & eligible_naturalized
    condition_2_count = np.sum(person_weights[condition_2_mask])
    print(
        f"Condition 2 - Eligible naturalized citizens: {condition_2_count:,.0f} people qualify for Code 3"
    )
    population_log.append(
        {
            "step": "Condition 2",
            "description": "Eligible naturalized citizens qualify for Code 3",
            "population": condition_2_count,
        }
    )

    # CONDITION 3: Medicare Recipients
    has_medicare = person.MCARE == 1
    condition_3_mask = potentially_undocumented & has_medicare
    condition_3_count = np.sum(person_weights[condition_3_mask])
    print(
        f"Condition 3 - Medicare recipients: {condition_3_count:,.0f} people qualify for Code 3"
    )
    population_log.append(
        {
            "step": "Condition 3",
            "description": "Medicare recipients qualify for Code 3",
            "population": condition_3_count,
        }
    )

    # CONDITION 4: Federal Retirement Benefits
    has_federal_pension = np.isin(person.PEN_SC1, [3]) | np.isin(
        person.PEN_SC2, [3]
    )  # Federal government pension
    condition_4_mask = potentially_undocumented & has_federal_pension
    condition_4_count = np.sum(person_weights[condition_4_mask])
    print(
        f"Condition 4 - Federal retirement benefits: {condition_4_count:,.0f} people qualify for Code 3"
    )
    population_log.append(
        {
            "step": "Condition 4",
            "description": "Federal retirement benefits qualify for Code 3",
            "population": condition_4_count,
        }
    )

    # CONDITION 5: Social Security Disability
    has_ss_disability = np.isin(person.RESNSS1, [2]) | np.isin(
        person.RESNSS2, [2]
    )  # Disabled (adult or child)
    condition_5_mask = potentially_undocumented & has_ss_disability
    condition_5_count = np.sum(person_weights[condition_5_mask])
    print(
        f"Condition 5 - Social Security disability: {condition_5_count:,.0f} people qualify for Code 3"
    )
    population_log.append(
        {
            "step": "Condition 5",
            "description": "Social Security disability qualify for Code 3",
            "population": condition_5_count,
        }
    )

    # CONDITION 6: Indian Health Service Coverage
    has_ihs = person.IHSFLG == 1
    condition_6_mask = potentially_undocumented & has_ihs
    condition_6_count = np.sum(person_weights[condition_6_mask])
    print(
        f"Condition 6 - Indian Health Service coverage: {condition_6_count:,.0f} people qualify for Code 3"
    )
    population_log.append(
        {
            "step": "Condition 6",
            "description": "Indian Health Service coverage qualify for Code 3",
            "population": condition_6_count,
        }
    )

    # CONDITION 7: Medicaid Recipients (simplified - no state adjustments)
    has_medicaid = person.CAID == 1
    condition_7_mask = potentially_undocumented & has_medicaid
    condition_7_count = np.sum(person_weights[condition_7_mask])
    print(
        f"Condition 7 - Medicaid recipients: {condition_7_count:,.0f} people qualify for Code 3"
    )
    population_log.append(
        {
            "step": "Condition 7",
            "description": "Medicaid recipients qualify for Code 3",
            "population": condition_7_count,
        }
    )

    # CONDITION 8: CHAMPVA Recipients
    has_champva = person.CHAMPVA == 1
    condition_8_mask = potentially_undocumented & has_champva
    condition_8_count = np.sum(person_weights[condition_8_mask])
    print(
        f"Condition 8 - CHAMPVA recipients: {condition_8_count:,.0f} people qualify for Code 3"
    )
    population_log.append(
        {
            "step": "Condition 8",
            "description": "CHAMPVA recipients qualify for Code 3",
            "population": condition_8_count,
        }
    )

    # CONDITION 9: Military Health Insurance
    has_military_insurance = person.MIL == 1
    condition_9_mask = potentially_undocumented & has_military_insurance
    condition_9_count = np.sum(person_weights[condition_9_mask])
    print(
        f"Condition 9 - Military health insurance: {condition_9_count:,.0f} people qualify for Code 3"
    )
    population_log.append(
        {
            "step": "Condition 9",
            "description": "Military health insurance qualify for Code 3",
            "population": condition_9_count,
        }
    )

    # CONDITION 10: Government Employees
    is_government_worker = np.isin(
        person.PEIO1COW, [1, 2, 3]
    )  # Fed/state/local gov
    is_military_occupation = person.A_MJOCC == 11  # Military occupation
    is_government_employee = is_government_worker | is_military_occupation
    condition_10_mask = potentially_undocumented & is_government_employee
    condition_10_count = np.sum(person_weights[condition_10_mask])
    print(
        f"Condition 10 - Government employees: {condition_10_count:,.0f} people qualify for Code 3"
    )
    population_log.append(
        {
            "step": "Condition 10",
            "description": "Government employees qualify for Code 3",
            "population": condition_10_count,
        }
    )

    # CONDITION 11: Social Security Recipients
    has_social_security = person.SS_YN == 1
    condition_11_mask = potentially_undocumented & has_social_security
    condition_11_count = np.sum(person_weights[condition_11_mask])
    print(
        f"Condition 11 - Social Security recipients: {condition_11_count:,.0f} people qualify for Code 3"
    )
    population_log.append(
        {
            "step": "Condition 11",
            "description": "Social Security recipients qualify for Code 3",
            "population": condition_11_count,
        }
    )

    # CONDITION 12: Housing Assistance
    spm_housing_map = dict(zip(spm_unit.SPM_ID, spm_unit.SPM_CAPHOUSESUB))
    has_housing_assistance = person.SPM_ID.map(spm_housing_map).fillna(0) > 0
    condition_12_mask = potentially_undocumented & has_housing_assistance
    condition_12_count = np.sum(person_weights[condition_12_mask])
    print(
        f"Condition 12 - Housing assistance: {condition_12_count:,.0f} people qualify for Code 3"
    )
    population_log.append(
        {
            "step": "Condition 12",
            "description": "Housing assistance qualify for Code 3",
            "population": condition_12_count,
        }
    )

    # CONDITION 13: Veterans/Military Personnel
    is_veteran = person.PEAFEVER == 1
    is_current_military = person.A_MJOCC == 11
    is_military_connected = is_veteran | is_current_military
    condition_13_mask = potentially_undocumented & is_military_connected
    condition_13_count = np.sum(person_weights[condition_13_mask])
    print(
        f"Condition 13 - Veterans/Military personnel: {condition_13_count:,.0f} people qualify for Code 3"
    )
    population_log.append(
        {
            "step": "Condition 13",
            "description": "Veterans/Military personnel qualify for Code 3",
            "population": condition_13_count,
        }
    )

    # CONDITION 14: SSI Recipients (simplified - assumes all SSI is for recipient)
    has_ssi = person.SSI_YN == 1
    condition_14_mask = potentially_undocumented & has_ssi
    condition_14_count = np.sum(person_weights[condition_14_mask])
    print(
        f"Condition 14 - SSI recipients: {condition_14_count:,.0f} people qualify for Code 3"
    )
    population_log.append(
        {
            "step": "Condition 14",
            "description": "SSI recipients qualify for Code 3",
            "population": condition_14_count,
        }
    )

    # ============================================================================
    # CONSOLIDATED ASSIGNMENT OF ASSUMED DOCUMENTED STATUS
    # ============================================================================

    # Combine all conditions that indicate legal status
    assumed_documented = (
        arrived_before_1982
        | eligible_naturalized
        | has_medicare
        | has_federal_pension
        | has_ss_disability
        | has_ihs
        | has_medicaid
        | has_champva
        | has_military_insurance
        | is_government_employee
        | has_social_security
        | has_housing_assistance
        | is_military_connected
        | has_ssi
    )

    # Apply single assignment for all conditions
    ssn_card_type[potentially_undocumented & assumed_documented] = 3
    # print(f"Step 2 - Documented indicators: Moved {np.sum(person_weights[potentially_undocumented & assumed_documented]):,.0f} people from Code 0 to Code 3")

    # Calculate undocumented workers and students after ASEC conditions
    undocumented_workers_mask = (
        (ssn_card_type == 0)
        & noncitizens
        & ((person.WSAL_VAL > 0) | (person.SEMP_VAL > 0))
    )
    undocumented_students_mask = (
        (ssn_card_type == 0) & noncitizens & (person.A_HSCOL == 2)
    )
    undocumented_workers_count = np.sum(
        person_weights[undocumented_workers_mask]
    )
    undocumented_students_count = np.sum(
        person_weights[undocumented_students_mask]
    )

    after_conditions_code_0 = np.sum(person_weights[ssn_card_type == 0])
    print(f"After conditions - Code 0 people: {after_conditions_code_0:,.0f}")
    print(
        f"  - Undocumented workers before adjustment: {undocumented_workers_count:,.0f} (target: {undocumented_workers_target:,.0f})"
    )
    print(
        f"  - Undocumented students before adjustment: {undocumented_students_count:,.0f} (target: {undocumented_students_target:,.0f})"
    )

    population_log.append(
        {
            "step": "After conditions",
            "description": "Code 0 people",
            "population": after_conditions_code_0,
        }
    )
    population_log.append(
        {
            "step": "Before adjustment",
            "description": "Undocumented workers",
            "population": undocumented_workers_count,
        }
    )
    population_log.append(
        {
            "step": "Target",
            "description": "Undocumented workers target",
            "population": undocumented_workers_target,
        }
    )
    population_log.append(
        {
            "step": "Before adjustment",
            "description": "Undocumented students",
            "population": undocumented_students_count,
        }
    )
    population_log.append(
        {
            "step": "Target",
            "description": "Undocumented students target",
            "population": undocumented_students_target,
        }
    )

    # ============================================================================
    # CODE 2 NON-CITIZEN WITH WORK/STUDY AUTHORIZATION
    # ============================================================================

    # Code 2: Non-citizens with work/study authorization (likely valid EAD)
    # Only consider people still in Code 0 (undocumented) after ASEC conditions
    worker_mask = (
        (ssn_card_type != 3)
        & noncitizens
        & ((person.WSAL_VAL > 0) | (person.SEMP_VAL > 0))
    )
    student_mask = (ssn_card_type != 3) & noncitizens & (person.A_HSCOL == 2)

    # Calculate target-driven worker assignment
    # Target: 8.3 million undocumented workers (from Pew Research)
    # https://www.pewresearch.org/short-reads/2024/07/22/what-we-know-about-unauthorized-immigrants-living-in-the-us/

    # Get worker IDs
    worker_ids = person[worker_mask].index

    # Use function to select workers for EAD
    total_weighted_workers = np.sum(person_weights[worker_ids])
    selected_workers = select_random_subset_to_target(
        worker_ids,
        total_weighted_workers,
        undocumented_workers_target,
        random_seed=0,
    )

    # Calculate target-driven student assignment
    # Target: 21% of 1.9 million = ~399k undocumented students (from Higher Ed Immigration Portal)
    # https://www.higheredimmigrationportal.org/research/immigrant-origin-students-in-u-s-higher-education-updated-august-2024/

    student_ids = person[student_mask].index

    # Use function to select students for EAD
    total_weighted_students = np.sum(person_weights[student_ids])
    selected_students = select_random_subset_to_target(
        student_ids,
        total_weighted_students,
        undocumented_students_target,
        random_seed=1,
    )

    # Assign code 2
    ssn_card_type[selected_workers] = 2
    ssn_card_type[selected_students] = 2
    ead_workers_moved = np.sum(person_weights[selected_workers])
    ead_students_moved = np.sum(person_weights[selected_students])
    after_ead_code_0 = np.sum(person_weights[ssn_card_type == 0])

    print(
        f"Step 3 - EAD workers: Moved {ead_workers_moved:,.0f} people from Code 0 to Code 2"
    )
    print(
        f"Step 4 - EAD students: Moved {ead_students_moved:,.0f} people from Code 0 to Code 2"
    )
    print(f"After EAD assignment - Code 0 people: {after_ead_code_0:,.0f}")

    population_log.append(
        {
            "step": "Step 3 - EAD workers",
            "description": "Moved from Code 0 to Code 2",
            "population": ead_workers_moved,
        }
    )
    population_log.append(
        {
            "step": "Step 4 - EAD students",
            "description": "Moved from Code 0 to Code 2",
            "population": ead_students_moved,
        }
    )
    population_log.append(
        {
            "step": "After EAD assignment",
            "description": "Code 0 people",
            "population": after_ead_code_0,
        }
    )

    final_counts = pd.Series(ssn_card_type).value_counts().sort_index()

    # ============================================================================
    # PROBABILISTIC FAMILY CORRELATION ADJUSTMENT
    # ============================================================================

    # Probabilistic family correlation: Only move code 3 household members to code 0
    # if needed to hit the undocumented target. This preserves mixed-status families
    # (citizens living with undocumented) while still achieving target-driven correlation.

    # Use existing household data
    person_household_ids = cps["person_household_id"]

    # Track before state
    code_0_before = np.sum(person_weights[ssn_card_type == 0])

    # Calculate how many more undocumented people we need to hit target
    current_undocumented = code_0_before
    undocumented_needed = max(0, undocumented_target - current_undocumented)

    print(
        f"Current undocumented: {current_undocumented:,.0f}, Target: {undocumented_target:,.0f}"
    )
    print(f"Additional undocumented needed: {undocumented_needed:,.0f}")

    families_adjusted = 0

    if undocumented_needed > 0:
        # Identify households with mixed status (code 0 + code 3 members)
        mixed_household_candidates = []

        unique_households = np.unique(person_household_ids)

        for household_id in unique_households:
            household_mask = person_household_ids == household_id
            household_ssn_codes = ssn_card_type[household_mask]

            # Check if household has both undocumented (code 0) AND code 3 members
            has_undocumented = (household_ssn_codes == 0).any()
            has_code3 = (household_ssn_codes == 3).any()

            if has_undocumented and has_code3:
                # Find code 3 indices in this household
                household_indices = np.where(household_mask)[0]
                code_3_indices = household_indices[household_ssn_codes == 3]
                mixed_household_candidates.extend(code_3_indices)

        # Randomly select from eligible code 3 members in mixed households to hit target
        if len(mixed_household_candidates) > 0:
            mixed_household_candidates = np.array(mixed_household_candidates)
            candidate_weights = person_weights[mixed_household_candidates]

            # Use probabilistic selection to hit target
            selected_indices = select_random_subset_to_target(
                mixed_household_candidates,
                current_undocumented,
                undocumented_target,
                random_seed=100,  # Different seed for family correlation
            )

            if len(selected_indices) > 0:
                ssn_card_type[selected_indices] = 0
                families_adjusted = len(selected_indices)
                print(
                    f"Selected {len(selected_indices)} people from {len(mixed_household_candidates)} candidates in mixed households"
                )
            else:
                print(
                    "No additional family members selected (target already reached)"
                )
        else:
            print("No mixed-status households found for family correlation")
    else:
        print(
            "No additional undocumented people needed - target already reached"
        )

    # Calculate the weighted impact
    code_0_after = np.sum(person_weights[ssn_card_type == 0])
    weighted_change = code_0_after - code_0_before

    print(
        f"Step 5 - Probabilistic family correlation: Changed {weighted_change:,.0f} people from Code 3 to Code 0"
    )
    print(f"After family correlation - Code 0 people: {code_0_after:,.0f}")

    population_log.append(
        {
            "step": "Step 5 - Family correlation",
            "description": "Changed from Code 3 to Code 0",
            "population": weighted_change,
        }
    )
    population_log.append(
        {
            "step": "After family correlation",
            "description": "Code 0 people",
            "population": code_0_after,
        }
    )

    def get_arrival_year_midpoint(peinusyr):
        """
        Map PEINUSYR codes to arrival year midpoints.
        Returns a numpy array of estimated arrival years.
        """
        arrival_year_map = {
            1: 1945,  # Before 1950
            2: 1955,  # 1950-1959
            3: 1962,  # 1960-1964
            4: 1967,  # 1965-1969
            5: 1972,  # 1970-1974
            6: 1977,  # 1975-1979
            7: 1981,  # 1980-1981
            8: 1983,  # 1982-1983
            9: 1985,  # 1984-1985
            10: 1987,  # 1986-1987
            11: 1989,  # 1988-1989
            12: 1991,  # 1990-1991
            13: 1993,  # 1992-1993
            14: 1995,  # 1994-1995
            15: 1997,  # 1996-1997
            16: 1999,  # 1998-1999
            17: 2001,  # 2000-2001
            18: 2003,  # 2002-2003
            19: 2005,  # 2004-2005
            20: 2007,  # 2006-2007
            21: 2009,  # 2008-2009
            22: 2011,  # 2010-2011
            23: 2013,  # 2012-2013
            24: 2015,  # 2014-2015
            25: 2017,  # 2016-2017
            26: 2019,  # 2018-2019
            27: 2021,  # 2020-2021
            28: 2023,  # 2022-2023
            29: 2024,  # 2024-2025
        }

        # Vectorized mapping with default value of 2024
        return np.vectorize(arrival_year_map.get)(peinusyr, 2024)

    # NEW IMMIGRATION-STATUS TAGS FOR OBFBA
    birth = person.PENATVTY  # two-digit nativity flag

    # Calculate arrival years once for all logic
    arrival_years = get_arrival_year_midpoint(person.PEINUSYR)
    years_in_us = time_period - arrival_years
    age_at_entry = np.maximum(0, person.A_AGE - years_in_us)

    # start every non-citizen as LPR so no UNSET survives
    immigration_status = np.full(
        len(person), "LEGAL_PERMANENT_RESIDENT", dtype="U32"
    )

    # 1. Undocumented: SSN card type 0 who arrived 1982 or later
    arrived_before_1982 = np.isin(person.PEINUSYR, [1, 2, 3, 4, 5, 6, 7])
    undoc_mask = (ssn_card_type == 0) & (~arrived_before_1982)
    immigration_status[undoc_mask] = "UNDOCUMENTED"

    COUNTRY_CODES = {
        "COFA": {511, 512},  # Micronesia, Marshall Islands. Palau not listed
        "CUBAN_HAITIAN": {327, 332},
    }

    mask = (ssn_card_type != 0) & np.isin(birth, list(COUNTRY_CODES["COFA"]))
    immigration_status[mask] = "LEGAL_PERMANENT_RESIDENT"

    # 3. Cuban / Haitian entrants (created by Congress in 1980)
    # Only those who arrived after 1980
    CUBAN_HAITIAN_ARRIVAL_CUTOFF = 1980
    cuban_haitian_mask = (
        (ssn_card_type != 0)
        & np.isin(birth, list(COUNTRY_CODES["CUBAN_HAITIAN"]))
        & (arrival_years >= CUBAN_HAITIAN_ARRIVAL_CUTOFF)
    )
    immigration_status[cuban_haitian_mask] = "CUBAN_HAITIAN_ENTRANT"

    # DACA eligibility constants
    # Source: https://www.uscis.gov/humanitarian/consideration-of-deferred-action-for-childhood-arrivals-daca
    DACA_LATEST_ARRIVAL_YEAR = 2007  # Must have arrived before June 15, 2007
    DACA_MAX_AGE_AT_ENTRY = 16  # Must have arrived before 16th birthday
    DACA_MIN_CURRENT_AGE = 15  # Must be at least 15 years old to apply

    # 4. DACA
    daca_mask = (
        (ssn_card_type == 2)  # Temporary/unauthorized status
        & (arrival_years <= DACA_LATEST_ARRIVAL_YEAR)
        & (age_at_entry < DACA_MAX_AGE_AT_ENTRY)
        & (person.A_AGE >= DACA_MIN_CURRENT_AGE)
    )
    immigration_status[daca_mask] = "DACA"

    # 5. Recent humanitarian parole/asylee/refugee (Code 3, ≤ 5 yrs)
    recent_refugee_mask = (ssn_card_type == 3) & (years_in_us <= 5)
    immigration_status[recent_refugee_mask] = "REFUGEE"

    # 6. Temp non-qualified (Code 2 not caught by DACA rule)
    mask = (ssn_card_type == 2) & (
        immigration_status == "LEGAL_PERMANENT_RESIDENT"
    )
    immigration_status[mask] = "TPS"

    # Final write (all values now in ImmigrationStatus Enum)
    cps["immigration_status"] = immigration_status.astype("S")
    # ============================================================================
    # CONVERT TO STRING LABELS AND STORE
    # ============================================================================

    code_to_str = {
        0: "NONE",  # Likely undocumented immigrants
        1: "CITIZEN",  # US citizens
        2: "NON_CITIZEN_VALID_EAD",  # Non-citizens with work/study authorization
        3: "OTHER_NON_CITIZEN",  # Non-citizens with indicators of legal status
    }
    ssn_card_type_str = (
        pd.Series(ssn_card_type).map(code_to_str).astype("S").values
    )
    cps["ssn_card_type"] = ssn_card_type_str

    # Final population summary
    print(f"\nFinal populations:")
    for code, label in code_to_str.items():
        pop = np.sum(person_weights[ssn_card_type == code])
        print(f"  Code {code} ({label}): {pop:,.0f}")
        population_log.append(
            {
                "step": "Final",
                "description": f"Code {code} ({label})",
                "population": pop,
            }
        )

    final_undocumented = np.sum(person_weights[ssn_card_type == 0])
    print(
        f"Total undocumented (Code 0): {final_undocumented:,.0f} (target: {undocumented_target:,.0f})"
    )
    population_log.append(
        {
            "step": "Final",
            "description": "Total undocumented (Code 0)",
            "population": final_undocumented,
        }
    )
    population_log.append(
        {
            "step": "Final",
            "description": "Undocumented target",
            "population": undocumented_target,
        }
    )

    # Save population log to CSV
    log_df = pd.DataFrame(population_log)
    csv_path = DOCS_FOLDER / "asec_population_log.csv"
    DOCS_FOLDER.mkdir(exist_ok=True)
    log_df.to_csv(csv_path, index=False)
    print(f"Population log saved to: {csv_path}")

    # Update documentation with actual numbers
    _update_documentation_with_numbers(log_df, DOCS_FOLDER)


def _update_documentation_with_numbers(log_df, docs_dir):
    """Update the documentation file with actual population numbers from CSV"""
    doc_path = docs_dir / "SSN_statuses_imputation.ipynb"

    if not doc_path.exists():
        print(f"Documentation file not found at: {doc_path}")
        return

    # Create mapping of step/description to population for easy lookup
    data_map = {}
    for _, row in log_df.iterrows():
        key = (row["step"], row["description"])
        data_map[key] = row["population"]

    # Read the documentation file
    with open(doc_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Define replacements based on our logging structure
    replacements = {
        "- **Step 0 - Initial**: Code 0 people = *[Run cps.py to populate]*": lambda: f"- **Step 0 - Initial**: Code 0 people = {data_map.get(('Step 0 - Initial', 'Code 0 people'), 0):,.0f}",
        "- **Step 1 - Citizens**: Moved to Code 1 = *[Run cps.py to populate]*": lambda: f"- **Step 1 - Citizens**: Moved to Code 1 = {data_map.get(('Step 1 - Citizens', 'Moved to Code 1'), 0):,.0f}",
        "- **ASEC Conditions**: Current Code 0 people = *[Run cps.py to populate]*": lambda: f"- **ASEC Conditions**: Current Code 0 people = {data_map.get(('ASEC Conditions', 'Current Code 0 people'), 0):,.0f}",
        "- **After conditions**: Code 0 people = *[Run cps.py to populate]*": lambda: f"- **After conditions**: Code 0 people = {data_map.get(('After conditions', 'Code 0 people'), 0):,.0f}",
        "- **Before adjustment**: Undocumented workers = *[Run cps.py to populate]*": lambda: f"- **Before adjustment**: Undocumented workers = {data_map.get(('Before adjustment', 'Undocumented workers'), 0):,.0f}",
        "- **Target**: Undocumented workers target = *[Run cps.py to populate]*": lambda: f"- **Target**: Undocumented workers target = {data_map.get(('Target', 'Undocumented workers target'), 0):,.0f}",
        "- **Before adjustment**: Undocumented students = *[Run cps.py to populate]*": lambda: f"- **Before adjustment**: Undocumented students = {data_map.get(('Before adjustment', 'Undocumented students'), 0):,.0f}",
        "- **Target**: Undocumented students target = *[Run cps.py to populate]*": lambda: f"- **Target**: Undocumented students target = {data_map.get(('Target', 'Undocumented students target'), 0):,.0f}",
        "- **Step 3 - EAD workers**: Moved from Code 0 to Code 2 = *[Run cps.py to populate]*": lambda: f"- **Step 3 - EAD workers**: Moved from Code 0 to Code 2 = {data_map.get(('Step 3 - EAD workers', 'Moved from Code 0 to Code 2'), 0):,.0f}",
        "- **Step 4 - EAD students**: Moved from Code 0 to Code 2 = *[Run cps.py to populate]*": lambda: f"- **Step 4 - EAD students**: Moved from Code 0 to Code 2 = {data_map.get(('Step 4 - EAD students', 'Moved from Code 0 to Code 2'), 0):,.0f}",
        "- **After EAD assignment**: Code 0 people = *[Run cps.py to populate]*": lambda: f"- **After EAD assignment**: Code 0 people = {data_map.get(('After EAD assignment', 'Code 0 people'), 0):,.0f}",
        "- **Step 5 - Family correlation**: Changed from Code 3 to Code 0 = *[Run cps.py to populate]*": lambda: f"- **Step 5 - Family correlation**: Changed from Code 3 to Code 0 = {data_map.get(('Step 5 - Family correlation', 'Changed from Code 3 to Code 0'), 0):,.0f}",
        "- **After family correlation**: Code 0 people = *[Run cps.py to populate]*": lambda: f"- **After family correlation**: Code 0 people = {data_map.get(('After family correlation', 'Code 0 people'), 0):,.0f}",
        "- **Final**: Code 0 (NONE) = *[Run cps.py to populate]*": lambda: f"- **Final**: Code 0 (NONE) = {data_map.get(('Final', 'Code 0 (NONE)'), 0):,.0f}",
        "- **Final**: Code 1 (CITIZEN) = *[Run cps.py to populate]*": lambda: f"- **Final**: Code 1 (CITIZEN) = {data_map.get(('Final', 'Code 1 (CITIZEN)'), 0):,.0f}",
        "- **Final**: Code 2 (NON_CITIZEN_VALID_EAD) = *[Run cps.py to populate]*": lambda: f"- **Final**: Code 2 (NON_CITIZEN_VALID_EAD) = {data_map.get(('Final', 'Code 2 (NON_CITIZEN_VALID_EAD)'), 0):,.0f}",
        "- **Final**: Code 3 (OTHER_NON_CITIZEN) = *[Run cps.py to populate]*": lambda: f"- **Final**: Code 3 (OTHER_NON_CITIZEN) = {data_map.get(('Final', 'Code 3 (OTHER_NON_CITIZEN)'), 0):,.0f}",
        "- **Final**: Total undocumented (Code 0) = *[Run cps.py to populate]*": lambda: f"- **Final**: Total undocumented (Code 0) = {data_map.get(('Final', 'Total undocumented (Code 0)'), 0):,.0f}",
        "- **Final**: Undocumented target = *[Run cps.py to populate]*": lambda: f"- **Final**: Undocumented target = {data_map.get(('Final', 'Undocumented target'), 0):,.0f}",
    }

    # Apply replacements
    for old_text, replacement_func in replacements.items():
        if old_text in content:
            content = content.replace(old_text, replacement_func())

    # Write updated content back to file
    with open(doc_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"Documentation updated with population numbers: {doc_path}")


def add_tips(self, cps: h5py.File):
    self.save_dataset(cps)
    from policyengine_us import Microsimulation

    sim = Microsimulation(dataset=self)
    cps = sim.calculate_dataframe(
        [
            "person_id",
            "household_id",
            "employment_income",
            "age",
            "household_weight",
        ],
        2025,
    )
    cps = pd.DataFrame(cps)

    cps["is_under_18"] = cps.age < 18
    cps["is_under_6"] = cps.age < 6
    cps["count_under_18"] = (
        cps.groupby("household_id")["is_under_18"]
        .sum()
        .loc[cps.household_id.values]
        .values
    )
    cps["count_under_6"] = (
        cps.groupby("household_id")["is_under_6"]
        .sum()
        .loc[cps.household_id.values]
        .values
    )
    cps = pd.DataFrame(cps)

    # Impute tips

    from policyengine_us_data.datasets.sipp import get_tip_model

    model = get_tip_model()

    cps["tip_income"] = model.predict(
        X_test=cps,
        mean_quantile=0.5,
    ).tip_income.values

    self.save_dataset(cps)


def add_overtime_occupation(cps: h5py.File, person: DataFrame) -> None:
    """Add occupation categories relevant to overtime eligibility calculations.
    Based on:
    https://www.law.cornell.edu/uscode/text/29/213
    https://www.congress.gov/crs-product/IF12480
    """
    cps["has_never_worked"] = person.POCCU2 == 53
    cps["is_military"] = person.POCCU2 == 52
    cps["is_computer_scientist"] = person.POCCU2 == 8
    cps["is_farmer_fisher"] = person.POCCU2 == 41
    cps["is_executive_administrative_professional"] = person.POCCU2.isin(
        [
            1,  # Chief executives, and managers
            2,  # Compensation, human resources, and infrastructure managers
            3,  # All other managers
            5,  # Business operations specialists
            6,  # Accountants and auditors
            7,  # Financial specialists
            9,  # Mathematical science occupations
            10,  # Architects, except naval
            11,  # Surveyors, cartographers, & photogrammetrists
            12,  # Engineering technologists and technicians
            13,  # Earth scientists
            14,  # Economists
            15,  # Psychologists, and other social scientists
            16,  # Health and safety specialists
            18,  # Lawyers, judges, magistrates, and other judicial workers
            19,  # Paralegals and all other legal support workers
            25,  # Registered nurses, therapists, and specific pathologists
            26,  # Veterinarians
            27,  # Health technicians and other healthcare practitioners
            28,  # Healthcare support occupations
            29,  # First-line supervisors of protective service workers
            34,  # First-line supervisors of housekeeping and janitorial workers
            36,  # Supervisors of personal care and service workers
            38,  # First-line supervisors of retail/non-retail sales workers
            39,  # Sales and related occupations
            40,  # Office & administrative support occupations
            42,  # First-line supervisors of construction trades workers
            50,  # Supervisors of transportation and flight related workers
        ]
    )


def add_auto_loan_interest_and_net_worth(self, cps: h5py.File) -> None:
    """ "Add auto loan balance, interest and net_worth variable."""
    self.save_dataset(cps)
    cps_data = self.load_dataset()

    # Access raw CPS for additional variables
    raw_data_instance = self.raw_cps(require=True)
    raw_data = raw_data_instance.load()
    person_data = raw_data.person

    # Preprocess the CPS for imputation
    lengths = {k: len(v) for k, v in cps_data.items()}
    var_len = cps_data["person_household_id"].shape[0]
    vars_of_interest = [name for name, ln in lengths.items() if ln == var_len]
    agg_data = pd.DataFrame({n: cps_data[n] for n in vars_of_interest})
    agg_data["interest_dividend_income"] = np.sum(
        [
            agg_data["taxable_interest_income"],
            agg_data["tax_exempt_interest_income"],
            agg_data["qualified_dividend_income"],
            agg_data["non_qualified_dividend_income"],
        ],
        axis=0,
    )
    agg_data["social_security_pension_income"] = np.sum(
        [
            agg_data["tax_exempt_private_pension_income"],
            agg_data["taxable_private_pension_income"],
            agg_data["social_security_retirement"],
        ],
        axis=0,
    )

    agg = (
        agg_data.groupby("person_household_id")[
            [
                "employment_income",
                "interest_dividend_income",
                "social_security_pension_income",
            ]
        ]
        .sum()
        .rename(
            columns={
                "employment_income": "household_employment_income",
                "interest_dividend_income": "household_interest_dividend_income",
                "social_security_pension_income": "household_social_security_pension_income",
            }
        )
        .reset_index()
    )

    def create_scf_reference_person_mask(cps_data, raw_person_data):
        """
        Create a boolean mask identifying SCF-style reference persons.

        SCF Reference Person Definition:
        - Single adult in household without a couple
        - In households with couples: male in mixed-sex couple OR older person in same-sex couple
        """
        all_persons_data = pd.DataFrame(
            {
                "person_household_id": cps_data["person_household_id"],
                "age": cps_data["age"],
            }
        )

        # Add sex variable (PESEX=2 means female in CPS)
        all_persons_data["is_female"] = (raw_person_data.A_SEX == 2).values

        # Add marital status (A_MARITL codes: 1,2 = married with spouse present/absent)
        all_persons_data["is_married"] = raw_person_data.A_MARITL.isin(
            [1, 2]
        ).values

        # Define adults as age 18+
        all_persons_data["is_adult"] = all_persons_data["age"] >= 18

        # Count adults per household
        adults_per_household = (
            all_persons_data[all_persons_data["is_adult"]]
            .groupby("person_household_id")
            .size()
            .reset_index(name="n_adults")
        )
        all_persons_data = all_persons_data.merge(
            adults_per_household, on="person_household_id", how="left"
        )

        # Identify couple households (households with exactly 2 married adults)
        married_adults_per_household = (
            all_persons_data[
                (all_persons_data["is_adult"])
                & (all_persons_data["is_married"])
            ]
            .groupby("person_household_id")
            .size()
        )

        couple_households = married_adults_per_household[
            (married_adults_per_household == 2)
            & (
                all_persons_data.groupby("person_household_id")[
                    "n_adults"
                ].first()
                == 2
            )
        ].index

        all_persons_data["is_couple_household"] = all_persons_data[
            "person_household_id"
        ].isin(couple_households)

        def determine_reference_person(group):
            """Determine reference person for a household group."""
            adults = group[group["is_adult"]]

            if len(adults) == 0:
                # No adults - select the oldest person regardless of age
                reference_idx = group["age"].idxmax()
                result = pd.Series([False] * len(group), index=group.index)
                result[reference_idx] = True
                return result

            elif len(adults) == 1:
                # Only one adult - they are the reference person
                result = pd.Series([False] * len(group), index=group.index)
                result[adults.index[0]] = True
                return result

            elif group["is_couple_household"].iloc[0] and len(adults) == 2:
                # Couple household with 2 adults
                couple_adults = adults.copy()

                # Check if same-sex couple
                if couple_adults["is_female"].nunique() == 1:
                    # Same-sex couple - choose older person
                    reference_idx = couple_adults["age"].idxmax()
                else:
                    # Mixed-sex couple - choose male (is_female = False)
                    male_adults = couple_adults[~couple_adults["is_female"]]
                    if len(male_adults) > 0:
                        reference_idx = male_adults.index[0]
                    else:
                        # Fallback to older person
                        reference_idx = couple_adults["age"].idxmax()

                result = pd.Series([False] * len(group), index=group.index)
                result[reference_idx] = True
                return result

            else:
                # Multiple adults but not a couple household
                # Use the oldest adult as reference person
                reference_idx = adults["age"].idxmax()
                result = pd.Series([False] * len(group), index=group.index)
                result[reference_idx] = True
                return result

        # Apply the reference person logic to each household
        all_persons_data["is_scf_reference_person"] = (
            all_persons_data.groupby("person_household_id")
            .apply(determine_reference_person, include_groups=False)
            .reset_index(level=0, drop=True)
        )

        return all_persons_data["is_scf_reference_person"].values

    mask = create_scf_reference_person_mask(cps_data, person_data)
    mask_len = mask.shape[0]

    cps_data = {
        var: data[mask] if data.shape[0] == mask_len else data
        for var, data in cps_data.items()
    }

    CPS_RACE_MAPPING = {
        1: 1,  # White only -> WHITE
        2: 2,  # Black only -> BLACK/AFRICAN-AMERICAN
        3: 5,  # American Indian, Alaskan Native only -> OTHER
        4: 4,  # Asian only -> ASIAN
        5: 5,  # Hawaiian/Pacific Islander only -> OTHER
        6: 5,  # White-Black -> OTHER
        7: 5,  # White-AI -> OTHER
        8: 5,  # White-Asian -> OTHER
        9: 3,  # White-HP -> HISPANIC
        10: 5,  # Black-AI -> OTHER
        11: 5,  # Black-Asian -> OTHER
        12: 3,  # Black-HP -> HISPANIC
        13: 5,  # AI-Asian -> OTHER
        14: 5,  # AI-HP -> OTHER
        15: 3,  # Asian-HP -> HISPANIC
        16: 5,  # White-Black-AI -> OTHER
        17: 5,  # White-Black-Asian -> OTHER
        18: 5,  # White-Black-HP -> OTHER
        19: 5,  # White-AI-Asian -> OTHER
        20: 5,  # White-AI-HP -> OTHER
        21: 5,  # White-Asian-HP -> OTHER
        22: 5,  # Black-AI-Asian -> OTHER
        23: 5,  # White-Black-AI-Asian -> OTHER
        24: 5,  # White-AI-Asian-HP -> OTHER
        25: 5,  # Other 3 race comb. -> OTHER
        26: 5,  # Other 4 or 5 race comb. -> OTHER
    }

    # Apply the mapping to recode the race values
    cps_data["cps_race"] = np.vectorize(CPS_RACE_MAPPING.get)(
        cps_data["cps_race"]
    )

    lengths = {k: len(v) for k, v in cps_data.items()}
    var_len = cps_data["person_household_id"].shape[0]
    vars_of_interest = [name for name, ln in lengths.items() if ln == var_len]
    receiver_data = pd.DataFrame({n: cps_data[n] for n in vars_of_interest})

    receiver_data = receiver_data.merge(
        agg[
            [
                "person_household_id",
                "household_employment_income",
                "household_interest_dividend_income",
                "household_social_security_pension_income",
            ]
        ],
        on="person_household_id",
        how="left",
    )
    receiver_data.drop("employment_income", axis=1, inplace=True)

    receiver_data.rename(
        columns={
            "household_employment_income": "employment_income",
            "household_interest_dividend_income": "interest_dividend_income",
            "household_social_security_pension_income": "social_security_pension_income",
        },
        inplace=True,
    )

    # Add is_married variable for household heads based on raw person data
    reference_persons = person_data[mask]
    receiver_data["is_married"] = reference_persons.A_MARITL.isin(
        [1, 2]
    ).values

    # Impute auto loan balance from the SCF
    from policyengine_us_data.datasets.scf.scf import SCF_2022

    scf_dataset = SCF_2022()
    scf_data = scf_dataset.load_dataset()
    scf_data = pd.DataFrame({key: scf_data[key] for key in scf_data.keys()})

    PREDICTORS = [
        "age",
        "is_female",
        "cps_race",
        "is_married",
        "own_children_in_household",
        "employment_income",
        "interest_dividend_income",
        "social_security_pension_income",
    ]
    IMPUTED_VARIABLES = ["networth", "auto_loan_balance", "auto_loan_interest"]
    weights = ["wgt"]

    donor_data = scf_data[PREDICTORS + IMPUTED_VARIABLES + weights].copy()

    from microimpute.models.qrf import QRF
    import logging
    import os

    # Set root logger level
    log_level = os.getenv("PYTHON_LOG_LEVEL", "WARNING")

    # Specifically target the microimpute logger
    logging.getLogger("microimpute").setLevel(getattr(logging, log_level))

    qrf_model = QRF()
    donor_data = donor_data.sample(frac=0.5, random_state=42).reset_index(
        drop=True
    )
    fitted_model = qrf_model.fit(
        X_train=donor_data,
        predictors=PREDICTORS,
        imputed_variables=IMPUTED_VARIABLES,
        weight_col=weights[0],
        tune_hyperparameters=False,
    )
    imputations = fitted_model.predict(X_test=receiver_data)

    for var in IMPUTED_VARIABLES:
        cps[var] = imputations[var]

    cps["net_worth"] = cps["networth"]
    del cps["networth"]

    self.save_dataset(cps)


class CPS_2019(CPS):
    name = "cps_2019"
    label = "CPS 2019"
    raw_cps = CensusCPS_2019
    previous_year_raw_cps = CensusCPS_2018
    file_path = STORAGE_FOLDER / "cps_2019.h5"
    time_period = 2019
    frac = 0.5


class CPS_2020(CPS):
    name = "cps_2020"
    label = "CPS 2020"
    raw_cps = CensusCPS_2020
    previous_year_raw_cps = CensusCPS_2019
    file_path = STORAGE_FOLDER / "cps_2020.h5"
    time_period = 2020
    frac = 0.5


class CPS_2021(CPS):
    name = "cps_2021"
    label = "CPS 2021"
    raw_cps = CensusCPS_2021
    previous_year_raw_cps = CensusCPS_2020
    file_path = STORAGE_FOLDER / "cps_2021_v1_6_1.h5"
    time_period = 2021
    frac = 0.5


class CPS_2022(CPS):
    name = "cps_2022"
    label = "CPS 2022"
    raw_cps = CensusCPS_2022
    previous_year_raw_cps = CensusCPS_2021
    file_path = STORAGE_FOLDER / "cps_2022_v1_6_1.h5"
    time_period = 2022
    frac = 0.5


class CPS_2023(CPS):
    name = "cps_2023"
    label = "CPS 2023"
    raw_cps = CensusCPS_2023
    previous_year_raw_cps = CensusCPS_2022
    file_path = STORAGE_FOLDER / "cps_2023.h5"
    time_period = 2023
    frac = 0.5


class CPS_2024(CPS):
    name = "cps_2024"
    label = "CPS 2024"
    raw_cps = CensusCPS_2024
    previous_year_raw_cps = CensusCPS_2023
    file_path = STORAGE_FOLDER / "cps_2024.h5"
    time_period = 2024
    frac = 0.5


class CPS_2025(CPS):
    name = "cps_2025"
    label = "CPS 2025 (2024-based)"
    file_path = STORAGE_FOLDER / "cps_2025.h5"
    time_period = 2025
    frac = 1


# The below datasets are a very naïve way of preventing downsampling in the
# Pooled 3-Year CPS. They should be replaced by a more sustainable approach.
# If these are still here on July 1, 2025, please open an issue and raise at standup.
class CPS_2021_Full(CPS):
    name = "cps_2021_full"
    label = "CPS 2021 (full)"
    raw_cps = CensusCPS_2021
    previous_year_raw_cps = CensusCPS_2020
    file_path = STORAGE_FOLDER / "cps_2021_full.h5"
    time_period = 2021


class CPS_2022_Full(CPS):
    name = "cps_2022_full"
    label = "CPS 2022 (full)"
    raw_cps = CensusCPS_2022
    previous_year_raw_cps = CensusCPS_2021
    file_path = STORAGE_FOLDER / "cps_2022_full.h5"
    time_period = 2022


class CPS_2023_Full(CPS):
    name = "cps_2023_full"
    label = "CPS 2023 (full)"
    raw_cps = CensusCPS_2023
    previous_year_raw_cps = CensusCPS_2022
    file_path = STORAGE_FOLDER / "cps_2023_full.h5"
    time_period = 2023


class PooledCPS(Dataset):
    data_format = Dataset.ARRAYS
    input_datasets: list
    time_period: int

    def generate(self):
        data = [
            input_dataset(require=True).load_dataset()
            for input_dataset in self.input_datasets
        ]
        time_periods = [dataset.time_period for dataset in self.input_datasets]
        data = [
            uprate_cps_data(data, time_period, self.time_period)
            for data, time_period in zip(data, time_periods)
        ]

        new_data = {}

        for i in range(len(data)):
            for variable in data[i]:
                data_values = data[i][variable]
                if variable not in new_data:
                    new_data[variable] = data_values
                elif "_id" in variable:
                    previous_max = new_data[variable].max()
                    new_data[variable] = np.concatenate(
                        [
                            new_data[variable],
                            data_values + previous_max,
                        ]
                    )
                else:
                    new_data[variable] = np.concatenate(
                        [
                            new_data[variable],
                            data_values,
                        ]
                    )

        new_data["household_weight"] = new_data["household_weight"] / len(
            self.input_datasets
        )

        self.save_dataset(new_data)


class Pooled_3_Year_CPS_2023(PooledCPS):
    label = "CPS 2023 (3-year pooled)"
    name = "pooled_3_year_cps_2023"
    file_path = STORAGE_FOLDER / "pooled_3_year_cps_2023.h5"
    input_datasets = [
        CPS_2021_Full,
        CPS_2022_Full,
        CPS_2023_Full,
    ]
    time_period = 2023
    url = "hf://policyengine/policyengine-us-data/pooled_3_year_cps_2023.h5"


if __name__ == "__main__":
    if test_lite:
        CPS_2024().generate()
        CPS_2025().generate()
    else:
        CPS_2021().generate()
        CPS_2022().generate()
        CPS_2023().generate()
        CPS_2024().generate()
        CPS_2025().generate()
        CPS_2021_Full().generate()
        CPS_2022_Full().generate()
        CPS_2023_Full().generate()
        Pooled_3_Year_CPS_2023().generate()
