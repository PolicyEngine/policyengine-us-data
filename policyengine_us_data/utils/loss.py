import pandas as pd
from .soi import pe_to_soi, get_soi
import numpy as np
from policyengine_us_data.storage import STORAGE_FOLDER


def fmt(x):
    if x == -np.inf:
        return "-inf"
    if x == np.inf:
        return "inf"
    if x < 1e3:
        return f"{x:.0f}"
    if x < 1e6:
        return f"{x/1e3:.0f}k"
    if x < 1e9:
        return f"{x/1e6:.0f}m"
    return f"{x/1e9:.1f}bn"


def build_loss_matrix(dataset: type, time_period):
    loss_matrix = pd.DataFrame()
    df = pe_to_soi(dataset, time_period)
    agi = df["adjusted_gross_income"].values
    filer = df["is_tax_filer"].values
    taxable = df["total_income_tax"].values > 0
    soi_subset = get_soi(time_period)
    targets_array = []
    agi_level_targeted_variables = [
        "adjusted_gross_income",
        "count",
        "employment_income",
        "business_net_profits",
        "capital_gains_gross",
        "ordinary_dividends",
        "partnership_and_s_corp_income",
        "qualified_dividends",
        "taxable_interest_income",
        "total_pension_income",
        "total_social_security",
    ]
    aggregate_level_targeted_variables = [
        "business_net_losses",
        "capital_gains_distributions",
        "capital_gains_losses",
        "estate_income",
        "estate_losses",
        "exempt_interest",
        "ira_distributions",
        "partnership_and_s_corp_losses",
        "rent_and_royalty_net_income",
        "rent_and_royalty_net_losses",
        "taxable_pension_income",
        "taxable_social_security",
        "unemployment_compensation",
    ]
    aggregate_level_targeted_variables = [
        variable
        for variable in aggregate_level_targeted_variables
        if variable in df.columns
    ]
    soi_subset = soi_subset[
        soi_subset.Variable.isin(agi_level_targeted_variables)
        | (
            soi_subset.Variable.isin(aggregate_level_targeted_variables)
            & (soi_subset["AGI lower bound"] == -np.inf)
            & (soi_subset["AGI upper bound"] == np.inf)
        )
    ]
    for _, row in soi_subset.iterrows():
        if not row["Taxable only"]:
            continue  # exclude non "taxable returns" statistics

        mask = (
            (agi >= row["AGI lower bound"])
            * (agi < row["AGI upper bound"])
            * filer
        ) > 0

        if row["Filing status"] == "Single":
            mask *= df["filing_status"].values == "SINGLE"
        elif row["Filing status"] == "Married Filing Jointly/Surviving Spouse":
            mask *= df["filing_status"].values == "JOINT"
        elif row["Filing status"] == "Head of Household":
            mask *= df["filing_status"].values == "HEAD_OF_HOUSEHOLD"
        elif row["Filing status"] == "Married Filing Separately":
            mask *= df["filing_status"].values == "SEPARATE"

        values = df[row["Variable"]].values

        if row["Taxable only"]:
            mask *= taxable

        if row["Count"]:
            values = (values > 0).astype(float)

        agi_range_label = (
            f"{fmt(row['AGI lower bound'])}-{fmt(row['AGI upper bound'])}"
        )
        taxable_label = (
            "taxable" if row["Taxable only"] else "all" + " returns"
        )
        filing_status_label = row["Filing status"]

        variable_label = row["Variable"].replace("_", " ")

        if row["Count"] and not row["Variable"] == "count":
            label = (
                f"irs/{variable_label}/count/AGI in "
                f"{agi_range_label}/{taxable_label}/{filing_status_label}"
            )
        elif row["Variable"] == "count":
            label = (
                f"irs/{variable_label}/count/AGI in "
                f"{agi_range_label}/{taxable_label}/{filing_status_label}"
            )
        else:
            label = (
                f"irs/{variable_label}/total/AGI in "
                f"{agi_range_label}/{taxable_label}/{filing_status_label}"
            )

        if label not in loss_matrix.columns:
            loss_matrix[label] = mask * values
            targets_array.append(row["Value"])

    # Convert tax-unit level df to household-level df

    from policyengine_us import Microsimulation

    sim = Microsimulation(dataset=dataset)
    hh_id = sim.calculate("household_id", map_to="person")
    tax_unit_hh_id = sim.map_result(
        hh_id, "person", "tax_unit", how="value_from_first_person"
    )

    loss_matrix = loss_matrix.groupby(tax_unit_hh_id).sum()

    hh_id = sim.calculate("household_id").values
    loss_matrix = loss_matrix.loc[hh_id]

    # Census single-year age population projections

    populations = pd.read_csv(STORAGE_FOLDER / "np2023_d5_mid.csv")
    populations = populations[populations.SEX == 0][populations.RACE_HISP == 0]
    populations = (
        populations.groupby("YEAR")
        .sum()[[f"POP_{i}" for i in range(0, 86)]]
        .T[time_period]
        .values
    )  # Array of [age_0_pop, age_1_pop, ...] for the given year
    age = sim.calculate("age").values
    for year in range(len(populations)):
        label = f"census/population_by_age/{year}"
        loss_matrix[label] = sim.map_result(
            (age >= year) * (age < year + 1), "person", "household"
        )
        targets_array.append(populations[year])

    # CBO projections

    PROGRAMS = [
        "income_tax",
        "snap",
        "social_security",
        "ssi",
        "unemployment_compensation",
    ]

    for variable_name in PROGRAMS:
        label = f"cbo/{variable_name}"
        loss_matrix[label] = sim.calculate(
            variable_name, map_to="household"
        ).values
        if any(loss_matrix[label].isna()):
            raise ValueError(f"Missing values for {label}")
        targets_array.append(
            sim.tax_benefit_system.parameters(
                time_period
            ).calibration.gov.cbo._children[variable_name]
        )

    # CPS-derived statistics
    # Medical expenses, sum of spm thresholds
    # Child support expenses

    CPS_DERIVED_TOTALS_2024 = {
        "health_insurance_premiums_without_medicare_part_b": 385e9,
        "other_medical_expenses": 278e9,
        "medicare_part_b_premiums": 112e9,
        "over_the_counter_health_expenses": 72e9,
        "spm_unit_spm_threshold": 3_945e9,
        "child_support_expense": 33e9,
        "child_support_received": 33e9,
        "spm_unit_capped_work_childcare_expenses": 348e9,
        "spm_unit_capped_housing_subsidy": 35e9,
        "tanf": 9e9,
        # Alimony could be targeted via SOI
        "alimony_income": 13e9,
        "alimony_expense": 13e9,
    }

    for variable_name, target in CPS_DERIVED_TOTALS_2024.items():
        label = f"census/{variable_name}"
        loss_matrix[label] = sim.calculate(
            variable_name, map_to="household"
        ).values
        if any(loss_matrix[label].isna()):
            raise ValueError(f"Missing values for {label}")
        targets_array.append(target)

    # Healthcare spending by age

    healthcare = pd.read_csv(STORAGE_FOLDER / "healthcare_spending.csv")

    for _, row in healthcare.iterrows():
        age_lower_bound = int(row["age_10_year_lower_bound"])
        in_age_range = (age >= age_lower_bound) * (age < age_lower_bound + 10)
        for expense_type in [
            "health_insurance_premiums_without_medicare_part_b",
            "over_the_counter_health_expenses",
            "other_medical_expenses",
            "medicare_part_b_premiums",
        ]:
            label = f"census/{expense_type}/age_{age_lower_bound}_to_{age_lower_bound+9}"
            value = sim.calculate(expense_type).values
            loss_matrix[label] = sim.map_result(
                in_age_range * value, "person", "household"
            )
            targets_array.append(row[expense_type])

    # AGI by SPM threshold totals

    spm_threshold_agi = pd.read_csv(STORAGE_FOLDER / "spm_threshold_agi.csv")

    for _, row in spm_threshold_agi.iterrows():
        spm_unit_agi = sim.calculate(
            "adjusted_gross_income", map_to="spm_unit"
        ).values
        spm_threshold = sim.calculate("spm_unit_spm_threshold").values
        in_threshold_range = (spm_threshold >= row["lower_spm_threshold"]) * (
            spm_threshold < row["upper_spm_threshold"]
        )
        label = f"census/agi_in_spm_threshold_decile_{int(row['decile'])}"
        loss_matrix[label] = sim.map_result(
            in_threshold_range * spm_unit_agi, "spm_unit", "household"
        )
        targets_array.append(row["adjusted_gross_income"])

        label = f"census/count_in_spm_threshold_decile_{int(row['decile'])}"
        loss_matrix[label] = sim.map_result(
            in_threshold_range, "spm_unit", "household"
        )
        targets_array.append(row["count"])

    if any(loss_matrix.isna().sum() > 0):
        raise ValueError("Some targets are missing from the loss matrix")

    if any(pd.isna(targets_array)):
        raise ValueError("Some targets are missing from the targets array")

    return loss_matrix, np.array(targets_array)
