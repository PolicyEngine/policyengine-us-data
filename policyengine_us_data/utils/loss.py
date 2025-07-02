import pandas as pd
from .soi import pe_to_soi, get_soi
import numpy as np
from policyengine_us_data.storage import STORAGE_FOLDER
from policyengine_core.reforms import Reform


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


# CPS-derived statistics
# Medical expenses, sum of spm thresholds
# Child support expenses

HARD_CODED_TOTALS = {
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
    # Rough estimate, not CPS derived
    "real_estate_taxes": 500e9,  # Rough estimate between 350bn and 600bn total property tax collections
    "rent": 735e9,  # ACS total uprated by CPI
    # Table 5A from https://www.irs.gov/statistics/soi-tax-stats-individual-information-return-form-w2-statistics
    # shows $38,316,190,000 in Box 7: Social security tips (2018)
    # Wages and salaries grew 32% from 2018 to 2023: https://fred.stlouisfed.org/graph/?g=1J0CC
    # Assume 40% through 2024
    "tip_income": 38e9 * 1.4,
}


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

        if row["AGI upper bound"] <= 10_000:
            continue

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
    sim.default_calculation_period = time_period
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

    # 1. Medicaid Spending
    label = "hhs/medicaid_spending"
    loss_matrix[label] = sim.calculate("medicaid", map_to="household").values
    MEDICAID_SPENDING_2024 = 9e11
    targets_array.append(MEDICAID_SPENDING_2024)

    # 2. Medicaid Enrollment
    label = "hhs/medicaid_enrollment"
    on_medicaid = (
        sim.calculate(
            "medicaid",  # or your enrollee flag
            map_to="person",
            period=time_period,
        ).values
        > 0
    ).astype(int)
    loss_matrix[label] = sim.map_result(on_medicaid, "person", "household")
    MEDICAID_ENROLLMENT_2024 = 72_429_055  # target lives (not thousands)
    targets_array.append(MEDICAID_ENROLLMENT_2024)

    # National ACA Spending
    label = "gov/aca_spending"
    loss_matrix[label] = sim.calculate(
        "aca_ptc", map_to="household", period=2025
    ).values
    ACA_SPENDING_2024 = 9.8e10  # 2024 outlays on PTC
    targets_array.append(ACA_SPENDING_2024)

    # National ACA Enrollment (people receiving a PTC)
    label = "gov/aca_enrollment"
    on_ptc = (
        sim.calculate("aca_ptc", map_to="person", period=2025).values > 0
    ).astype(int)
    loss_matrix[label] = sim.map_result(on_ptc, "person", "household")

    ACA_PTC_ENROLLMENT_2024 = 19_743_689  # people enrolled
    targets_array.append(ACA_PTC_ENROLLMENT_2024)

    # Treasury EITC

    loss_matrix["treasury/eitc"] = sim.calculate(
        "eitc", map_to="household"
    ).values
    eitc_spending = (
        sim.tax_benefit_system.parameters.calibration.gov.treasury.tax_expenditures.eitc
    )
    targets_array.append(eitc_spending(time_period))

    # IRS EITC filers and totals by child counts
    eitc_stats = pd.read_csv(STORAGE_FOLDER / "eitc.csv")

    eitc_spending_uprating = eitc_spending(time_period) / eitc_spending(2021)
    population = (
        sim.tax_benefit_system.parameters.calibration.gov.census.populations.total
    )
    population_uprating = population(time_period) / population(2021)

    for _, row in eitc_stats.iterrows():
        returns_label = (
            f"irs/eitc/returns/count_children_{row['count_children']}"
        )
        eitc_eligible_children = sim.calculate("eitc_child_count").values
        eitc = sim.calculate("eitc").values
        if row["count_children"] < 2:
            meets_child_criteria = (
                eitc_eligible_children == row["count_children"]
            )
        else:
            meets_child_criteria = (
                eitc_eligible_children >= row["count_children"]
            )
        loss_matrix[returns_label] = sim.map_result(
            (eitc > 0) * meets_child_criteria,
            "tax_unit",
            "household",
        )
        targets_array.append(row["eitc_returns"] * population_uprating)

        spending_label = (
            f"irs/eitc/spending/count_children_{row['count_children']}"
        )
        loss_matrix[spending_label] = sim.map_result(
            eitc * meets_child_criteria,
            "tax_unit",
            "household",
        )
        targets_array.append(row["eitc_total"] * eitc_spending_uprating)

    # Hard-coded totals
    for variable_name, target in HARD_CODED_TOTALS.items():
        label = f"census/{variable_name}"
        loss_matrix[label] = sim.calculate(
            variable_name, map_to="household"
        ).values
        if any(loss_matrix[label].isna()):
            raise ValueError(f"Missing values for {label}")
        targets_array.append(target)

    # Negative household market income total rough estimate from the IRS SOI PUF

    market_income = sim.calculate("household_market_income").values
    loss_matrix["irs/negative_household_market_income_total"] = (
        market_income * (market_income < 0)
    )
    targets_array.append(-138e9)

    loss_matrix["irs/negative_household_market_income_count"] = (
        market_income < 0
    ).astype(float)
    targets_array.append(3e6)

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

    # Population by state and population under 5 by state

    state_population = pd.read_csv(STORAGE_FOLDER / "population_by_state.csv")

    for _, row in state_population.iterrows():
        in_state = sim.calculate("state_code", map_to="person") == row["state"]
        label = f"census/population_by_state/{row['state']}"
        loss_matrix[label] = sim.map_result(in_state, "person", "household")
        targets_array.append(row["population"])

        under_5 = sim.calculate("age").values < 5
        in_state_under_5 = in_state * under_5
        label = f"census/population_under_5_by_state/{row['state']}"
        loss_matrix[label] = sim.map_result(
            in_state_under_5, "person", "household"
        )
        targets_array.append(row["population_under_5"])

    age = sim.calculate("age").values
    infants = (age >= 0) & (age < 1)
    label = "census/infants"
    loss_matrix[label] = sim.map_result(infants, "person", "household")
    # Total number of infants in the 1 Year ACS
    INFANTS_2023 = 3_491_679
    INFANTS_2022 = 3_437_933
    # Assume infant population grows at the same rate from 2023.
    infants_2024 = INFANTS_2023 * (INFANTS_2023 / INFANTS_2022)
    targets_array.append(infants_2024)

    networth = sim.calculate("net_worth").values
    label = "net_worth/total"
    loss_matrix[label] = networth
    # Federal Reserve estimate of $160 trillion in 2024Q4
    # https://fred.stlouisfed.org/series/BOGZ1FL192090005Q
    NET_WORTH_2024 = 160e12
    targets_array.append(NET_WORTH_2024)

    # SALT tax expenditure targeting

    _add_tax_expenditure_targets(
        dataset, time_period, sim, loss_matrix, targets_array
    )

    if any(loss_matrix.isna().sum() > 0):
        raise ValueError("Some targets are missing from the loss matrix")

    if any(pd.isna(targets_array)):
        raise ValueError("Some targets are missing from the targets array")

    # SSN Card Type calibration
    for card_type_str in ["NONE"]:  # SSN card types as strings
        ssn_type_mask = sim.calculate("ssn_card_type").values == card_type_str

        # Overall count by SSN card type
        label = f"ssa/ssn_card_type_{card_type_str.lower()}_count"
        loss_matrix[label] = sim.map_result(
            ssn_type_mask, "person", "household"
        )

        # Target undocumented population by year based on various sources
        if card_type_str == "NONE":
            undocumented_targets = {
                2022: 11.0e6,  # Official DHS Office of Homeland Security Statistics estimate for 1 Jan 2022
                # https://ohss.dhs.gov/sites/default/files/2024-06/2024_0418_ohss_estimates-of-the-unauthorized-immigrant-population-residing-in-the-united-states-january-2018%25E2%2580%2593january-2022.pdf
                2023: 12.2e6,  # Center for Migration Studies ACS-based residual estimate (published May 2025)
                # https://cmsny.org/publications/the-undocumented-population-in-the-united-states-increased-to-12-million-in-2023/
                2024: 13.0e6,  # Reuters synthesis of experts ahead of 2025 change ("~13-14 million") - central value
                # https://www.reuters.com/data/who-are-immigrants-who-could-be-targeted-trumps-mass-deportation-plans-2024-12-18/
                2025: 13.0e6,  # Same midpoint carried forward - CBP data show 95% drop in border apprehensions
            }
            if time_period <= 2022:
                target_count = 11.0e6  # Use 2022 value for earlier years
            elif time_period >= 2025:
                target_count = 13.0e6  # Use 2025 value for later years
            else:
                target_count = undocumented_targets[time_period]

        targets_array.append(target_count)

    # ACA spending by state
    spending_by_state = pd.read_csv(
        STORAGE_FOLDER / "aca_spending_and_enrollment_2024.csv"
    )
    # Monthly to yearly
    spending_by_state["spending"] = spending_by_state["spending"] * 12
    # Adjust to match national target
    spending_by_state["spending"] = spending_by_state["spending"] * (
        ACA_SPENDING_2024 / spending_by_state["spending"].sum()
    )

    for _, row in spending_by_state.iterrows():
        # Households located in this state
        in_state = (
            sim.calculate("state_code", map_to="household").values
            == row["state"]
        )

        # ACA PTC amounts for every household (2025)
        aca_value = sim.calculate(
            "aca_ptc", map_to="household", period=2025
        ).values

        # Add a loss-matrix entry and matching target
        label = f"irs/aca_spending/{row['state'].lower()}"
        loss_matrix[label] = aca_value * in_state
        annual_target = row["spending"]
        if any(loss_matrix[label].isna()):
            raise ValueError(f"Missing values for {label}")
        targets_array.append(annual_target)

    # Marketplace enrollment by state (targets in thousands)
    enrollment_by_state = pd.read_csv(
        STORAGE_FOLDER / "aca_spending_and_enrollment_2024.csv"
    )

    # One-time pulls so we don’t re-compute inside the loop
    state_person = sim.calculate("state_code", map_to="person").values

    # Flag people in households that actually receive any PTC (> 0)
    in_tax_unit_with_aca = (
        sim.calculate("aca_ptc", map_to="person", period=2025).values > 0
    )
    is_aca_eligible = sim.calculate(
        "is_aca_ptc_eligible", map_to="person", period=2025
    ).values
    is_enrolled = in_tax_unit_with_aca & is_aca_eligible

    for _, row in enrollment_by_state.iterrows():
        # People who both live in the state and have marketplace coverage
        in_state = state_person == row["state"]
        in_state_enrolled = in_state & is_enrolled

        label = f"irs/aca_enrollment/{row['state'].lower()}"
        loss_matrix[label] = sim.map_result(
            in_state_enrolled, "person", "household"
        )
        if any(loss_matrix[label].isna()):
            raise ValueError(f"Missing values for {label}")

        # Convert to thousands for the target
        targets_array.append(row["enrollment"])

    # State 10-year age targets

    age_targets = pd.read_csv(STORAGE_FOLDER / "age_state.csv")

    for state in age_targets.GEO_NAME.unique():
        state_mask = state_person == state
        for age_range in age_targets.columns[2:]:
            if "+" in age_range:
                # Handle the "85+" case
                age_lower_bound = int(age_range.replace("+", ""))
                age_upper_bound = np.inf
            else:
                age_lower_bound, age_upper_bound = map(
                    int, age_range.split("-")
                )

            age_mask = (age >= age_lower_bound) & (age <= age_upper_bound)
            label = f"census/age/{state}/{age_range}"
            loss_matrix[label] = sim.map_result(
                state_mask * age_mask, "person", "household"
            )
            target_value = age_targets.loc[
                age_targets.GEO_NAME == state, age_range
            ].values[0]
            targets_array.append(target_value)

    agi_state_target_names, agi_state_targets = _add_agi_state_targets()
    targets_array.extend(agi_state_targets)
    loss_matrix = _add_agi_metric_columns(loss_matrix, sim)

    targets_array, loss_matrix = _add_state_real_estate_taxes(
        loss_matrix, targets_array, sim
    )

    return loss_matrix, np.array(targets_array)


def _add_tax_expenditure_targets(
    dataset,
    time_period,
    baseline_simulation,
    loss_matrix: pd.DataFrame,
    targets_array: list,
):
    from policyengine_us import Microsimulation

    income_tax_b = baseline_simulation.calculate(
        "income_tax", map_to="household"
    ).values

    # Dictionary of itemized deductions and their target values
    # (in billions for 2024, per the 2024 JCT Tax Expenditures report)
    # https://www.jct.gov/publications/2024/jcx-48-24/
    ITEMIZED_DEDUCTIONS = {
        "salt_deduction": 21.247e9,
        "medical_expense_deduction": 11.4e9,
        "charitable_deduction": 65.301e9,
        "interest_deduction": 24.8e9,
        "qualified_business_income_deduction": 63.1e9,
    }

    def make_repeal_class(deduction_var):
        # Create a custom Reform subclass that neutralizes the given deduction.
        class RepealDeduction(Reform):
            def apply(self):
                self.neutralize_variable(deduction_var)

        return RepealDeduction

    for deduction, target in ITEMIZED_DEDUCTIONS.items():
        # Generate the custom repeal class for the current deduction.
        RepealDeduction = make_repeal_class(deduction)

        # Run the microsimulation using the repeal reform.
        simulation = Microsimulation(dataset=dataset, reform=RepealDeduction)
        simulation.default_calculation_period = time_period

        # Calculate the baseline and reform income tax values.
        income_tax_r = simulation.calculate(
            "income_tax", map_to="household"
        ).values

        # Compute the tax expenditure (TE) values.
        te_values = income_tax_r - income_tax_b

        # Record the TE difference and the corresponding target value.
        loss_matrix[f"jct/{deduction}_expenditure"] = te_values
        targets_array.append(target)

def get_agi_band_label(lower: float, upper: float) -> str:
    """Get the label for the AGI band based on lower and upper bounds."""
    if lower <= 0:
        return f"-inf_{int(upper)}"
    elif np.isposinf(upper):
        return f"{int(lower)}_inf"
    else:
        return f"{int(lower)}_{int(upper)}"
    


def get_national_income_tax_from_puf(puf, year = 2025):
    """
    Calculate the national total for income tax after credits from PUF data.
    """
    # Convert PUF to SOI format
    df = puf_to_soi(puf, year)
    
    # Calculate weighted national total
    national_total = (df["income_tax_after_credits"] * df["weight"]).sum()
    
    return national_total


def _add_agi_state_targets():
    """
    Create an aggregate target matrix for the appropriate geographic area
    with federal income tax calibrated to national total.
    """
    soi_targets = pd.read_csv(STORAGE_FOLDER / "agi_state.csv")
    
    # Calibrate federal income tax amounts to national PUF total
    fed_tax_mask = (
        (soi_targets["VARIABLE"] == "federal_income_tax/amount") & 
        (soi_targets["IS_COUNT"] == 0)
    )
    
    if fed_tax_mask.any():
        # Get national total from PUF
        
        # You'll need to have access to the puf data and year
        national_total = get_national_income_tax_from_puf(puf)  
      
        # Calculate current state sum for federal income tax
        state_sum = soi_targets.loc[fed_tax_mask, "VALUE"].sum()
        
        # Calculate scaling factor
        scaling_factor = national_total / state_sum
        
        # Scale federal income tax values
        soi_targets.loc[fed_tax_mask, "VALUE"] *= scaling_factor
        
        # Verify the scaling worked
        assert np.isclose(
            soi_targets.loc[fed_tax_mask, "VALUE"].sum(), 
            national_total, 
            rtol=1e-8
        ), f"Federal income tax totals do not sum to national target: {soi_targets.loc[fed_tax_mask, 'VALUE'].sum()} vs {national_total}"
    
    # Create target names
    soi_targets["target_name"] = (
        soi_targets["GEO_NAME"]
        + "/"
        + soi_targets["VARIABLE"]
        + "/"
        + soi_targets.apply(
            lambda r: get_agi_band_label(
                r["AGI_LOWER_BOUND"], r["AGI_UPPER_BOUND"]
            ),
            axis=1,
        )
    )
    
    target_names = soi_targets["target_name"].tolist()
    target_values = soi_targets["VALUE"].astype(float).tolist()
    return target_names, target_values


def add_agi_metric_columns(
    loss_matrix: pd.DataFrame, 
    sim,
):
    """
    Add AGI & National Income tax metric columns by state to the loss_matrix from the SOI.
    """
    soi_targets = pd.read_csv(STORAGE_FOLDER / "agi_state.csv")
    
    # AGI is at tax unit level
    agi = sim.calculate("adjusted_gross_income").values
    
    # Federal income tax is at household level - need to map to tax unit from household
    federal_income_tax_household = sim.calculate("federal_income_tax").values
    federal_income_tax = sim.map_result(
        federal_income_tax_household, "household", "tax_unit", how="divide_by_num_of_tax_units"
    )
    
    # State is at person level - map to tax unit
    state = sim.calculate("state_code", map_to="person").values
    state = sim.map_result(
        state, "person", "tax_unit", how="value_from_first_person"
    )
    
    for _, r in soi_targets.iterrows():
        lower, upper = r.AGI_LOWER_BOUND, r.AGI_UPPER_BOUND
        band = get_agi_band_label(lower, upper)
        
        in_state = state == r.GEO_NAME
        in_band = (agi > lower) & (agi <= upper)
        
        # Determine which variable to use based on the target
        if "adjusted_gross_income" in r.VARIABLE:
            base_value = agi
        elif "federal_income_tax" in r.VARIABLE:
            base_value = federal_income_tax
        else:
            continue
            
        if r.IS_COUNT:
            metric = (in_state & in_band & (base_value > 0)).astype(float)
        else:
            metric = np.where(in_state & in_band, base_value, 0.0)
            
        # Map back to household level for the loss matrix
        metric = sim.map_result(metric, "tax_unit", "household")
        
        col_name = f"{r.GEO_NAME}/{r.VARIABLE}/{band}"
        loss_matrix[col_name] = metric
        
    return loss_matrix


def _add_state_real_estate_taxes(loss_matrix, targets_list, sim):
    """
    Add state real estate taxes to the loss matrix and targets list.
    """
    # Read the real estate taxes data
    real_estate_taxes_targets = pd.read_csv(
        STORAGE_FOLDER / "real_estate_taxes_by_state_acs.csv"
    )
    national_total = HARD_CODED_TOTALS["real_estate_taxes"]
    state_sum = real_estate_taxes_targets["real_estate_taxes_bn"].sum() * 1e9
    national_to_state_diff = national_total / state_sum
    real_estate_taxes_targets["real_estate_taxes_bn"] *= national_to_state_diff
    real_estate_taxes_targets["real_estate_taxes_bn"] = (
        real_estate_taxes_targets["real_estate_taxes_bn"] * 1e9
    )

    assert np.isclose(
        real_estate_taxes_targets["real_estate_taxes_bn"].sum(),
        national_total,
        rtol=1e-8,
    ), "Real estate tax totals do not sum to national target"

    targets_list.extend(
        real_estate_taxes_targets["real_estate_taxes_bn"].tolist()
    )

    real_estate_taxes = sim.calculate(
        "real_estate_taxes", map_to="household"
    ).values
    state = sim.calculate("state_code", map_to="household").values

    for _, r in real_estate_taxes_targets.iterrows():
        in_state = (state == r["state_code"]).astype(float)
        label = f"real_estate_taxes/{r['state_code']}"
        loss_matrix[label] = real_estate_taxes * in_state

    return targets_list, loss_matrix
