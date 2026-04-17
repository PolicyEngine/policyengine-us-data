import gc

import pandas as pd
import numpy as np
import logging
import sqlite3

from policyengine_us_data.storage import CALIBRATION_FOLDER, STORAGE_FOLDER
from policyengine_us_data.storage.calibration_targets.pull_soi_targets import (
    STATE_ABBR_TO_FIPS,
)
from policyengine_us_data.storage.calibration_targets.soi_metadata import (
    RETIREMENT_CONTRIBUTION_TARGETS,
)
from policyengine_us_data.utils.cms_medicare import (
    get_beneficiary_paid_medicare_part_b_premiums_target,
)
from policyengine_us_data.db.etl_irs_soi import get_national_geography_soi_target
from policyengine_core.reforms import Reform
from policyengine_us_data.utils.soi import pe_to_soi, get_soi

# National calibration targets consumed by build_loss_matrix().
# These values are specific to 2024 — they should NOT be applied to
# other years without re-sourcing.  They are duplicated in
# db/etl_national_targets.py which loads them into policy_data.db.
# A future PR should wire build_loss_matrix() to read from the
# database so this dict can be deleted.  See PR #488.

HARD_CODED_TOTALS = {
    "health_insurance_premiums_without_medicare_part_b": 385e9,
    "other_medical_expenses": 278e9,
    "medicare_part_b_premiums": get_beneficiary_paid_medicare_part_b_premiums_target(
        2024
    ),
    "over_the_counter_health_expenses": 72e9,
    "spm_unit_spm_threshold": 3_945e9,
    "child_support_expense": 33e9,
    "child_support_received": 33e9,
    "spm_unit_capped_work_childcare_expenses": 348e9,
    "spm_unit_capped_housing_subsidy": 35e9,
    "tanf": 7_788_317_474.55,
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
    # SSA benefit-type totals for 2024, derived from:
    # - Total OASDI: $1,452B (CBO projection)
    # - OASI trust fund: $1,227.4B in 2023
    #   https://www.ssa.gov/OACT/STATS/table4a3.html
    # - DI trust fund: $151.9B in 2023
    #   https://www.ssa.gov/OACT/STATS/table4a3.html
    # - SSA 2024 fact sheet type shares: retired+deps=78.5%,
    #   survivors=11.0%, disabled+deps=10.5%
    #   https://www.ssa.gov/OACT/FACTS/
    # - SSA Annual Statistical Supplement Table 5.A1
    #   https://www.ssa.gov/policy/docs/statcomps/supplement/2024/5a.html
    "social_security_retirement": 1_060e9,  # ~73% of total
    "social_security_disability": 148e9,  # ~10.2% (disabled workers)
    "social_security_survivors": 160e9,  # ~11.0% (widows, children of deceased)
    "social_security_dependents": 84e9,  # ~5.8% (spouses/children of retired+disabled)
    # Retirement contribution calibration targets.
    #
    # traditional_ira_contributions: IRS SOI Publication 1304, Table 1.4
    # (TY 2023), "IRA payments" deduction — $13.77B (col DU, row
    # "All returns, total"). This is the actual above-the-line
    # deduction claimed on returns. The variable flows directly into
    # the ALD with no deductibility logic in policyengine-us, so the
    # target must match the deduction, not total contributions.
    # https://www.irs.gov/statistics/soi-tax-stats-individual-statistical-tables-by-size-of-adjusted-gross-income
    "traditional_ira_contributions": RETIREMENT_CONTRIBUTION_TARGETS[
        "traditional_ira_contributions"
    ]["value"],
    # traditional_401k_contributions & roth_401k_contributions:
    # BEA/FRED National Income Accounts. Total DC employer+employee
    # = $815.4B (Y351RC1A027NBEA), employer-only = $247.5B
    # (W351RC0A144NBEA), employee elective deferrals = $567.9B.
    # Split into traditional/Roth using estimated 15% Roth dollar
    # share (Vanguard How America Saves 2024: 18% participation,
    # ~15% dollar share; PSCA 67th Annual Survey: 21% participation).
    # Traditional: $567.9B × 85% = $482.7B
    # Roth: $567.9B × 15% = $85.2B
    # https://fred.stlouisfed.org/series/Y351RC1A027NBEA
    # https://fred.stlouisfed.org/series/W351RC0A144NBEA
    # https://corporate.vanguard.com/content/dam/corp/research/pdf/how_america_saves_report_2024.pdf
    "traditional_401k_contributions": 482.7e9,
    "roth_401k_contributions": 85.2e9,
    # self_employed_pension_contribution_ald: IRS SOI Publication
    # 1304, Table 1.4 (TY 2023), "Payments to a Keogh plan" —
    # $30.13B (col DM, row "All returns, total"). Includes
    # SEP-IRAs, SIMPLE-IRAs, and traditional Keogh/HR-10 plans.
    # Targeting the ALD (not the input) because policyengine-us
    # applies a min(contributions, SE_income) cap.
    # https://www.irs.gov/statistics/soi-tax-stats-individual-statistical-tables-by-size-of-adjusted-gross-income
    "self_employed_pension_contribution_ald": RETIREMENT_CONTRIBUTION_TARGETS[
        "self_employed_pension_contribution_ald"
    ]["value"],
    # roth_ira_contributions: IRS SOI IRA Accumulation Tables 5 & 6
    # (TY 2022, latest published). Total Roth IRA contributions =
    # $34.95B (10.04M contributors). Direct administrative source.
    # https://www.irs.gov/statistics/soi-tax-stats-accumulation-and-distribution-of-individual-retirement-arrangements
    "roth_ira_contributions": RETIREMENT_CONTRIBUTION_TARGETS["roth_ira_contributions"][
        "value"
    ],
}

ACA_SPENDING_TARGETS = {
    2024: 98e9,
}

ACA_ENROLLMENT_TARGETS = {
    2024: 19_743_689,
}

MEDICAID_SPENDING_TARGETS = {
    2024: 9e11,
    # CMS projects Medicaid spending growth of 7.4% in 2025.
    # Apply that projection to 2024 Medicaid spending of $931.7B.
    # Source: CMS National Health Expenditure projections, 2024-2033.
    2025: 931.7e9 * 1.074,
}

MEDICAID_ENROLLMENT_TARGETS = {
    2024: 72_429_055,
}


def fmt(x):
    if x == -np.inf:
        return "-inf"
    if x == np.inf:
        return "inf"
    if x < 1e3:
        return f"{x:.0f}"
    if x < 1e6:
        return f"{x / 1e3:.0f}k"
    if x < 1e9:
        return f"{x / 1e6:.0f}m"
    return f"{x / 1e9:.1f}bn"


def _parse_constraint_value(value):
    if value == "True":
        return True
    if value == "False":
        return False
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return float(value)
        except (TypeError, ValueError):
            return value


def _apply_constraint(values, operation: str, raw_value: str):
    if operation == "in":
        allowed_values = [part.strip() for part in raw_value.split("|")]
        return np.isin(values, allowed_values)

    value = _parse_constraint_value(raw_value)
    if operation in ("equals", "==", "="):
        return values == value
    if operation in ("greater_than", ">"):
        return values > value
    if operation in ("greater_than_or_equal", ">="):
        return values >= value
    if operation in ("less_than", "<"):
        return values < value
    if operation in ("less_than_or_equal", "<="):
        return values <= value
    if operation in ("not_equals", "!=", "<>"):
        return values != value

    raise ValueError(f"Unsupported stratum constraint operation: {operation}")


def _geo_label_from_ucgid(ucgid_str: str) -> str:
    if ucgid_str in (None, "", "0100000US"):
        return "nation"
    return f"geo/{ucgid_str}"


def _add_liheap_targets_from_db(loss_matrix, targets_list, sim, time_period):
    db_path = STORAGE_FOLDER / "calibration" / "policy_data.db"
    if not db_path.exists():
        return targets_list, loss_matrix

    query = """
        SELECT
            t.target_id,
            t.variable,
            t.value AS target_value,
            s.notes,
            sc.constraint_variable,
            sc.operation,
            sc.value AS constraint_value
        FROM targets t
        JOIN strata s
            ON s.stratum_id = t.stratum_id
        JOIN stratum_constraints sc
            ON sc.stratum_id = s.stratum_id
        WHERE
            t.active = 1
            AND t.reform_id = 0
            AND t.period = ?
            AND s.notes LIKE '%LIHEAP%'
        ORDER BY t.target_id
    """

    with sqlite3.connect(db_path) as conn:
        target_rows = pd.read_sql_query(query, conn, params=[time_period])

    if target_rows.empty:
        return targets_list, loss_matrix

    household_values_cache = {
        "household_weight": sim.calculate("household_weight").values
    }

    def get_household_values(variable: str):
        if variable not in household_values_cache:
            household_values_cache[variable] = sim.calculate(
                variable,
                map_to="household",
            ).values
        return household_values_cache[variable]

    n_households = len(household_values_cache["household_weight"])

    for _, target_df in target_rows.groupby("target_id", sort=False):
        mask = np.ones(n_households, dtype=bool)
        for row in target_df.itertuples(index=False):
            if (
                row.constraint_variable == "ucgid_str"
                and row.constraint_value == "0100000US"
            ):
                continue
            values = get_household_values(row.constraint_variable)
            mask &= _apply_constraint(
                values,
                row.operation,
                row.constraint_value,
            )

        variable = target_df["variable"].iat[0]
        if variable == "household_count":
            metric = mask.astype(float)
        else:
            metric = np.where(mask, get_household_values(variable), 0.0)

        ucgid_constraints = target_df.loc[
            target_df.constraint_variable == "ucgid_str", "constraint_value"
        ]
        geo_label = _geo_label_from_ucgid(
            ucgid_constraints.iat[0] if not ucgid_constraints.empty else None
        )
        label = f"{geo_label}/db/liheap/{variable}"
        loss_matrix[label] = metric
        targets_list.append(target_df["target_value"].iat[0])

    logging.info(
        f"Loaded {target_rows['target_id'].nunique()} LIHEAP targets from the local targets DB"
    )

    return targets_list, loss_matrix


def _best_available_year(targets_by_year: dict, requested_year: int) -> int:
    if not targets_by_year:
        raise ValueError("No target years available")
    eligible_years = [year for year in targets_by_year if year <= requested_year]
    if not eligible_years:
        return min(targets_by_year)
    return max(eligible_years)


def _load_yeared_target_csv(
    prefix: str, requested_year: int
) -> tuple[pd.DataFrame, int]:
    candidates = {}
    for path in CALIBRATION_FOLDER.glob(f"{prefix}_*.csv"):
        suffix = path.stem.removeprefix(f"{prefix}_")
        if suffix.isdigit():
            candidates[int(suffix)] = path

    data_year = _best_available_year(candidates, requested_year)
    return pd.read_csv(candidates[data_year]), data_year


def _load_aca_spending_and_enrollment_targets(
    requested_year: int,
) -> tuple[pd.DataFrame, int]:
    return _load_yeared_target_csv("aca_spending_and_enrollment", requested_year)


def _load_medicaid_enrollment_targets(
    requested_year: int,
) -> tuple[pd.DataFrame, int]:
    return _load_yeared_target_csv("medicaid_enrollment", requested_year)


def _get_aca_national_targets(requested_year: int) -> tuple[float, float, int]:
    targets, data_year = _load_aca_spending_and_enrollment_targets(requested_year)
    if data_year in ACA_SPENDING_TARGETS and data_year in ACA_ENROLLMENT_TARGETS:
        return (
            ACA_SPENDING_TARGETS[data_year],
            ACA_ENROLLMENT_TARGETS[data_year],
            data_year,
        )

    # Newer CMS ACA state files encode monthly total APTC spending by state and
    # APTC enrollment counts. Annualize the spending for the national target.
    return (
        float(targets["spending"].sum() * 12),
        float(targets["enrollment"].sum()),
        data_year,
    )


def _get_medicaid_national_targets(requested_year: int) -> tuple[float, float, int]:
    targets, data_year = _load_medicaid_enrollment_targets(requested_year)
    spending_year = _best_available_year(MEDICAID_SPENDING_TARGETS, data_year)
    enrollment_target = MEDICAID_ENROLLMENT_TARGETS.get(
        data_year, float(targets["enrollment"].sum())
    )
    return (
        MEDICAID_SPENDING_TARGETS[spending_year],
        enrollment_target,
        data_year,
    )


def _add_ctc_targets(loss_matrix, targets_list, sim, time_period):
    """Add legacy national CTC component amount and recipient-count targets."""
    for variable in ("refundable_ctc", "non_refundable_ctc"):
        target = get_national_geography_soi_target(variable, time_period)

        label = f"nation/irs/{variable}"
        loss_matrix[label] = sim.calculate(variable, map_to="household").values
        if any(pd.isna(loss_matrix[label])):
            raise ValueError(f"Missing values for {label}")
        targets_list.append(target["amount"])

        label = f"nation/irs/{variable}_count"
        amount = sim.calculate(variable).values
        loss_matrix[label] = sim.map_result(
            (amount > 0).astype(float),
            "tax_unit",
            "household",
        )
        if any(pd.isna(loss_matrix[label])):
            raise ValueError(f"Missing values for {label}")
        targets_list.append(target["count"])

    return targets_list, loss_matrix


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
        # The current SOI source only exposes taxable-only aggregate targets for
        # mortgage-interest deductions, not the AGI-bin detail used above.
        "mortgage_interest_deductions",
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
            (agi >= row["AGI lower bound"]) * (agi < row["AGI upper bound"]) * filer
        ) > 0

        if row["Filing status"] == "Single":
            mask *= df["filing_status"].values == "SINGLE"
        elif row["Filing status"] == "Married Filing Jointly/Surviving Spouse":
            mask *= np.isin(df["filing_status"].values, ["JOINT", "SURVIVING_SPOUSE"])
        elif row["Filing status"] == "Head of Household":
            mask *= df["filing_status"].values == "HEAD_OF_HOUSEHOLD"
        elif row["Filing status"] == "Married Filing Separately":
            mask *= df["filing_status"].values == "SEPARATE"

        values = df[row["Variable"]].values

        if row["Taxable only"]:
            mask *= taxable

        if row["Count"]:
            values = (values > 0).astype(float)

        agi_range_label = f"{fmt(row['AGI lower bound'])}-{fmt(row['AGI upper bound'])}"
        taxable_label = "taxable" if row["Taxable only"] else "all" + " returns"
        filing_status_label = row["Filing status"]

        variable_label = row["Variable"].replace("_", " ")

        if row["Count"] and not row["Variable"] == "count":
            label = (
                f"nation/irs/{variable_label}/count/AGI in "
                f"{agi_range_label}/{taxable_label}/{filing_status_label}"
            )
        elif row["Variable"] == "count":
            label = (
                f"nation/irs/{variable_label}/count/AGI in "
                f"{agi_range_label}/{taxable_label}/{filing_status_label}"
            )
        else:
            label = (
                f"nation/irs/{variable_label}/total/AGI in "
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

    populations = pd.read_csv(CALIBRATION_FOLDER / "np2023_d5_mid.csv")
    populations = populations[populations.SEX == 0][populations.RACE_HISP == 0]
    populations = (
        populations.groupby("YEAR")
        .sum()[[f"POP_{i}" for i in range(0, 86)]]
        .T[time_period]
        .values
    )  # Array of [age_0_pop, age_1_pop, ...] for the given year
    age = sim.calculate("age").values
    for year in range(len(populations)):
        label = f"nation/census/population_by_age/{year}"
        loss_matrix[label] = sim.map_result(
            (age >= year) * (age < year + 1), "person", "household"
        )
        targets_array.append(populations[year])

    # CBO projections
    # Note: income_tax_positive matches CBO's receipts definition where
    # refundable credit payments in excess of liability are classified as
    # outlays, not negative receipts. See: https://www.cbo.gov/publication/43767

    CBO_PROGRAMS = [
        "income_tax_positive",
        "snap",
        "social_security",
        "ssi",
        "unemployment_compensation",
    ]

    # Mapping from variable name to CBO parameter name (when different)
    CBO_PARAM_NAME_MAP = {
        "income_tax_positive": "income_tax",
    }

    for variable_name in CBO_PROGRAMS:
        label = f"nation/cbo/{variable_name}"
        loss_matrix[label] = sim.calculate(variable_name, map_to="household").values
        if any(loss_matrix[label].isna()):
            raise ValueError(f"Missing values for {label}")
        param_name = CBO_PARAM_NAME_MAP.get(variable_name, variable_name)
        targets_array.append(
            sim.tax_benefit_system.parameters(
                time_period
            ).calibration.gov.cbo._children[param_name]
        )

    # 1. Medicaid Spending
    medicaid_spending_target, medicaid_enrollment_target, _ = (
        _get_medicaid_national_targets(time_period)
    )

    label = "nation/hhs/medicaid_spending"
    loss_matrix[label] = sim.calculate("medicaid", map_to="household").values
    targets_array.append(medicaid_spending_target)

    # 2. Medicaid Enrollment
    label = "nation/hhs/medicaid_enrollment"
    on_medicaid = (
        sim.calculate(
            "medicaid",  # or your enrollee flag
            map_to="person",
            period=time_period,
        ).values
        > 0
    ).astype(int)
    loss_matrix[label] = sim.map_result(on_medicaid, "person", "household")
    targets_array.append(medicaid_enrollment_target)

    # National ACA Spending
    aca_spending_target, aca_enrollment_target, _ = _get_aca_national_targets(
        time_period
    )

    label = "nation/gov/aca_spending"
    loss_matrix[label] = sim.calculate(
        "aca_ptc", map_to="household", period=time_period
    ).values
    targets_array.append(aca_spending_target)

    # National ACA Enrollment (people receiving a PTC)
    label = "nation/gov/aca_enrollment"
    on_ptc = (
        sim.calculate("aca_ptc", map_to="person", period=time_period).values > 0
    ).astype(int)
    loss_matrix[label] = sim.map_result(on_ptc, "person", "household")

    targets_array.append(aca_enrollment_target)

    # Treasury EITC

    loss_matrix["nation/treasury/eitc"] = sim.calculate(
        "eitc", map_to="household"
    ).values
    eitc_spending = (
        sim.tax_benefit_system.parameters.calibration.gov.treasury.tax_expenditures.eitc
    )
    targets_array.append(eitc_spending(time_period))

    # IRS EITC filers and totals by child counts
    eitc_stats = pd.read_csv(CALIBRATION_FOLDER / "eitc.csv")

    eitc_spending_uprating = eitc_spending(time_period) / eitc_spending(2021)
    population = (
        sim.tax_benefit_system.parameters.calibration.gov.census.populations.total
    )
    population_uprating = population(time_period) / population(2021)

    for _, row in eitc_stats.iterrows():
        returns_label = (
            f"nation/irs/eitc/returns/count_children_{row['count_children']}"
        )
        eitc_eligible_children = sim.calculate("eitc_child_count").values
        eitc = sim.calculate("eitc").values
        # IRS Pub 1304 Table 2.5 reports EITC returns by exclusive
        # qualifying-child categories: 0, 1, 2, and "3 or more". Row 3
        # represents 3+ since EITC caps qualifying children at 3.
        if row["count_children"] < 3:
            meets_child_criteria = eitc_eligible_children == row["count_children"]
        else:
            meets_child_criteria = eitc_eligible_children >= row["count_children"]
        loss_matrix[returns_label] = sim.map_result(
            (eitc > 0) * meets_child_criteria,
            "tax_unit",
            "household",
        )
        targets_array.append(row["eitc_returns"] * population_uprating)

        spending_label = (
            f"nation/irs/eitc/spending/count_children_{row['count_children']}"
        )
        loss_matrix[spending_label] = sim.map_result(
            eitc * meets_child_criteria,
            "tax_unit",
            "household",
        )
        targets_array.append(row["eitc_total"] * eitc_spending_uprating)

    targets_array, loss_matrix = _add_ctc_targets(
        loss_matrix,
        targets_array,
        sim,
        time_period,
    )

    # Tax filer counts by AGI band (SOI Table 1.1). Calibrates total
    # filers (not just taxable returns), with granular bands sourced
    # from the latest SOI year <= calibration year to avoid hardcoding
    # stale 2015 values.
    soi_all = pd.read_csv(CALIBRATION_FOLDER / "soi_targets.csv")
    soi_count_rows = soi_all[
        (soi_all["Variable"] == "count")
        & (soi_all["Filing status"] == "All")
        & (~soi_all["Full population"])
        & (~soi_all["Taxable only"])
        & (soi_all["Year"] <= time_period)
    ]
    soi_latest_year = int(soi_count_rows["Year"].max())
    soi_filer_bands = (
        soi_count_rows[soi_count_rows["Year"] == soi_latest_year]
        .sort_values("AGI lower bound")
        .reset_index(drop=True)
    )

    agi_tu = sim.calculate("adjusted_gross_income").values
    is_filer_tu = sim.calculate("tax_unit_is_filer").values > 0

    for _, row in soi_filer_bands.iterrows():
        agi_lower = row["AGI lower bound"]
        agi_upper = row["AGI upper bound"]
        in_band = (agi_tu >= agi_lower) & (agi_tu < agi_upper)
        label = f"nation/soi/filer_count/agi_{fmt(agi_lower)}_{fmt(agi_upper)}"
        loss_matrix[label] = sim.map_result(
            (is_filer_tu & in_band).astype(float),
            "tax_unit",
            "household",
        )
        targets_array.append(row["Value"])

    # Hard-coded totals
    for variable_name, target in HARD_CODED_TOTALS.items():
        label = f"nation/census/{variable_name}"
        loss_matrix[label] = sim.calculate(variable_name, map_to="household").values
        if any(loss_matrix[label].isna()):
            raise ValueError(f"Missing values for {label}")
        targets_array.append(target)

    # Negative household market income total rough estimate from the IRS SOI PUF

    market_income = sim.calculate("household_market_income").values
    loss_matrix["nation/irs/negative_household_market_income_total"] = market_income * (
        market_income < 0
    )
    targets_array.append(-138e9)

    loss_matrix["nation/irs/negative_household_market_income_count"] = (
        market_income < 0
    ).astype(float)
    targets_array.append(3e6)

    # Healthcare spending by age.
    # Each row targets a decade of ages (lower_bound to lower_bound + 9).
    # The top row is treated as unbounded (age >= lower_bound) so the
    # 90+ population is constrained by an age-specific target rather than
    # only by the national total. See issue #768.

    healthcare = pd.read_csv(CALIBRATION_FOLDER / "healthcare_spending.csv")
    top_age_lower_bound = int(healthcare["age_10_year_lower_bound"].max())

    for _, row in healthcare.iterrows():
        age_lower_bound = int(row["age_10_year_lower_bound"])
        is_top_bucket = age_lower_bound == top_age_lower_bound
        if is_top_bucket:
            in_age_range = age >= age_lower_bound
            label_suffix = f"age_{age_lower_bound}_plus"
        else:
            in_age_range = (age >= age_lower_bound) * (age < age_lower_bound + 10)
            label_suffix = f"age_{age_lower_bound}_to_{age_lower_bound + 9}"
        for expense_type in [
            "health_insurance_premiums_without_medicare_part_b",
            "over_the_counter_health_expenses",
            "other_medical_expenses",
            "medicare_part_b_premiums",
        ]:
            label = f"nation/census/{expense_type}/{label_suffix}"
            value = sim.calculate(expense_type).values
            loss_matrix[label] = sim.map_result(
                in_age_range * value, "person", "household"
            )
            targets_array.append(row[expense_type])

    # AGI by SPM threshold totals

    spm_threshold_agi = pd.read_csv(CALIBRATION_FOLDER / "spm_threshold_agi.csv")

    for _, row in spm_threshold_agi.iterrows():
        spm_unit_agi = sim.calculate("adjusted_gross_income", map_to="spm_unit").values
        spm_threshold = sim.calculate("spm_unit_spm_threshold").values
        in_threshold_range = (spm_threshold >= row["lower_spm_threshold"]) * (
            spm_threshold < row["upper_spm_threshold"]
        )
        label = f"nation/census/agi_in_spm_threshold_decile_{int(row['decile'])}"
        loss_matrix[label] = sim.map_result(
            in_threshold_range * spm_unit_agi, "spm_unit", "household"
        )
        targets_array.append(row["adjusted_gross_income"])

        label = f"nation/census/count_in_spm_threshold_decile_{int(row['decile'])}"
        loss_matrix[label] = sim.map_result(in_threshold_range, "spm_unit", "household")
        targets_array.append(row["count"])

    # Population by state and population under 5 by state

    state_population = pd.read_csv(CALIBRATION_FOLDER / "population_by_state.csv")

    for _, row in state_population.iterrows():
        in_state = sim.calculate("state_code", map_to="person") == row["state"]
        label = f"state/census/population_by_state/{row['state']}"
        loss_matrix[label] = sim.map_result(in_state, "person", "household")
        targets_array.append(row["population"])

        under_5 = sim.calculate("age").values < 5
        in_state_under_5 = in_state * under_5
        label = f"state/census/population_under_5_by_state/{row['state']}"
        loss_matrix[label] = sim.map_result(in_state_under_5, "person", "household")
        targets_array.append(row["population_under_5"])

    age = sim.calculate("age").values
    infants = (age >= 0) & (age < 1)
    label = "nation/census/infants"
    loss_matrix[label] = sim.map_result(infants, "person", "household")
    # Total number of infants in the 1 Year ACS
    INFANTS_2023 = 3_491_679
    INFANTS_2022 = 3_437_933
    # Assume infant population grows at the same rate from 2023.
    infants_2024 = INFANTS_2023 * (INFANTS_2023 / INFANTS_2022)
    targets_array.append(infants_2024)

    networth = sim.calculate("net_worth").values
    label = "nation/net_worth/total"
    loss_matrix[label] = networth
    # Federal Reserve estimate of $160 trillion in 2024Q4
    # https://fred.stlouisfed.org/series/BOGZ1FL192090005Q
    NET_WORTH_2024 = 160e12
    targets_array.append(NET_WORTH_2024)

    # SALT tax expenditure targeting

    _add_tax_expenditure_targets(dataset, time_period, sim, loss_matrix, targets_array)

    if any(loss_matrix.isna().sum() > 0):
        raise ValueError("Some targets are missing from the loss matrix")

    if any(pd.isna(targets_array)):
        raise ValueError("Some targets are missing from the targets array")

    # SSN Card Type calibration
    for card_type_str in ["NONE"]:  # SSN card types as strings
        ssn_type_mask = sim.calculate("ssn_card_type").values == card_type_str

        # Overall count by SSN card type
        label = f"nation/ssa/ssn_card_type_{card_type_str.lower()}_count"
        loss_matrix[label] = sim.map_result(ssn_type_mask, "person", "household")

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
    spending_by_state, _ = _load_aca_spending_and_enrollment_targets(time_period)
    # Monthly to yearly
    spending_by_state["spending"] = spending_by_state["spending"] * 12
    # Adjust to match national target
    spending_by_state["spending"] = spending_by_state["spending"] * (
        aca_spending_target / spending_by_state["spending"].sum()
    )

    for _, row in spending_by_state.iterrows():
        # Households located in this state
        in_state = (
            sim.calculate("state_code", map_to="household").values == row["state"]
        )

        # ACA PTC amounts for every household at time_period.
        aca_value = sim.calculate(
            "aca_ptc", map_to="household", period=time_period
        ).values

        # Add a loss-matrix entry and matching target. Prefix `state/`
        # so `reweight()` correctly classifies this as a state-level
        # (non-national) target via `startswith("nation/")`.
        label = f"state/irs/aca_spending/{row['state'].lower()}"
        loss_matrix[label] = aca_value * in_state
        annual_target = row["spending"]
        if any(loss_matrix[label].isna()):
            raise ValueError(f"Missing values for {label}")
        targets_array.append(annual_target)

    # Marketplace enrollment by state (targets in thousands)
    enrollment_by_state, _ = _load_aca_spending_and_enrollment_targets(time_period)

    # One-time pulls so we don’t re-compute inside the loop
    state_person = sim.calculate("state_code", map_to="person").values

    # Flag people in households that actually receive any PTC (> 0)
    in_tax_unit_with_aca = (
        sim.calculate("aca_ptc", map_to="person", period=time_period).values > 0
    )
    is_aca_eligible = sim.calculate(
        "is_aca_ptc_eligible", map_to="person", period=time_period
    ).values
    is_enrolled = in_tax_unit_with_aca & is_aca_eligible

    for _, row in enrollment_by_state.iterrows():
        # People who both live in the state and have marketplace coverage
        in_state = state_person == row["state"]
        in_state_enrolled = in_state & is_enrolled

        label = f"state/irs/aca_enrollment/{row['state'].lower()}"
        loss_matrix[label] = sim.map_result(in_state_enrolled, "person", "household")
        if any(loss_matrix[label].isna()):
            raise ValueError(f"Missing values for {label}")

        # Convert to thousands for the target
        targets_array.append(row["enrollment"])

    # Medicaid enrollment by state

    enrollment_by_state, _ = _load_medicaid_enrollment_targets(time_period)

    # One-time pulls so we don’t re-compute inside the loop
    state_person = sim.calculate("state_code", map_to="person").values

    # Flag people in households that actually receive medicaid
    has_medicaid = sim.calculate(
        "medicaid_enrolled", map_to="person", period=time_period
    )
    is_medicaid_eligible = sim.calculate(
        "is_medicaid_eligible", map_to="person", period=time_period
    ).values
    is_enrolled = has_medicaid & is_medicaid_eligible

    for _, row in enrollment_by_state.iterrows():
        # People who both live in the state and have marketplace coverage
        in_state = state_person == row["state"]
        in_state_enrolled = in_state & is_enrolled

        # Prefix `state/` so `reweight()` correctly classifies this as a
        # state-level (non-national) target — matches the sibling
        # ACA enrollment label on line 849.
        label = f"state/irs/medicaid_enrollment/{row['state'].lower()}"
        loss_matrix[label] = sim.map_result(in_state_enrolled, "person", "household")
        if any(loss_matrix[label].isna()):
            raise ValueError(f"Missing values for {label}")

        # Convert to thousands for the target
        targets_array.append(row["enrollment"])

        logging.info(
            f"Targeting Medicaid enrollment for {row['state']} "
            f"with target {row['enrollment']:.0f}k"
        )

    # State 10-year age targets

    age_targets = pd.read_csv(CALIBRATION_FOLDER / "age_state.csv")

    for state in age_targets.GEO_NAME.unique():
        state_mask = state_person == state
        for age_range in age_targets.columns[2:]:
            if "+" in age_range:
                # Handle the "85+" case
                age_lower_bound = int(age_range.replace("+", ""))
                age_upper_bound = np.inf
            else:
                age_lower_bound, age_upper_bound = map(int, age_range.split("-"))

            age_mask = (age >= age_lower_bound) & (age <= age_upper_bound)
            label = f"state/census/age/{state}/{age_range}"
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

    snap_state_target_names, snap_state_targets = _add_snap_state_targets(sim)
    targets_array.extend(snap_state_targets)
    loss_matrix = _add_snap_metric_columns(loss_matrix, sim)

    targets_array, loss_matrix = _add_liheap_targets_from_db(
        loss_matrix, targets_array, sim, time_period
    )

    del sim, df
    gc.collect()

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
        income_tax_r = simulation.calculate("income_tax", map_to="household").values

        # Compute the tax expenditure (TE) values.
        te_values = income_tax_r - income_tax_b

        # Record the TE difference and the corresponding target value.
        loss_matrix[f"nation/jct/{deduction}_expenditure"] = te_values
        targets_array.append(target)


def get_agi_band_label(lower: float, upper: float) -> str:
    """Get the label for the AGI band based on lower and upper bounds."""
    if lower <= 0:
        return f"-inf_{int(upper)}"
    elif np.isposinf(upper):
        return f"{int(lower)}_inf"
    else:
        return f"{int(lower)}_{int(upper)}"


def _add_agi_state_targets():
    """
    Create an aggregate target matrix for the appropriate geographic area
    """

    soi_targets = pd.read_csv(CALIBRATION_FOLDER / "agi_state.csv")

    soi_targets["target_name"] = (
        "state/"
        + soi_targets["GEO_NAME"]
        + "/"
        + soi_targets["VARIABLE"]
        + "/"
        + soi_targets.apply(
            lambda r: get_agi_band_label(r["AGI_LOWER_BOUND"], r["AGI_UPPER_BOUND"]),
            axis=1,
        )
    )

    target_names = soi_targets["target_name"].tolist()
    target_values = soi_targets["VALUE"].astype(float).tolist()
    return target_names, target_values


def _add_agi_metric_columns(
    loss_matrix: pd.DataFrame,
    sim,
):
    """
    Add AGI metric columns to the loss_matrix.
    """
    soi_targets = pd.read_csv(CALIBRATION_FOLDER / "agi_state.csv")

    agi = sim.calculate("adjusted_gross_income").values
    state = sim.calculate("state_code", map_to="person").values
    state = sim.map_result(state, "person", "tax_unit", how="value_from_first_person")

    for _, r in soi_targets.iterrows():
        lower, upper = r.AGI_LOWER_BOUND, r.AGI_UPPER_BOUND
        band = get_agi_band_label(lower, upper)

        in_state = state == r.GEO_NAME
        # Use the same [lower, upper) boundary convention as the main SOI
        # loop in build_loss_matrix() (the SOI targets use half-open bands
        # starting at the lower bound).
        in_band = (agi >= lower) & (agi < upper)

        if r.IS_COUNT:
            metric = (in_state & in_band & (agi > 0)).astype(float)
        else:
            metric = np.where(in_state & in_band, agi, 0.0)

        metric = sim.map_result(metric, "tax_unit", "household")

        col_name = f"state/{r.GEO_NAME}/{r.VARIABLE}/{band}"
        loss_matrix[col_name] = metric

    return loss_matrix


def _add_state_real_estate_taxes(loss_matrix, targets_list, sim):
    """
    Add state real estate taxes to the loss matrix and targets list.
    """
    # Read the real estate taxes data
    real_estate_taxes_targets = pd.read_csv(
        CALIBRATION_FOLDER / "real_estate_taxes_by_state_acs.csv"
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

    targets_list.extend(real_estate_taxes_targets["real_estate_taxes_bn"].tolist())

    real_estate_taxes = sim.calculate("real_estate_taxes", map_to="household").values
    state = sim.calculate("state_code", map_to="household").values

    for _, r in real_estate_taxes_targets.iterrows():
        in_state = (state == r["state_code"]).astype(float)
        label = f"state/real_estate_taxes/{r['state_code']}"
        loss_matrix[label] = real_estate_taxes * in_state

    return targets_list, loss_matrix


def _add_snap_state_targets(sim):
    """
    Add snap targets at the state level, adjusted in aggregate to the sim
    """
    snap_targets = pd.read_csv(CALIBRATION_FOLDER / "snap_state.csv")
    time_period = sim.default_calculation_period

    national_cost_target = sim.tax_benefit_system.parameters(
        time_period
    ).calibration.gov.cbo._children["snap"]
    ratio = snap_targets[["Cost"]].sum().values[0] / national_cost_target
    snap_targets[["CostAdj"]] = snap_targets[["Cost"]] / ratio
    assert np.round(snap_targets[["CostAdj"]].sum().values[0]) == national_cost_target

    cost_targets = snap_targets.copy()[["GEO_ID", "CostAdj"]]
    cost_targets["target_name"] = cost_targets["GEO_ID"].str[-4:] + "/snap-cost"

    hh_targets = snap_targets.copy()[["GEO_ID", "Households"]]
    hh_targets["target_name"] = snap_targets["GEO_ID"].str[-4:] + "/snap-hhs"

    target_names = (
        cost_targets["target_name"].tolist() + hh_targets["target_name"].tolist()
    )
    target_values = (
        cost_targets["CostAdj"].astype(float).tolist()
        + hh_targets["Households"].astype(float).tolist()
    )
    return target_names, target_values


def _add_snap_metric_columns(
    loss_matrix: pd.DataFrame,
    sim,
):
    """
    Add SNAP metric columns to the loss_matrix.
    """
    snap_targets = pd.read_csv(CALIBRATION_FOLDER / "snap_state.csv")

    snap_cost = sim.calculate("snap_reported", map_to="household").values
    snap_hhs = (sim.calculate("snap_reported", map_to="household").values > 0).astype(
        int
    )

    state = sim.calculate("state_code", map_to="person").values
    state = sim.map_result(state, "person", "household", how="value_from_first_person")
    state_fips = pd.Series(state).apply(lambda s: STATE_ABBR_TO_FIPS[s])

    for _, r in snap_targets.iterrows():
        in_state = state_fips == r.GEO_ID[-2:]
        metric = np.where(in_state, snap_cost, 0.0)
        col_name = f"{r.GEO_ID[-4:]}/snap-cost"
        loss_matrix[col_name] = metric

    for _, r in snap_targets.iterrows():
        in_state = state_fips == r.GEO_ID[-2:]
        metric = np.where(in_state, snap_hhs, 0.0)
        col_name = f"{r.GEO_ID[-4:]}/snap-hhs"
        loss_matrix[col_name] = metric

    return loss_matrix


def print_reweighting_diagnostics(optimised_weights, loss_matrix, targets_array, label):
    # Convert all inputs to NumPy arrays right at the start
    optimised_weights_np = (
        optimised_weights.numpy()
        if hasattr(optimised_weights, "numpy")
        else np.asarray(optimised_weights)
    )
    loss_matrix_np = (
        loss_matrix.numpy()
        if hasattr(loss_matrix, "numpy")
        else np.asarray(loss_matrix)
    )
    targets_array_np = (
        targets_array.numpy()
        if hasattr(targets_array, "numpy")
        else np.asarray(targets_array)
    )

    logging.info(f"\n\n---{label}: reweighting quick diagnostics----\n")
    logging.info(
        f"{np.sum(optimised_weights_np == 0)} are zero, "
        f"{np.sum(optimised_weights_np != 0)} weights are nonzero"
    )

    # All subsequent calculations use the guaranteed NumPy versions
    estimate = optimised_weights_np @ loss_matrix_np

    rel_error = (((estimate - targets_array_np) + 1) / (targets_array_np + 1)) ** 2
    within_10_percent_mask = np.abs(estimate - targets_array_np) <= (
        0.10 * np.abs(targets_array_np)
    )
    percent_within_10 = np.mean(within_10_percent_mask) * 100
    logging.info(
        f"rel_error: min: {np.min(rel_error):.2f}\n"
        f"max: {np.max(rel_error):.2f}\n"
        f"mean: {np.mean(rel_error):.2f}\n"
        f"median: {np.median(rel_error):.2f}\n"
        f"Within 10% of target: {percent_within_10:.2f}%"
    )
    logging.info("Relative error over 100% for:")
    for i in np.where(rel_error > 1)[0]:
        # Keep this check, as Tensors won't have a .columns attribute
        if hasattr(loss_matrix, "columns"):
            logging.info(f"target_name: {loss_matrix.columns[i]}")
        else:
            logging.info(f"target_index: {i}")

        logging.info(f"target_value: {targets_array_np[i]}")
        logging.info(f"estimate_value: {estimate[i]}")
        logging.info(f"has rel_error: {rel_error[i]:.2f}\n")
    logging.info("---End of reweighting quick diagnostics------")
    return percent_within_10
