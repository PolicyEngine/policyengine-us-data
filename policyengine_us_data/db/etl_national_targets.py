import warnings

from sqlmodel import Session, create_engine
import pandas as pd

from policyengine_us_data.storage import STORAGE_FOLDER
from policyengine_us_data.db.create_database_tables import (
    Stratum,
    StratumConstraint,
    Target,
)
from policyengine_us_data.storage.calibration_targets.soi_metadata import (
    RETIREMENT_CONTRIBUTION_TARGETS,
)
from policyengine_us_data.utils.census_spm import (
    build_census_spm_capped_housing_subsidy_target,
)
from policyengine_us_data.utils.db import (
    DEFAULT_YEAR,
    etl_argparser,
)


def extract_national_targets(year: int = DEFAULT_YEAR):
    """
    Extract national calibration targets from various sources.

    Parameters
    ----------
    year : int
        Target year for calibration data.

    Returns
    -------
    dict
        Dictionary containing:
        - direct_sum_targets: Variables that can be summed directly
        - tax_filer_targets: Tax-related variables requiring filer constraint
        - tax_expenditure_targets: Variables targeted via repeal-based tax expenditures
        - conditional_count_targets: Enrollment counts requiring constraints
        - cbo_targets: List of CBO projection targets
        - treasury_targets: List of Treasury/JCT targets
        - time_period: The target year
    """
    from policyengine_us import CountryTaxBenefitSystem

    time_period = year
    print(f"Using time_period: {time_period}")

    tax_benefit_system = CountryTaxBenefitSystem()

    # Hardcoded dollar targets are specific to 2024 and should be
    # labeled as such.  Only CBO/Treasury parameter lookups use the
    # dynamic time_period derived from the dataset.
    HARDCODED_YEAR = 2024
    if time_period != HARDCODED_YEAR:
        warnings.warn(
            f"Dataset year ({time_period}) != HARDCODED_YEAR "
            f"({HARDCODED_YEAR}). Hardcoded dollar targets may "
            f"be stale and need re-sourcing."
        )

    # Separate tax-related targets that need filer constraint
    tax_filer_targets = []

    # These JCT values are tax expenditures, not baseline deduction totals.
    # They must be matched against repeal-based income tax deltas in the
    # unified calibration path.
    raw_tax_expenditure_targets = [
        {
            "reform_id": 1,
            "variable": "salt_deduction",
            "value": 21.247e9,
            "source": "Joint Committee on Taxation",
            "notes": "SALT deduction tax expenditure",
            "year": HARDCODED_YEAR,
        },
        {
            "reform_id": 2,
            "variable": "medical_expense_deduction",
            "value": 11.4e9,
            "source": "Joint Committee on Taxation",
            "notes": "Medical expense deduction tax expenditure",
            "year": HARDCODED_YEAR,
        },
        {
            "reform_id": 3,
            "variable": "charitable_deduction",
            "value": 65.301e9,
            "source": "Joint Committee on Taxation",
            "notes": "Charitable deduction tax expenditure",
            "year": HARDCODED_YEAR,
        },
        {
            "reform_id": 4,
            "variable": "deductible_mortgage_interest",
            "value": 24.8e9,
            "source": "Joint Committee on Taxation",
            "notes": "Mortgage interest deduction tax expenditure",
            "year": HARDCODED_YEAR,
        },
        {
            "reform_id": 5,
            "variable": "qualified_business_income_deduction",
            "value": 63.1e9,
            "source": "Joint Committee on Taxation",
            "notes": "QBI deduction tax expenditure",
            "year": HARDCODED_YEAR,
        },
    ]
    tax_expenditure_targets = [{**target} for target in raw_tax_expenditure_targets]

    direct_sum_targets = [
        {
            "variable": "alimony_income",
            "value": 13e9,
            "source": "Survey-reported (post-TCJA grandfathered)",
            "notes": "Alimony received - survey reported, not tax-filer restricted",
            "year": HARDCODED_YEAR,
        },
        {
            "variable": "alimony_expense",
            "value": 13e9,
            "source": "Survey-reported (post-TCJA grandfathered)",
            "notes": "Alimony paid - survey reported, not tax-filer restricted",
            "year": HARDCODED_YEAR,
        },
        {
            "variable": "medicaid",
            "value": 871.7e9,
            "source": "https://www.cms.gov/files/document/highlights.pdf",
            "notes": "CMS 2023 highlights document - total Medicaid spending",
            "year": HARDCODED_YEAR,
        },
        {
            "variable": "net_worth",
            "value": 160e12,
            "source": "Federal Reserve SCF",
            "notes": "Total household net worth",
            "year": HARDCODED_YEAR,
        },
        {
            "variable": "health_insurance_premiums_without_medicare_part_b",
            "value": 385e9,
            "source": "MEPS/NHEA",
            "notes": "Health insurance premiums excluding Medicare Part B",
            "year": HARDCODED_YEAR,
        },
        {
            "variable": "other_medical_expenses",
            "value": 278e9,
            "source": "MEPS/NHEA",
            "notes": "Out-of-pocket medical expenses",
            "year": HARDCODED_YEAR,
        },
        {
            "variable": "medicare_part_b_premiums",
            "value": 112e9,
            "source": "CMS Medicare data",
            "notes": "Medicare Part B premium payments",
            "year": HARDCODED_YEAR,
        },
        {
            "variable": "over_the_counter_health_expenses",
            "value": 72e9,
            "source": "Consumer Expenditure Survey",
            "notes": "OTC health products and supplies",
            "year": HARDCODED_YEAR,
        },
        {
            "variable": "child_support_expense",
            "value": 33e9,
            "source": "Census Bureau",
            "notes": "Child support payments",
            "year": HARDCODED_YEAR,
        },
        {
            "variable": "child_support_received",
            "value": 33e9,
            "source": "Census Bureau",
            "notes": "Child support received",
            "year": HARDCODED_YEAR,
        },
        {
            "variable": "spm_unit_capped_work_childcare_expenses",
            "value": 348e9,
            "source": "Census Bureau SPM",
            "notes": "Work and childcare expenses for SPM",
            "year": HARDCODED_YEAR,
        },
        build_census_spm_capped_housing_subsidy_target(HARDCODED_YEAR),
        {
            "variable": "tanf",
            "value": 9e9,
            "source": "HHS/ACF",
            "notes": "TANF cash assistance",
            "year": HARDCODED_YEAR,
        },
        {
            "variable": "real_estate_taxes",
            "value": 500e9,
            "source": "Census Bureau",
            "notes": "Property taxes paid",
            "year": HARDCODED_YEAR,
        },
        {
            "variable": "rent",
            "value": 735e9,
            "source": "Census Bureau/BLS",
            "notes": "Rental payments",
            "year": HARDCODED_YEAR,
        },
        {
            "variable": "tip_income",
            "value": 53.2e9,
            "source": "IRS Form W-2 Box 7 statistics",
            "notes": "Social security tips uprated 40% to account for underreporting",
            "year": HARDCODED_YEAR,
        },
        # SSA benefit-type totals derived from trust fund data and
        # SSA fact sheet type shares
        {
            "variable": "social_security_retirement",
            "value": 1_060e9,
            "source": "https://www.ssa.gov/OACT/STATS/table4a3.html",
            "notes": "~73% of total OASDI ($1,452B CBO projection)",
            "year": HARDCODED_YEAR,
        },
        {
            "variable": "social_security_disability",
            "value": 148e9,
            "source": "https://www.ssa.gov/OACT/STATS/table4a3.html",
            "notes": "~10.2% of total OASDI (disabled workers)",
            "year": HARDCODED_YEAR,
        },
        {
            "variable": "social_security_survivors",
            "value": 160e9,
            "source": "https://www.ssa.gov/OACT/FACTS/",
            "notes": "~11.0% of total OASDI (widows, children of deceased)",
            "year": HARDCODED_YEAR,
        },
        {
            "variable": "social_security_dependents",
            "value": 84e9,
            "source": "https://www.ssa.gov/OACT/FACTS/",
            "notes": "~5.8% of total OASDI (spouses/children of retired+disabled)",
            "year": HARDCODED_YEAR,
        },
        # Retirement contribution targets — see issue #553
        {
            "variable": "traditional_ira_contributions",
            "value": RETIREMENT_CONTRIBUTION_TARGETS["traditional_ira_contributions"][
                "value"
            ],
            "source": RETIREMENT_CONTRIBUTION_TARGETS["traditional_ira_contributions"][
                "source"
            ],
            "notes": RETIREMENT_CONTRIBUTION_TARGETS["traditional_ira_contributions"][
                "notes"
            ],
            "year": HARDCODED_YEAR,
        },
        {
            "variable": "traditional_401k_contributions",
            "value": 482.7e9,
            "source": "https://fred.stlouisfed.org/series/Y351RC1A027NBEA",
            "notes": "BEA/FRED employee DC deferrals ($567.9B) x 85% traditional share (Vanguard HAS 2024)",
            "year": HARDCODED_YEAR,
        },
        {
            "variable": "roth_401k_contributions",
            "value": 85.2e9,
            "source": "https://fred.stlouisfed.org/series/Y351RC1A027NBEA",
            "notes": "BEA/FRED employee DC deferrals ($567.9B) x 15% Roth share (Vanguard HAS 2024)",
            "year": HARDCODED_YEAR,
        },
        {
            "variable": "self_employed_pension_contribution_ald",
            "value": RETIREMENT_CONTRIBUTION_TARGETS[
                "self_employed_pension_contribution_ald"
            ]["value"],
            "source": RETIREMENT_CONTRIBUTION_TARGETS[
                "self_employed_pension_contribution_ald"
            ]["source"],
            "notes": RETIREMENT_CONTRIBUTION_TARGETS[
                "self_employed_pension_contribution_ald"
            ]["notes"],
            "year": HARDCODED_YEAR,
        },
        {
            "variable": "roth_ira_contributions",
            "value": RETIREMENT_CONTRIBUTION_TARGETS["roth_ira_contributions"]["value"],
            "source": RETIREMENT_CONTRIBUTION_TARGETS["roth_ira_contributions"][
                "source"
            ],
            "notes": RETIREMENT_CONTRIBUTION_TARGETS["roth_ira_contributions"]["notes"],
            "year": HARDCODED_YEAR,
        },
    ]

    # Conditional count targets - these need strata with constraints
    # Store with actual source year
    conditional_count_targets = [
        {
            "constraint_variable": "medicaid",
            "person_count": 72_429_055,
            "source": "CMS/HHS administrative data",
            "notes": "Medicaid enrollment count",
            "year": HARDCODED_YEAR,
        },
        {
            "constraint_variable": "aca_ptc",
            "person_count": 19_743_689,
            "source": "CMS marketplace data",
            "notes": "ACA Premium Tax Credit recipients",
            "year": HARDCODED_YEAR,
        },
        {
            "constraint_variable": "spm_unit_energy_subsidy_reported",
            "target_variable": "household_count",
            "household_count": 5_939_605,
            "source": "https://liheappm.acf.gov/sites/default/files/private/congress/profiles/2023/FY2023AllStates%28National%29Profile-508Compliant.pdf",
            "notes": "LIHEAP total households served by state programs",
            "year": 2023,
        },
        {
            "constraint_variable": "spm_unit_energy_subsidy_reported",
            "target_variable": "household_count",
            "household_count": 5_876_646,
            "source": "https://liheappm.acf.gov/sites/default/files/private/congress/profiles/2024/FY2024_AllStates%28National%29_Profile.pdf",
            "notes": "LIHEAP total households served by state programs",
            "year": 2024,
        },
    ]

    # Add SSN card type NONE targets for multiple years
    # Based on loss.py lines 445-460
    ssn_none_targets_by_year = [
        {
            "constraint_variable": "ssn_card_type",
            "constraint_value": "NONE",  # Need to specify the value we're checking for
            "person_count": 11.0e6,
            "source": "DHS Office of Homeland Security Statistics",
            "notes": "Undocumented population estimate for Jan 1, 2022",
            "year": 2022,
        },
        {
            "constraint_variable": "ssn_card_type",
            "constraint_value": "NONE",
            "person_count": 12.2e6,
            "source": "Center for Migration Studies ACS-based residual estimate",
            "notes": "Undocumented population estimate (published May 2025)",
            "year": 2023,
        },
        {
            "constraint_variable": "ssn_card_type",
            "constraint_value": "NONE",
            "person_count": 13.0e6,
            "source": "Reuters synthesis of experts",
            "notes": "Undocumented population central estimate (~13-14 million)",
            "year": 2024,
        },
        {
            "constraint_variable": "ssn_card_type",
            "constraint_value": "NONE",
            "person_count": 13.0e6,
            "source": "Reuters synthesis of experts",
            "notes": "Same midpoint carried forward - CBP data show 95% drop in border apprehensions",
            "year": 2025,
        },
    ]

    conditional_count_targets.extend(ssn_none_targets_by_year)

    # CBO projection targets - use time_period derived from dataset
    cbo_vars = [
        # Note: income_tax_positive matches CBO's receipts definition
        # where refundable credit payments in excess of liability are
        # classified as outlays, not negative receipts. See:
        # https://www.cbo.gov/publication/43767
        "income_tax_positive",
        "snap",
        "social_security",
        "ssi",
        "unemployment_compensation",
    ]

    # Mapping from target variable to CBO parameter name (when different)
    cbo_param_name_map = {
        "income_tax_positive": "income_tax",  # CBO param is income_tax
    }

    cbo_targets = []
    for variable_name in cbo_vars:
        param_name = cbo_param_name_map.get(variable_name, variable_name)
        try:
            value = tax_benefit_system.parameters(
                time_period
            ).calibration.gov.cbo._children[param_name]
            cbo_targets.append(
                {
                    "variable": variable_name,
                    "value": float(value),
                    "source": "CBO Budget Projections",
                    "notes": f"CBO projection for {variable_name}",
                    "year": time_period,
                }
            )
        except (KeyError, AttributeError) as e:
            print(
                f"Warning: Could not extract CBO parameter for "
                f"{variable_name} (param: {param_name}): {e}"
            )

    # Treasury/JCT targets (EITC) - use time_period derived from dataset
    try:
        eitc_value = tax_benefit_system.parameters.calibration.gov.treasury.tax_expenditures.eitc(
            time_period
        )
        treasury_targets = [
            {
                "variable": "eitc",
                "value": float(eitc_value),
                "source": "Treasury/JCT Tax Expenditures",
                "notes": "EITC tax expenditure",
                "year": time_period,
            }
        ]
    except (KeyError, AttributeError) as e:
        print(f"Warning: Could not extract Treasury EITC parameter: {e}")
        treasury_targets = []

    return {
        "direct_sum_targets": direct_sum_targets,
        "tax_filer_targets": tax_filer_targets,
        "tax_expenditure_targets": tax_expenditure_targets,
        "conditional_count_targets": conditional_count_targets,
        "cbo_targets": cbo_targets,
        "treasury_targets": treasury_targets,
        "time_period": time_period,
    }


def transform_national_targets(raw_targets):
    """
    Transform extracted targets into standardized format for loading.

    Parameters
    ----------
    raw_targets : dict
        Dictionary from extract_national_targets()

    Returns
    -------
    tuple
        (direct_targets_df, tax_filer_df, tax_expenditure_df, conditional_targets)
        - direct_targets_df: DataFrame with direct sum targets
        - tax_filer_df: DataFrame with tax-related targets needing filer constraint
        - tax_expenditure_df: DataFrame with reform-based tax expenditure targets
        - conditional_targets: List of conditional count targets
    """

    # Process direct sum targets (non-tax items and some CBO items)
    # Note: income_tax_positive from CBO and eitc from Treasury need
    # filer constraint
    cbo_non_tax = [
        t for t in raw_targets["cbo_targets"] if t["variable"] != "income_tax_positive"
    ]
    cbo_tax = [
        t for t in raw_targets["cbo_targets"] if t["variable"] == "income_tax_positive"
    ]

    all_direct_targets = raw_targets["direct_sum_targets"] + cbo_non_tax

    # Tax-related targets that need filer constraint
    all_tax_filer_targets = (
        raw_targets["tax_filer_targets"]
        + cbo_tax
        + raw_targets["treasury_targets"]  # EITC
    )

    direct_df = (
        pd.DataFrame(all_direct_targets) if all_direct_targets else pd.DataFrame()
    )
    tax_filer_df = (
        pd.DataFrame(all_tax_filer_targets) if all_tax_filer_targets else pd.DataFrame()
    )
    tax_expenditure_df = (
        pd.DataFrame(raw_targets["tax_expenditure_targets"])
        if raw_targets["tax_expenditure_targets"]
        else pd.DataFrame()
    )

    # Conditional targets stay as list for special processing
    conditional_targets = raw_targets["conditional_count_targets"]

    return direct_df, tax_filer_df, tax_expenditure_df, conditional_targets


def load_national_targets(
    direct_targets_df,
    tax_filer_df,
    tax_expenditure_df,
    conditional_targets,
):
    """
    Load national targets into the database.

    Parameters
    ----------
    direct_targets_df : pd.DataFrame
        DataFrame with direct sum target data
    tax_filer_df : pd.DataFrame
        DataFrame with tax-related targets needing filer constraint
    tax_expenditure_df : pd.DataFrame
        DataFrame with reform-based tax expenditure targets
    conditional_targets : list
        List of conditional count targets requiring strata
    """

    DATABASE_URL = f"sqlite:///{STORAGE_FOLDER / 'calibration' / 'policy_data.db'}"
    engine = create_engine(DATABASE_URL)

    with Session(engine) as session:
        # Get the national stratum
        us_stratum = (
            session.query(Stratum).filter(Stratum.parent_stratum_id.is_(None)).first()
        )

        if not us_stratum:
            raise ValueError(
                "National stratum not found. Run create_initial_strata.py first."
            )

        # Process direct sum targets
        for _, target_data in direct_targets_df.iterrows():
            target_year = target_data["year"]
            # Check if target already exists
            existing_target = (
                session.query(Target)
                .filter(
                    Target.stratum_id == us_stratum.stratum_id,
                    Target.variable == target_data["variable"],
                    Target.period == target_year,
                )
                .first()
            )

            # Combine source info into notes
            notes_parts = []
            if pd.notna(target_data.get("notes")):
                notes_parts.append(target_data["notes"])
            notes_parts.append(f"Source: {target_data.get('source', 'Unknown')}")
            combined_notes = " | ".join(notes_parts)

            if existing_target:
                # Update existing target
                existing_target.value = target_data["value"]
                existing_target.notes = combined_notes
                existing_target.source = "PolicyEngine"
                print(f"Updated target: {target_data['variable']}")
            else:
                # Create new target
                target = Target(
                    stratum_id=us_stratum.stratum_id,
                    variable=target_data["variable"],
                    period=target_year,
                    value=target_data["value"],
                    active=True,
                    source="PolicyEngine",
                    notes=combined_notes,
                )
                session.add(target)
                print(f"Added target: {target_data['variable']}")

        # Process tax-related targets that need filer constraint
        if not tax_filer_df.empty:
            # Get or create the national filer stratum
            national_filer_stratum = (
                session.query(Stratum)
                .filter(
                    Stratum.parent_stratum_id == us_stratum.stratum_id,
                    Stratum.notes == "United States - Tax Filers",
                )
                .first()
            )

            if not national_filer_stratum:
                # Create national filer stratum
                national_filer_stratum = Stratum(
                    parent_stratum_id=us_stratum.stratum_id,
                    notes="United States - Tax Filers",
                )
                national_filer_stratum.constraints_rel = [
                    StratumConstraint(
                        constraint_variable="tax_unit_is_filer",
                        operation="==",
                        value="1",
                    )
                ]
                session.add(national_filer_stratum)
                session.flush()
                print("Created national filer stratum")

            # Add tax-related targets to filer stratum
            for _, target_data in tax_filer_df.iterrows():
                target_year = target_data["year"]
                # Check if target already exists
                existing_target = (
                    session.query(Target)
                    .filter(
                        Target.stratum_id == national_filer_stratum.stratum_id,
                        Target.variable == target_data["variable"],
                        Target.period == target_year,
                    )
                    .first()
                )

                # Combine source info into notes
                notes_parts = []
                if pd.notna(target_data.get("notes")):
                    notes_parts.append(target_data["notes"])
                notes_parts.append(f"Source: {target_data.get('source', 'Unknown')}")
                combined_notes = " | ".join(notes_parts)

                if existing_target:
                    # Update existing target
                    existing_target.value = target_data["value"]
                    existing_target.notes = combined_notes
                    existing_target.source = "PolicyEngine"
                    print(f"Updated filer target: {target_data['variable']}")
                else:
                    # Create new target
                    target = Target(
                        stratum_id=national_filer_stratum.stratum_id,
                        variable=target_data["variable"],
                        period=target_year,
                        value=target_data["value"],
                        active=True,
                        source="PolicyEngine",
                        notes=combined_notes,
                    )
                    session.add(target)
                    print(f"Added filer target: {target_data['variable']}")

        # Process reform-based tax expenditure targets.
        if not tax_expenditure_df.empty:
            migrated_strata = (
                session.query(Stratum)
                .filter(
                    Stratum.parent_stratum_id == us_stratum.stratum_id,
                    Stratum.notes.in_(
                        [
                            "United States - Tax Filers",
                            "United States - Itemizing Tax Filers",
                        ]
                    ),
                )
                .all()
            )
            migrated_stratum_ids = [s.stratum_id for s in migrated_strata]

            for _, target_data in tax_expenditure_df.iterrows():
                target_year = target_data["year"]
                target_reform_id = int(target_data["reform_id"])

                # Clean up incorrectly scoped baseline rows from older DBs.
                if migrated_stratum_ids:
                    stale_targets = (
                        session.query(Target)
                        .filter(
                            Target.stratum_id.in_(migrated_stratum_ids),
                            Target.variable == target_data["variable"],
                            Target.period == target_year,
                            Target.reform_id == 0,
                            Target.active,
                        )
                        .all()
                    )
                    for stale_target in stale_targets:
                        stale_target.active = False

                existing_target = (
                    session.query(Target)
                    .filter(
                        Target.stratum_id == us_stratum.stratum_id,
                        Target.variable == target_data["variable"],
                        Target.period == target_year,
                        Target.reform_id == target_reform_id,
                    )
                    .first()
                )

                notes_parts = []
                if pd.notna(target_data.get("notes")):
                    notes_parts.append(target_data["notes"])
                notes_parts.append(
                    "Modeled as repeal-based income tax expenditure target"
                )
                notes_parts.append(f"Source: {target_data.get('source', 'Unknown')}")
                combined_notes = " | ".join(notes_parts)

                if existing_target:
                    existing_target.value = target_data["value"]
                    existing_target.notes = combined_notes
                    existing_target.source = "PolicyEngine"
                    existing_target.active = True
                    print(f"Updated tax expenditure target: {target_data['variable']}")
                else:
                    target = Target(
                        stratum_id=us_stratum.stratum_id,
                        variable=target_data["variable"],
                        period=target_year,
                        reform_id=target_reform_id,
                        value=target_data["value"],
                        active=True,
                        source="PolicyEngine",
                        notes=combined_notes,
                    )
                    session.add(target)
                    session.flush()

                    persisted = (
                        session.query(Target)
                        .filter(Target.target_id == target.target_id)
                        .first()
                    )
                    if persisted.reform_id != target_reform_id:
                        print(
                            f"  WARNING: {target_data['variable']} persisted "
                            f"with reform_id={persisted.reform_id}, "
                            f"correcting to {target_reform_id}"
                        )
                        persisted.reform_id = target_reform_id
                        session.flush()

                    print(f"Added tax expenditure target: {target_data['variable']}")

        # Process conditional count targets (enrollment counts)
        for cond_target in conditional_targets:
            constraint_var = cond_target["constraint_variable"]
            target_year = cond_target["year"]
            target_variable = cond_target.get("target_variable", "person_count")
            target_value = cond_target.get(target_variable)

            # Determine constraint details
            if constraint_var == "medicaid":
                stratum_notes = "National Medicaid Enrollment"
                constraint_operation = ">"
                constraint_value = "0"
            elif constraint_var == "aca_ptc":
                stratum_notes = "National ACA Premium Tax Credit Recipients"
                constraint_operation = ">"
                constraint_value = "0"
            elif constraint_var == "spm_unit_energy_subsidy_reported":
                stratum_notes = "National LIHEAP Recipient Households"
                constraint_operation = ">"
                constraint_value = "0"
            elif constraint_var == "ssn_card_type":
                stratum_notes = "National Undocumented Population"
                constraint_operation = "=="
                constraint_value = cond_target.get("constraint_value", "NONE")
            else:
                stratum_notes = f"National {constraint_var} Recipients"
                constraint_operation = ">"
                constraint_value = "0"

            # Check if this stratum already exists
            existing_stratum = (
                session.query(Stratum)
                .filter(
                    Stratum.parent_stratum_id == us_stratum.stratum_id,
                    Stratum.notes == stratum_notes,
                )
                .first()
            )

            if existing_stratum:
                # Update the existing target in this stratum
                existing_target = (
                    session.query(Target)
                    .filter(
                        Target.stratum_id == existing_stratum.stratum_id,
                        Target.variable == target_variable,
                        Target.period == target_year,
                    )
                    .first()
                )

                if existing_target:
                    existing_target.value = target_value
                    existing_target.source = "PolicyEngine"
                    print(f"Updated enrollment target for {constraint_var}")
                else:
                    # Add new target to existing stratum
                    new_target = Target(
                        stratum_id=existing_stratum.stratum_id,
                        variable=target_variable,
                        period=target_year,
                        value=target_value,
                        active=True,
                        source="PolicyEngine",
                        notes=f"{cond_target['notes']} | Source: {cond_target['source']}",
                    )
                    session.add(new_target)
                    print(f"Added enrollment target for {constraint_var}")
            else:
                # Create new stratum with constraint
                new_stratum = Stratum(
                    parent_stratum_id=us_stratum.stratum_id,
                    notes=stratum_notes,
                )

                # Add constraint
                new_stratum.constraints_rel = [
                    StratumConstraint(
                        constraint_variable=constraint_var,
                        operation=constraint_operation,
                        value=constraint_value,
                    )
                ]

                # Add target
                new_stratum.targets_rel = [
                    Target(
                        variable=target_variable,
                        period=target_year,
                        value=target_value,
                        active=True,
                        source="PolicyEngine",
                        notes=f"{cond_target['notes']} | Source: {cond_target['source']}",
                    )
                ]

                session.add(new_stratum)
                print(f"Created stratum and target for {constraint_var} enrollment")

        session.commit()

        tax_exp_vars = [
            "salt_deduction",
            "charitable_deduction",
            "deductible_mortgage_interest",
            "medical_expense_deduction",
            "qualified_business_income_deduction",
        ]
        bad_targets = (
            session.query(Target)
            .join(Stratum, Target.stratum_id == Stratum.stratum_id)
            .filter(
                Target.variable.in_(tax_exp_vars),
                Target.active == True,
                Stratum.parent_stratum_id == None,
                Target.reform_id == 0,
            )
            .all()
        )
        if bad_targets:
            bad_names = [t.variable for t in bad_targets]
            raise ValueError(
                f"Post-commit check failed: tax expenditure targets "
                f"have reform_id=0 in root stratum: {bad_names}"
            )

        total_targets = (
            len(direct_targets_df)
            + len(tax_filer_df)
            + len(tax_expenditure_df)
            + len(conditional_targets)
        )
        print(f"\nSuccessfully loaded {total_targets} national targets")
        print(f"  - {len(direct_targets_df)} direct sum targets")
        print(f"  - {len(tax_filer_df)} tax filer targets")
        print(f"  - {len(tax_expenditure_df)} tax expenditure targets")
        print(f"  - {len(conditional_targets)} enrollment count targets (as strata)")


def main():
    """Main ETL pipeline for national targets."""
    _, year = etl_argparser("ETL for national calibration targets")

    # Extract
    print("Extracting national targets...")
    raw_targets = extract_national_targets(year=year)
    time_period = raw_targets["time_period"]
    print(f"Using time_period={time_period} for CBO/Treasury targets")

    # Transform
    print("Transforming targets...")
    (
        direct_targets_df,
        tax_filer_df,
        tax_expenditure_df,
        conditional_targets,
    ) = transform_national_targets(raw_targets)

    # Load
    print("Loading targets into database...")
    load_national_targets(
        direct_targets_df,
        tax_filer_df,
        tax_expenditure_df,
        conditional_targets,
    )

    print("\nETL pipeline complete!")


if __name__ == "__main__":
    main()
