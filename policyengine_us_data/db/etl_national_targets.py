from sqlmodel import Session, create_engine
import pandas as pd

from policyengine_us_data.storage import STORAGE_FOLDER
from policyengine_us_data.db.create_database_tables import (
    Stratum,
    StratumConstraint,
    Target,
    SourceType,
)
from policyengine_us_data.utils.db_metadata import (
    get_or_create_source,
)


def extract_national_targets():
    """
    Extract national calibration targets from various sources.
    
    Returns
    -------
    dict
        Dictionary containing:
        - direct_sum_targets: Variables that can be summed directly
        - conditional_count_targets: Enrollment counts requiring constraints
        - cbo_targets: List of CBO projection targets
        - treasury_targets: List of Treasury/JCT targets
    """
    
    # Initialize PolicyEngine for parameter access
    from policyengine_us import Microsimulation
    sim = Microsimulation(dataset="hf://policyengine/policyengine-us-data/cps_2023.h5")
    
    # Direct sum targets - these are regular variables that can be summed
    # Store with their actual source year (2024 for hardcoded values from loss.py)
    HARDCODED_YEAR = 2024
    
    direct_sum_targets = [
        {
            "variable": "medicaid",
            "value": 871.7e9,
            "source": "https://www.cms.gov/files/document/highlights.pdf",
            "notes": "CMS 2023 highlights document - total Medicaid spending",
            "year": HARDCODED_YEAR
        },
        {
            "variable": "net_worth",
            "value": 160e12,
            "source": "Federal Reserve SCF",
            "notes": "Total household net worth",
            "year": HARDCODED_YEAR
        },
        {
            "variable": "salt_deduction",
            "value": 21.247e9,
            "source": "Joint Committee on Taxation",
            "notes": "SALT deduction tax expenditure",
            "year": HARDCODED_YEAR
        },
        {
            "variable": "medical_expense_deduction",
            "value": 11.4e9,
            "source": "Joint Committee on Taxation",
            "notes": "Medical expense deduction tax expenditure",
            "year": HARDCODED_YEAR
        },
        {
            "variable": "charitable_deduction",
            "value": 65.301e9,
            "source": "Joint Committee on Taxation",
            "notes": "Charitable deduction tax expenditure",
            "year": HARDCODED_YEAR
        },
        {
            "variable": "interest_deduction",
            "value": 24.8e9,
            "source": "Joint Committee on Taxation",
            "notes": "Mortgage interest deduction tax expenditure",
            "year": HARDCODED_YEAR
        },
        {
            "variable": "qualified_business_income_deduction",
            "value": 63.1e9,
            "source": "Joint Committee on Taxation",
            "notes": "QBI deduction tax expenditure",
            "year": HARDCODED_YEAR
        },
        {
            "variable": "health_insurance_premiums_without_medicare_part_b",
            "value": 385e9,
            "source": "MEPS/NHEA",
            "notes": "Health insurance premiums excluding Medicare Part B",
            "year": HARDCODED_YEAR
        },
        {
            "variable": "other_medical_expenses",
            "value": 278e9,
            "source": "MEPS/NHEA",
            "notes": "Out-of-pocket medical expenses",
            "year": HARDCODED_YEAR
        },
        {
            "variable": "medicare_part_b_premiums",
            "value": 112e9,
            "source": "CMS Medicare data",
            "notes": "Medicare Part B premium payments",
            "year": HARDCODED_YEAR
        },
        {
            "variable": "over_the_counter_health_expenses",
            "value": 72e9,
            "source": "Consumer Expenditure Survey",
            "notes": "OTC health products and supplies",
            "year": HARDCODED_YEAR
        },
        {
            "variable": "child_support_expense",
            "value": 33e9,
            "source": "Census Bureau",
            "notes": "Child support payments",
            "year": HARDCODED_YEAR
        },
        {
            "variable": "child_support_received",
            "value": 33e9,
            "source": "Census Bureau",
            "notes": "Child support received",
            "year": HARDCODED_YEAR
        },
        {
            "variable": "spm_unit_capped_work_childcare_expenses",
            "value": 348e9,
            "source": "Census Bureau SPM",
            "notes": "Work and childcare expenses for SPM",
            "year": HARDCODED_YEAR
        },
        {
            "variable": "spm_unit_capped_housing_subsidy",
            "value": 35e9,
            "source": "HUD/Census",
            "notes": "Housing subsidies",
            "year": HARDCODED_YEAR
        },
        {
            "variable": "tanf",
            "value": 9e9,
            "source": "HHS/ACF",
            "notes": "TANF cash assistance",
            "year": HARDCODED_YEAR
        },
        {
            "variable": "alimony_income",
            "value": 13e9,
            "source": "IRS Statistics of Income",
            "notes": "Alimony received",
            "year": HARDCODED_YEAR
        },
        {
            "variable": "alimony_expense",
            "value": 13e9,
            "source": "IRS Statistics of Income",
            "notes": "Alimony paid",
            "year": HARDCODED_YEAR
        },
        {
            "variable": "real_estate_taxes",
            "value": 500e9,
            "source": "Census Bureau",
            "notes": "Property taxes paid",
            "year": HARDCODED_YEAR
        },
        {
            "variable": "rent",
            "value": 735e9,
            "source": "Census Bureau/BLS",
            "notes": "Rental payments",
            "year": HARDCODED_YEAR
        },
        {
            "variable": "tip_income",
            "value": 53.2e9,
            "source": "IRS Form W-2 Box 7 statistics",
            "notes": "Social security tips uprated 40% to account for underreporting",
            "year": HARDCODED_YEAR
        }
    ]
    
    # Conditional count targets - these need strata with constraints
    # Store with actual source year
    conditional_count_targets = [
        {
            "constraint_variable": "medicaid",
            "stratum_group_id": 5,  # Medicaid strata group
            "person_count": 72_429_055,
            "source": "CMS/HHS administrative data",
            "notes": "Medicaid enrollment count",
            "year": HARDCODED_YEAR
        },
        {
            "constraint_variable": "aca_ptc",
            "stratum_group_id": None,  # Will use a generic stratum or create new group
            "person_count": 19_743_689,
            "source": "CMS marketplace data",
            "notes": "ACA Premium Tax Credit recipients",
            "year": HARDCODED_YEAR
        }
    ]
    
    # Add SSN card type NONE targets for multiple years
    # Based on loss.py lines 445-460
    ssn_none_targets_by_year = [
        {
            "constraint_variable": "ssn_card_type",
            "constraint_value": "NONE",  # Need to specify the value we're checking for
            "stratum_group_id": 7,  # New group for SSN card type
            "person_count": 11.0e6,
            "source": "DHS Office of Homeland Security Statistics",
            "notes": "Undocumented population estimate for Jan 1, 2022",
            "year": 2022
        },
        {
            "constraint_variable": "ssn_card_type",
            "constraint_value": "NONE",
            "stratum_group_id": 7,
            "person_count": 12.2e6,
            "source": "Center for Migration Studies ACS-based residual estimate",
            "notes": "Undocumented population estimate (published May 2025)",
            "year": 2023
        },
        {
            "constraint_variable": "ssn_card_type",
            "constraint_value": "NONE",
            "stratum_group_id": 7,
            "person_count": 13.0e6,
            "source": "Reuters synthesis of experts",
            "notes": "Undocumented population central estimate (~13-14 million)",
            "year": 2024
        },
        {
            "constraint_variable": "ssn_card_type",
            "constraint_value": "NONE",
            "stratum_group_id": 7,
            "person_count": 13.0e6,
            "source": "Reuters synthesis of experts",
            "notes": "Same midpoint carried forward - CBP data show 95% drop in border apprehensions",
            "year": 2025
        }
    ]
    
    conditional_count_targets.extend(ssn_none_targets_by_year)
    
    # CBO projection targets - get for a specific year
    CBO_YEAR = 2023  # Year the CBO projections are for
    cbo_vars = [
        "income_tax",
        "snap",
        "social_security",
        "ssi",
        "unemployment_compensation",
    ]
    
    cbo_targets = []
    for variable_name in cbo_vars:
        try:
            value = (
                sim.tax_benefit_system
                   .parameters(CBO_YEAR)
                   .calibration
                   .gov
                   .cbo
                   ._children[variable_name]
            )
            cbo_targets.append({
                "variable": variable_name,
                "value": float(value),
                "source": "CBO Budget Projections",
                "notes": f"CBO projection for {variable_name}",
                "year": CBO_YEAR
            })
        except (KeyError, AttributeError) as e:
            print(f"Warning: Could not extract CBO parameter for {variable_name}: {e}")
    
    # Treasury/JCT targets (EITC) - get for a specific year
    TREASURY_YEAR = 2023
    try:
        eitc_value = (
            sim.tax_benefit_system.parameters
               .calibration
               .gov
               .treasury
               .tax_expenditures
               .eitc(TREASURY_YEAR)
        )
        treasury_targets = [{
            "variable": "eitc",
            "value": float(eitc_value),
            "source": "Treasury/JCT Tax Expenditures",
            "notes": "EITC tax expenditure",
            "year": TREASURY_YEAR
        }]
    except (KeyError, AttributeError) as e:
        print(f"Warning: Could not extract Treasury EITC parameter: {e}")
        treasury_targets = []
    
    return {
        "direct_sum_targets": direct_sum_targets,
        "conditional_count_targets": conditional_count_targets,
        "cbo_targets": cbo_targets,
        "treasury_targets": treasury_targets
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
        (direct_targets_df, conditional_targets)
        - direct_targets_df: DataFrame with direct sum targets
        - conditional_targets: List of conditional count targets
    """
    
    # Process direct sum targets
    all_direct_targets = (
        raw_targets["direct_sum_targets"] +
        raw_targets["cbo_targets"] +
        raw_targets["treasury_targets"]
    )
    
    direct_df = pd.DataFrame(all_direct_targets)
    
    # Conditional targets stay as list for special processing
    conditional_targets = raw_targets["conditional_count_targets"]
    
    return direct_df, conditional_targets


def load_national_targets(direct_targets_df, conditional_targets):
    """
    Load national targets into the database.
    
    Parameters
    ----------
    direct_targets_df : pd.DataFrame
        DataFrame with direct sum target data
    conditional_targets : list
        List of conditional count targets requiring strata
    year : int
        Year for the targets
    """
    
    DATABASE_URL = f"sqlite:///{STORAGE_FOLDER / 'policy_data.db'}"
    engine = create_engine(DATABASE_URL)
    
    with Session(engine) as session:
        # Get or create the calibration source
        calibration_source = get_or_create_source(
            session,
            name="PolicyEngine Calibration Targets",
            source_type=SourceType.HARDCODED,
            vintage="Mixed (2023-2024)",
            description="National calibration targets from various authoritative sources",
            url=None,
            notes="Aggregated from CMS, IRS, CBO, Treasury, and other federal sources"
        )
        
        # Get the national stratum
        us_stratum = session.query(Stratum).filter(
            Stratum.parent_stratum_id == None
        ).first()
        
        if not us_stratum:
            raise ValueError("National stratum not found. Run create_initial_strata.py first.")
        
        # Process direct sum targets
        for _, target_data in direct_targets_df.iterrows():
            target_year = target_data["year"]
            # Check if target already exists
            existing_target = session.query(Target).filter(
                Target.stratum_id == us_stratum.stratum_id,
                Target.variable == target_data["variable"],
                Target.period == target_year
            ).first()
            
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
                print(f"Updated target: {target_data['variable']}")
            else:
                # Create new target
                target = Target(
                    stratum_id=us_stratum.stratum_id,
                    variable=target_data["variable"],
                    period=target_year,
                    value=target_data["value"],
                    source_id=calibration_source.source_id,
                    active=True,
                    notes=combined_notes
                )
                session.add(target)
                print(f"Added target: {target_data['variable']}")
        
        # Process conditional count targets (enrollment counts)
        for cond_target in conditional_targets:
            constraint_var = cond_target["constraint_variable"]
            stratum_group_id = cond_target.get("stratum_group_id")
            target_year = cond_target["year"]
            
            # Determine stratum group ID and constraint details
            if constraint_var == "medicaid":
                stratum_group_id = 5  # Medicaid strata group
                stratum_notes = "National Medicaid Enrollment"
                constraint_operation = ">"
                constraint_value = "0"
            elif constraint_var == "aca_ptc":
                stratum_group_id = 6  # EITC group or could create new ACA group
                stratum_notes = "National ACA Premium Tax Credit Recipients"
                constraint_operation = ">"
                constraint_value = "0"
            elif constraint_var == "ssn_card_type":
                stratum_group_id = 7  # SSN card type group
                stratum_notes = "National Undocumented Population"
                constraint_operation = "="
                constraint_value = cond_target.get("constraint_value", "NONE")
            else:
                stratum_notes = f"National {constraint_var} Recipients"
                constraint_operation = ">"
                constraint_value = "0"
            
            # Check if this stratum already exists
            existing_stratum = session.query(Stratum).filter(
                Stratum.parent_stratum_id == us_stratum.stratum_id,
                Stratum.stratum_group_id == stratum_group_id,
                Stratum.notes == stratum_notes
            ).first()
            
            if existing_stratum:
                # Update the existing target in this stratum
                existing_target = session.query(Target).filter(
                    Target.stratum_id == existing_stratum.stratum_id,
                    Target.variable == "person_count",
                    Target.period == target_year
                ).first()
                
                if existing_target:
                    existing_target.value = cond_target["person_count"]
                    print(f"Updated enrollment target for {constraint_var}")
                else:
                    # Add new target to existing stratum
                    new_target = Target(
                        stratum_id=existing_stratum.stratum_id,
                        variable="person_count",
                        period=target_year,
                        value=cond_target["person_count"],
                        source_id=calibration_source.source_id,
                        active=True,
                        notes=f"{cond_target['notes']} | Source: {cond_target['source']}"
                    )
                    session.add(new_target)
                    print(f"Added enrollment target for {constraint_var}")
            else:
                # Create new stratum with constraint
                new_stratum = Stratum(
                    parent_stratum_id=us_stratum.stratum_id,
                    stratum_group_id=stratum_group_id,
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
                        variable="person_count",
                        period=target_year,
                        value=cond_target["person_count"],
                        source_id=calibration_source.source_id,
                        active=True,
                        notes=f"{cond_target['notes']} | Source: {cond_target['source']}"
                    )
                ]
                
                session.add(new_stratum)
                print(f"Created stratum and target for {constraint_var} enrollment")
        
        session.commit()
        
        total_targets = len(direct_targets_df) + len(conditional_targets)
        print(f"\nSuccessfully loaded {total_targets} national targets")
        print(f"  - {len(direct_targets_df)} direct sum targets")
        print(f"  - {len(conditional_targets)} enrollment count targets (as strata)")


def main():
    """Main ETL pipeline for national targets."""
    
    # Extract
    print("Extracting national targets...")
    raw_targets = extract_national_targets()
    
    # Transform
    print("Transforming targets...")
    direct_targets_df, conditional_targets = transform_national_targets(raw_targets)
    
    # Load
    print("Loading targets into database...")
    load_national_targets(direct_targets_df, conditional_targets)
    
    print("\nETL pipeline complete!")


if __name__ == "__main__":
    main()