from sqlmodel import Session, create_engine

from policyengine_us_data.storage import STORAGE_FOLDER
from policyengine_us_data.db.create_database_tables import (
    Stratum,
    Target,
    SourceType,
)
from policyengine_us_data.utils.db_metadata import (
    get_or_create_source,
    get_or_create_variable_group,
    get_or_create_variable_metadata,
)


def main():
    DATABASE_URL = f"sqlite:///{STORAGE_FOLDER / 'policy_data.db'}"
    engine = create_engine(DATABASE_URL)
    default_period = 2023  # If I can choose, I'll get them for 2023
    
    with Session(engine) as session:
        # Get or create the hardcoded calibration source
        calibration_source = get_or_create_source(
            session,
            name="PolicyEngine Calibration Targets",
            source_type=SourceType.HARDCODED,
            vintage="2024",
            description="Hardcoded calibration targets from various sources",
            url=None,
            notes="National totals from CPS-derived statistics, IRS, and other sources"
        )
        
        # Create variable groups for different types of hardcoded targets
        medical_group = get_or_create_variable_group(
            session,
            name="medical_expenses",
            category="expense",
            is_histogram=False,
            is_exclusive=False,
            aggregation_method="sum",
            display_order=9,
            description="Medical expenses and health insurance premiums"
        )
        
        other_income_group = get_or_create_variable_group(
            session,
            name="other_income",
            category="income",
            is_histogram=False,
            is_exclusive=False,
            aggregation_method="sum",
            display_order=10,
            description="Other income sources (tips, etc.)"
        )
        
        # Create variable metadata
        medical_vars = [
            ("health_insurance_premiums_without_medicare_part_b", "Health Insurance Premiums (non-Medicare)", 1),
            ("other_medical_expenses", "Other Medical Expenses", 2),
            ("medicare_part_b_premiums", "Medicare Part B Premiums", 3),
        ]
        
        for var_name, display_name, order in medical_vars:
            get_or_create_variable_metadata(
                session,
                variable=var_name,
                group=medical_group,
                display_name=display_name,
                display_order=order,
                units="dollars"
            )
        
        # Child support and tip income
        get_or_create_variable_metadata(
            session,
            variable="child_support_expense",
            group=None,  # Doesn't fit neatly into a group
            display_name="Child Support Expense",
            display_order=1,
            units="dollars"
        )
        
        get_or_create_variable_metadata(
            session,
            variable="tip_income",
            group=other_income_group,
            display_name="Tip Income",
            display_order=1,
            units="dollars"
        )
        
        # Get the national stratum
        us_stratum = session.query(Stratum).filter(
            Stratum.parent_stratum_id == None
        ).first()
        
        if not us_stratum:
            raise ValueError("National stratum not found. Run create_initial_strata.py first.")
        
        national_targets = [
            {
                "variable": "medicaid",
                "operation": "sum",
                "value": 871.7e9,
                "source": "https://www.cms.gov/files/document/highlights.pdf",
                "notes": "CMS 2023 highlights document",
                "year": 2023
            },
            {
                "variable": "medicaid_enrollment",
                "operation": "person_count",
                "value": 72_429_055,
                "source": "loss.py",
                "notes": "Can hook up to an authoritative source later",
                "year": 2024
            },
            {
                "variable": "aca_ptc",
                "operation": "person_count",
                "value": 19_743_689,
                "source": "loss.py",
                "notes": "ACA Premium Tax Credit. Can hook up to an authoritative source later",
                "year": 2024
            },
            {
                "variable": "net_worth",
                "operation": "sum",
                "value": 160e12,
                "source": "loss.py",
                "notes": "Can hook up to an authoritative source later",
                "year": 2024
            },
            {
                "variable": "salt_deduction",
                "operation": "sum",
                "value": 21.247e9,
                "source": "loss.py",
                "notes": "Can hook up to an authoritative source later",
                "year": 2024
            },
            {
                "variable": "medical_expense_deduction",
                "operation": "sum",
                "value": 11.4e9,
                "source": "loss.py",
                "notes": "Can hook up to an authoritative source later",
                "year": 2024
            },
            {
                "variable": "charitable_deduction",
                "operation": "sum",
                "value": 65.301e9,
                "source": "loss.py",
                "notes": "Can hook up to an authoritative source later",
                "year": 2024
            },
            {
                "variable": "interest_deduction",
                "operation": "sum",
                "value": 24.8e9,
                "source": "loss.py",
                "notes": "Can hook up to an authoritative source later",
                "year": 2024
            },
            {
                "variable": "qualified_business_income_deduction",
                "operation": "sum",
                "value": 63.1e9,
                "source": "loss.py",
                "notes": "Can hook up to an authoritative source later",
                "year": 2024
            },
            {
                "variable": "health_insurance_premiums_without_medicare_part_b",
                "operation": "sum",
                "value": 385e9,
                "source": "loss.py",
                "notes": "Temporary hard-coded",
                "year": 2024
            },
            {
                "variable": "other_medical_expenses",
                "operation": "sum",
                "value": 278e9,
                "source": "loss.py",
                "notes": "Temporary hard-coded",
                "year": 2024
            },
            {
                "variable": "medicare_part_b_premiums",
                "operation": "sum",
                "value": 112e9,
                "source": "loss.py",
                "notes": "Temporary hard-coded",
                "year": 2024
            },
            {
                "variable": "over_the_counter_health_expenses",
                "operation": "sum",
                "value": 72e9,
                "source": "loss.py",
                "notes": "Temporary hard-coded",
                "year": 2024
            },
            {
                "variable": "child_support_expense",
                "operation": "sum",
                "value": 33e9,
                "source": "loss.py",
                "notes": "Temporary hard-coded",
                "year": 2024
            },
            {
                "variable": "child_support_received",
                "operation": "sum",
                "value": 33e9,
                "source": "loss.py",
                "notes": "Temporary hard-coded",
                "year": 2024
            },
            {
                "variable": "spm_unit_capped_work_childcare_expenses",
                "operation": "sum",
                "value": 348e9,
                "source": "loss.py",
                "notes": "Temporary hard-coded",
                "year": 2024
            },
            {
                "variable": "spm_unit_capped_housing_subsidy",
                "operation": "sum",
                "value": 35e9,
                "source": "loss.py",
                "notes": "Temporary hard-coded",
                "year": 2024
            },
            {
                "variable": "tanf",
                "operation": "sum",
                "value": 9e9,
                "source": "loss.py",
                "notes": "Temporary hard-coded",
                "year": 2024
            },
            {
                "variable": "alimony_income",
                "operation": "sum",
                "value": 13e9,
                "source": "loss.py",
                "notes": "Temporary hard-coded",
                "year": 2024
            },
            {
                "variable": "alimony_expense",
                "operation": "sum",
                "value": 13e9,
                "source": "loss.py",
                "notes": "Temporary hard-coded",
                "year": 2024
            },
            {
                "variable": "real_estate_taxes",
                "operation": "sum",
                "value": 500e9,
                "source": "loss.py",
                "notes": "Temporary hard-coded",
                "year": 2024
            },
            {
                "variable": "rent",
                "operation": "sum",
                "value": 735e9,
                "source": "loss.py",
                "notes": "Temporary hard-coded",
                "year": 2024
            },
            {
                "variable": "tip_income",
                "operation": "sum",
                "value": 53.2e9,  # 38e9 * 1.4 as per the calculation in loss.py
                "source": "IRS Form W-2 Box 7 statistics, uprated 40% to 2024",
                "notes": "Social security tips from W-2 forms",
                "year": 2024
            }
        ]

    # Treasury targets -----
    national_targets.append(
        {
            "variable": "eitc",
            "operation": "sum",
            "value": (
                sim.tax_benefit_system.parameters
                   .calibration
                   .gov
                   .treasury
                   .tax_expenditures
                   .eitc(default_period)
            ),
            "source": "IRS Form W-2 Box 7 statistics, uprated 40% to 2024",
            "notes": "Social security tips from W-2 forms",
            "year": default_period 
        }
    )


    # CBO targets ----
    
    from policyengine_us import Microsimulation
    sim = Microsimulation(dataset = "hf://policyengine/policyengine-us-data/cps_2023.h5")
    
    CBO_VARS = [
        "income_tax",
        "snap",
        "social_security",
        "ssi",
        "unemployment_compensation",
    ]
    
    for variable_name in CBO_VARS:
        national_targets.append({
            "variable": variable_name,
            "operation": "sum",
            "value": (
                sim.tax_benefit_system
                   .parameters(default_period)
                   .calibration
                   .gov
                   .cbo
                   ._children[variable_name]
                ),
            "source": "policyengine-us",
            "notes": "",
            "year": default_period 
        })
    
        
        for target_data in national_targets:
            existing_target = session.query(Target).filter(
                Target.stratum_id == us_stratum.stratum_id,
                Target.variable == target_data["variable"],
                Target.period == default_period
            ).first()
            
            if existing_target:
                # Update existing target
                existing_target.value = target_data["value"]
                # Combine operation and source info into notes
                notes_parts = []
                if target_data.get("notes"):
                    notes_parts.append(target_data["notes"])
                notes_parts.append(f"Operation: {target_data['operation']}")
                notes_parts.append(f"Source: {target_data.get('source', 'Unknown')}")
                existing_target.notes = " | ".join(notes_parts)
                print(f"Updated target: {target_data['variable']}")
            else:
                # Create new target
                # Combine operation and source info into notes
                notes_parts = []
                if target_data.get("notes"):
                    notes_parts.append(target_data["notes"])
                notes_parts.append(f"Operation: {target_data['operation']}")
                notes_parts.append(f"Source: {target_data.get('source', 'Unknown')}")
                
                target = Target(
                    stratum_id=us_stratum.stratum_id,
                    variable=target_data["variable"],
                    period=default_period,
                    value=target_data["value"],
                    source_id=calibration_source.source_id,
                    active=True,
                    notes=" | ".join(notes_parts)
                )
                session.add(target)
                print(f"Added target: {target_data['variable']}")
        
        session.commit()
        print(f"\nSuccessfully loaded {len(national_targets)} national targets")
        

if __name__ == "__main__":
    main()
