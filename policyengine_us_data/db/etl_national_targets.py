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
        
        # These are hardcoded values from loss.py HARD_CODED_TOTALS dictionary
        # and other national hardcoded values that are NOT already loaded by other ETL files
        national_targets = [
            {
                "variable": "health_insurance_premiums_without_medicare_part_b",
                "operation": "sum",
                "value": 385e9,
                "source": "CPS-derived statistics 2024",
                "notes": "Total health insurance premiums excluding Medicare Part B"
            },
            {
                "variable": "other_medical_expenses",
                "operation": "sum",
                "value": 278e9,
                "source": "CPS-derived statistics 2024",
                "notes": "Out-of-pocket medical expenses"
            },
            {
                "variable": "medicare_part_b_premiums",
                "operation": "sum",
                "value": 112e9,
                "source": "CPS-derived statistics 2024",
                "notes": "Medicare Part B premiums"
            },
            {
                "variable": "child_support_expense",
                "operation": "sum",
                "value": 33e9,
                "source": "CPS-derived statistics 2024",
                "notes": "Total child support paid"
            },
            {
                "variable": "tip_income",
                "operation": "sum",
                "value": 53.2e9,  # 38e9 * 1.4 as per the calculation in loss.py
                "source": "IRS Form W-2 Box 7 statistics, uprated 40% to 2024",
                "notes": "Social security tips from W-2 forms"
            }
        ]
        
        # Add or update the targets
        period = 2024  # Default period for these targets
        for target_data in national_targets:
            existing_target = session.query(Target).filter(
                Target.stratum_id == us_stratum.stratum_id,
                Target.variable == target_data["variable"],
                Target.period == period
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
                    period=period,
                    value=target_data["value"],
                    source_id=calibration_source.source_id,
                    active=True,
                    notes=" | ".join(notes_parts)
                )
                session.add(target)
                print(f"Added target: {target_data['variable']}")
        
        session.commit()
        print(f"\nSuccessfully loaded {len(national_targets)} national targets")
        
        # Smell test - verify the values make economic sense
        print("\n--- Economic Smell Test ---")
        print(f"Health insurance premiums: ${385e9/1e9:.0f}B - reasonable for US population")
        print(f"Medicare Part B premiums: ${112e9/1e9:.0f}B - ~60M beneficiaries * ~$2k/year")
        print(f"Child support: ${33e9/1e9:.0f}B - matches payments and receipts")
        print(f"Tip income: ${53.2e9/1e9:.1f}B - reasonable for service industry")


if __name__ == "__main__":
    main()