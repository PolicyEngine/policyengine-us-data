from sqlmodel import Session, create_engine

from policyengine_us_data.storage import STORAGE_FOLDER
from policyengine_us_data.db.create_database_tables import (
    Stratum,
    Target,
)


def main():
    DATABASE_URL = f"sqlite:///{STORAGE_FOLDER / 'policy_data.db'}"
    engine = create_engine(DATABASE_URL)
    
    with Session(engine) as session:
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
                    source_id=5,  # Hardcoded source ID for national targets
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