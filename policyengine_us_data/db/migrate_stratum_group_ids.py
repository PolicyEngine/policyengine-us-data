"""
TODO: what is this file? Do we still need it?


Migration script to update stratum_group_id values to represent conceptual categories.

New scheme:
- 1: Geographic (US, states, congressional districts)
- 2: Age-based strata
- 3: Income/AGI-based strata
- 4: SNAP recipient strata
- 5: Medicaid enrollment strata
- 6: EITC recipient strata
"""

from sqlmodel import Session, create_engine, select
from policyengine_us_data.storage import STORAGE_FOLDER
from policyengine_us_data.db.create_database_tables import (
    Stratum,
    StratumConstraint,
)


def migrate_stratum_group_ids():
    """Update stratum_group_id values based on constraint variables."""

    DATABASE_URL = (
        f"sqlite:///{STORAGE_FOLDER / 'calibration' / 'policy_data.db'}"
    )
    engine = create_engine(DATABASE_URL)

    with Session(engine) as session:
        print("Starting stratum_group_id migration...")
        print("=" * 60)

        # Track updates
        updates = {
            "Geographic": 0,
            "Age": 0,
            "Income/AGI": 0,
            "SNAP": 0,
            "Medicaid": 0,
            "EITC": 0,
        }

        # Get all strata
        all_strata = session.exec(select(Stratum)).unique().all()

        for stratum in all_strata:
            # Get constraints for this stratum
            constraints = session.exec(
                select(StratumConstraint).where(
                    StratumConstraint.stratum_id == stratum.stratum_id
                )
            ).all()

            # Determine new group_id based on constraints
            constraint_vars = [c.constraint_variable for c in constraints]

            # Geographic strata (no demographic constraints)
            if not constraint_vars or all(
                cv in ["state_fips", "congressional_district_geoid"]
                for cv in constraint_vars
            ):
                if stratum.stratum_group_id != 1:
                    stratum.stratum_group_id = 1
                    updates["Geographic"] += 1

            # Age strata
            elif "age" in constraint_vars:
                if stratum.stratum_group_id != 2:
                    stratum.stratum_group_id = 2
                    updates["Age"] += 1

            # Income/AGI strata
            elif "adjusted_gross_income" in constraint_vars:
                if stratum.stratum_group_id != 3:
                    stratum.stratum_group_id = 3
                    updates["Income/AGI"] += 1

            # SNAP strata
            elif "snap" in constraint_vars:
                if stratum.stratum_group_id != 4:
                    stratum.stratum_group_id = 4
                    updates["SNAP"] += 1

            # Medicaid strata
            elif "medicaid_enrolled" in constraint_vars:
                if stratum.stratum_group_id != 5:
                    stratum.stratum_group_id = 5
                    updates["Medicaid"] += 1

            # EITC strata
            elif "eitc_child_count" in constraint_vars:
                if stratum.stratum_group_id != 6:
                    stratum.stratum_group_id = 6
                    updates["EITC"] += 1

        # Commit changes
        session.commit()

        # Report results
        print("\nMigration complete!")
        print("-" * 60)
        print("Updates made:")
        for category, count in updates.items():
            if count > 0:
                print(f"  {category:15}: {count:5} strata updated")

        # Verify final counts
        print("\nFinal stratum_group_id distribution:")
        print("-" * 60)

        group_names = {
            1: "Geographic",
            2: "Age",
            3: "Income/AGI",
            4: "SNAP",
            5: "Medicaid",
            6: "EITC",
        }

        for group_id, name in group_names.items():
            count = len(
                session.exec(
                    select(Stratum).where(Stratum.stratum_group_id == group_id)
                )
                .unique()
                .all()
            )
            print(f"  Group {group_id} ({name:12}): {count:5} strata")

        print("\n✅ Migration successful!")


if __name__ == "__main__":
    migrate_stratum_group_ids()
