"""
Validation script to ensure the parent-child hierarchy is working correctly.
Checks geographic and age strata relationships.
"""

import sys
from sqlmodel import Session, create_engine, select
from policyengine_us_data.storage import STORAGE_FOLDER
from policyengine_us_data.db.create_database_tables import (
    Stratum,
    StratumConstraint,
)


def validate_geographic_hierarchy(session):
    """Validate the geographic hierarchy: US -> States -> Congressional Districts"""

    print("\n" + "=" * 60)
    print("VALIDATING GEOGRAPHIC HIERARCHY")
    print("=" * 60)

    errors = []

    # Check US stratum exists and has no parent
    us_stratum = session.exec(
        select(Stratum).where(Stratum.parent_stratum_id == None)
    ).first()

    if not us_stratum:
        errors.append(
            "ERROR: No US-level stratum found (should have parent_stratum_id = None)"
        )
    else:
        print(
            f"✓ US stratum found: {us_stratum.notes} (ID: {us_stratum.stratum_id})"
        )

        # Check it has no constraints
        us_constraints = session.exec(
            select(StratumConstraint).where(
                StratumConstraint.stratum_id == us_stratum.stratum_id
            )
        ).all()

        if us_constraints:
            errors.append(
                f"ERROR: US stratum has {len(us_constraints)} constraints, should have 0"
            )
        else:
            print("✓ US stratum has no constraints (correct)")

    # States: children of US with state_fips constraint
    us_children = (
        session.exec(
            select(Stratum).where(
                Stratum.parent_stratum_id == us_stratum.stratum_id,
            )
        )
        .unique()
        .all()
    )
    states = []
    for child in us_children:
        constraints = session.exec(
            select(StratumConstraint).where(
                StratumConstraint.stratum_id == child.stratum_id
            )
        ).all()
        constraint_vars = {c.constraint_variable for c in constraints}
        if constraint_vars == {"state_fips"}:
            states.append(child)

    print(f"\n✓ Found {len(states)} state strata")
    if len(states) != 51:  # 50 states + DC
        errors.append(
            f"WARNING: Expected 51 states (including DC), found {len(states)}"
        )

    # Verify each state has proper constraints
    state_ids = {}
    for state in states[:5]:  # Sample first 5 states
        constraints = session.exec(
            select(StratumConstraint).where(
                StratumConstraint.stratum_id == state.stratum_id
            )
        ).all()

        state_fips_constraint = [
            c for c in constraints if c.constraint_variable == "state_fips"
        ]
        if not state_fips_constraint:
            errors.append(
                f"ERROR: State '{state.notes}' has no state_fips constraint"
            )
        else:
            state_ids[state.stratum_id] = state.notes
            print(
                f"  - {state.notes}: state_fips = {state_fips_constraint[0].value}"
            )

    # Check congressional districts
    print("\nChecking Congressional Districts...")

    # CDs: strata with only geographic constraints including
    # congressional_district_geoid
    all_strata = session.exec(select(Stratum)).unique().all()
    all_cds = []
    for s in all_strata:
        constraints = session.exec(
            select(StratumConstraint).where(
                StratumConstraint.stratum_id == s.stratum_id
            )
        ).all()
        constraint_vars = {c.constraint_variable for c in constraints}
        if (
            "congressional_district_geoid" in constraint_vars
            and constraint_vars
            <= {"state_fips", "congressional_district_geoid"}
        ):
            all_cds.append(s)

    print(f"✓ Found {len(all_cds)} congressional/delegate districts")
    if len(all_cds) != 436:
        errors.append(
            f"WARNING: Expected 436 congressional districts (including DC delegate), found {len(all_cds)}"
        )

    # Verify CDs are children of correct states (spot check)
    wyoming_id = None
    for state in states:
        if "Wyoming" in state.notes:
            wyoming_id = state.stratum_id
            break

    if wyoming_id:
        # Check Wyoming's congressional district
        wyoming_children = (
            session.exec(
                select(Stratum).where(
                    Stratum.parent_stratum_id == wyoming_id,
                )
            )
            .unique()
            .all()
        )
        wyoming_cds = []
        for child in wyoming_children:
            constraints = session.exec(
                select(StratumConstraint).where(
                    StratumConstraint.stratum_id == child.stratum_id
                )
            ).all()
            cvars = {c.constraint_variable for c in constraints}
            if "congressional_district_geoid" in cvars and cvars <= {
                "state_fips",
                "congressional_district_geoid",
            }:
                wyoming_cds.append(child)

        if len(wyoming_cds) != 1:
            errors.append(
                f"ERROR: Wyoming should have 1 CD, found {len(wyoming_cds)}"
            )
        else:
            print(f"✓ Wyoming has correct number of CDs: 1")

        # Verify no other state's CDs are incorrectly parented to Wyoming
        wrong_parent_cds = (
            session.exec(
                select(Stratum).where(
                    Stratum.parent_stratum_id == wyoming_id,
                    ~Stratum.notes.like("%Wyoming%"),
                    Stratum.notes.like("%Congressional%"),
                )
            )
            .unique()
            .all()
        )

        if wrong_parent_cds:
            errors.append(
                f"ERROR: Found {len(wrong_parent_cds)} non-Wyoming CDs incorrectly parented to Wyoming"
            )
            for cd in wrong_parent_cds[:5]:
                errors.append(f"  - {cd.notes}")
        else:
            print(
                "✓ No congressional districts incorrectly parented to Wyoming"
            )

    return errors


def validate_demographic_strata(session):
    """Validate demographic strata are properly attached to geographic strata"""

    print("\n" + "=" * 60)
    print("VALIDATING DEMOGRAPHIC STRATA")
    print("=" * 60)

    errors = []

    from sqlalchemy import text

    # Expected strata per geographic area for each domain variable.
    # Most domains have strata at all 488 geo areas (1 national + 51 states + 436 CDs).
    # medicaid_enrolled is state-level only (51 states)
    # because CD-level survey data is disabled pending 119th Congress
    # district code remapping (see etl_medicaid.py TODO).
    # the national medicaid target actually uses the `medicaid` (expense) variable
    expected_counts = {
        "age": 18 * 488,
        "adjusted_gross_income": 9 * 488,
        "snap": 1 * 488,
        "medicaid_enrolled": 1 * 51,
        "eitc_child_count": 4 * 488,
    }

    # Use stratum_domain view to get actual counts
    result = session.execute(
        text(
            "SELECT domain_variable, "
            "COUNT(DISTINCT stratum_id) as cnt "
            "FROM stratum_domain GROUP BY domain_variable"
        )
    ).fetchall()

    domain_counts = {row[0]: row[1] for row in result}

    print(f"\nDomain variables found via stratum_domain view:")
    for domain, count in sorted(domain_counts.items()):
        print(f"  {domain}: {count} strata")

    # Validate expected counts for known domains
    for domain, expected_total in expected_counts.items():
        actual = domain_counts.get(domain, 0)
        if actual == expected_total:
            print(f"✓ {domain}: {actual} strata")
        elif actual == 0:
            errors.append(
                f"ERROR: {domain} has no strata, " f"expected {expected_total}"
            )
        else:
            errors.append(
                f"WARNING: {domain} has {actual} strata, "
                f"expected {expected_total}"
            )

    # Identify geographic strata (those with only geographic
    # constraints or no constraints) for parent validation
    geo_stratum_ids = set()
    all_strata = session.exec(select(Stratum)).unique().all()
    for s in all_strata:
        constraints = session.exec(
            select(StratumConstraint).where(
                StratumConstraint.stratum_id == s.stratum_id
            )
        ).all()
        cvars = {c.constraint_variable for c in constraints}
        if cvars <= {
            "state_fips",
            "congressional_district_geoid",
        }:
            geo_stratum_ids.add(s.stratum_id)

    # Check parent relationships for a sample of demographic strata
    print("\nChecking parent relationships (sample):")
    domain_stratum_ids = session.execute(
        text("SELECT DISTINCT stratum_id FROM stratum_domain")
    ).fetchall()
    domain_ids = [row[0] for row in domain_stratum_ids]

    sample_strata = [session.get(Stratum, sid) for sid in domain_ids[:100]]

    correct_parents = 0
    wrong_parents = 0
    no_parents = 0

    for stratum in sample_strata:
        if stratum is None:
            continue
        if stratum.parent_stratum_id:
            if stratum.parent_stratum_id in geo_stratum_ids:
                correct_parents += 1
            else:
                wrong_parents += 1
                errors.append(
                    f"ERROR: Stratum {stratum.stratum_id} "
                    f"has non-geographic parent "
                    f"{stratum.parent_stratum_id}"
                )
        else:
            no_parents += 1
            errors.append(
                f"ERROR: Stratum {stratum.stratum_id} " f"has no parent"
            )

    print(f"  Sample of {len(sample_strata)} " f"demographic strata:")
    print(f"    - With geographic parent: {correct_parents}")
    print(f"    - With wrong parent: {wrong_parents}")
    print(f"    - With no parent: {no_parents}")

    return errors


def validate_constraint_uniqueness(session):
    """Check that constraint combinations produce unique hashes"""

    print("\n" + "=" * 60)
    print("VALIDATING CONSTRAINT UNIQUENESS")
    print("=" * 60)

    errors = []

    # Check for duplicate definition_hashes
    all_strata = session.exec(select(Stratum)).unique().all()
    hash_counts = {}

    for stratum in all_strata:
        if stratum.definition_hash in hash_counts:
            hash_counts[stratum.definition_hash].append(stratum)
        else:
            hash_counts[stratum.definition_hash] = [stratum]

    duplicates = {
        h: strata for h, strata in hash_counts.items() if len(strata) > 1
    }

    if duplicates:
        errors.append(
            f"ERROR: Found {len(duplicates)} duplicate definition_hashes"
        )
        for hash_val, strata in list(duplicates.items())[:3]:  # Show first 3
            errors.append(
                f"  Hash {hash_val[:10]}... appears {len(strata)} times:"
            )
            for s in strata[:3]:
                errors.append(f"    - ID {s.stratum_id}: {s.notes[:50]}")
    else:
        print(f"✓ All {len(all_strata)} strata have unique definition_hashes")

    return errors


def main():
    """Run all validation checks"""

    DATABASE_URL = (
        f"sqlite:///{STORAGE_FOLDER / 'calibration' / 'policy_data.db'}"
    )
    engine = create_engine(DATABASE_URL)

    all_errors = []

    with Session(engine) as session:
        # Run validation checks
        all_errors.extend(validate_geographic_hierarchy(session))
        all_errors.extend(validate_demographic_strata(session))
        all_errors.extend(validate_constraint_uniqueness(session))

    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    if all_errors:
        print(f"\n❌ Found {len(all_errors)} issues:\n")
        for error in all_errors:
            print(f"  {error}")
        sys.exit(1)
    else:
        print("\n✅ All validation checks passed!")
        print("   - Geographic hierarchy is correct")
        print("   - Demographic strata properly organized and attached")
        print("   - All constraint combinations are unique")
        sys.exit(0)


if __name__ == "__main__":
    main()
