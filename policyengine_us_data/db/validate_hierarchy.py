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
    
    print("\n" + "="*60)
    print("VALIDATING GEOGRAPHIC HIERARCHY")
    print("="*60)
    
    errors = []
    
    # Check US stratum exists and has no parent
    us_stratum = session.exec(
        select(Stratum).where(
            Stratum.stratum_group_id == 1,
            Stratum.parent_stratum_id == None
        )
    ).first()
    
    if not us_stratum:
        errors.append("ERROR: No US-level stratum found (should have parent_stratum_id = None)")
    else:
        print(f"✓ US stratum found: {us_stratum.notes} (ID: {us_stratum.stratum_id})")
        
        # Check it has no constraints
        us_constraints = session.exec(
            select(StratumConstraint).where(
                StratumConstraint.stratum_id == us_stratum.stratum_id
            )
        ).all()
        
        if us_constraints:
            errors.append(f"ERROR: US stratum has {len(us_constraints)} constraints, should have 0")
        else:
            print("✓ US stratum has no constraints (correct)")
    
    # Check states
    states = session.exec(
        select(Stratum).where(
            Stratum.stratum_group_id == 1,
            Stratum.parent_stratum_id == us_stratum.stratum_id
        )
    ).unique().all()
    
    print(f"\n✓ Found {len(states)} state strata")
    if len(states) != 51:  # 50 states + DC
        errors.append(f"WARNING: Expected 51 states (including DC), found {len(states)}")
    
    # Verify each state has proper constraints
    state_ids = {}
    for state in states[:5]:  # Sample first 5 states
        constraints = session.exec(
            select(StratumConstraint).where(
                StratumConstraint.stratum_id == state.stratum_id
            )
        ).all()
        
        state_fips_constraint = [c for c in constraints if c.constraint_variable == "state_fips"]
        if not state_fips_constraint:
            errors.append(f"ERROR: State '{state.notes}' has no state_fips constraint")
        else:
            state_ids[state.stratum_id] = state.notes
            print(f"  - {state.notes}: state_fips = {state_fips_constraint[0].value}")
    
    # Check congressional districts
    print("\nChecking Congressional Districts...")
    
    # Count total CDs (including delegate districts)
    all_cds = session.exec(
        select(Stratum).where(
            Stratum.stratum_group_id == 1,
            (Stratum.notes.like("%Congressional District%") | Stratum.notes.like("%Delegate District%"))
        )
    ).unique().all()
    
    print(f"✓ Found {len(all_cds)} congressional/delegate districts")
    if len(all_cds) != 436:
        errors.append(f"WARNING: Expected 436 congressional districts (including DC delegate), found {len(all_cds)}")
    
    # Verify CDs are children of correct states (spot check)
    wyoming_id = None
    for state in states:
        if "Wyoming" in state.notes:
            wyoming_id = state.stratum_id
            break
    
    if wyoming_id:
        # Check Wyoming's congressional district
        wyoming_cds = session.exec(
            select(Stratum).where(
                Stratum.stratum_group_id == 1,
                Stratum.parent_stratum_id == wyoming_id,
                Stratum.notes.like("%Congressional%")
            )
        ).unique().all()
        
        if len(wyoming_cds) != 1:
            errors.append(f"ERROR: Wyoming should have 1 CD, found {len(wyoming_cds)}")
        else:
            print(f"✓ Wyoming has correct number of CDs: 1")
            
        # Verify no other state's CDs are incorrectly parented to Wyoming
        wrong_parent_cds = session.exec(
            select(Stratum).where(
                Stratum.stratum_group_id == 1,
                Stratum.parent_stratum_id == wyoming_id,
                ~Stratum.notes.like("%Wyoming%"),
                Stratum.notes.like("%Congressional%")
            )
        ).unique().all()
        
        if wrong_parent_cds:
            errors.append(f"ERROR: Found {len(wrong_parent_cds)} non-Wyoming CDs incorrectly parented to Wyoming")
            for cd in wrong_parent_cds[:5]:
                errors.append(f"  - {cd.notes}")
        else:
            print("✓ No congressional districts incorrectly parented to Wyoming")
    
    return errors


def validate_demographic_strata(session):
    """Validate demographic strata are properly attached to geographic strata"""
    
    print("\n" + "="*60)
    print("VALIDATING DEMOGRAPHIC STRATA")
    print("="*60)
    
    errors = []
    
    # Group names for the new scheme
    group_names = {
        2: ("Age", 18),
        3: ("Income/AGI", 9),
        4: ("SNAP", 1),
        5: ("Medicaid", 1),
        6: ("EITC", 4),
    }
    
    # Validate each demographic group
    for group_id, (name, expected_per_geo) in group_names.items():
        strata = session.exec(
            select(Stratum).where(Stratum.stratum_group_id == group_id)
        ).unique().all()
        
        expected_total = expected_per_geo * 488  # 488 geographic areas
        print(f"\n{name} strata (group {group_id}):")
        print(f"  Found: {len(strata)}")
        print(f"  Expected: {expected_total} ({expected_per_geo} × 488 geographic areas)")
        
        if len(strata) != expected_total:
            errors.append(f"WARNING: {name} has {len(strata)} strata, expected {expected_total}")
    
    
    # Check parent relationships for a sample of demographic strata
    print("\nChecking parent relationships (sample):")
    sample_strata = session.exec(
        select(Stratum).where(Stratum.stratum_group_id > 1)  # All demographic groups
    ).unique().all()[:100]  # Take first 100
    
    correct_parents = 0
    wrong_parents = 0
    no_parents = 0
    
    for stratum in sample_strata:
        if stratum.parent_stratum_id:
            parent = session.get(Stratum, stratum.parent_stratum_id)
            if parent and parent.stratum_group_id == 1:  # Geographic parent
                correct_parents += 1
            else:
                wrong_parents += 1
                errors.append(f"ERROR: Stratum {stratum.stratum_id} has non-geographic parent")
        else:
            no_parents += 1
            errors.append(f"ERROR: Stratum {stratum.stratum_id} has no parent")
    
    print(f"  Sample of {len(sample_strata)} demographic strata:")
    print(f"    - With geographic parent: {correct_parents}")
    print(f"    - With wrong parent: {wrong_parents}")
    print(f"    - With no parent: {no_parents}")
    
    return errors


def validate_constraint_uniqueness(session):
    """Check that constraint combinations produce unique hashes"""
    
    print("\n" + "="*60)
    print("VALIDATING CONSTRAINT UNIQUENESS")
    print("="*60)
    
    errors = []
    
    # Check for duplicate definition_hashes
    all_strata = session.exec(select(Stratum)).unique().all()
    hash_counts = {}
    
    for stratum in all_strata:
        if stratum.definition_hash in hash_counts:
            hash_counts[stratum.definition_hash].append(stratum)
        else:
            hash_counts[stratum.definition_hash] = [stratum]
    
    duplicates = {h: strata for h, strata in hash_counts.items() if len(strata) > 1}
    
    if duplicates:
        errors.append(f"ERROR: Found {len(duplicates)} duplicate definition_hashes")
        for hash_val, strata in list(duplicates.items())[:3]:  # Show first 3
            errors.append(f"  Hash {hash_val[:10]}... appears {len(strata)} times:")
            for s in strata[:3]:
                errors.append(f"    - ID {s.stratum_id}: {s.notes[:50]}")
    else:
        print(f"✓ All {len(all_strata)} strata have unique definition_hashes")
    
    return errors


def main():
    """Run all validation checks"""
    
    DATABASE_URL = f"sqlite:///{STORAGE_FOLDER / 'policy_data.db'}"
    engine = create_engine(DATABASE_URL)
    
    all_errors = []
    
    with Session(engine) as session:
        # Run validation checks
        all_errors.extend(validate_geographic_hierarchy(session))
        all_errors.extend(validate_demographic_strata(session))
        all_errors.extend(validate_constraint_uniqueness(session))
    
    # Summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    
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