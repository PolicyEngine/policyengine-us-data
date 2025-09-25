#!/usr/bin/env python
"""
Comprehensive verification script for geo-stacked calibration (states and congressional districts).
Consolidates all key verification checks into one place.
"""

import sys
import argparse
from pathlib import Path
from sqlalchemy import create_engine, text
import numpy as np
import pandas as pd
import pickle
from scipy import sparse as sp
from policyengine_us import Microsimulation
from policyengine_us_data.datasets.cps.geo_stacking_calibration.metrics_matrix_geo_stacking_sparse import SparseGeoStackingMatrixBuilder

# Setup
DB_PATH = '/home/baogorek/devl/policyengine-us-data/policyengine_us_data/storage/policy_data.db'
DB_URI = f"sqlite:///{DB_PATH}"


def verify_target_counts(geo_level='congressional_district'):
    """Verify expected target counts for states or CDs."""
    print("=" * 70)
    print(f"TARGET COUNT VERIFICATION - {geo_level.upper()}")
    print("=" * 70)
    
    engine = create_engine(DB_URI)
    builder = SparseGeoStackingMatrixBuilder(DB_URI, time_period=2023)
    
    if geo_level == 'congressional_district':
        # Get all CDs
        query = """
        SELECT DISTINCT sc.value as cd_geoid
        FROM strata s
        JOIN stratum_constraints sc ON s.stratum_id = sc.stratum_id
        WHERE s.stratum_group_id = 1
          AND sc.constraint_variable = "congressional_district_geoid"
        ORDER BY sc.value
        """
        
        with engine.connect() as conn:
            result = conn.execute(text(query)).fetchall()
            all_geos = [row[0] for row in result]
        
        print(f"Total CDs found: {len(all_geos)}")
        
        # Get unique states for CDs
        unique_states = set()
        for cd in all_geos:
            state_fips = builder.get_state_fips_for_cd(cd)
            unique_states.add(state_fips)
        
        print(f"Unique states: {len(unique_states)}")
        
        # Calculate expected targets
        print("\n=== Expected Target Counts ===")
        categories = [
            ("National", 5),
            ("CD Age (18 × 436)", 18 * 436),
            ("CD Medicaid (1 × 436)", 436),
            ("CD SNAP household (1 × 436)", 436),
            ("State SNAP costs", len(unique_states)),
            ("CD AGI distribution (9 × 436)", 9 * 436),
            ("CD IRS SOI (50 × 436)", 50 * 436)
        ]
        
        running_total = 0
        for name, count in categories:
            running_total += count
            print(f"{name:30} {count:6,}  (running total: {running_total:6,})")
        
        expected_total = 30576
        
    else:  # state
        states_to_calibrate = [
            '1', '2', '4', '5', '6', '8', '9', '10', '11', '12', '13', '15', '16', '17', '18', 
            '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', 
            '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '44', '45', '46', '47', 
            '48', '49', '50', '51', '53', '54', '55', '56'
        ]
        all_geos = states_to_calibrate
        print(f"Total states: {len(all_geos)}")
        
        # Calculate expected targets for states
        print("\n=== Expected Target Counts ===")
        categories = [
            ("State Age (18 × 52)", 18 * 52),
            ("State SNAP (1 × 52)", 52),
            ("State Medicaid (1 × 52)", 52),
            ("State AGI distribution (9 × 52)", 9 * 52),
            ("National SSN targets", 1),
            ("National targets", 4)
        ]
        
        running_total = 0
        for name, count in categories:
            running_total += count
            print(f"{name:30} {count:6,}  (running total: {running_total:6,})")
        
        expected_total = 1497
    
    print(f"\n=== Total Expected: {running_total:,} ===")
    print(f"Expected target: {expected_total:,}")
    print(f"Match: {running_total == expected_total}")
    
    return running_total == expected_total


def verify_target_periods():
    """Check target periods in database."""
    print("\n" + "=" * 70)
    print("TARGET PERIOD VERIFICATION")
    print("=" * 70)
    
    engine = create_engine(DB_URI)
    
    # Check national target periods
    query = """
    SELECT DISTINCT period, COUNT(*) as count, 
           GROUP_CONCAT(DISTINCT variable) as sample_variables
    FROM targets t
    JOIN strata s ON t.stratum_id = s.stratum_id
    WHERE s.stratum_group_id = 2  -- National strata
    GROUP BY period
    ORDER BY period
    """
    
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
        print("\nNational target periods:")
        print(df.to_string())
    
    # Check CD target periods
    query = """
    SELECT DISTINCT t.period, COUNT(*) as count
    FROM targets t
    JOIN strata s ON t.stratum_id = s.stratum_id
    WHERE s.stratum_group_id = 1  -- Geographic
      AND EXISTS (
        SELECT 1 FROM stratum_constraints sc
        WHERE sc.stratum_id = s.stratum_id
          AND sc.constraint_variable = 'congressional_district_geoid'
      )
    GROUP BY t.period
    ORDER BY t.period
    LIMIT 5
    """
    
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
        print("\nCongressional district target periods (sample):")
        print(df.to_string())


def verify_ssn_constraint():
    """Verify SSN constraint is applied correctly."""
    print("\n" + "=" * 70)
    print("SSN CONSTRAINT VERIFICATION")
    print("=" * 70)
    
    engine = create_engine(DB_URI)
    builder = SparseGeoStackingMatrixBuilder(DB_URI, time_period=2023)
    
    # Load simulation
    sim = Microsimulation(dataset="hf://policyengine/policyengine-us-data/cps_2023.h5")
    
    # Check person-level SSN data
    person_mask = (sim.calculate('ssn_card_type', 2023) == 'NONE')
    person_weights = sim.calculate('person_weight', 2023).values
    
    print(f"Persons with ssn_card_type == 'NONE': {person_mask.sum():,}")
    print(f"Weighted count: {(person_mask * person_weights).sum():,.0f}")
    print(f"Expected 2023 target: 12,200,000")
    
    # Get national targets to check for SSN
    national_targets = builder.get_national_targets(sim)
    
    # Check for SSN targets
    ssn_targets = national_targets[
        (national_targets['constraint_variable'] == 'ssn_card_type') &
        (national_targets['constraint_value'] == 'NONE')
    ]
    
    if not ssn_targets.empty:
        print(f"\n✓ Found SSN targets in national targets:")
        for _, row in ssn_targets.iterrows():
            print(f"  Period {row['period']}: {row['value']:,.0f}")
    else:
        print("\n❌ No SSN targets found in national targets")
    
    # Test constraint application
    constraint_df = pd.DataFrame([{
        'constraint_variable': 'ssn_card_type',
        'operation': '=',
        'value': 'NONE'
    }])
    
    nonzero_indices, nonzero_values = builder.apply_constraints_to_sim_sparse(
        sim, constraint_df, 'person_count'
    )
    
    total_persons = nonzero_values.sum()
    print(f"\nConstraint application result: {total_persons:,.0f} persons")
    
    return abs(total_persons - 12200000) / 12200000 < 0.1  # Within 10%


def test_snap_cascading(num_geos=5, geo_level='congressional_district'):
    """Test that state SNAP costs cascade correctly."""
    print("\n" + "=" * 70)
    print(f"SNAP CASCADING TEST ({geo_level.upper()}, {num_geos} samples)")
    print("=" * 70)
    
    engine = create_engine(DB_URI)
    builder = SparseGeoStackingMatrixBuilder(DB_URI, time_period=2023)
    
    if geo_level == 'congressional_district':
        query = """
        SELECT DISTINCT sc.value as geo_id
        FROM strata s
        JOIN stratum_constraints sc ON s.stratum_id = sc.stratum_id
        WHERE s.stratum_group_id = 1
          AND sc.constraint_variable = "congressional_district_geoid"
        ORDER BY sc.value
        LIMIT :limit
        """
    else:
        query = """
        SELECT DISTINCT value as geo_id
        FROM (VALUES ('6'), ('48'), ('36'), ('12'), ('17')) AS t(value)
        LIMIT :limit
        """
    
    with engine.connect() as conn:
        result = conn.execute(text(query), {'limit': num_geos}).fetchall()
        test_geos = [row[0] for row in result]
    
    print(f"Testing with {geo_level}s: {test_geos}")
    
    # Load simulation
    dataset_uri = "/home/baogorek/devl/policyengine-us-data/policyengine_us_data/storage/stratified_extended_cps_2023.h5"
    sim = Microsimulation(dataset=dataset_uri)
    
    # Build matrix
    targets_df, X_sparse, household_id_mapping = builder.build_stacked_matrix_sparse(
        geo_level, 
        test_geos,
        sim
    )
    
    # Check state SNAP costs
    state_snap_costs = targets_df[
        (targets_df['geographic_level'] == 'state') & 
        (targets_df['variable'] == 'snap')
    ]
    
    print(f"\nState SNAP cost targets found: {len(state_snap_costs)}")
    if not state_snap_costs.empty:
        print("State SNAP costs by state (first 5):")
        for _, row in state_snap_costs.head().iterrows():
            print(f"  State {row['geographic_id']}: ${row['value']:,.0f}")
    
    print(f"\nMatrix shape: {X_sparse.shape}")
    print(f"Number of targets: {len(targets_df)}")
    
    return len(state_snap_costs) > 0


def check_loaded_targets(pkl_file=None, geo_level='congressional_district'):
    """Check targets from a saved pickle file."""
    if pkl_file is None:
        if geo_level == 'congressional_district':
            pkl_file = '/home/baogorek/Downloads/cd_calibration_data/cd_targets_df.pkl'
        else:
            pkl_file = '/home/baogorek/Downloads/state_calibration_data/state_targets_df.pkl'
    
    if not Path(pkl_file).exists():
        print(f"\nPickle file not found: {pkl_file}")
        return False
    
    print("\n" + "=" * 70)
    print(f"LOADED TARGETS CHECK ({geo_level.upper()})")
    print("=" * 70)
    
    with open(pkl_file, 'rb') as f:
        targets_df = pickle.load(f)
    
    print(f"Total targets loaded: {len(targets_df):,}")
    
    # Breakdown by geographic level
    for level in ['national', 'state', 'congressional_district']:
        count = len(targets_df[targets_df['geographic_level'] == level])
        if count > 0:
            print(f"  {level}: {count:,}")
    
    # Check for specific target types
    agi_targets = targets_df[
        (targets_df['description'].str.contains('adjusted_gross_income', na=False)) &
        (targets_df['variable'] == 'person_count')
    ]
    print(f"\nAGI distribution targets: {len(agi_targets):,}")
    
    state_snap = targets_df[
        (targets_df['geographic_level'] == 'state') & 
        (targets_df['variable'] == 'snap')
    ]
    print(f"State SNAP cost targets: {len(state_snap)}")
    
    irs_income_tax = targets_df[targets_df['variable'] == 'income_tax']
    print(f"Income tax targets: {len(irs_income_tax)}")
    
    return True


def main():
    """Run verification checks based on command line arguments."""
    parser = argparse.ArgumentParser(description='Verify geo-stacked calibration')
    parser.add_argument('--geo', choices=['state', 'congressional_district', 'cd'], 
                        default='congressional_district',
                        help='Geographic level to verify (default: congressional_district)')
    parser.add_argument('--skip-ssn', action='store_true',
                        help='Skip SSN constraint verification')
    parser.add_argument('--skip-snap', action='store_true',
                        help='Skip SNAP cascading test')
    parser.add_argument('--pkl-file', type=str,
                        help='Path to targets pickle file to check')
    
    args = parser.parse_args()
    
    # Normalize geo level
    geo_level = 'congressional_district' if args.geo == 'cd' else args.geo
    
    print("\n" + "=" * 70)
    print(f"CALIBRATION VERIFICATION - {geo_level.upper()}")
    print("=" * 70)
    
    results = {}
    
    # 1. Verify target counts
    results['target_counts'] = verify_target_counts(geo_level)
    
    # 2. Verify target periods
    verify_target_periods()
    
    # 3. Verify SSN constraint (only for state level)
    if not args.skip_ssn and geo_level == 'state':
        results['ssn_constraint'] = verify_ssn_constraint()
    
    # 4. Test SNAP cascading
    if not args.skip_snap:
        results['snap_cascading'] = test_snap_cascading(num_geos=5, geo_level=geo_level)
    
    # 5. Check loaded targets if file exists
    if args.pkl_file or Path(f'/home/baogorek/Downloads/{geo_level}_calibration_data').exists():
        results['loaded_targets'] = check_loaded_targets(args.pkl_file, geo_level)
    
    # Summary
    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)
    
    for check, passed in results.items():
        status = "✓" if passed else "❌"
        print(f"{status} {check.replace('_', ' ').title()}: {'PASSED' if passed else 'FAILED'}")
    
    if all(results.values()):
        print("\n✅ All verification checks passed!")
    else:
        print("\n❌ Some checks failed - review output above")
        sys.exit(1)


if __name__ == "__main__":
    main()