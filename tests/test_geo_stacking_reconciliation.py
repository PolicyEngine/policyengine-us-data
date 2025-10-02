#!/usr/bin/env python3
"""
Unit tests for geo-stacking reconciliation logic.

These are self-contained tests that verify the reconciliation of
targets across geographic hierarchies (CD -> State -> National).
"""

import unittest
from unittest.mock import Mock, MagicMock, patch
import pandas as pd
import numpy as np


class TestReconciliationLogic(unittest.TestCase):
    """Test reconciliation of hierarchical targets."""
    
    def test_age_reconciliation_cd_to_state(self):
        """Test that CD age targets are adjusted to match state totals."""
        # Create mock CD targets for California
        cd_geoids = ['601', '602', '603']
        age_bins = ['age_0_4', 'age_5_9', 'age_10_14']
        
        # CD targets (survey-based, undercount state totals)
        cd_targets = []
        for cd in cd_geoids:
            for age_bin in age_bins:
                cd_targets.append({
                    'geographic_id': cd,
                    'stratum_group_id': 2,  # Age
                    'variable': 'person_count',
                    'constraint': age_bin,
                    'value': 10000,  # Each CD has 10,000 per age bin
                    'source': 'survey'
                })
        
        cd_df = pd.DataFrame(cd_targets)
        
        # State targets (administrative, authoritative)
        state_targets = []
        for age_bin in age_bins:
            state_targets.append({
                'geographic_id': '6',  # California FIPS
                'stratum_group_id': 2,
                'variable': 'person_count',
                'constraint': age_bin,
                'value': 33000,  # State total: 33,000 per age bin (10% higher)
                'source': 'administrative'
            })
        
        state_df = pd.DataFrame(state_targets)
        
        # Calculate reconciliation factors
        reconciliation_factors = {}
        for age_bin in age_bins:
            cd_sum = cd_df[cd_df['constraint'] == age_bin]['value'].sum()
            state_val = state_df[state_df['constraint'] == age_bin]['value'].iloc[0]
            reconciliation_factors[age_bin] = state_val / cd_sum if cd_sum > 0 else 1.0
        
        # Apply reconciliation
        reconciled_cd_df = cd_df.copy()
        reconciled_cd_df['original_value'] = reconciled_cd_df['value']
        reconciled_cd_df['reconciliation_factor'] = reconciled_cd_df['constraint'].map(reconciliation_factors)
        reconciled_cd_df['value'] = reconciled_cd_df['original_value'] * reconciled_cd_df['reconciliation_factor']
        
        # Verify reconciliation
        for age_bin in age_bins:
            reconciled_sum = reconciled_cd_df[reconciled_cd_df['constraint'] == age_bin]['value'].sum()
            state_val = state_df[state_df['constraint'] == age_bin]['value'].iloc[0]
            
            self.assertAlmostEqual(
                reconciled_sum, state_val, 2,
                f"Reconciled CD sum for {age_bin} should match state total"
            )
            
            # Check factor is correct (should be 1.1 = 33000/30000)
            factor = reconciliation_factors[age_bin]
            self.assertAlmostEqual(
                factor, 1.1, 4,
                f"Reconciliation factor for {age_bin} should be 1.1"
            )
    
    def test_medicaid_reconciliation_survey_to_admin(self):
        """Test Medicaid reconciliation from survey to administrative data."""
        # CD-level survey data (typically undercounts)
        cd_geoids = ['601', '602', '603', '604', '605']
        
        cd_medicaid = pd.DataFrame({
            'geographic_id': cd_geoids,
            'stratum_group_id': [5] * 5,  # Medicaid group
            'variable': ['person_count'] * 5,
            'value': [45000, 48000, 42000, 50000, 40000],  # Survey counts
            'source': ['survey'] * 5
        })
        
        cd_total = cd_medicaid['value'].sum()  # 225,000
        
        # State-level administrative data (authoritative)
        state_medicaid = pd.DataFrame({
            'geographic_id': ['6'],  # California
            'stratum_group_id': [5],
            'variable': ['person_count'],
            'value': [270000],  # 20% higher than survey
            'source': ['administrative']
        })
        
        state_total = state_medicaid['value'].iloc[0]
        
        # Calculate reconciliation
        reconciliation_factor = state_total / cd_total
        expected_factor = 1.2  # 270000 / 225000
        
        self.assertAlmostEqual(
            reconciliation_factor, expected_factor, 4,
            "Reconciliation factor should be 1.2"
        )
        
        # Apply reconciliation
        cd_medicaid['reconciliation_factor'] = reconciliation_factor
        cd_medicaid['original_value'] = cd_medicaid['value']
        cd_medicaid['value'] = cd_medicaid['value'] * reconciliation_factor
        
        # Verify total matches
        reconciled_total = cd_medicaid['value'].sum()
        self.assertAlmostEqual(
            reconciled_total, state_total, 2,
            "Reconciled CD total should match state administrative total"
        )
        
        # Verify each CD was scaled proportionally
        for i, cd in enumerate(cd_geoids):
            original = cd_medicaid.iloc[i]['original_value']
            reconciled = cd_medicaid.iloc[i]['value']
            expected_reconciled = original * expected_factor
            
            self.assertAlmostEqual(
                reconciled, expected_reconciled, 2,
                f"CD {cd} should be scaled by factor {expected_factor}"
            )
    
    def test_snap_household_reconciliation(self):
        """Test SNAP household count reconciliation."""
        # CD-level SNAP household counts
        cd_geoids = ['601', '602', '603']
        
        cd_snap = pd.DataFrame({
            'geographic_id': cd_geoids,
            'stratum_group_id': [4] * 3,  # SNAP group
            'variable': ['household_count'] * 3,
            'value': [20000, 25000, 18000],  # Survey counts
            'source': ['survey'] * 3
        })
        
        cd_total = cd_snap['value'].sum()  # 63,000
        
        # State-level administrative SNAP households
        state_snap = pd.DataFrame({
            'geographic_id': ['6'],
            'stratum_group_id': [4],
            'variable': ['household_count'],
            'value': [69300],  # 10% higher
            'source': ['administrative']
        })
        
        state_total = state_snap['value'].iloc[0]
        
        # Calculate and apply reconciliation
        factor = state_total / cd_total
        cd_snap['reconciled_value'] = cd_snap['value'] * factor
        
        # Verify
        self.assertAlmostEqual(
            factor, 1.1, 4,
            "SNAP reconciliation factor should be 1.1"
        )
        
        reconciled_total = cd_snap['reconciled_value'].sum()
        self.assertAlmostEqual(
            reconciled_total, state_total, 2,
            "Reconciled SNAP totals should match state administrative data"
        )
    
    def test_no_reconciliation_when_no_higher_level(self):
        """Test that targets are not modified when no higher-level data exists."""
        # CD targets with no corresponding state data
        cd_targets = pd.DataFrame({
            'geographic_id': ['601', '602'],
            'stratum_group_id': [999, 999],  # Some group without state targets
            'variable': ['custom_var', 'custom_var'],
            'value': [1000, 2000],
            'source': ['survey', 'survey']
        })
        
        # No state targets available
        state_targets = pd.DataFrame()  # Empty
        
        # Reconciliation should not change values
        reconciled = cd_targets.copy()
        reconciled['reconciliation_factor'] = 1.0  # No change
        
        # Verify no change
        for i in range(len(cd_targets)):
            self.assertEqual(
                reconciled.iloc[i]['value'], cd_targets.iloc[i]['value'],
                "Values should not change when no higher-level data exists"
            )
            self.assertEqual(
                reconciled.iloc[i]['reconciliation_factor'], 1.0,
                "Reconciliation factor should be 1.0 when no adjustment needed"
            )
    
    def test_undercount_percentage_calculation(self):
        """Test calculation of undercount percentages."""
        # Survey total: 900,000
        # Admin total: 1,000,000
        # Undercount: 100,000 (10%)
        
        survey_total = 900000
        admin_total = 1000000
        
        undercount = admin_total - survey_total
        undercount_pct = (undercount / admin_total) * 100
        
        self.assertAlmostEqual(
            undercount_pct, 10.0, 2,
            "Undercount percentage should be 10%"
        )
        
        # Alternative calculation using factor
        factor = admin_total / survey_total
        undercount_pct_alt = (1 - 1/factor) * 100
        
        self.assertAlmostEqual(
            undercount_pct_alt, 10.0, 2,
            "Alternative undercount calculation should also give 10%"
        )
    
    def test_hierarchical_reconciliation_order(self):
        """Test that reconciliation preserves hierarchical consistency."""
        # National -> State -> CD hierarchy
        
        # National target
        national_total = 1000000
        
        # State targets (should sum to national)
        state_targets = pd.DataFrame({
            'state_fips': ['6', '36', '48'],  # CA, NY, TX
            'value': [400000, 350000, 250000]
        })
        
        # CD targets (should sum to respective states)
        cd_targets = pd.DataFrame({
            'cd_geoid': ['601', '602', '3601', '3602', '4801'],
            'state_fips': ['6', '6', '36', '36', '48'],
            'value': [180000, 200000, 160000, 170000, 240000]  # Slightly off from state totals
        })
        
        # Step 1: Reconcile states to national
        state_sum = state_targets['value'].sum()
        self.assertEqual(state_sum, national_total, "States should sum to national")
        
        # Step 2: Reconcile CDs to states
        for state_fips in ['6', '36', '48']:
            state_total = state_targets[state_targets['state_fips'] == state_fips]['value'].iloc[0]
            cd_state_mask = cd_targets['state_fips'] == state_fips
            cd_state_sum = cd_targets[cd_state_mask]['value'].sum()
            
            if cd_state_sum > 0:
                factor = state_total / cd_state_sum
                cd_targets.loc[cd_state_mask, 'reconciled_value'] = (
                    cd_targets.loc[cd_state_mask, 'value'] * factor
                )
        
        # Verify hierarchical consistency
        for state_fips in ['6', '36', '48']:
            state_total = state_targets[state_targets['state_fips'] == state_fips]['value'].iloc[0]
            cd_state_mask = cd_targets['state_fips'] == state_fips
            cd_reconciled_sum = cd_targets[cd_state_mask]['reconciled_value'].sum()
            
            self.assertAlmostEqual(
                cd_reconciled_sum, state_total, 2,
                f"Reconciled CDs in state {state_fips} should sum to state total"
            )
        
        # Verify grand total
        total_reconciled = cd_targets['reconciled_value'].sum()
        self.assertAlmostEqual(
            total_reconciled, national_total, 2,
            "All reconciled CDs should sum to national total"
        )


class TestReconciliationEdgeCases(unittest.TestCase):
    """Test edge cases in reconciliation logic."""
    
    def test_zero_survey_values(self):
        """Test handling of zero values in survey data."""
        cd_targets = pd.DataFrame({
            'geographic_id': ['601', '602', '603'],
            'value': [0, 1000, 2000]  # First CD has zero
        })
        
        state_total = 3300  # 10% higher than non-zero sum
        
        # Calculate factor based on non-zero values
        non_zero_sum = cd_targets[cd_targets['value'] > 0]['value'].sum()
        factor = state_total / non_zero_sum if non_zero_sum > 0 else 1.0
        
        # Apply reconciliation
        cd_targets['reconciled'] = cd_targets['value'] * factor
        
        # Zero should remain zero
        self.assertEqual(
            cd_targets.iloc[0]['reconciled'], 0,
            "Zero values should remain zero after reconciliation"
        )
        
        # Non-zero values should be scaled
        self.assertAlmostEqual(
            cd_targets.iloc[1]['reconciled'], 1100, 2,
            "Non-zero values should be scaled appropriately"
        )
    
    def test_missing_geographic_coverage(self):
        """Test when some CDs are missing from survey data."""
        # Only 3 of 5 CDs have data
        cd_targets = pd.DataFrame({
            'geographic_id': ['601', '602', '603'],
            'value': [30000, 35000, 25000]
        })
        
        # State total covers all 5 CDs
        state_total = 150000  # Implies 60,000 for missing CDs
        
        # Can only reconcile the CDs we have
        cd_sum = cd_targets['value'].sum()
        available_ratio = cd_sum / state_total  # 90,000 / 150,000 = 0.6
        
        self.assertAlmostEqual(
            available_ratio, 0.6, 4,
            "Available CDs represent 60% of state total"
        )
        
        # Options for handling:
        # 1. Scale up existing CDs (not recommended - distorts distribution)
        # 2. Flag as incomplete coverage (recommended)
        # 3. Impute missing CDs first, then reconcile
        
        # Test option 2: Flag incomplete coverage
        coverage_threshold = 0.8  # Require 80% coverage
        has_sufficient_coverage = available_ratio >= coverage_threshold
        
        self.assertFalse(
            has_sufficient_coverage,
            "Should flag insufficient coverage when <80% of CDs present"
        )
    
    def test_negative_values(self):
        """Test handling of negative values (should not occur but test anyway)."""
        cd_targets = pd.DataFrame({
            'geographic_id': ['601', '602'],
            'value': [-1000, 2000]  # Negative value (data error)
        })
        
        # Should either:
        # 1. Raise an error
        # 2. Treat as zero
        # 3. Take absolute value
        
        # Test option 2: Treat negatives as zero
        cd_targets['cleaned_value'] = cd_targets['value'].apply(lambda x: max(0, x))
        
        self.assertEqual(
            cd_targets.iloc[0]['cleaned_value'], 0,
            "Negative values should be treated as zero"
        )
        
        self.assertEqual(
            cd_targets.iloc[1]['cleaned_value'], 2000,
            "Positive values should remain unchanged"
        )


if __name__ == '__main__':
    unittest.main()