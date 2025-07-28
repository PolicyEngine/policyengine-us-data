"""
Validate transfer program modeling in Enhanced CPS.

This script addresses concerns about benefit underreporting,
program interactions, and geographic variation.
"""

import pandas as pd
import numpy as np
from policyengine_us import Microsimulation
from policyengine_us_data.datasets.cps.enhanced_cps import EnhancedCPS


def analyze_benefit_underreporting():
    """Analyze impact of CPS benefit underreporting on results."""
    
    sim = Microsimulation(dataset=EnhancedCPS, dataset_year=2022)
    
    programs = {
        'snap': {
            'variable': 'snap',
            'admin_total': 114.1,  # billions, from USDA
            'participation_rate': 0.82  # from USDA
        },
        'ssi': {
            'variable': 'ssi',
            'admin_total': 56.7,  # from SSA
            'participation_rate': 0.93
        },
        'tanf': {
            'variable': 'tanf',
            'admin_total': 7.1,  # from HHS
            'participation_rate': 0.23
        },
        'housing': {
            'variable': 'housing_benefit',
            'admin_total': 50.3,  # from HUD
            'participation_rate': 0.76
        }
    }
    
    results = []
    for program, info in programs.items():
        # Calculate totals
        benefit = sim.calculate(info['variable'], 2022)
        weight = sim.calculate('household_weight', 2022)
        
        # Total benefits
        total = (benefit * weight).sum() / 1e9  # billions
        
        # Participation
        participants = (benefit > 0).sum()
        weighted_participants = ((benefit > 0) * weight).sum() / 1e6  # millions
        
        # Underreporting factor
        underreporting = info['admin_total'] / total if total > 0 else np.inf
        
        results.append({
            'program': program,
            'enhanced_cps_total': total,
            'admin_total': info['admin_total'],
            'underreporting_factor': underreporting,
            'participants_millions': weighted_participants,
            'mean_benefit': benefit[benefit > 0].mean() if (benefit > 0).any() else 0
        })
    
    return pd.DataFrame(results)


def validate_program_interactions():
    """Validate joint participation in multiple programs."""
    
    sim = Microsimulation(dataset=EnhancedCPS, dataset_year=2022)
    
    # Get program benefits
    snap = sim.calculate('snap', 2022) > 0
    medicaid = sim.calculate('medicaid', 2022) > 0
    ssi = sim.calculate('ssi', 2022) > 0
    tanf = sim.calculate('tanf', 2022) > 0
    housing = sim.calculate('housing_benefit', 2022) > 0
    
    weight = sim.calculate('household_weight', 2022)
    
    # Calculate joint participation rates
    interactions = []
    
    # Two-way interactions
    programs = {
        'snap': snap,
        'medicaid': medicaid, 
        'ssi': ssi,
        'tanf': tanf,
        'housing': housing
    }
    
    for prog1_name, prog1 in programs.items():
        for prog2_name, prog2 in programs.items():
            if prog1_name < prog2_name:  # Avoid duplicates
                joint = prog1 & prog2
                joint_rate = (joint * weight).sum() / weight.sum() * 100
                
                # Conditional probabilities
                prob_2_given_1 = (joint * weight).sum() / (prog1 * weight).sum() * 100 if (prog1 * weight).sum() > 0 else 0
                prob_1_given_2 = (joint * weight).sum() / (prog2 * weight).sum() * 100 if (prog2 * weight).sum() > 0 else 0
                
                interactions.append({
                    'program_1': prog1_name,
                    'program_2': prog2_name,
                    'joint_participation_rate': joint_rate,
                    f'{prog2_name}_given_{prog1_name}': prob_2_given_1,
                    f'{prog1_name}_given_{prog2_name}': prob_1_given_2
                })
    
    return pd.DataFrame(interactions)


def analyze_benefit_cliffs():
    """Analyze effective marginal tax rates and benefit cliffs."""
    
    sim = Microsimulation(dataset=EnhancedCPS, dataset_year=2022)
    
    # Select sample households
    earnings = sim.calculate('employment_income', 2022)
    has_children = sim.calculate('num_children', 2022) > 0
    weight = sim.calculate('household_weight', 2022)
    
    # Focus on near-poor households with children
    median_earnings = np.median(earnings[earnings > 0])
    sample = (earnings > 0) & (earnings < median_earnings) & has_children
    
    if not sample.any():
        return pd.DataFrame()
    
    results = []
    
    # Calculate EMTR at different earnings levels
    for pct_increase in [0, 10, 20, 30, 40, 50]:
        # Create reform increasing earnings
        def earnings_reform(parameters):
            parameters.simulation.earnings_multiplier = 1 + pct_increase / 100
            return parameters
        
        reformed_sim = Microsimulation(
            dataset=EnhancedCPS,
            dataset_year=2022,
            reform=earnings_reform
        )
        
        # Calculate net income change
        original_net = sim.calculate('household_net_income', 2022)
        reformed_net = reformed_sim.calculate('household_net_income', 2022)
        
        # Calculate implicit EMTR
        earnings_change = earnings * pct_increase / 100
        net_change = reformed_net - original_net
        
        emtr = np.where(
            earnings_change > 0,
            1 - (net_change / earnings_change),
            0
        )
        
        # Focus on sample
        sample_emtr = emtr[sample]
        sample_weight = weight[sample]
        
        # Calculate statistics
        mean_emtr = np.average(sample_emtr, weights=sample_weight)
        median_emtr = np.median(sample_emtr)
        high_emtr = np.average(sample_emtr > 0.8, weights=sample_weight) * 100
        
        results.append({
            'earnings_increase_%': pct_increase,
            'mean_emtr': mean_emtr * 100,
            'median_emtr': median_emtr * 100,
            'pct_facing_cliff': high_emtr
        })
    
    return pd.DataFrame(results)


def validate_state_benefits():
    """Validate state-level benefit totals."""
    
    sim = Microsimulation(dataset=EnhancedCPS, dataset_year=2022)
    
    # Get state identifiers
    state = sim.calculate('state_code', 2022)
    weight = sim.calculate('household_weight', 2022)
    
    # Calculate state totals for major programs
    programs = ['snap', 'medicaid', 'tanf', 'unemployment_compensation']
    
    state_totals = []
    
    for state_code in range(1, 57):  # All states + DC
        if (state == state_code).any():
            state_data = {'state_code': state_code}
            
            for program in programs:
                benefit = sim.calculate(program, 2022)
                state_total = (benefit[state == state_code] * 
                             weight[state == state_code]).sum() / 1e9
                state_data[f'{program}_billions'] = state_total
            
            state_totals.append(state_data)
    
    return pd.DataFrame(state_totals)


def analyze_aca_subsidies():
    """Analyze ACA premium tax credit modeling."""
    
    sim = Microsimulation(dataset=EnhancedCPS, dataset_year=2022)
    
    # Get relevant variables
    ptc = sim.calculate('premium_tax_credit', 2022)
    income = sim.calculate('household_income', 2022)
    fpl = sim.calculate('household_fpg', 2022)
    weight = sim.calculate('household_weight', 2022)
    
    # Calculate income relative to FPL
    income_fpl_ratio = income / fpl
    
    # Analyze by FPL bracket
    brackets = [
        (0, 1.38, '100-138% FPL'),
        (1.38, 2.0, '138-200% FPL'),
        (2.0, 2.5, '200-250% FPL'),
        (2.5, 4.0, '250-400% FPL'),
        (4.0, np.inf, '>400% FPL')
    ]
    
    results = []
    for min_fpl, max_fpl, label in brackets:
        mask = (income_fpl_ratio >= min_fpl) & (income_fpl_ratio < max_fpl)
        
        if mask.any():
            total_ptc = (ptc[mask] * weight[mask]).sum() / 1e9
            recipients = ((ptc > 0) & mask).sum()
            weighted_recipients = (((ptc > 0) & mask) * weight).sum() / 1e6
            mean_ptc = ptc[(ptc > 0) & mask].mean() if ((ptc > 0) & mask).any() else 0
            
            results.append({
                'income_bracket': label,
                'total_ptc_billions': total_ptc,
                'recipients_millions': weighted_recipients,
                'mean_annual_ptc': mean_ptc
            })
    
    return pd.DataFrame(results)


def generate_benefit_validation_report():
    """Generate comprehensive benefit program validation report."""
    
    print("Enhanced CPS Benefit Program Validation Report")
    print("=" * 60)
    
    # Benefit underreporting
    print("\n1. Benefit Underreporting Analysis")
    print("-" * 40)
    underreporting_df = analyze_benefit_underreporting()
    print(underreporting_df.to_string(index=False))
    
    print("\nKey findings:")
    avg_underreporting = underreporting_df['underreporting_factor'].mean()
    print(f"- Average underreporting factor: {avg_underreporting:.2f}x")
    print("- Suggests need for benefit imputation or reweighting adjustment")
    
    # Program interactions
    print("\n\n2. Program Interaction Analysis")
    print("-" * 40)
    interactions_df = validate_program_interactions()
    print(interactions_df.to_string(index=False))
    
    # Benefit cliffs
    print("\n\n3. Effective Marginal Tax Rates")
    print("-" * 40)
    emtr_df = analyze_benefit_cliffs()
    if not emtr_df.empty:
        print(emtr_df.to_string(index=False))
        print(f"\n- Households facing >80% EMTR: {emtr_df['pct_facing_cliff'].max():.1f}%")
    
    # State validation
    print("\n\n4. Top 10 States by SNAP Benefits")
    print("-" * 40)
    state_df = validate_state_benefits()
    top_states = state_df.nlargest(10, 'snap_billions')[['state_code', 'snap_billions']]
    print(top_states.to_string(index=False))
    
    # ACA analysis
    print("\n\n5. ACA Premium Tax Credit Distribution")
    print("-" * 40)
    aca_df = analyze_aca_subsidies()
    print(aca_df.to_string(index=False))
    
    # Save results
    underreporting_df.to_csv('validation/benefit_underreporting.csv', index=False)
    interactions_df.to_csv('validation/program_interactions.csv', index=False)
    emtr_df.to_csv('validation/effective_marginal_tax_rates.csv', index=False)
    state_df.to_csv('validation/state_benefit_totals.csv', index=False)
    aca_df.to_csv('validation/aca_ptc_analysis.csv', index=False)
    
    print("\n\nValidation results saved to validation/ directory")


if __name__ == "__main__":
    generate_benefit_validation_report()