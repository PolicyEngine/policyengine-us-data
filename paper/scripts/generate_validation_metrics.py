"""
Generate validation metrics for the enhanced CPS paper.

This script computes all validation metrics comparing the Enhanced CPS
to the baseline CPS and PUF datasets. Results are saved as CSV files
for inclusion in the paper.
"""

import pandas as pd
import numpy as np
from policyengine_us import Microsimulation
from policyengine_us_data.datasets.cps.enhanced_cps import EnhancedCPS
from policyengine_us_data.datasets.cps.cps import CPS
from policyengine_us_data.datasets.irs.puf import PUF
import json
import os


def calculate_validation_metrics(year: int = 2024):
    """
    Calculate validation metrics across all three datasets.
    
    Args:
        year: Tax year to analyze
        
    Returns:
        dict: Validation results by dataset and metric type
    """
    results = {}
    
    # Initialize datasets
    print(f"Loading datasets for {year}...")
    
    try:
        # Load each dataset
        enhanced_cps = EnhancedCPS(year=year)
        baseline_cps = CPS(year=year)
        puf = PUF(year=year)
        
        # Get the loss matrix targets that Enhanced CPS was calibrated to
        from policyengine_us_data.utils.loss import build_loss_matrix
        
        loss_matrix, targets, names = build_loss_matrix(
            EnhancedCPS, str(year)
        )
        
        print(f"Found {len(targets)} calibration targets")
        
        # Calculate how well each dataset matches the targets
        for dataset_name, dataset in [
            ("Enhanced CPS", enhanced_cps),
            ("Baseline CPS", baseline_cps), 
            ("PUF", puf)
        ]:
            print(f"\nCalculating metrics for {dataset_name}...")
            
            # Create microsimulation
            sim = Microsimulation(dataset=dataset)
            
            # Calculate achieved values for each target
            # This is placeholder - actual implementation would compute
            # each target value using the microsimulation
            
            results[dataset_name] = {
                "total_targets": len(targets),
                "dataset_year": year,
                "status": "TO BE CALCULATED"
            }
            
    except Exception as e:
        print(f"Error loading datasets: {e}")
        print("Creating placeholder results...")
        
        results = {
            "Enhanced CPS": {
                "total_targets": "[TO BE CALCULATED]",
                "outperforms_cps_pct": "[TO BE CALCULATED]",
                "outperforms_puf_pct": "[TO BE CALCULATED]",
            },
            "Baseline CPS": {
                "total_targets": "[TO BE CALCULATED]",
            },
            "PUF": {
                "total_targets": "[TO BE CALCULATED]",
            }
        }
    
    return results


def calculate_poverty_metrics(year: int = 2024):
    """
    Calculate poverty metrics for each dataset.
    
    Args:
        year: Tax year to analyze
        
    Returns:
        pd.DataFrame: Poverty rates by dataset
    """
    print(f"\nCalculating poverty metrics for {year}...")
    
    # Placeholder implementation
    # Actual implementation would calculate SPM poverty rates
    
    results = pd.DataFrame({
        "Dataset": ["CPS", "PUF", "Enhanced CPS"],
        "SPM Poverty Rate": ["[TO BE CALCULATED]", "[TO BE CALCULATED]", "[TO BE CALCULATED]"],
        "Child Poverty Rate": ["[TO BE CALCULATED]", "[TO BE CALCULATED]", "[TO BE CALCULATED]"],
        "Senior Poverty Rate": ["[TO BE CALCULATED]", "[TO BE CALCULATED]", "[TO BE CALCULATED]"],
    })
    
    return results


def calculate_income_distribution_metrics(year: int = 2024):
    """
    Calculate income distribution metrics.
    
    Args:
        year: Tax year to analyze
        
    Returns:
        pd.DataFrame: Distribution metrics by dataset
    """
    print(f"\nCalculating income distribution metrics for {year}...")
    
    # Placeholder implementation
    results = pd.DataFrame({
        "Dataset": ["CPS", "PUF", "Enhanced CPS"],
        "Gini Coefficient": ["[TO BE CALCULATED]", "[TO BE CALCULATED]", "[TO BE CALCULATED]"],
        "Top 1% Share": ["[TO BE CALCULATED]", "[TO BE CALCULATED]", "[TO BE CALCULATED]"],
        "Top 10% Share": ["[TO BE CALCULATED]", "[TO BE CALCULATED]", "[TO BE CALCULATED]"],
        "Bottom 50% Share": ["[TO BE CALCULATED]", "[TO BE CALCULATED]", "[TO BE CALCULATED]"],
    })
    
    return results


def calculate_policy_reform_impacts(year: int = 2024):
    """
    Calculate revenue impacts of top rate reform.
    
    Args:
        year: Tax year to analyze
        
    Returns:
        pd.DataFrame: Revenue projections by dataset
    """
    print(f"\nCalculating policy reform impacts for {year}...")
    
    # Placeholder for top marginal rate increase from 37% to 39.6%
    results = pd.DataFrame({
        "Dataset": ["CPS", "PUF", "Enhanced CPS"],
        "Revenue Impact ($B)": ["[TO BE CALCULATED]", "[TO BE CALCULATED]", "[TO BE CALCULATED]"],
        "Affected Tax Units": ["[TO BE CALCULATED]", "[TO BE CALCULATED]", "[TO BE CALCULATED]"],
        "Average Tax Increase": ["[TO BE CALCULATED]", "[TO BE CALCULATED]", "[TO BE CALCULATED]"],
    })
    
    return results


def main():
    """Generate all paper results."""
    
    # Create results directory
    results_dir = "paper/results"
    os.makedirs(results_dir, exist_ok=True)
    
    print("Generating validation metrics for PolicyEngine Enhanced CPS paper")
    print("=" * 70)
    
    # Generate validation metrics
    validation_results = calculate_validation_metrics()
    
    # Save as JSON for reference
    with open(f"{results_dir}/validation_summary.json", "w") as f:
        json.dump(validation_results, f, indent=2)
    
    # Generate poverty metrics table
    poverty_df = calculate_poverty_metrics()
    poverty_df.to_csv(f"{results_dir}/poverty_metrics.csv", index=False)
    
    # Generate income distribution table  
    dist_df = calculate_income_distribution_metrics()
    dist_df.to_csv(f"{results_dir}/income_distribution.csv", index=False)
    
    # Generate policy reform table
    reform_df = calculate_policy_reform_impacts()
    reform_df.to_csv(f"{results_dir}/policy_reform_impacts.csv", index=False)
    
    print(f"\nResults saved to {results_dir}/")
    print("\nNOTE: All metrics marked as [TO BE CALCULATED] require full")
    print("dataset generation and microsimulation runs to compute actual values.")
    

if __name__ == "__main__":
    main()