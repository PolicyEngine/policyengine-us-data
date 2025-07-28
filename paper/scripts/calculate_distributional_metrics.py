"""
Calculate actual distributional metrics from Enhanced CPS datasets.

This script computes Gini coefficients, income shares, and other
distributional statistics when the full datasets are available.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def calculate_gini(values, weights):
    """
    Calculate Gini coefficient for weighted data.
    
    Args:
        values: Income/wealth values
        weights: Sample weights
        
    Returns:
        float: Gini coefficient between 0 and 1
    """
    # Remove any nan values
    mask = ~(np.isnan(values) | np.isnan(weights))
    values = values[mask]
    weights = weights[mask]
    
    # Sort by income
    sorted_indices = np.argsort(values)
    sorted_values = values[sorted_indices]
    sorted_weights = weights[sorted_indices]
    
    # Calculate cumulative proportions
    cumsum_weights = np.cumsum(sorted_weights)
    total_weight = cumsum_weights[-1]
    prop_weights = cumsum_weights / total_weight
    
    # Calculate income-weighted cumulative proportions  
    weighted_values = sorted_values * sorted_weights
    cumsum_weighted_values = np.cumsum(weighted_values)
    total_weighted_value = cumsum_weighted_values[-1]
    prop_values = cumsum_weighted_values / total_weighted_value
    
    # Calculate Gini using trapezoidal rule
    # Add (0,0) point
    prop_weights = np.concatenate([[0], prop_weights])
    prop_values = np.concatenate([[0], prop_values])
    
    # Area under Lorenz curve
    area_under_lorenz = np.trapz(prop_values, prop_weights)
    
    # Gini = (0.5 - area_under_lorenz) / 0.5 = 1 - 2 * area_under_lorenz
    gini = 1 - 2 * area_under_lorenz
    
    return gini


def calculate_top_shares(values, weights, percentiles=[90, 99]):
    """
    Calculate income shares for top percentiles.
    
    Args:
        values: Income values
        weights: Sample weights
        percentiles: List of percentiles (e.g., [90, 99] for top 10% and 1%)
        
    Returns:
        dict: Share of total income for each top group
    """
    # Remove any nan values
    mask = ~(np.isnan(values) | np.isnan(weights))
    values = values[mask]
    weights = weights[mask]
    
    # Calculate total weighted income
    total_income = np.sum(values * weights)
    
    # Calculate weighted percentiles
    shares = {}
    for p in percentiles:
        threshold = weighted_percentile(values, weights, p)
        mask = values >= threshold
        top_income = np.sum(values[mask] * weights[mask])
        shares[f"top_{100-p}%"] = top_income / total_income
    
    return shares


def weighted_percentile(values, weights, percentile):
    """
    Calculate weighted percentile.
    
    Args:
        values: Data values
        weights: Sample weights
        percentile: Percentile to calculate (0-100)
        
    Returns:
        float: Value at the given percentile
    """
    # Sort by values
    sorted_indices = np.argsort(values)
    sorted_values = values[sorted_indices]
    sorted_weights = weights[sorted_indices]
    
    # Calculate cumulative weight proportions
    cumsum = np.cumsum(sorted_weights)
    total = cumsum[-1]
    proportions = cumsum / total
    
    # Find value at percentile
    idx = np.searchsorted(proportions, percentile / 100.0)
    if idx >= len(sorted_values):
        return sorted_values[-1]
    return sorted_values[idx]


def load_and_calculate_metrics(dataset_name, year=2024):
    """
    Load dataset and calculate distributional metrics.
    
    Args:
        dataset_name: Name of dataset class
        year: Year to analyze
        
    Returns:
        dict: Calculated metrics
    """
    try:
        # Import the dataset
        if dataset_name == "CPS":
            from policyengine_us_data.datasets.cps.cps import CPS
            dataset = CPS(year=year)
        elif dataset_name == "EnhancedCPS":
            from policyengine_us_data.datasets.cps.enhanced_cps import EnhancedCPS
            dataset = EnhancedCPS(year=year)
        elif dataset_name == "PUF":
            from policyengine_us_data.datasets.irs.puf import PUF
            dataset = PUF(year=year)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        # Load the data
        from policyengine_us import Microsimulation
        sim = Microsimulation(dataset=dataset)
        
        # Get income and weights at appropriate level
        if dataset_name == "PUF":
            # PUF is at tax unit level
            income = sim.calculate("adjusted_gross_income", period=year)
            weights = sim.calculate("tax_unit_weight", period=year)
            level = "tax_unit"
        else:
            # CPS datasets can do household level
            income = sim.calculate("household_net_income", period=year)
            weights = sim.calculate("household_weight", period=year)
            level = "household"
        
        # Calculate metrics
        gini = calculate_gini(income, weights)
        shares = calculate_top_shares(income, weights, [90, 99])
        
        return {
            "dataset": dataset_name,
            "level": level,
            "gini": gini,
            "top_10_share": shares.get("top_10%", None),
            "top_1_share": shares.get("top_1%", None),
            "status": "calculated"
        }
        
    except Exception as e:
        print(f"Error calculating metrics for {dataset_name}: {e}")
        return {
            "dataset": dataset_name,
            "level": "unknown",
            "gini": "[TBC]",
            "top_10_share": "[TBC]",
            "top_1_share": "[TBC]",
            "status": "error"
        }


def save_metrics_to_csv(metrics_list, output_path):
    """Save calculated metrics to CSV for reference."""
    df = pd.DataFrame(metrics_list)
    df.to_csv(output_path, index=False)
    print(f"Saved metrics to {output_path}")


def main():
    """Calculate all distributional metrics."""
    print("Calculating distributional metrics for Enhanced CPS paper...")
    print("=" * 70)
    
    # Calculate metrics for each dataset
    datasets = ["CPS", "EnhancedCPS", "PUF"]
    metrics = []
    
    for dataset_name in datasets:
        print(f"\nProcessing {dataset_name}...")
        result = load_and_calculate_metrics(dataset_name)
        metrics.append(result)
        
        if result["status"] == "calculated":
            print(f"  Gini: {result['gini']:.3f}")
            print(f"  Top 10% share: {result['top_10_share']:.3f}")
            print(f"  Top 1% share: {result['top_1_share']:.3f}")
        else:
            print(f"  Unable to calculate - using placeholders")
    
    # Save results
    output_dir = Path("paper/results")
    output_dir.mkdir(exist_ok=True)
    save_metrics_to_csv(metrics, output_dir / "distributional_metrics.csv")
    
    print("\nNote: To get actual values, ensure datasets are generated with 'make data'")


if __name__ == "__main__":
    main()