"""
Calculate actual performance metrics by comparing datasets to calibration targets.

This script computes how well each dataset matches the administrative targets
that the Enhanced CPS was calibrated to achieve.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Tuple


def calculate_target_achievement(
    loss_matrix: np.ndarray,
    weights: np.ndarray,
    targets: np.ndarray,
    target_names: List[str],
) -> pd.DataFrame:
    """
    Calculate how well a dataset achieves calibration targets.

    Args:
        loss_matrix: Matrix of household contributions to targets
        weights: Household weights
        targets: Target values to achieve
        target_names: Names of each target

    Returns:
        DataFrame with target achievement metrics
    """
    # Calculate achieved values
    achieved = loss_matrix.T @ weights

    # Calculate absolute relative error for each target
    relative_errors = np.abs(achieved - targets) / targets

    # Create results dataframe
    results = pd.DataFrame(
        {
            "Target": target_names,
            "Target_Value": targets,
            "Achieved_Value": achieved,
            "Absolute_Relative_Error": relative_errors,
            "Percent_Error": relative_errors * 100,
        }
    )

    return results


def compare_dataset_performance(
    enhanced_results: pd.DataFrame,
    baseline_results: pd.DataFrame,
    puf_results: pd.DataFrame,
) -> Dict[str, float]:
    """
    Compare Enhanced CPS performance to baseline datasets.

    Args:
        enhanced_results: Target achievement for Enhanced CPS
        baseline_results: Target achievement for baseline CPS
        puf_results: Target achievement for PUF

    Returns:
        Dictionary with comparison metrics
    """
    # Calculate where Enhanced CPS outperforms each dataset
    enhanced_better_than_cps = (
        enhanced_results["Absolute_Relative_Error"]
        < baseline_results["Absolute_Relative_Error"]
    ).mean() * 100

    enhanced_better_than_puf = (
        enhanced_results["Absolute_Relative_Error"]
        < puf_results["Absolute_Relative_Error"]
    ).mean() * 100

    # Calculate average improvement by target category
    categories = {
        "IRS Income": lambda x: "employment_income" in x
        or "capital_gains" in x,
        "Demographics": lambda x: "age_" in x or "population" in x,
        "Programs": lambda x: "snap" in x or "social_security" in x,
        "Tax Expenditures": lambda x: "salt" in x or "charitable" in x,
    }

    category_improvements = {}
    for cat_name, cat_filter in categories.items():
        mask = enhanced_results["Target"].apply(cat_filter)
        if mask.any():
            enhanced_avg = enhanced_results.loc[mask, "Percent_Error"].mean()
            baseline_avg = baseline_results.loc[mask, "Percent_Error"].mean()
            improvement = (baseline_avg - enhanced_avg) / baseline_avg * 100
            category_improvements[cat_name] = improvement

    return {
        "outperforms_cps_pct": enhanced_better_than_cps,
        "outperforms_puf_pct": enhanced_better_than_puf,
        "category_improvements": category_improvements,
    }


def generate_validation_report():
    """
    Generate comprehensive validation report.

    NOTE: This requires the actual loss matrices and weights from
    running the full dataset generation pipeline.
    """
    results_dir = Path("paper/results")
    results_dir.mkdir(exist_ok=True)

    # Placeholder for when we have actual data
    validation_summary = {
        "note": "Results require full dataset generation",
        "total_targets": "[TO BE CALCULATED]",
        "enhanced_cps": {
            "outperforms_cps_pct": "[TO BE CALCULATED]",
            "outperforms_puf_pct": "[TO BE CALCULATED]",
            "mean_absolute_error": "[TO BE CALCULATED]",
        },
        "category_performance": {
            "IRS_income_components": "[TO BE CALCULATED]",
            "demographic_targets": "[TO BE CALCULATED]",
            "program_participation": "[TO BE CALCULATED]",
            "tax_expenditures": "[TO BE CALCULATED]",
        },
    }

    # Save results
    with open(results_dir / "validation_metrics.json", "w") as f:
        json.dump(validation_summary, f, indent=2)

    print(f"Validation report saved to {results_dir}/validation_metrics.json")
    print("\nTo generate actual results:")
    print("1. Run 'make data' to generate all datasets")
    print("2. Run this script again with access to the loss matrices")


if __name__ == "__main__":
    generate_validation_report()
