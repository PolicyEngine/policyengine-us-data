"""
Run QRF diagnostics on actual PolicyEngine data to generate the statistics
cited in the methodology section.
"""

import numpy as np
import pandas as pd
from policyengine_us_data import CPS_2024, ExtendedCPS_2024, PUF_2024
from policyengine_us import Microsimulation
import warnings

warnings.filterwarnings("ignore")

# Import the diagnostic functions
import sys

sys.path.append("/Users/maxghenis/PolicyEngine/policyengine-us-data")
from validation.qrf_diagnostics import (
    analyze_common_support,
    validate_qrf_accuracy,
    test_joint_distribution_preservation,
    create_diagnostic_plots,
)


def load_dataset_as_dataframe(dataset_class):
    """Load a PolicyEngine dataset and convert to DataFrame for analysis."""
    print(f"Loading {dataset_class.__name__}...")

    # Create microsimulation
    sim = Microsimulation(dataset=dataset_class)

    # Define variables to extract
    demographic_vars = [
        "age",
        "sex",
        "filing_status",
        "tax_unit_dependents",
        "is_tax_unit_head",
        "is_tax_unit_spouse",
        "is_tax_unit_dependent",
    ]

    income_vars = [
        "employment_income",
        "interest_income",
        "dividend_income",
        "business_income",
        "capital_gains",
        "pension_income",
        "social_security",
    ]

    # Extract data
    data = {}

    # Demographics
    for var in demographic_vars:
        try:
            if var == "age":
                data[var] = sim.calculate(var, 2024, map_to="person").values
            elif var == "sex":
                # Convert sex to binary (1 for male, 0 for female)
                sex_values = sim.calculate(var, 2024, map_to="person").values
                data[var] = (sex_values == "MALE").astype(int)
            elif var == "tax_unit_dependents":
                data["num_dependents"] = sim.calculate(
                    var, 2024, map_to="tax_unit"
                ).values
            else:
                data[var] = sim.calculate(var, 2024, map_to="person").values
        except:
            print(f"Could not extract {var}")
            data[var] = np.zeros(len(sim.calculate("person_id", 2024).values))

    # Income variables
    for var in income_vars:
        try:
            # Map common names
            if var == "employment_income":
                actual_var = "earned_income"
            elif var == "business_income":
                actual_var = "self_employment_income"
            elif var == "capital_gains":
                actual_var = "long_term_capital_gains"
            else:
                actual_var = var

            data[var.replace("_income", "")] = sim.calculate(
                actual_var, 2024, map_to="person"
            ).values
        except:
            print(f"Could not extract {var}")
            data[var.replace("_income", "")] = np.zeros(
                len(sim.calculate("person_id", 2024).values)
            )

    # Add weights
    data["weight"] = sim.calculate("person_weight", 2024).values

    return pd.DataFrame(data)


def calculate_variance_explained(puf_data, predictors, target_vars):
    """Calculate R-squared for each target variable using the predictors."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import r2_score

    results = {}

    X = puf_data[predictors].fillna(0)

    for target in target_vars:
        if target in puf_data.columns:
            y = puf_data[target].fillna(0)

            # Fit random forest to get variance explained
            rf = RandomForestRegressor(n_estimators=100, random_state=42)

            # Only use non-zero values for income variables
            if target in ["wages", "capital", "retirement"]:
                mask = y > 0
                if mask.sum() > 100:  # Need enough samples
                    X_subset = X[mask]
                    y_subset = y[mask]

                    rf.fit(X_subset, y_subset)
                    y_pred = rf.predict(X_subset)
                    r2 = r2_score(y_subset, y_pred)
                    results[target] = r2
                else:
                    results[target] = 0.0
            else:
                if len(y) > 100:
                    rf.fit(X, y)
                    y_pred = rf.predict(X)
                    r2 = r2_score(y, y_pred)
                    results[target] = r2
                else:
                    results[target] = 0.0

    return results


def main():
    """Run full diagnostic analysis on real data."""

    # Load datasets
    print("Loading CPS 2024 data...")
    cps_data = load_dataset_as_dataframe(CPS_2024)

    print("\nLoading PUF 2024 data...")
    puf_data = load_dataset_as_dataframe(PUF_2024)

    print("\nLoading Enhanced CPS 2024 data...")
    enhanced_data = load_dataset_as_dataframe(ExtendedCPS_2024)

    # Define predictors
    predictors = [
        "age",
        "sex",
        "filing_status",
        "num_dependents",
        "is_tax_unit_head",
        "is_tax_unit_spouse",
        "is_tax_unit_dependent",
    ]

    # Ensure predictors exist in all datasets
    for pred in predictors:
        if pred not in cps_data.columns:
            cps_data[pred] = 0
        if pred not in puf_data.columns:
            puf_data[pred] = 0
        if pred not in enhanced_data.columns:
            enhanced_data[pred] = 0

    print("\n" + "=" * 60)
    print("QUANTILE REGRESSION FOREST DIAGNOSTIC REPORT")
    print("=" * 60)

    # 1. Common support analysis
    print("\n1. COMMON SUPPORT ANALYSIS")
    print("-" * 40)

    support_results = analyze_common_support(cps_data, puf_data, predictors)
    print(support_results.round(3).to_string())

    print("\nSummary:")
    print(
        f"- Average overlap coefficient: {support_results['overlap_coefficient'].mean():.3f}"
    )
    print(
        f"- All overlap coefficients > 0.85: {(support_results['overlap_coefficient'] > 0.85).all()}"
    )
    print(
        f"- Variables with SMD > 0.25: {(support_results['standardized_mean_diff'] > 0.25).sum()}"
    )
    print(
        f"- All SMDs < 0.25: {(support_results['standardized_mean_diff'] < 0.25).all()}"
    )
    print(
        f"- Variables with significant KS test (p<0.05): {(support_results['ks_pvalue'] < 0.05).sum()}"
    )
    print(
        f"- All KS tests non-significant (p>0.05): {(support_results['ks_pvalue'] > 0.05).all()}"
    )

    # 2. Variance explained analysis
    print("\n\n2. VARIANCE EXPLAINED BY PREDICTORS")
    print("-" * 40)

    # Map target variables
    target_map = {
        "wages": "employment",
        "capital": "capital_gains",
        "retirement": "pension",
    }

    variance_results = calculate_variance_explained(
        puf_data, predictors, list(target_map.values())
    )

    print("R-squared values:")
    for display_name, actual_name in target_map.items():
        if actual_name in variance_results:
            print(
                f"- {display_name.capitalize()}: {variance_results[actual_name]*100:.0f}%"
            )

    # 3. Joint distribution preservation
    print("\n\n3. JOINT DISTRIBUTION PRESERVATION")
    print("-" * 40)

    var_pairs = [
        ("employment", "age"),
        ("interest", "dividend"),
        ("capital_gains", "business"),
        ("pension", "social_security"),
    ]

    # Filter to pairs that exist in data
    valid_pairs = []
    for v1, v2 in var_pairs:
        if v1 in enhanced_data.columns and v2 in enhanced_data.columns:
            if v1 in puf_data.columns and v2 in puf_data.columns:
                valid_pairs.append((v1, v2))

    if valid_pairs:
        joint_results = test_joint_distribution_preservation(
            puf_data, enhanced_data, valid_pairs
        )
        print(joint_results.round(3).to_string(index=False))

        print("\nSummary:")
        print(
            f"- All correlation differences < 0.05: {(joint_results['correlation_diff'] < 0.05).all()}"
        )
        print(
            f"- Average correlation difference: {joint_results['correlation_diff'].mean():.3f}"
        )

    # Save results
    print("\n\nSAVING RESULTS...")
    print("-" * 40)

    # Create validation directory if it doesn't exist
    import os

    os.makedirs("validation/outputs", exist_ok=True)

    # Save detailed results
    support_results.to_csv("validation/outputs/common_support_analysis.csv")
    print(
        "✓ Saved common support analysis to validation/outputs/common_support_analysis.csv"
    )

    with open("validation/outputs/variance_explained.txt", "w") as f:
        f.write("Variance Explained by Predictors\n")
        f.write("=" * 40 + "\n\n")
        for display_name, actual_name in target_map.items():
            if actual_name in variance_results:
                f.write(
                    f"{display_name.capitalize()}: {variance_results[actual_name]*100:.0f}%\n"
                )
    print(
        "✓ Saved variance explained results to validation/outputs/variance_explained.txt"
    )

    if valid_pairs:
        joint_results.to_csv(
            "validation/outputs/joint_distribution_tests.csv", index=False
        )
        print(
            "✓ Saved joint distribution tests to validation/outputs/joint_distribution_tests.csv"
        )

    # Create summary report
    with open("validation/outputs/qrf_diagnostics_summary.txt", "w") as f:
        f.write("QRF DIAGNOSTICS SUMMARY\n")
        f.write("=" * 60 + "\n")
        f.write(f"Generated: {pd.Timestamp.now()}\n\n")

        f.write("1. COMMON SUPPORT\n")
        f.write("-" * 40 + "\n")
        f.write(
            f"Average overlap coefficient: {support_results['overlap_coefficient'].mean():.3f}\n"
        )
        f.write(
            f"All overlap coefficients > 0.85: {(support_results['overlap_coefficient'] > 0.85).all()}\n"
        )
        f.write(
            f"All SMDs < 0.25: {(support_results['standardized_mean_diff'] < 0.25).all()}\n"
        )
        f.write(
            f"All KS tests p > 0.05: {(support_results['ks_pvalue'] > 0.05).all()}\n\n"
        )

        f.write("2. VARIANCE EXPLAINED\n")
        f.write("-" * 40 + "\n")
        for display_name, actual_name in target_map.items():
            if actual_name in variance_results:
                f.write(
                    f"{display_name.capitalize()}: {variance_results[actual_name]*100:.0f}%\n"
                )

        if valid_pairs:
            f.write("\n3. JOINT DISTRIBUTIONS\n")
            f.write("-" * 40 + "\n")
            f.write(
                f"All correlation differences < 0.05: {(joint_results['correlation_diff'] < 0.05).all()}\n"
            )
            f.write(
                f"Average correlation difference: {joint_results['correlation_diff'].mean():.3f}\n"
            )

    print("✓ Saved summary to validation/outputs/qrf_diagnostics_summary.txt")

    print("\n" + "=" * 60)
    print("DIAGNOSTICS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
