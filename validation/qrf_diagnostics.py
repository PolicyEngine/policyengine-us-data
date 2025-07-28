"""
Diagnostics for Quantile Regression Forest imputation methodology.

This script provides validation of the QRF approach including:
- Common support analysis
- Out-of-sample prediction accuracy
- Joint distribution preservation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from quantile_forest import RandomForestQuantileRegressor
from scipy import stats


def analyze_common_support(cps_data, puf_data, predictors):
    """Analyze overlap in predictor distributions between CPS and PUF."""
    
    results = {}
    
    for var in predictors:
        # Calculate distributions
        cps_dist = cps_data[var].dropna()
        puf_dist = puf_data[var].dropna()
        
        # Overlap coefficient (Weitzman 1970)
        # OVL = sum(min(f(x), g(x))) where f,g are densities
        bins = np.histogram_bin_edges(
            np.concatenate([cps_dist, puf_dist]), bins=50
        )
        
        cps_hist, _ = np.histogram(cps_dist, bins=bins, density=True)
        puf_hist, _ = np.histogram(puf_dist, bins=bins, density=True)
        
        bin_width = bins[1] - bins[0]
        overlap = np.sum(np.minimum(cps_hist, puf_hist)) * bin_width
        
        # Kolmogorov-Smirnov test
        ks_stat, ks_pval = stats.ks_2samp(cps_dist, puf_dist)
        
        # Standardized mean difference
        smd = (cps_dist.mean() - puf_dist.mean()) / np.sqrt(
            (cps_dist.var() + puf_dist.var()) / 2
        )
        
        results[var] = {
            'overlap_coefficient': overlap,
            'ks_statistic': ks_stat,
            'ks_pvalue': ks_pval,
            'standardized_mean_diff': abs(smd),
            'cps_mean': cps_dist.mean(),
            'puf_mean': puf_dist.mean(),
            'cps_std': cps_dist.std(),
            'puf_std': puf_dist.std()
        }
    
    return pd.DataFrame(results).T


def validate_qrf_accuracy(puf_data, predictors, target_vars, n_estimators=100):
    """Validate QRF out-of-sample prediction accuracy."""
    
    # Split data
    X = puf_data[predictors]
    results = {}
    
    for target in target_vars[:5]:  # Sample of variables
        y = puf_data[target]
        
        # Remove missing
        mask = ~(X.isna().any(axis=1) | y.isna())
        X_clean = X[mask]
        y_clean = y[mask]
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_clean, y_clean, test_size=0.2, random_state=42
        )
        
        # Fit QRF
        qrf = RandomForestQuantileRegressor(
            n_estimators=n_estimators,
            random_state=42
        )
        qrf.fit(X_train, y_train)
        
        # Predictions at multiple quantiles
        quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
        predictions = {}
        
        for q in quantiles:
            pred = qrf.predict(X_test, quantiles=[q])
            predictions[f'q{int(q*100)}'] = pred.flatten()
        
        # Calculate metrics
        median_pred = predictions['q50']
        mae = np.mean(np.abs(y_test - median_pred))
        rmse = np.sqrt(np.mean((y_test - median_pred)**2))
        
        # Coverage of prediction intervals
        coverage_90 = np.mean(
            (y_test >= predictions['q10']) & 
            (y_test <= predictions['q90'])
        )
        coverage_50 = np.mean(
            (y_test >= predictions['q25']) & 
            (y_test <= predictions['q75'])
        )
        
        # Compare to simple methods
        # Hot-deck (random sampling from training)
        hotdeck_pred = np.random.choice(y_train, size=len(y_test))
        hotdeck_mae = np.mean(np.abs(y_test - hotdeck_pred))
        
        # Linear regression
        from sklearn.linear_model import LinearRegression
        lr = LinearRegression().fit(X_train, y_train)
        lr_pred = lr.predict(X_test)
        lr_mae = np.mean(np.abs(y_test - lr_pred))
        
        results[target] = {
            'qrf_mae': mae,
            'qrf_rmse': rmse,
            'hotdeck_mae': hotdeck_mae,
            'linear_mae': lr_mae,
            'qrf_improvement_vs_hotdeck': (hotdeck_mae - mae) / hotdeck_mae * 100,
            'qrf_improvement_vs_linear': (lr_mae - mae) / lr_mae * 100,
            'coverage_90pct': coverage_90,
            'coverage_50pct': coverage_50
        }
    
    return pd.DataFrame(results).T


def test_joint_distribution_preservation(original_data, imputed_data, var_pairs):
    """Test whether joint distributions are preserved in imputation."""
    
    results = []
    
    for var1, var2 in var_pairs:
        # Original correlation
        orig_corr = original_data[[var1, var2]].corr().iloc[0, 1]
        
        # Imputed correlation
        imp_corr = imputed_data[[var1, var2]].corr().iloc[0, 1]
        
        # Copula-based dependence (Kendall's tau)
        orig_tau = stats.kendalltau(
            original_data[var1].dropna(), 
            original_data[var2].dropna()
        )[0]
        imp_tau = stats.kendalltau(
            imputed_data[var1].dropna(),
            imputed_data[var2].dropna()
        )[0]
        
        # Joint distribution test (2D KS test approximation)
        # Using average of marginal KS statistics
        ks1 = stats.ks_2samp(
            original_data[var1].dropna(),
            imputed_data[var1].dropna()
        )[0]
        ks2 = stats.ks_2samp(
            original_data[var2].dropna(),
            imputed_data[var2].dropna()
        )[0]
        joint_ks = (ks1 + ks2) / 2
        
        results.append({
            'variable_pair': f'{var1}-{var2}',
            'original_correlation': orig_corr,
            'imputed_correlation': imp_corr,
            'correlation_diff': abs(orig_corr - imp_corr),
            'original_kendall_tau': orig_tau,
            'imputed_kendall_tau': imp_tau,
            'tau_diff': abs(orig_tau - imp_tau),
            'joint_ks_statistic': joint_ks
        })
    
    return pd.DataFrame(results)


def create_diagnostic_plots(cps_data, puf_data, predictors):
    """Create diagnostic plots for common support."""
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for i, var in enumerate(predictors[:8]):
        ax = axes[i]
        
        # Plot distributions
        cps_dist = cps_data[var].dropna()
        puf_dist = puf_data[var].dropna()
        
        # Normalize for comparison
        ax.hist(cps_dist, bins=30, alpha=0.5, density=True, 
                label='CPS', color='blue')
        ax.hist(puf_dist, bins=30, alpha=0.5, density=True,
                label='PUF', color='red')
        
        ax.set_title(f'{var}')
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig('validation/common_support_diagnostics.png', dpi=300)
    plt.close()


def generate_qrf_diagnostic_report(cps_data, puf_data, imputed_data):
    """Generate comprehensive QRF diagnostic report."""
    
    predictors = [
        'age', 'sex', 'filing_status', 'num_dependents',
        'is_tax_unit_head', 'is_tax_unit_spouse', 'is_tax_unit_dependent'
    ]
    
    target_vars = [
        'wages', 'interest', 'dividends', 'business_income',
        'capital_gains', 'pension_income', 'social_security'
    ]
    
    print("Quantile Regression Forest Diagnostic Report")
    print("=" * 60)
    
    # Common support analysis
    print("\n1. Common Support Analysis")
    print("-" * 40)
    support_df = analyze_common_support(cps_data, puf_data, predictors)
    print(support_df.to_string())
    
    # Interpretation
    print("\nInterpretation:")
    print(f"- Average overlap coefficient: {support_df['overlap_coefficient'].mean():.3f}")
    print(f"- Variables with SMD > 0.25: {(support_df['standardized_mean_diff'] > 0.25).sum()}")
    print(f"- Variables with significant KS test (p<0.05): {(support_df['ks_pvalue'] < 0.05).sum()}")
    
    # QRF accuracy
    print("\n\n2. Out-of-Sample Prediction Accuracy")
    print("-" * 40)
    accuracy_df = validate_qrf_accuracy(puf_data, predictors, target_vars)
    print(accuracy_df.to_string())
    
    print("\nSummary:")
    print(f"- Average QRF improvement vs hot-deck: {accuracy_df['qrf_improvement_vs_hotdeck'].mean():.1f}%")
    print(f"- Average QRF improvement vs linear: {accuracy_df['qrf_improvement_vs_linear'].mean():.1f}%")
    print(f"- Average 90% coverage: {accuracy_df['coverage_90pct'].mean():.3f}")
    
    # Joint distribution preservation
    print("\n\n3. Joint Distribution Preservation")
    print("-" * 40)
    var_pairs = [
        ('wages', 'age'),
        ('interest', 'dividends'),
        ('capital_gains', 'business_income'),
        ('pension_income', 'social_security')
    ]
    
    joint_df = test_joint_distribution_preservation(
        puf_data, imputed_data, var_pairs
    )
    print(joint_df.to_string(index=False))
    
    # Create diagnostic plots
    create_diagnostic_plots(cps_data, puf_data, predictors)
    print("\n\nDiagnostic plots saved to validation/common_support_diagnostics.png")
    
    # Save results
    support_df.to_csv('validation/common_support_analysis.csv')
    accuracy_df.to_csv('validation/qrf_accuracy_metrics.csv')
    joint_df.to_csv('validation/joint_distribution_tests.csv', index=False)
    
    print("\nAll diagnostic results saved to validation/ directory")


if __name__ == "__main__":
    # Load data (placeholder - would load actual data in practice)
    print("Loading data...")
    # cps_data = pd.read_csv('data/cps_2022.csv')
    # puf_data = pd.read_csv('data/puf_2015.csv')
    # imputed_data = pd.read_csv('data/extended_cps_2022.csv')
    
    # generate_qrf_diagnostic_report(cps_data, puf_data, imputed_data)