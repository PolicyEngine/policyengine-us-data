import numpy as np
import pandas as pd


def iterative_proportional_fitting(
    X, y, w_initial, max_iters=100, tol=1e-6, verbose=True
):
    """
    Fast iterative proportional fitting (raking) for reweighting.

    Args:
        X: Design matrix (n_households x n_features)
        y: Target vector (n_features,)
        w_initial: Initial weights (n_households,)
        max_iters: Maximum iterations
        tol: Convergence tolerance
        verbose: Print progress

    Returns:
        w_new: New weights (n_households,)
        info: Dictionary with convergence info
    """
    w = w_initial.copy()
    n_features = X.shape[1]

    for iter_num in range(max_iters):
        predictions = X.T @ w

        adjustment_factors = y / (predictions + 1e-10)

        w_new = w.copy()
        for i in range(len(w)):
            household_features = X[i, :]
            relevant_adjustments = adjustment_factors[household_features > 0]
            if len(relevant_adjustments) > 0:
                adjustment = np.prod(
                    relevant_adjustments
                    ** (
                        household_features[household_features > 0]
                        / household_features.sum()
                    )
                )
                w_new[i] *= adjustment

        rel_change = np.abs(w_new - w).max() / (np.abs(w).max() + 1e-10)
        w = w_new

        if verbose and (iter_num % 10 == 0 or rel_change < tol):
            predictions_new = X.T @ w
            rel_errors = np.abs(predictions_new - y) / y
            max_rel_error = rel_errors.max()
            print(
                f"Iteration {iter_num:3d}: Max relative error = {max_rel_error:.6f}, Weight change = {rel_change:.6e}"
            )

        if rel_change < tol:
            if verbose:
                print(f"Converged in {iter_num + 1} iterations")
            break

    predictions_final = X.T @ w
    predictions_initial = X.T @ w_initial

    info = {
        "success": True,
        "iterations": iter_num + 1,
        "predictions_initial": predictions_initial,
        "predictions_new": predictions_final,
        "relative_errors_initial": (predictions_initial - y) / y,
        "relative_errors_new": (predictions_final - y) / y,
        "weight_ratio": w / w_initial,
    }

    return w, info


def calibrate_greg(
    calibrator,
    X,
    y_target,
    baseline_weights,
    ss_values=None,
    ss_target=None,
    payroll_values=None,
    payroll_target=None,
    h6_income_values=None,
    h6_revenue_target=None,
    n_ages=86,
):
    """
    Calibrate weights using GREG method via samplics.

    Args:
        calibrator: SampleWeight instance from samplics
        X: Design matrix (n_households x n_ages)
        y_target: Target age distribution
        baseline_weights: Initial household weights
        ss_values: Optional Social Security values per household
        ss_target: Optional Social Security target total
        payroll_values: Optional taxable payroll values per household
        payroll_target: Optional taxable payroll target total
        h6_income_values: Optional H6 reform income values per household
        h6_revenue_target: Optional H6 reform total revenue impact target
        n_ages: Number of age groups

    Returns:
        w_new: Calibrated weights
        iterations: Number of iterations (always 1 for GREG)
    """
    controls = {}
    for age_idx in range(n_ages):
        controls[f"age_{age_idx}"] = y_target[age_idx]

    # Build auxiliary variables dataframe if any continuous constraints are provided
    needs_aux_df = (
        (ss_values is not None and ss_target is not None)
        or (payroll_values is not None and payroll_target is not None)
        or (h6_income_values is not None and h6_revenue_target is not None)
    )

    if needs_aux_df:
        age_cols = {f"age_{i}": X[:, i] for i in range(n_ages)}
        aux_df = pd.DataFrame(age_cols)

        if ss_values is not None and ss_target is not None:
            aux_df["ss_total"] = ss_values
            controls["ss_total"] = ss_target

        if payroll_values is not None and payroll_target is not None:
            aux_df["payroll_total"] = payroll_values
            controls["payroll_total"] = payroll_target

        # H6 reform revenue impact as a simple linear constraint
        if h6_income_values is not None and h6_revenue_target is not None:
            aux_df["h6_revenue"] = h6_income_values
            controls["h6_revenue"] = h6_revenue_target

        aux_vars = aux_df
    else:
        aux_vars = X

    w_new = calibrator.calibrate(
        samp_weight=baseline_weights,
        aux_vars=aux_vars,
        control=controls,
    )

    return w_new, 1


def calibrate_weights(
    X,
    y_target,
    baseline_weights,
    method="ipf",
    calibrator=None,
    ss_values=None,
    ss_target=None,
    payroll_values=None,
    payroll_target=None,
    h6_income_values=None,
    h6_revenue_target=None,
    n_ages=86,
    max_iters=100,
    tol=1e-6,
    verbose=False,
):
    """
    Unified interface for weight calibration.

    Args:
        X: Design matrix (n_households x n_features)
        y_target: Target vector
        baseline_weights: Initial weights
        method: 'ipf' or 'greg'
        calibrator: Required if method='greg'
        ss_values: Optional SS values (for GREG with SS)
        ss_target: Optional SS target (for GREG with SS)
        payroll_values: Optional payroll values (for GREG with payroll)
        payroll_target: Optional payroll target (for GREG with payroll)
        h6_income_values: Optional H6 reform income values per household
        h6_revenue_target: Optional H6 reform total revenue impact target
        n_ages: Number of age groups
        max_iters: Max iterations for IPF
        tol: Convergence tolerance for IPF
        verbose: Print progress

    Returns:
        w_new: Calibrated weights
        iterations: Number of iterations
    """
    if method == "greg":
        if calibrator is None:
            raise ValueError("calibrator required for GREG method")
        try:
            return calibrate_greg(
                calibrator,
                X,
                y_target,
                baseline_weights,
                ss_values,
                ss_target,
                payroll_values,
                payroll_target,
                h6_income_values,
                h6_revenue_target,
                n_ages,
            )
        except Exception as e:
            if verbose:
                print(f"GREG failed: {e}, falling back to IPF")
            w_new, info = iterative_proportional_fitting(
                X, y_target, baseline_weights, max_iters, tol, verbose
            )
            return w_new, info["iterations"]
    else:
        w_new, info = iterative_proportional_fitting(
            X, y_target, baseline_weights, max_iters, tol, verbose
        )
        return w_new, info["iterations"]
