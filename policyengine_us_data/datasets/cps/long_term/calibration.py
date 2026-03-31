import numpy as np
import pandas as pd
from scipy import optimize, sparse


def _pct_error(achieved, target):
    if target == 0:
        return 0.0 if achieved == 0 else float("inf")
    return (achieved - target) / target * 100


def _relative_errors(achieved, target):
    target = np.asarray(target, dtype=float)
    achieved = np.asarray(achieved, dtype=float)
    denominator = np.maximum(np.abs(target), 1e-10)
    return (achieved - target) / denominator


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
            rel_errors = np.abs(_relative_errors(predictions_new, y))
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
        "relative_errors_initial": _relative_errors(predictions_initial, y),
        "relative_errors_new": _relative_errors(predictions_final, y),
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
    oasdi_tob_values=None,
    oasdi_tob_target=None,
    hi_tob_values=None,
    hi_tob_target=None,
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
        oasdi_tob_values: Optional OASDI TOB revenue values per household
        oasdi_tob_target: Optional OASDI TOB revenue target total
        hi_tob_values: Optional HI TOB revenue values per household
        hi_tob_target: Optional HI TOB revenue target total
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
        or (oasdi_tob_values is not None and oasdi_tob_target is not None)
        or (hi_tob_values is not None and hi_tob_target is not None)
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

        if oasdi_tob_values is not None and oasdi_tob_target is not None:
            aux_df["oasdi_tob"] = oasdi_tob_values
            controls["oasdi_tob"] = oasdi_tob_target

        if hi_tob_values is not None and hi_tob_target is not None:
            aux_df["hi_tob"] = hi_tob_values
            controls["hi_tob"] = hi_tob_target

        aux_vars = aux_df
    else:
        aux_vars = X

    w_new = calibrator.calibrate(
        samp_weight=baseline_weights,
        aux_vars=aux_vars,
        control=controls,
    )

    return w_new, 1


def _build_constraint_dataframe_and_controls(
    X,
    y_target,
    *,
    ss_values=None,
    ss_target=None,
    payroll_values=None,
    payroll_target=None,
    h6_income_values=None,
    h6_revenue_target=None,
    oasdi_tob_values=None,
    oasdi_tob_target=None,
    hi_tob_values=None,
    hi_tob_target=None,
    n_ages=86,
):
    controls = {}
    age_cols = {f"age_{i}": X[:, i].astype(float) for i in range(n_ages)}
    aux_df = pd.DataFrame(age_cols)

    for age_idx in range(n_ages):
        controls[f"age_{age_idx}"] = float(y_target[age_idx])

    if ss_values is not None and ss_target is not None:
        aux_df["ss_total"] = np.asarray(ss_values, dtype=float)
        controls["ss_total"] = float(ss_target)

    if payroll_values is not None and payroll_target is not None:
        aux_df["payroll_total"] = np.asarray(payroll_values, dtype=float)
        controls["payroll_total"] = float(payroll_target)

    if h6_income_values is not None and h6_revenue_target is not None:
        aux_df["h6_revenue"] = np.asarray(h6_income_values, dtype=float)
        controls["h6_revenue"] = float(h6_revenue_target)

    if oasdi_tob_values is not None and oasdi_tob_target is not None:
        aux_df["oasdi_tob"] = np.asarray(oasdi_tob_values, dtype=float)
        controls["oasdi_tob"] = float(oasdi_tob_target)

    if hi_tob_values is not None and hi_tob_target is not None:
        aux_df["hi_tob"] = np.asarray(hi_tob_values, dtype=float)
        controls["hi_tob"] = float(hi_tob_target)

    return aux_df, controls


def calibrate_entropy(
    X,
    y_target,
    baseline_weights,
    ss_values=None,
    ss_target=None,
    payroll_values=None,
    payroll_target=None,
    h6_income_values=None,
    h6_revenue_target=None,
    oasdi_tob_values=None,
    oasdi_tob_target=None,
    hi_tob_values=None,
    hi_tob_target=None,
    n_ages=86,
    max_iters=500,
    tol=1e-10,
):
    """
    Positive calibration via entropy balancing.

    Finds strictly positive weights minimizing KL divergence from the baseline
    weights while matching all requested calibration constraints.
    """

    aux_df, controls = _build_constraint_dataframe_and_controls(
        X,
        y_target,
        ss_values=ss_values,
        ss_target=ss_target,
        payroll_values=payroll_values,
        payroll_target=payroll_target,
        h6_income_values=h6_income_values,
        h6_revenue_target=h6_revenue_target,
        oasdi_tob_values=oasdi_tob_values,
        oasdi_tob_target=oasdi_tob_target,
        hi_tob_values=hi_tob_values,
        hi_tob_target=hi_tob_target,
        n_ages=n_ages,
    )

    A = aux_df.to_numpy(dtype=float)
    targets = np.array(list(controls.values()), dtype=float)
    scales = np.maximum(
        np.maximum(np.abs(targets), np.abs(A.T @ baseline_weights)),
        1.0,
    )
    A_scaled = A / scales
    targets_scaled = targets / scales

    baseline_weights = np.asarray(baseline_weights, dtype=float)
    gram = A_scaled.T @ (baseline_weights[:, None] * A_scaled)
    gram += np.eye(gram.shape[0]) * 1e-12
    beta0 = np.linalg.solve(gram, targets_scaled - (A_scaled.T @ baseline_weights))

    def objective_gradient_hessian(beta):
        eta = np.clip(A_scaled @ beta, -700, 700)
        exp_eta = np.exp(eta)
        weights = baseline_weights * exp_eta
        objective = float(np.sum(weights) - targets_scaled @ beta)
        gradient = A_scaled.T @ weights - targets_scaled
        hessian = A_scaled.T @ (weights[:, None] * A_scaled)
        return objective, gradient, hessian

    def solve_with_root(beta_start):
        result = optimize.root(
            lambda z: objective_gradient_hessian(z)[1],
            beta_start,
            jac=lambda z: objective_gradient_hessian(z)[2],
            method="hybr",
            options={"xtol": tol},
        )
        if not result.success:
            return None
        _, gradient, _ = objective_gradient_hessian(result.x)
        max_error = float(
            np.max(100 * np.abs(gradient) / np.maximum(np.abs(targets_scaled), 1e-12))
        )
        if max_error > tol * 100:
            return None
        return result.x, result.nfev

    def infeasibility_error(prefix):
        feasibility = assess_nonnegative_feasibility(A, targets)
        if feasibility["success"]:
            return RuntimeError(
                f"{prefix}. Nonnegative exact calibration appears infeasible under current support; "
                f"best achievable max relative constraint error is "
                f"{feasibility['best_case_max_pct_error']:.3f}%."
            )
        return RuntimeError(
            f"{prefix}. Nonnegative feasibility diagnostic could not certify a solution: "
            f"{feasibility['message']}"
        )

    beta = beta0.copy()
    iterations = 0
    final_max_error = float("inf")

    for iterations in range(1, max_iters + 1):
        objective, gradient, hessian = objective_gradient_hessian(beta)
        final_max_error = float(
            np.max(100 * np.abs(gradient) / np.maximum(np.abs(targets_scaled), 1e-12))
        )
        if final_max_error <= tol * 100:
            break

        hessian += np.eye(hessian.shape[0]) * 1e-12
        try:
            delta = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            delta = np.linalg.lstsq(hessian, gradient, rcond=None)[0]

        step = 1.0
        while step >= 1e-8:
            candidate = beta - step * delta
            candidate_objective, candidate_gradient, _ = objective_gradient_hessian(candidate)
            candidate_max_error = float(
                np.max(
                    100
                    * np.abs(candidate_gradient)
                    / np.maximum(np.abs(targets_scaled), 1e-12)
                )
            )
            if np.isfinite(candidate_objective) and (
                candidate_objective <= objective + 1e-12
                or candidate_max_error < final_max_error
            ):
                beta = candidate
                break
            step /= 2.0

        if step < 1e-8:
            root_solution = solve_with_root(beta)
            if root_solution is not None:
                beta, root_iterations = root_solution
                iterations += int(root_iterations)
                break
            raise infeasibility_error(
                "Entropy calibration line search failed to find a descent step"
            )
    else:
        root_solution = solve_with_root(beta)
        if root_solution is None:
            raise infeasibility_error(
                "Entropy calibration failed: "
                f"max constraint error remained {final_max_error:.6f}% "
                f"after {max_iters} iterations"
            )
        beta, root_iterations = root_solution
        iterations += int(root_iterations)

    eta = np.clip(A_scaled @ beta, -700, 700)
    weights = baseline_weights * np.exp(eta)
    return weights, iterations


def calibrate_entropy_approximate(
    X,
    y_target,
    baseline_weights,
    ss_values=None,
    ss_target=None,
    payroll_values=None,
    payroll_target=None,
    h6_income_values=None,
    h6_revenue_target=None,
    oasdi_tob_values=None,
    oasdi_tob_target=None,
    hi_tob_values=None,
    hi_tob_target=None,
    n_ages=86,
):
    """
    Approximate nonnegative calibration via minimax relative-error LP.

    This is the robust fallback when exact positive entropy calibration is
    infeasible under the current support. It returns the best nonnegative
    weight vector available under the requested constraints.
    """

    aux_df, controls = _build_constraint_dataframe_and_controls(
        X,
        y_target,
        ss_values=ss_values,
        ss_target=ss_target,
        payroll_values=payroll_values,
        payroll_target=payroll_target,
        h6_income_values=h6_income_values,
        h6_revenue_target=h6_revenue_target,
        oasdi_tob_values=oasdi_tob_values,
        oasdi_tob_target=oasdi_tob_target,
        hi_tob_values=hi_tob_values,
        hi_tob_target=hi_tob_target,
        n_ages=n_ages,
    )

    A = aux_df.to_numpy(dtype=float)
    targets = np.array(list(controls.values()), dtype=float)
    feasibility = assess_nonnegative_feasibility(A, targets, return_weights=True)
    weights = feasibility.get("weights")
    if not feasibility["success"] or weights is None:
        raise RuntimeError(
            "Approximate nonnegative calibration failed: "
            f"{feasibility['message']}"
        )

    return np.asarray(weights, dtype=float), 1, feasibility


def assess_nonnegative_feasibility(A, targets, *, return_weights=False):
    """
    Solve for the minimum uniform relative error achievable with nonnegative weights.

    Returns a dict with `success` and `best_case_max_pct_error`.
    """
    A = np.asarray(A, dtype=float)
    targets = np.asarray(targets, dtype=float)
    if A.shape[1] == len(targets):
        constraint_by_unit = A.T
    elif A.shape[0] == len(targets):
        constraint_by_unit = A
    else:
        raise ValueError(
            "Constraint matrix shape does not match targets: "
            f"{A.shape} vs {targets.shape}"
        )

    scales = np.maximum(np.abs(targets), 1.0)
    A_rel = constraint_by_unit / scales[:, None]
    b_rel = targets / scales

    constraint_matrix = sparse.csr_matrix(A_rel)
    epsilon_column = sparse.csc_matrix(np.ones((constraint_matrix.shape[0], 1)))
    A_ub = sparse.vstack(
        [
            sparse.hstack([constraint_matrix, -epsilon_column]),
            sparse.hstack([-constraint_matrix, -epsilon_column]),
        ],
        format="csc",
    )
    b_ub = np.concatenate([b_rel, -b_rel])
    c = np.zeros(constraint_matrix.shape[1] + 1)
    c[-1] = 1.0
    bounds = [(0, None)] * constraint_matrix.shape[1] + [(0, None)]

    result = optimize.linprog(
        c,
        A_ub=A_ub,
        b_ub=b_ub,
        bounds=bounds,
        method="highs",
    )

    result_dict = {
        "success": bool(result.success),
        "best_case_max_pct_error": (
            float(result.x[-1] * 100) if result.success else None
        ),
        "status": int(result.status),
        "message": result.message,
    }
    if return_weights:
        result_dict["weights"] = (
            np.asarray(result.x[:-1], dtype=float) if result.success else None
        )
    return result_dict


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
    oasdi_tob_values=None,
    oasdi_tob_target=None,
    hi_tob_values=None,
    hi_tob_target=None,
    n_ages=86,
    max_iters=100,
    tol=1e-6,
    verbose=False,
    allow_fallback_to_ipf=True,
    allow_approximate_entropy=False,
    approximate_max_error_pct=None,
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
        oasdi_tob_values: Optional OASDI TOB revenue values per household
        oasdi_tob_target: Optional OASDI TOB revenue target total
        hi_tob_values: Optional HI TOB revenue values per household
        hi_tob_target: Optional HI TOB revenue target total
        n_ages: Number of age groups
        max_iters: Max iterations for IPF
        tol: Convergence tolerance for IPF
        verbose: Print progress

    Returns:
        w_new: Calibrated weights
        iterations: Number of iterations
        audit: Metadata about calibration method selection
    """
    audit = {
        "method_requested": method,
        "method_used": method,
        "greg_attempted": method == "greg",
        "greg_error": None,
        "fell_back_to_ipf": False,
        "lp_fallback_used": False,
        "approximate_solution_used": False,
        "approximation_method": None,
        "approximate_solution_error_pct": None,
    }

    if method == "greg":
        if calibrator is None:
            raise ValueError("calibrator required for GREG method")
        try:
            w_new, iterations = calibrate_greg(
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
                oasdi_tob_values,
                oasdi_tob_target,
                hi_tob_values,
                hi_tob_target,
                n_ages,
            )
            return w_new, iterations, audit
        except Exception as e:
            audit["greg_error"] = str(e)
            if not allow_fallback_to_ipf:
                raise RuntimeError(
                    "GREG calibration failed while fallback was disabled"
                ) from e
            if verbose:
                print(f"GREG failed: {e}, falling back to IPF")
            w_new, info = iterative_proportional_fitting(
                X, y_target, baseline_weights, max_iters, tol, verbose
            )
            audit["method_used"] = "ipf"
            audit["fell_back_to_ipf"] = True
            return w_new, info["iterations"], audit
    elif method == "entropy":
        try:
            w_new, iterations = calibrate_entropy(
                X,
                y_target,
                baseline_weights,
                ss_values=ss_values,
                ss_target=ss_target,
                payroll_values=payroll_values,
                payroll_target=payroll_target,
                h6_income_values=h6_income_values,
                h6_revenue_target=h6_revenue_target,
                oasdi_tob_values=oasdi_tob_values,
                oasdi_tob_target=oasdi_tob_target,
                hi_tob_values=hi_tob_values,
                hi_tob_target=hi_tob_target,
                n_ages=n_ages,
                max_iters=max_iters * 5,
                tol=max(tol, 1e-10),
            )
            return w_new, iterations, audit
        except RuntimeError as error:
            audit["entropy_error"] = str(error)
            w_new, iterations, feasibility = calibrate_entropy_approximate(
                X,
                y_target,
                baseline_weights,
                ss_values=ss_values,
                ss_target=ss_target,
                payroll_values=payroll_values,
                payroll_target=payroll_target,
                h6_income_values=h6_income_values,
                h6_revenue_target=h6_revenue_target,
                oasdi_tob_values=oasdi_tob_values,
                oasdi_tob_target=oasdi_tob_target,
                hi_tob_values=hi_tob_values,
                hi_tob_target=hi_tob_target,
                n_ages=n_ages,
            )
            approximate_error_pct = float(feasibility["best_case_max_pct_error"])
            if approximate_error_pct <= max(tol * 100, 1e-6):
                audit["lp_fallback_used"] = True
                audit["approximation_method"] = "lp_minimax_exact"
                audit["approximate_solution_error_pct"] = approximate_error_pct
                return w_new, iterations, audit

            if not allow_approximate_entropy:
                raise

            if (
                approximate_max_error_pct is not None
                and approximate_error_pct > approximate_max_error_pct
            ):
                raise RuntimeError(
                    "Approximate entropy fallback exceeded allowable error: "
                    f"{approximate_error_pct:.3f}% > {approximate_max_error_pct:.3f}%"
                ) from error

            audit["lp_fallback_used"] = True
            audit["approximate_solution_used"] = True
            audit["approximation_method"] = "lp_minimax"
            audit["approximate_solution_error_pct"] = approximate_error_pct
            return w_new, iterations, audit
    else:
        w_new, info = iterative_proportional_fitting(
            X, y_target, baseline_weights, max_iters, tol, verbose
        )
        return w_new, info["iterations"], audit


def build_calibration_audit(
    *,
    X,
    y_target,
    weights,
    baseline_weights,
    calibration_event,
    ss_values=None,
    ss_target=None,
    payroll_values=None,
    payroll_target=None,
    h6_income_values=None,
    h6_revenue_target=None,
    oasdi_tob_values=None,
    oasdi_tob_target=None,
    hi_tob_values=None,
    hi_tob_target=None,
):
    achieved_ages = X.T @ weights
    age_errors = np.abs(achieved_ages - y_target) / np.maximum(np.abs(y_target), 1e-10) * 100

    neg_mask = weights < 0
    negative_values = np.abs(weights[neg_mask])

    audit = dict(calibration_event)
    audit.update(
        {
            "age_max_pct_error": float(age_errors.max()),
            "negative_weight_count": int(neg_mask.sum()),
            "negative_weight_pct": float(100 * neg_mask.sum() / len(weights)),
            "largest_negative_weight": float(negative_values.max()) if negative_values.size else 0.0,
            "constraints": {},
            "baseline_weight_sum": float(np.sum(baseline_weights)),
            "calibrated_weight_sum": float(np.sum(weights)),
            "max_constraint_pct_error": 0.0,
        }
    )

    constraint_specs = [
        ("ss_total", ss_values, ss_target),
        ("payroll_total", payroll_values, payroll_target),
        ("h6_revenue", h6_income_values, h6_revenue_target),
        ("oasdi_tob", oasdi_tob_values, oasdi_tob_target),
        ("hi_tob", hi_tob_values, hi_tob_target),
    ]

    for name, values, target in constraint_specs:
        if values is None or target is None:
            continue
        achieved = float(np.sum(values * weights))
        audit["constraints"][name] = {
            "target": float(target),
            "achieved": achieved,
            "error": achieved - float(target),
            "pct_error": float(_pct_error(achieved, float(target))),
        }

    if audit["constraints"]:
        audit["max_constraint_pct_error"] = float(
            max(
                abs(stats["pct_error"])
                for stats in audit["constraints"].values()
            )
        )

    return audit
