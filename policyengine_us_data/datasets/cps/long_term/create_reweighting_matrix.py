import sqlite3
import pandas as pd
import numpy as np

from policyengine_us import Microsimulation

# Database connection
db_path = '/home/baogorek/devl/sep/policyengine-us-data/policyengine_us_data/storage/policy_data.db'

def get_national_age_targets():
    """
    Extract the 18x1 vector of national age demographic targets
    for use as y-variables in the reweighting regression.

    Returns:
        np.ndarray: 18x1 vector of population counts by age bracket
        pd.DataFrame: DataFrame with age ranges and counts for reference
    """

    # SQL query to extract national age demographic targets
    # - stratum_group_id = 2 represents age strata
    # - parent_stratum_id = 1 represents national level
    # - Use 2023 data which is the most recent complete year
    query = """
    SELECT
        s.stratum_id,
        s.notes,
        t.variable,
        t.value,
        t.period,
        src.name AS source_name
    FROM strata s
    JOIN targets t ON s.stratum_id = t.stratum_id
    JOIN sources src ON t.source_id = src.source_id
    WHERE s.stratum_group_id = 2  -- Age strata
        AND s.parent_stratum_id = 1  -- National level
        AND t.period = 2023  -- Most recent complete year
        AND s.notes LIKE 'Age:%'  -- Ensure we only get age strata, not totals
    ORDER BY s.stratum_id;
    """

    # Connect and execute query
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(query, conn)
    conn.close()

    # Extract age ranges from notes column
    df['age_range'] = df['notes'].str.extract(r'Age: ([\d-]+)')

    # Define the expected age ranges in order
    expected_age_ranges = [
        '0-4', '5-9', '10-14', '15-19', '20-24', '25-29',
        '30-34', '35-39', '40-44', '45-49', '50-54', '55-59',
        '60-64', '65-69', '70-74', '75-79', '80-84', '85-999'
    ]

    # Create a mapping for sorting
    age_range_order = {age: i for i, age in enumerate(expected_age_ranges)}
    df['sort_order'] = df['age_range'].map(age_range_order)

    # Sort by the predefined order
    df_sorted = df.sort_values('sort_order')

    # Extract the 18x1 vector of population counts
    y_targets = df_sorted['value'].values

    # Create reference DataFrame
    reference_df = pd.DataFrame({
        'age_range': df_sorted['age_range'].values,
        'target_value': y_targets
    })

    # Verify we have exactly 18 age groups
    assert len(y_targets) == 18, f"Expected 18 age groups, got {len(y_targets)}"

    return y_targets, reference_df


def create_reweighting_target_matrix():
    """
    Create the full target matrix for reweighting regression.
    Currently includes national age demographics (18x1).
    Can be extended to include other target variables.

    Returns:
        np.ndarray: Target matrix (currently 18x1, expandable)
        list: List of target descriptions
    """

    # Get age targets
    age_targets, age_ref_df = get_national_age_targets()

    # Initialize target matrix - start with age targets
    # This can be expanded horizontally to include other targets
    target_matrix = age_targets.reshape(-1, 1)

    # Track what each row represents
    target_descriptions = [f"Age {row['age_range']}" for _, row in age_ref_df.iterrows()]

    return target_matrix, target_descriptions


def iterative_proportional_fitting(X, y, w_initial, max_iters=100, tol=1e-6, verbose=True):
    """
    Fast iterative proportional fitting (raking) for reweighting.
    Much faster than optimization for large datasets.

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
        # Current predictions
        predictions = X.T @ w

        # Calculate adjustment factors for each constraint
        adjustment_factors = y / (predictions + 1e-10)

        # Apply adjustments to weights
        # For each household, multiply weight by geometric mean of relevant adjustments
        w_new = w.copy()
        for i in range(len(w)):
            # Get adjustments for features present in this household
            household_features = X[i, :]
            relevant_adjustments = adjustment_factors[household_features > 0]
            if len(relevant_adjustments) > 0:
                # Geometric mean of adjustments
                adjustment = np.prod(relevant_adjustments ** (household_features[household_features > 0] / household_features.sum()))
                w_new[i] *= adjustment

        # Check convergence
        rel_change = np.abs(w_new - w).max() / (np.abs(w).max() + 1e-10)
        w = w_new

        if verbose and (iter_num % 10 == 0 or rel_change < tol):
            predictions_new = X.T @ w
            rel_errors = np.abs(predictions_new - y) / y
            max_rel_error = rel_errors.max()
            print(f"Iteration {iter_num:3d}: Max relative error = {max_rel_error:.6f}, Weight change = {rel_change:.6e}")

        if rel_change < tol:
            if verbose:
                print(f"Converged in {iter_num + 1} iterations")
            break

    # Final predictions
    predictions_final = X.T @ w
    predictions_initial = X.T @ w_initial

    info = {
        'success': True,
        'iterations': iter_num + 1,
        'predictions_initial': predictions_initial,
        'predictions_new': predictions_final,
        'relative_errors_initial': (predictions_initial - y) / y,
        'relative_errors_new': (predictions_final - y) / y,
        'weight_ratio': w / w_initial
    }

    return w, info


def create_age_design_matrix(sim):
    """
    Create the design matrix X for age groups from microsimulation data.

    Args:
        sim: PolicyEngine Microsimulation object

    Returns:
        np.ndarray: Design matrix (n_households x 18) where each row indicates
                    count of people in each age bracket for that household
        np.ndarray: Weights vector (n_households,)
    """

    # Get person-level age data
    age_person = sim.calculate("age")

    # Get household ID for each person - need to get the actual person's household
    person_household_id = sim.calculate("person_household_id")

    # Get unique household IDs
    household_ids_unique = np.unique(person_household_id.values)
    n_households = len(household_ids_unique)

    # Define age brackets (matching the targets)
    age_brackets = [
        (0, 5),    # 0-4
        (5, 10),   # 5-9
        (10, 15),  # 10-14
        (15, 20),  # 15-19
        (20, 25),  # 20-24
        (25, 30),  # 25-29
        (30, 35),  # 30-34
        (35, 40),  # 35-39
        (40, 45),  # 40-44
        (45, 50),  # 45-49
        (50, 55),  # 50-54
        (55, 60),  # 55-59
        (60, 65),  # 60-64
        (65, 70),  # 65-69
        (70, 75),  # 70-74
        (75, 80),  # 75-79
        (80, 85),  # 80-84
        (85, 999), # 85+
    ]
    n_brackets = len(age_brackets)

    # Initialize design matrix
    X = np.zeros((n_households, n_brackets))

    # Map household IDs to row indices
    hh_id_to_idx = {hh_id: idx for idx, hh_id in enumerate(household_ids_unique)}

    # Fill design matrix: count people in each age bracket per household
    for person_idx in range(len(age_person)):
        age = age_person.values[person_idx]
        hh_id = person_household_id.values[person_idx]
        hh_idx = hh_id_to_idx[hh_id]

        # Find which age bracket this person belongs to
        for bracket_idx, (lower, upper) in enumerate(age_brackets):
            if lower <= age < upper:
                X[hh_idx, bracket_idx] += 1  # Count person in this bracket
                break

    # Extract household weights (aligned with household order)
    # Get person weights to extract household weight (same for all members of household)
    person_weight = sim.calculate("person_weight")
    weights = np.zeros(n_households)
    for idx, hh_id in enumerate(household_ids_unique):
        # Get weight for this household from any of its members
        hh_mask = person_household_id.values == hh_id
        hh_weight_value = person_weight.values[hh_mask][0]
        weights[idx] = hh_weight_value

    return X, weights


if __name__ == "__main__":

    # Get the target vector (y)
    y_age, age_df = get_national_age_targets()

    # Load microsimulation
    sim = Microsimulation(dataset="hf://policyengine/policyengine-us-data/enhanced_cps_2024.h5")

    # Create design matrix
    X, weights = create_age_design_matrix(sim)

    # Calculate initialized current weighted totals (X'w)
    current_totals = X.T @ weights

    print("\n" + "="*60)
    print("CURRENT VS TARGET COMPARISON")
    print("="*60)
    print(f"{'Age Bracket':<12} {'Current':>15} {'Target':>15} {'Difference':>15}")
    print("-"*60)

    age_labels = age_df['age_range'].values
    for i, label in enumerate(age_labels):
        current = current_totals[i]
        target = y_age[i]
        diff = current - target
        print(f"{label:<12} {current:15,.0f} {target:15,.0f} {diff:+15,.0f}")

    print("-"*60)
    print(f"{'TOTAL':<12} {current_totals.sum():15,.0f} {y_age.sum():15,.0f} {(current_totals.sum() - y_age.sum()):+15,.0f}")

    # Perform reweighting
    print("\n" + "="*60)
    print("PERFORMING REWEIGHTING (IPF/Raking)")
    print("="*60)

    w_new, info = iterative_proportional_fitting(X, y_age, weights,
                                                  max_iters=100,
                                                  tol=1e-6,
                                                  verbose=True)

    # Display results
    print("\n" + "="*60)
    print("REWEIGHTING RESULTS")
    print("="*60)
    print(f"{'Age Bracket':<12} {'Initial':>15} {'Target':>15} {'Reweighted':>15} {'Rel Error':>12}")
    print("-"*70)

    for i, label in enumerate(age_labels):
        initial = info['predictions_initial'][i]
        target = y_age[i]
        reweighted = info['predictions_new'][i]
        rel_error = info['relative_errors_new'][i]
        print(f"{label:<12} {initial:15,.0f} {target:15,.0f} {reweighted:15,.0f} {rel_error:12.4%}")

    print("-"*70)
    print(f"{'TOTAL':<12} {info['predictions_initial'].sum():15,.0f} {y_age.sum():15,.0f} {info['predictions_new'].sum():15,.0f}")

    # Weight statistics
    print("\n" + "="*60)
    print("WEIGHT STATISTICS")
    print("="*60)
    print(f"Min weight ratio: {info['weight_ratio'].min():.4f}")
    print(f"Max weight ratio: {info['weight_ratio'].max():.4f}")
    print(f"Mean weight ratio: {info['weight_ratio'].mean():.4f}")
    print(f"Std weight ratio: {info['weight_ratio'].std():.4f}")
    print(f"Weights kept positive: {np.all(w_new > 0)}")
    print(f"Total weight preserved: {np.abs(w_new.sum() - weights.sum()) < 1e-6}")

    # Overall performance
    print("\n" + "="*60)
    print("OVERALL PERFORMANCE")
    print("="*60)
    initial_rel_error = np.sqrt(np.mean(info['relative_errors_initial']**2))
    final_rel_error = np.sqrt(np.mean(info['relative_errors_new']**2))
    print(f"Initial RMSE (relative): {initial_rel_error:.4%}")
    print(f"Final RMSE (relative): {final_rel_error:.4%}")
    print(f"Improvement: {(1 - final_rel_error/initial_rel_error):.2%}")

    save_numpy_arrays = False
    if save_numpy_arrays:
        np.save('X_age_design.npy', X)
        np.save('y_age_targets.npy', y_age)
        np.save('weights_initial.npy', weights)
        np.save('weights_reweighted.npy', w_new)
