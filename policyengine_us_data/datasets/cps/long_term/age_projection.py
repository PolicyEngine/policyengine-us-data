import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from create_reweighting_matrix import iterative_proportional_fitting

# Initial population vector for 2024 (from y_age)
y_age_2024 = np.array([
    18333697., 19799430., 21203879., 22168390., 21618383., 21906706.,
    23405056., 22650099., 22126485., 19859230., 20661941., 20198508.,
    21676036., 19026961., 15797857., 11318751.,  7041419.,  6122068.
])

# Age bracket labels
age_brackets = [
    '0-4', '5-9', '10-14', '15-19', '20-24', '25-29',
    '30-34', '35-39', '40-44', '45-49', '50-54', '55-59',
    '60-64', '65-69', '70-74', '75-79', '80-84', '85-999'
]

def create_simple_transition_matrix():
    """
    Create a simple transition matrix where people age 5 years per period.
    Each 5-year age bracket moves to the next one.
    """
    n = 18  # Number of age brackets
    T = np.zeros((n, n))

    # Basic aging: each bracket moves to the next
    for i in range(n - 1):
        T[i + 1, i] = 1.0  # Everyone in bracket i moves to bracket i+1

    # The oldest bracket (85-999) stays in the same bracket
    T[n - 1, n - 1] = 1.0

    return T


def create_realistic_transition_matrix(survival_rates=None):
    """
    Create a more realistic transition matrix with survival rates.

    Args:
        survival_rates: Optional array of 5-year survival rates for each bracket.
                       If None, uses simplified estimates.
    """
    n = 18
    T = np.zeros((n, n))

    # Default survival rates (5-year survival probabilities)
    # These are simplified estimates - real rates would come from life tables
    if survival_rates is None:
        survival_rates = np.array([
            0.995,  # 0-4 -> 5-9
            0.998,  # 5-9 -> 10-14
            0.998,  # 10-14 -> 15-19
            0.997,  # 15-19 -> 20-24
            0.996,  # 20-24 -> 25-29
            0.995,  # 25-29 -> 30-34
            0.994,  # 30-34 -> 35-39
            0.992,  # 35-39 -> 40-44
            0.988,  # 40-44 -> 45-49
            0.982,  # 45-49 -> 50-54
            0.972,  # 50-54 -> 55-59
            0.958,  # 55-59 -> 60-64
            0.935,  # 60-64 -> 65-69
            0.900,  # 65-69 -> 70-74
            0.850,  # 70-74 -> 75-79
            0.780,  # 75-79 -> 80-84
            0.650,  # 80-84 -> 85-999
            0.400,  # 85-999 -> 85-999 (stay in bracket with some survival)
        ])

    # Fill transition matrix with survival rates
    for i in range(n - 1):
        T[i + 1, i] = survival_rates[i]

    # Oldest bracket: some survive and stay
    T[n - 1, n - 1] = survival_rates[n - 1]

    return T


def project_population(initial_pop, transition_matrix, n_periods, births_per_period=None):
    """
    Project population forward using transition matrix.

    Args:
        initial_pop: Initial population vector
        transition_matrix: Transition matrix T
        n_periods: Number of 5-year periods to project
        births_per_period: Optional array of births for each period (added to 0-4 bracket)

    Returns:
        Array with population for each period (n_periods + 1 x n_brackets)
    """
    n_brackets = len(initial_pop)
    projections = np.zeros((n_periods + 1, n_brackets))
    projections[0] = initial_pop

    for t in range(n_periods):
        # Apply transition matrix
        projections[t + 1] = transition_matrix @ projections[t]

        # Add births if specified
        if births_per_period is not None:
            if isinstance(births_per_period, (int, float)):
                projections[t + 1, 0] += births_per_period
            else:
                projections[t + 1, 0] += births_per_period[t]

    return projections


def visualize_projections(projections, years, selected_brackets=None):
    """
    Visualize population projections over time using plotly.
    """
    if selected_brackets is None:
        selected_brackets = [0, 5, 10, 15, 17]  # Sample of brackets

    fig = go.Figure()

    for idx in selected_brackets:
        fig.add_trace(go.Scatter(
            x=years,
            y=projections[:, idx] / 1e6,
            mode='lines',
            name=age_brackets[idx],
            line=dict(width=2)
        ))

    fig.update_layout(
        title='Population Projections by Age Bracket',
        xaxis_title='Year',
        yaxis_title='Population (millions)',
        hovermode='x unified',
        width=900,
        height=500
    )

    return fig


def main():
    print("=" * 70)
    print("AGE COHORT PROJECTION SYSTEM")
    print("=" * 70)

    # 1. Simple projection (everyone just ages)
    print("\n1. SIMPLE AGING (No mortality)")
    print("-" * 40)

    T_simple = create_simple_transition_matrix()
    n_periods = 15  # 15 five-year periods = 75 years

    projections_simple = project_population(y_age_2024, T_simple, n_periods)

    # Show results for selected years
    for year_idx, year in enumerate([2024, 2049, 2074, 2099]):
        period = (year - 2024) // 5
        if period <= n_periods:
            print(f"\nYear {year} (Period {period}):")
            print(f"  Total population: {projections_simple[period].sum():,.0f}")
            print(f"  In 85+ bracket: {projections_simple[period, 17]:,.0f} "
                  f"({100 * projections_simple[period, 17] / projections_simple[period].sum():.1f}%)")

    # 2. Realistic projection with mortality
    print("\n\n2. REALISTIC PROJECTION (With mortality)")
    print("-" * 40)

    T_realistic = create_realistic_transition_matrix()

    # Add some births to maintain population
    annual_births = 3_800_000  # Approximate US births per year
    births_per_5year = annual_births * 5

    projections_realistic = project_population(
        y_age_2024, T_realistic, n_periods, births_per_5year
    )

    for year_idx, year in enumerate([2024, 2049, 2074, 2099]):
        period = (year - 2024) // 5
        if period <= n_periods:
            print(f"\nYear {year} (Period {period}):")
            print(f"  Total population: {projections_realistic[period].sum():,.0f}")
            print(f"  Age distribution:")
            for i in [0, 5, 10, 15, 17]:
                pct = 100 * projections_realistic[period, i] / projections_realistic[period].sum()
                print(f"    {age_brackets[i]:8} : {projections_realistic[period, i]:12,.0f} ({pct:5.1f}%)")

    # 3. Show transition matrix structure
    print("\n\n3. TRANSITION MATRIX STRUCTURE")
    print("-" * 40)
    print("\nSimple transition matrix (first 5x5):")
    print(T_simple[:5, :5])

    print("\nRealistic transition matrix (first 5x5):")
    print(T_realistic[:5, :5])

    # 4. Create DataFrame for analysis
    years = [2024 + 5*i for i in range(n_periods + 1)]

    df_projections = pd.DataFrame(
        projections_realistic,
        index=years,
        columns=age_brackets
    )

    print("\n\n4. PROJECTION SUMMARY TABLE")
    print("-" * 40)
    print("\nPopulation by age bracket (millions):")
    print(df_projections.iloc[::3, ::3] / 1e6)  # Every 3rd period and bracket

    # Create visualization
    years = [2024 + 5*i for i in range(n_periods + 1)]
    fig = visualize_projections(projections_realistic, years)
    fig.write_html('age_projections.html')
    print("\nVisualization saved to age_projections.html")

    # Optional: save results
    save_results = False
    if save_results:
        np.save('transition_matrix_simple.npy', T_simple)
        np.save('transition_matrix_realistic.npy', T_realistic)
        np.save('projections_simple.npy', projections_simple)
        np.save('projections_realistic.npy', projections_realistic)
        df_projections.to_csv('age_projections.csv')
        print("\nResults saved to files.")

    return T_realistic, projections_realistic, df_projections


def create_annual_transition_matrix():
    """
    Create annual transition matrix for year-by-year projections.
    Each year, 1/5 of each bracket ages into the next bracket.
    """
    n = 18
    T = np.zeros((n, n))

    # Annual survival rates (approximate - derived from 5-year rates)
    survival_rates_5yr = np.array([
        0.995,  # 0-4 -> 5-9
        0.998,  # 5-9 -> 10-14
        0.998,  # 10-14 -> 15-19
        0.997,  # 15-19 -> 20-24
        0.996,  # 20-24 -> 25-29
        0.995,  # 25-29 -> 30-34
        0.994,  # 30-34 -> 35-39
        0.992,  # 35-39 -> 40-44
        0.988,  # 40-44 -> 45-49
        0.982,  # 45-49 -> 50-54
        0.972,  # 50-54 -> 55-59
        0.958,  # 55-59 -> 60-64
        0.935,  # 60-64 -> 65-69
        0.900,  # 65-69 -> 70-74
        0.850,  # 70-74 -> 75-79
        0.780,  # 75-79 -> 80-84
        0.650,  # 80-84 -> 85-999
        0.400,  # 85-999 -> 85-999 (5-year survival in same bracket)
    ])

    # Convert to annual rates (5th root)
    annual_survival = survival_rates_5yr ** 0.2

    # Fraction that ages out each year (1/5 of the bracket)
    aging_fraction = 0.2

    # Build transition matrix
    for i in range(n):
        if i < n - 1:
            # Fraction that stays in current bracket
            T[i, i] = (1 - aging_fraction) * annual_survival[i]
            # Fraction that moves to next bracket
            T[i + 1, i] = aging_fraction * annual_survival[i]
        else:
            # Last bracket: everyone stays (with survival)
            T[i, i] = annual_survival[i]

    return T


def project_annual_targets():
    """
    Create 18x76 matrix of annual projections for 2025-2100.
    """
    print("\n" + "=" * 70)
    print("CREATING ANNUAL TARGETS MATRIX (2025-2100)")
    print("=" * 70)

    T = create_annual_transition_matrix()
    n_years = 76
    n_brackets = 18
    annual_births = 3_800_000

    # Initialize projections array
    projections = np.zeros((n_years + 1, n_brackets))
    projections[0] = y_age_2024

    for year in range(n_years):
        # Apply transition
        projections[year + 1] = T @ projections[year]
        # Add births to 0-4 bracket (full annual births, not divided)
        projections[year + 1, 0] += annual_births

    # Extract 2025-2100 (skip 2024)
    projections_2025_2100 = projections[1:, :]

    # Transpose to get 18 x 76 matrix
    target_matrix = projections_2025_2100.T

    print(f"\nTarget matrix shape: {target_matrix.shape}")
    print(f"  - Rows: {target_matrix.shape[0]} age brackets")
    print(f"  - Columns: {target_matrix.shape[1]} years (2025-2100)")

    # Save the matrix
    np.save('age_targets_2025_2100.npy', target_matrix)
    print(f"\nSaved: age_targets_2025_2100.npy")

    # Also save as CSV
    years = list(range(2025, 2101))
    df = pd.DataFrame(target_matrix, index=age_brackets, columns=years)
    df.to_csv('age_targets_2025_2100.csv')
    print(f"Saved: age_targets_2025_2100.csv")

    # Show sample
    print(f"\nSample values (2025, 2050, 2100):")
    for year in [2025, 2050, 2100]:
        idx = year - 2025
        total = target_matrix[:, idx].sum()
        print(f"  {year}: {total:,.0f} total population")

    return target_matrix


def reweight_all_years():
    """
    Apply IPF reweighting for each year 2025-2100 using the target matrix.
    """
    from policyengine_us import Microsimulation

    print("\n" + "=" * 70)
    print("REWEIGHTING FOR ALL YEARS (2025-2100)")
    print("=" * 70)

    # Load the target matrix
    target_matrix = np.load('age_targets_2025_2100.npy')
    print(f"Loaded target matrix: {target_matrix.shape}")

    # Load microsimulation and create design matrix
    print("\nLoading CPS microsimulation data...")
    sim = Microsimulation(dataset="hf://policyengine/policyengine-us-data/enhanced_cps_2024.h5")

    # Get person-level age data
    age_person = sim.calculate("age")
    person_household_id = sim.calculate("person_household_id")

    # Get unique household IDs
    household_ids_unique = np.unique(person_household_id.values)
    n_households = len(household_ids_unique)

    # Define age brackets
    age_brackets_ranges = [
        (0, 5), (5, 10), (10, 15), (15, 20), (20, 25), (25, 30),
        (30, 35), (35, 40), (40, 45), (45, 50), (50, 55), (55, 60),
        (60, 65), (65, 70), (70, 75), (75, 80), (80, 85), (85, 999)
    ]
    n_brackets = len(age_brackets_ranges)

    # Create design matrix
    print("Creating design matrix...")
    X = np.zeros((n_households, n_brackets))

    # Map household IDs to row indices
    hh_id_to_idx = {hh_id: idx for idx, hh_id in enumerate(household_ids_unique)}

    # Fill design matrix
    for person_idx in range(len(age_person)):
        age = age_person.values[person_idx]
        hh_id = person_household_id.values[person_idx]
        hh_idx = hh_id_to_idx[hh_id]

        # Find which age bracket this person belongs to
        for bracket_idx, (lower, upper) in enumerate(age_brackets_ranges):
            if lower <= age < upper:
                X[hh_idx, bracket_idx] += 1
                break

    # Get initial household weights
    person_weight = sim.calculate("person_weight")
    initial_weights = np.zeros(n_households)
    for idx, hh_id in enumerate(household_ids_unique):
        hh_mask = person_household_id.values == hh_id
        initial_weights[idx] = person_weight.values[hh_mask][0]

    # Initialize weights matrix (n_households x 76 years)
    n_years = target_matrix.shape[1]
    weights_matrix = np.zeros((n_households, n_years))

    print(f"\nRunning IPF for {n_years} years...")
    print("Year    Max Rel Error    Iterations")
    print("-" * 40)

    # Run IPF for each year
    for year_idx in range(n_years):
        year = 2025 + year_idx
        y_target = target_matrix[:, year_idx]

        # Run IPF (silent mode for individual years)
        w_new, info = iterative_proportional_fitting(
            X, y_target, initial_weights,
            max_iters=100, tol=1e-6, verbose=False
        )

        weights_matrix[:, year_idx] = w_new

        # Report progress every 5 years
        if year % 5 == 0 or year == 2025:
            max_error = np.max(np.abs(info['relative_errors_new']))
            print(f"{year}    {max_error:.6f}        {info['iterations']}")

    # Save weights matrix
    np.save('household_weights_2025_2100.npy', weights_matrix)
    print("\n" + "-" * 40)
    print(f"Saved: household_weights_2025_2100.npy")
    print(f"Shape: {weights_matrix.shape} ({n_households} households x {n_years} years)")

    # Summary statistics
    print("\n" + "=" * 70)
    print("WEIGHTS SUMMARY")
    print("=" * 70)

    # Weight ratios relative to initial
    weight_ratios = weights_matrix / initial_weights.reshape(-1, 1)

    print("\nWeight adjustment factors (relative to initial):")
    for year in [2025, 2050, 2075, 2100]:
        year_idx = year - 2025
        ratios = weight_ratios[:, year_idx]
        print(f"  {year}: min={ratios.min():.3f}, max={ratios.max():.3f}, "
              f"mean={ratios.mean():.3f}, std={ratios.std():.3f}")

    # Check total weights preserved
    print("\nTotal weight preservation check:")
    for year in [2025, 2050, 2075, 2100]:
        year_idx = year - 2025
        initial_total = initial_weights.sum()
        new_total = weights_matrix[:, year_idx].sum()
        pct_change = 100 * (new_total - initial_total) / initial_total
        print(f"  {year}: {new_total:,.0f} ({pct_change:+.2f}% from initial)")

    return weights_matrix, X, household_ids_unique


if __name__ == "__main__":
    # First run the original projections
    T, projections, df = main()

    # Create the annual targets matrix
    target_matrix = project_annual_targets()

    # Now do IPF reweighting for all years
    weights_matrix, design_matrix, household_ids = reweight_all_years()

    # Verify state transition properties
    print("\n\n5. TRANSITION MATRIX PROPERTIES")
    print("-" * 40)
    print(f"Matrix shape: {T.shape}")
    print(f"Column sums (should be ≤ 1 due to mortality):")
    col_sums = T.sum(axis=0)
    for i, s in enumerate(col_sums):
        print(f"  Column {i:2} ({age_brackets[i]:8}): {s:.4f}")

    # Calculate steady state (eigenvalue analysis)
    eigenvalues, eigenvectors = np.linalg.eig(T)
    print(f"\nLargest eigenvalue: {np.max(np.real(eigenvalues)):.4f}")
    print("(Should be < 1 with mortality, = 1 without)")