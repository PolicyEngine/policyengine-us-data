import numpy as np
import pandas as pd
from create_reweighting_matrix import iterative_proportional_fitting


def load_ssa_projections(end_year=2100):
    """
    Load SSA population projections from local CSV.

    Args:
        end_year: Final year to include (default 2100)

    Returns:
        86 x n_years matrix (ages 0-85+ x years 2025-end_year)
    """
    df = pd.read_csv('SSPopJul_TR2024.csv')

    df_future = df[(df['Year'] >= 2025) & (df['Year'] <= end_year)]

    # 86 age groups to match CPS (ages 0-84 individually, 85+ aggregated)
    # CPS top-codes age at 85, so we aggregate SSA ages 85-100 into "85+"
    n_ages = 86
    n_years = end_year - 2025 + 1
    target_matrix = np.zeros((n_ages, n_years))

    for year_idx, year in enumerate(range(2025, end_year + 1)):
        df_year = df_future[df_future['Year'] == year]

        for age in range(85):
            pop = df_year[df_year['Age'] == age]['Total'].values[0]
            target_matrix[age, year_idx] = pop

        pop_85plus = df_year[df_year['Age'] >= 85]['Total'].sum()
        target_matrix[85, year_idx] = pop_85plus

    return target_matrix


def reweight_all_years(end_year=2100):
    """
    Apply IPF reweighting for each year 2025-end_year using SSA projections.

    Args:
        end_year: Final year to include (default 2100)
    """
    from policyengine_us import Microsimulation

    print("\n" + "=" * 70)
    print(f"REWEIGHTING FOR ALL YEARS (2025-{end_year})")
    print("=" * 70)

    target_matrix = load_ssa_projections(end_year=end_year)
    print(f"Loaded target matrix: {target_matrix.shape}")

    print("\nLoading CPS microsimulation data...")
    sim = Microsimulation(dataset="hf://policyengine/policyengine-us-data/enhanced_cps_2024.h5")

    age_person = sim.calculate("age")
    person_household_id = sim.calculate("person_household_id")

    household_ids_unique = np.unique(person_household_id.values)
    n_households = len(household_ids_unique)

    n_ages = 86

    print("Creating design matrix...")
    X = np.zeros((n_households, n_ages))

    hh_id_to_idx = {hh_id: idx for idx, hh_id in enumerate(household_ids_unique)}

    for person_idx in range(len(age_person)):
        age = int(age_person.values[person_idx])
        hh_id = person_household_id.values[person_idx]
        hh_idx = hh_id_to_idx[hh_id]

        age_idx = min(age, 85)
        X[hh_idx, age_idx] += 1

    person_weight = sim.calculate("person_weight")
    initial_weights = np.zeros(n_households)
    for idx, hh_id in enumerate(household_ids_unique):
        hh_mask = person_household_id.values == hh_id
        initial_weights[idx] = person_weight.values[hh_mask][0]

    n_years = target_matrix.shape[1]
    weights_matrix = np.zeros((n_households, n_years))

    print(f"\nRunning IPF for {n_years} years...")
    print("Year    Max Rel Error    Iterations")
    print("-" * 40)

    for year_idx in range(n_years):
        year = 2025 + year_idx
        y_target = target_matrix[:, year_idx]

        w_new, info = iterative_proportional_fitting(
            X, y_target, initial_weights,
            max_iters=100, tol=1e-6, verbose=False
        )

        weights_matrix[:, year_idx] = w_new

        if year % 5 == 0 or year == 2025:
            max_error = np.max(np.abs(info['relative_errors_new']))
            print(f"{year}    {max_error:.6f}        {info['iterations']}")

    print("\n" + "-" * 40)
    print(f"Weights matrix shape: {weights_matrix.shape} ({n_households} households x {n_years} years)")

    print("\n" + "=" * 70)
    print("WEIGHTS SUMMARY")
    print("=" * 70)

    weight_ratios = weights_matrix / initial_weights.reshape(-1, 1)

    summary_years = [y for y in [2025, 2050, 2075, 2100] if y <= end_year]

    print("\nWeight adjustment factors (relative to initial):")
    for year in summary_years:
        year_idx = year - 2025
        ratios = weight_ratios[:, year_idx]
        print(f"  {year}: min={ratios.min():.3f}, max={ratios.max():.3f}, "
              f"mean={ratios.mean():.3f}, std={ratios.std():.3f}")

    print("\nTotal weight preservation check:")
    for year in summary_years:
        year_idx = year - 2025
        initial_total = initial_weights.sum()
        new_total = weights_matrix[:, year_idx].sum()
        pct_change = 100 * (new_total - initial_total) / initial_total
        print(f"  {year}: {new_total:,.0f} ({pct_change:+.2f}% from initial)")

    return weights_matrix, X, household_ids_unique


if __name__ == "__main__":
    weights_matrix, design_matrix, household_ids = reweight_all_years()
