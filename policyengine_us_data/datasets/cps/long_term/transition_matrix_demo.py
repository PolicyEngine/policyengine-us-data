import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

y_age_2024 = np.array(
    [
        18333697.0,
        19799430.0,
        21203879.0,
        22168390.0,
        21618383.0,
        21906706.0,
        23405056.0,
        22650099.0,
        22126485.0,
        19859230.0,
        20661941.0,
        20198508.0,
        21676036.0,
        19026961.0,
        15797857.0,
        11318751.0,
        7041419.0,
        6122068.0,
    ]
)

age_brackets = [
    "0-4",
    "5-9",
    "10-14",
    "15-19",
    "20-24",
    "25-29",
    "30-34",
    "35-39",
    "40-44",
    "45-49",
    "50-54",
    "55-59",
    "60-64",
    "65-69",
    "70-74",
    "75-79",
    "80-84",
    "85-999",
]


def create_simple_transition_matrix():
    """
    Create a simple transition matrix where people age 5 years per period.
    Each 5-year age bracket moves to the next one.
    """
    n = 18
    T = np.zeros((n, n))

    for i in range(n - 1):
        T[i + 1, i] = 1.0

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

    if survival_rates is None:
        survival_rates = np.array(
            [
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
            ]
        )

    for i in range(n - 1):
        T[i + 1, i] = survival_rates[i]

    T[n - 1, n - 1] = survival_rates[n - 1]

    return T


def project_population(
    initial_pop, transition_matrix, n_periods, births_per_period=None
):
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
        projections[t + 1] = transition_matrix @ projections[t]

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
        selected_brackets = [0, 5, 10, 15, 17]

    fig = go.Figure()

    for idx in selected_brackets:
        fig.add_trace(
            go.Scatter(
                x=years,
                y=projections[:, idx] / 1e6,
                mode="lines",
                name=age_brackets[idx],
                line=dict(width=2),
            )
        )

    fig.update_layout(
        title="Population Projections by Age Bracket",
        xaxis_title="Year",
        yaxis_title="Population (millions)",
        hovermode="x unified",
        width=900,
        height=500,
    )

    return fig


def main():
    print("=" * 70)
    print("AGE COHORT PROJECTION SYSTEM")
    print("=" * 70)

    print("\n1. SIMPLE AGING (No mortality)")
    print("-" * 40)

    T_simple = create_simple_transition_matrix()
    n_periods = 15

    projections_simple = project_population(y_age_2024, T_simple, n_periods)

    for year_idx, year in enumerate([2024, 2049, 2074, 2099]):
        period = (year - 2024) // 5
        if period <= n_periods:
            print(f"\nYear {year} (Period {period}):")
            print(
                f"  Total population: {projections_simple[period].sum():,.0f}"
            )
            print(
                f"  In 85+ bracket: {projections_simple[period, 17]:,.0f} "
                f"({100 * projections_simple[period, 17] / projections_simple[period].sum():.1f}%)"
            )

    print("\n\n2. REALISTIC PROJECTION (With mortality)")
    print("-" * 40)

    T_realistic = create_realistic_transition_matrix()

    annual_births = 3_800_000
    births_per_5year = annual_births * 5

    projections_realistic = project_population(
        y_age_2024, T_realistic, n_periods, births_per_5year
    )

    for year_idx, year in enumerate([2024, 2049, 2074, 2099]):
        period = (year - 2024) // 5
        if period <= n_periods:
            print(f"\nYear {year} (Period {period}):")
            print(
                f"  Total population: {projections_realistic[period].sum():,.0f}"
            )
            print(f"  Age distribution:")
            for i in [0, 5, 10, 15, 17]:
                pct = (
                    100
                    * projections_realistic[period, i]
                    / projections_realistic[period].sum()
                )
                print(
                    f"    {age_brackets[i]:8} : {projections_realistic[period, i]:12,.0f} ({pct:5.1f}%)"
                )

    print("\n\n3. TRANSITION MATRIX STRUCTURE")
    print("-" * 40)
    print("\nSimple transition matrix (first 5x5):")
    print(T_simple[:5, :5])

    print("\nRealistic transition matrix (first 5x5):")
    print(T_realistic[:5, :5])

    years = [2024 + 5 * i for i in range(n_periods + 1)]

    df_projections = pd.DataFrame(
        projections_realistic, index=years, columns=age_brackets
    )

    print("\n\n4. PROJECTION SUMMARY TABLE")
    print("-" * 40)
    print("\nPopulation by age bracket (millions):")
    print(df_projections.iloc[::3, ::3] / 1e6)

    years = [2024 + 5 * i for i in range(n_periods + 1)]
    fig = visualize_projections(projections_realistic, years)
    fig.write_html("age_projections.html")
    print("\nVisualization saved to age_projections.html")

    print("\n\n5. TRANSITION MATRIX PROPERTIES")
    print("-" * 40)
    print(f"Matrix shape: {T_realistic.shape}")
    print(f"Column sums (should be ≤ 1 due to mortality):")
    col_sums = T_realistic.sum(axis=0)
    for i, s in enumerate(col_sums):
        print(f"  Column {i:2} ({age_brackets[i]:8}): {s:.4f}")

    eigenvalues, eigenvectors = np.linalg.eig(T_realistic)
    print(f"\nLargest eigenvalue: {np.max(np.real(eigenvalues)):.4f}")
    print("(Should be < 1 with mortality, = 1 without)")

    return T_realistic, projections_realistic, df_projections


def create_annual_transition_matrix():
    """
    Create annual transition matrix for year-by-year projections.
    Each year, 1/5 of each bracket ages into the next bracket.
    """
    n = 18
    T = np.zeros((n, n))

    survival_rates_5yr = np.array(
        [
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
        ]
    )

    annual_survival = survival_rates_5yr**0.2

    aging_fraction = 0.2

    for i in range(n):
        if i < n - 1:
            T[i, i] = (1 - aging_fraction) * annual_survival[i]
            T[i + 1, i] = aging_fraction * annual_survival[i]
        else:
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

    projections = np.zeros((n_years + 1, n_brackets))
    projections[0] = y_age_2024

    for year in range(n_years):
        projections[year + 1] = T @ projections[year]
        projections[year + 1, 0] += annual_births

    projections_2025_2100 = projections[1:, :]

    target_matrix = projections_2025_2100.T

    print(f"\nTarget matrix shape: {target_matrix.shape}")
    print(f"  - Rows: {target_matrix.shape[0]} age brackets")
    print(f"  - Columns: {target_matrix.shape[1]} years (2025-2100)")

    print(f"\nSample values (2025, 2050, 2100):")
    for year in [2025, 2050, 2100]:
        idx = year - 2025
        total = target_matrix[:, idx].sum()
        print(f"  {year}: {total:,.0f} total population")

    return target_matrix


if __name__ == "__main__":
    T, projections, df = main()
    target_matrix = project_annual_targets()
