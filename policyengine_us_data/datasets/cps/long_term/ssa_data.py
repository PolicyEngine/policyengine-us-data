import numpy as np
import pandas as pd
from policyengine_us_data.storage import STORAGE_FOLDER


def load_ssa_age_projections(end_year=2100):
    """
    Load SSA population projections from package storage.

    Args:
        end_year: Final year to include (default 2100)

    Returns:
        86 x n_years matrix (ages 0-85+ x years 2025-end_year)
    """
    csv_path = STORAGE_FOLDER / "SSPopJul_TR2024.csv"
    df = pd.read_csv(csv_path)

    df_future = df[(df["Year"] >= 2025) & (df["Year"] <= end_year)]

    n_ages = 86
    n_years = end_year - 2025 + 1
    target_matrix = np.zeros((n_ages, n_years))

    for year_idx, year in enumerate(range(2025, end_year + 1)):
        df_year = df_future[df_future["Year"] == year]

        for age in range(85):
            pop = df_year[df_year["Age"] == age]["Total"].values[0]
            target_matrix[age, year_idx] = pop

        pop_85plus = df_year[df_year["Age"] >= 85]["Total"].sum()
        target_matrix[85, year_idx] = pop_85plus

    return target_matrix


def load_ssa_benefit_projections(year):
    """
    Load SSA Trustee Report projections for Social Security benefits.

    Args:
        year: Year to load benefits for

    Returns:
        Total OASDI benefits in nominal dollars
    """
    csv_path = STORAGE_FOLDER / "social_security_aux.csv"
    df = pd.read_csv(csv_path)

    row = df[df["year"] == year]
    nominal_billions = row["oasdi_cost_in_billion_nominal_usd"].values[0]
    return nominal_billions * 1e9


def load_taxable_payroll_projections(year):
    """
    Load SSA Trustee Report projections for taxable payroll.

    Args:
        year: Year to load taxable payroll for

    Returns:
        Total taxable payroll in nominal dollars
    """
    csv_path = STORAGE_FOLDER / "social_security_aux.csv"
    df = pd.read_csv(csv_path)

    row = df[df["year"] == year]
    nominal_billions = row["taxable_payroll_in_billion_nominal_usd"].values[0]
    return nominal_billions * 1e9
