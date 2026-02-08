import numpy as np
import pandas as pd
from policyengine_us_data.storage import STORAGE_FOLDER


def load_ssa_age_projections(start_year=2025, end_year=2100):
    """
    Load SSA population projections from package storage.

    Args:
        start_year: First year to include (default 2025)
        end_year: Final year to include (default 2100)

    Returns:
        86 x n_years matrix (ages 0-85+ x years start_year-end_year)
    """
    csv_path = STORAGE_FOLDER / "SSPopJul_TR2024.csv"
    df = pd.read_csv(csv_path)

    df_future = df[(df["Year"] >= start_year) & (df["Year"] <= end_year)]

    MAX_SINGLE_AGE = 85
    n_ages = MAX_SINGLE_AGE + 1
    n_years = end_year - start_year + 1
    target_matrix = np.zeros((n_ages, n_years))

    for year_idx, year in enumerate(range(start_year, end_year + 1)):
        df_year = df_future[df_future["Year"] == year]

        for age in range(MAX_SINGLE_AGE):
            pop = df_year[df_year["Age"] == age]["Total"].values[0]
            target_matrix[age, year_idx] = pop

        pop_85plus = df_year[df_year["Age"] >= MAX_SINGLE_AGE]["Total"].sum()
        target_matrix[MAX_SINGLE_AGE, year_idx] = pop_85plus

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


def load_h6_income_rate_change(year):
    """
    Load H6 reform income rate change target for a given year.

    Args:
        year: Year to load rate change for

    Returns:
        H6 income rate change as decimal (e.g., -0.0018 for -0.18%)
    """
    csv_path = STORAGE_FOLDER / "social_security_aux.csv"
    df = pd.read_csv(csv_path)

    row = df[df["year"] == year]
    # CSV stores as percentage (e.g., -0.18), convert to decimal
    return row["h6_income_rate_change"].values[0] / 100


def load_oasdi_tob_projections(year):
    """
    Load OASDI TOB (Taxation of Benefits) revenue target for a given year.

    Args:
        year: Year to load OASDI TOB revenue for

    Returns:
        Total OASDI TOB revenue in nominal dollars
    """
    csv_path = STORAGE_FOLDER / "social_security_aux.csv"
    df = pd.read_csv(csv_path)

    row = df[df["year"] == year]
    nominal_billions = row["oasdi_tob_billions_nominal_usd"].values[0]
    return nominal_billions * 1e9


def load_hi_tob_projections(year):
    """
    Load HI (Medicare) TOB revenue target for a given year.

    Args:
        year: Year to load HI TOB revenue for

    Returns:
        Total HI TOB revenue in nominal dollars
    """
    csv_path = STORAGE_FOLDER / "social_security_aux.csv"
    df = pd.read_csv(csv_path)

    row = df[df["year"] == year]
    nominal_billions = row["hi_tob_billions_nominal_usd"].values[0]
    return nominal_billions * 1e9
