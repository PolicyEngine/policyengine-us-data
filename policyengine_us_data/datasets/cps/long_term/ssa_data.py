import json
import os
from functools import lru_cache

import numpy as np
import pandas as pd
from policyengine_us_data.storage import STORAGE_FOLDER


LONG_TERM_TARGET_SOURCES_DIR = STORAGE_FOLDER / "long_term_target_sources"
LONG_TERM_TARGET_SOURCES_MANIFEST = LONG_TERM_TARGET_SOURCES_DIR / "sources.json"
DEFAULT_LONG_TERM_TARGET_SOURCE = "trustees_2025_current_law"
_CURRENT_LONG_TERM_TARGET_SOURCE = os.environ.get(
    "POLICYENGINE_US_DATA_LONG_TERM_TARGET_SOURCE",
    DEFAULT_LONG_TERM_TARGET_SOURCE,
)


@lru_cache(maxsize=1)
def _load_long_term_target_sources_manifest() -> dict:
    return json.loads(LONG_TERM_TARGET_SOURCES_MANIFEST.read_text(encoding="utf-8"))


def available_long_term_target_sources() -> list[str]:
    manifest = _load_long_term_target_sources_manifest()
    return sorted(manifest["sources"])


def get_long_term_target_source() -> str:
    return _CURRENT_LONG_TERM_TARGET_SOURCE


def set_long_term_target_source(source_name: str) -> None:
    global _CURRENT_LONG_TERM_TARGET_SOURCE
    _CURRENT_LONG_TERM_TARGET_SOURCE = resolve_long_term_target_source_name(source_name)


def resolve_long_term_target_source_name(source_name: str | None = None) -> str:
    manifest = _load_long_term_target_sources_manifest()
    candidate = source_name or _CURRENT_LONG_TERM_TARGET_SOURCE
    if candidate not in manifest["sources"]:
        valid = ", ".join(sorted(manifest["sources"]))
        raise ValueError(
            f"Unknown long-term target source {candidate!r}. Valid sources: {valid}"
        )
    return candidate


def describe_long_term_target_source(source_name: str | None = None) -> dict:
    manifest = _load_long_term_target_sources_manifest()
    resolved_name = resolve_long_term_target_source_name(source_name)
    source = dict(manifest["sources"][resolved_name])
    source["name"] = resolved_name
    return source


@lru_cache(maxsize=None)
def _load_long_term_target_frame(source_name: str) -> pd.DataFrame:
    source = describe_long_term_target_source(source_name)
    csv_path = LONG_TERM_TARGET_SOURCES_DIR / source["file"]
    return pd.read_csv(csv_path)


def _load_long_term_target_row(year: int, source_name: str | None = None) -> pd.Series:
    resolved_name = resolve_long_term_target_source_name(source_name)
    df = _load_long_term_target_frame(resolved_name)
    row = df[df["year"] == year]
    if row.empty:
        raise ValueError(
            f"Year {year} not found in long-term target source {resolved_name!r}"
        )
    return row.iloc[0]


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


def load_ssa_benefit_projections(year, source_name: str | None = None):
    """
    Load SSA Trustee Report projections for Social Security benefits.

    Args:
        year: Year to load benefits for

    Returns:
        Total OASDI benefits in nominal dollars
    """
    row = _load_long_term_target_row(year, source_name)
    nominal_billions = row["oasdi_cost_in_billion_nominal_usd"]
    return nominal_billions * 1e9


def load_taxable_payroll_projections(year, source_name: str | None = None):
    """
    Load SSA Trustee Report projections for taxable payroll.

    Args:
        year: Year to load taxable payroll for

    Returns:
        Total taxable payroll in nominal dollars
    """
    row = _load_long_term_target_row(year, source_name)
    nominal_billions = row["taxable_payroll_in_billion_nominal_usd"]
    return nominal_billions * 1e9


def load_h6_income_rate_change(year, source_name: str | None = None):
    """
    Load H6 reform income rate change target for a given year.

    Args:
        year: Year to load rate change for

    Returns:
        H6 income rate change as decimal (e.g., -0.0018 for -0.18%)
    """
    row = _load_long_term_target_row(year, source_name)
    # CSV stores as percentage (e.g., -0.18), convert to decimal
    return row["h6_income_rate_change"] / 100


def load_oasdi_tob_projections(year, source_name: str | None = None):
    """
    Load OASDI TOB (Taxation of Benefits) revenue target for a given year.

    Args:
        year: Year to load OASDI TOB revenue for

    Returns:
        Total OASDI TOB revenue in nominal dollars
    """
    row = _load_long_term_target_row(year, source_name)
    nominal_billions = row["oasdi_tob_billions_nominal_usd"]
    return nominal_billions * 1e9


def load_hi_tob_projections(year, source_name: str | None = None):
    """
    Load HI (Medicare) TOB revenue target for a given year.

    Args:
        year: Year to load HI TOB revenue for

    Returns:
        Total HI TOB revenue in nominal dollars
    """
    row = _load_long_term_target_row(year, source_name)
    nominal_billions = row["hi_tob_billions_nominal_usd"]
    return nominal_billions * 1e9
