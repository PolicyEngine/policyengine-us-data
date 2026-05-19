"""BEA regional wage targets used by US data calibration."""

from __future__ import annotations

import io
import logging
import zipfile

import numpy as np
import pandas as pd

from policyengine_us_data.calibration.calibration_utils import STATE_CODES
from policyengine_us_data.utils.http import get_with_exponential_backoff
from policyengine_us_data.utils.raw_cache import (
    is_cached,
    load_bytes,
    save_bytes,
)

logger = logging.getLogger(__name__)

BEA_REGIONAL_SAINC_ZIP_URL = "https://apps.bea.gov/regional/zip/SAINC.zip"
BEA_REGIONAL_SAINC_CACHE_FILE = "bea_regional_sainc.zip"
BEA_STATE_WAGES_TABLE = "SAINC4"
BEA_STATE_WAGES_LINE_CODE = 50
BEA_STATE_SUPPLEMENTS_LINE_CODE = 60
BEA_STATE_CONTRIBUTIONS_LINE_CODE = 36
BEA_STATE_RESIDENCE_ADJUSTMENT_LINE_CODE = 42
BEA_STATE_WAGES_SOURCE = "BEA Regional SAINC4"
BEA_STATE_WAGES_SOURCE_URL = BEA_REGIONAL_SAINC_ZIP_URL


def _load_sainc_zip_bytes() -> bytes:
    if is_cached(BEA_REGIONAL_SAINC_CACHE_FILE):
        logger.info("Using cached %s", BEA_REGIONAL_SAINC_CACHE_FILE)
        return load_bytes(BEA_REGIONAL_SAINC_CACHE_FILE)

    logger.info(
        "Downloading BEA regional SAINC data from %s", BEA_REGIONAL_SAINC_ZIP_URL
    )
    response = get_with_exponential_backoff(
        BEA_REGIONAL_SAINC_ZIP_URL,
        timeout=120,
        initial_wait_seconds=5,
    )
    save_bytes(BEA_REGIONAL_SAINC_CACHE_FILE, response.content)
    return response.content


def _state_fips_from_geo_fips(value) -> int:
    return int(str(value).strip().strip('"')) // 1000


def _best_available_year(columns, requested_year: int) -> int:
    years = sorted(int(column) for column in columns if str(column).isdigit())
    if not years:
        raise ValueError("BEA SAINC4 data has no annual year columns")
    eligible = [year for year in years if year <= int(requested_year)]
    return max(eligible) if eligible else years[0]


def _line_value(df: pd.DataFrame, line_code: int, year: int) -> float:
    rows = df[df["LineCode"] == line_code]
    if rows.empty:
        raise ValueError(f"BEA SAINC4 file is missing line code {line_code}")
    return float(rows[str(year)].iloc[0]) * 1_000_000


def extract_bea_state_wages_and_salaries(
    requested_year: int,
    *,
    zip_bytes: bytes | None = None,
) -> tuple[pd.DataFrame, int]:
    """Extract residence-adjusted state wage totals from BEA SAINC4.

    BEA reports SAINC4 line 50 wages and salaries on a place-of-work basis.
    For household residence-state calibration, allocate line 42's residence
    adjustment to wages in proportion to the place-of-work net-compensation
    components described in BEA's state personal-income distribution method.
    """
    zip_bytes = zip_bytes if zip_bytes is not None else _load_sainc_zip_bytes()
    state_codes = set(STATE_CODES.values())
    rows = []
    data_year = None

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zip_file:
        for name in sorted(zip_file.namelist()):
            if not name.startswith(f"{BEA_STATE_WAGES_TABLE}_") or not name.endswith(
                ".csv"
            ):
                continue

            state_code = name.split("_")[1]
            if state_code not in state_codes:
                continue

            with zip_file.open(name) as state_file:
                df = pd.read_csv(state_file)

            df["LineCode"] = pd.to_numeric(df["LineCode"], errors="coerce")
            file_year = _best_available_year(df.columns, requested_year)
            if data_year is None:
                data_year = file_year
            elif data_year != file_year:
                raise ValueError(
                    "BEA SAINC4 files resolved to inconsistent years: "
                    f"{data_year} and {file_year}"
                )

            wages = _line_value(df, BEA_STATE_WAGES_LINE_CODE, file_year)
            supplements = _line_value(df, BEA_STATE_SUPPLEMENTS_LINE_CODE, file_year)
            contributions = _line_value(
                df,
                BEA_STATE_CONTRIBUTIONS_LINE_CODE,
                file_year,
            )
            residence_adjustment = _line_value(
                df,
                BEA_STATE_RESIDENCE_ADJUSTMENT_LINE_CODE,
                file_year,
            )
            denominator = wages + supplements + contributions
            adjustment_share = wages / denominator if denominator else 0
            wages_residence_adjusted = wages + residence_adjustment * adjustment_share

            state_fips = _state_fips_from_geo_fips(df["GeoFIPS"].iloc[0])
            rows.append(
                {
                    "state_fips": state_fips,
                    "state_code": state_code,
                    "wages_and_salaries_place_of_work": wages,
                    "wages_and_salaries": wages_residence_adjusted,
                    "supplements_to_wages_and_salaries": supplements,
                    "contributions_for_government_social_insurance": contributions,
                    "residence_adjustment": residence_adjustment,
                }
            )

    if data_year is None:
        raise ValueError("No state SAINC4 wage files found in BEA regional zip")

    result = pd.DataFrame(rows).sort_values("state_fips").reset_index(drop=True)
    expected_states = set(STATE_CODES.values())
    missing = expected_states - set(result["state_code"])
    if missing:
        raise ValueError(f"BEA SAINC4 wage data missing states: {sorted(missing)}")

    return result, data_year


def scale_state_wages_to_national_total(
    wages: pd.DataFrame,
    national_total: float,
) -> pd.DataFrame:
    """Scale BEA state wage distribution to a national wage aggregate."""
    result = wages.copy()
    state_total = float(result["wages_and_salaries"].sum())
    if not np.isfinite(state_total) or state_total <= 0:
        raise ValueError("BEA state wage total must be positive before scaling")

    scale_factor = float(national_total) / state_total
    result["employment_income_before_lsr"] = result["wages_and_salaries"] * scale_factor
    result["scale_factor"] = scale_factor
    return result


def get_bea_state_wage_targets(
    requested_year: int,
    *,
    national_total: float,
) -> tuple[pd.DataFrame, int]:
    wages, data_year = extract_bea_state_wages_and_salaries(requested_year)
    return scale_state_wages_to_national_total(wages, national_total), data_year
