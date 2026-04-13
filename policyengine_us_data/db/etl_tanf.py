import io
import logging
import re

import pandas as pd
import requests
from sqlmodel import Session, create_engine

from policyengine_us_data.storage import STORAGE_FOLDER
from policyengine_us_data.db.create_database_tables import (
    Stratum,
    StratumConstraint,
    Target,
)
from policyengine_us_data.utils.census import STATE_NAME_TO_FIPS
from policyengine_us_data.utils.db import etl_argparser, get_geographic_strata
from policyengine_us_data.utils.raw_cache import is_cached, load_bytes, save_bytes

logger = logging.getLogger(__name__)

CASELOAD_PAGE_URL = "https://www.acf.hhs.gov/ofa/data/tanf-caseload-data-2024"
FINANCIAL_PAGE_URL = "https://www.acf.hhs.gov/ofa/data/tanf-financial-data-fy-2024"
CASELOAD_URL_PATTERN = re.compile(
    r"https://acf\.gov/sites/default/files/documents/ofa/fy\d{4}_tanf_caseload\.xlsx"
)
FINANCIAL_URL_PATTERN = re.compile(
    r"https://acf\.gov/sites/default/files/documents/ofa/fy-\d{4}-tanf-moe-financial-data\.xlsx"
)


def _download_acf_excel(page_url: str, cache_file: str, url_pattern: re.Pattern) -> bytes:
    if is_cached(cache_file):
        logger.info("Using cached %s", cache_file)
        return load_bytes(cache_file)

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36"
            )
        }
    )

    page_response = session.get(page_url, timeout=30)
    page_response.raise_for_status()
    match = url_pattern.search(page_response.text)
    if match is None:
        raise ValueError(f"Could not find TANF workbook URL on {page_url}")

    workbook_url = match.group(0)
    workbook_response = session.get(workbook_url, timeout=60)
    workbook_response.raise_for_status()
    save_bytes(cache_file, workbook_response.content)
    return workbook_response.content


def extract_tanf_caseload_data(year: int) -> pd.DataFrame:
    workbook = _download_acf_excel(
        CASELOAD_PAGE_URL,
        f"tanf_caseload_{year}.xlsx",
        CASELOAD_URL_PATTERN,
    )
    return pd.read_excel(io.BytesIO(workbook), sheet_name="TFam", header=3)


def transform_tanf_caseload_data(raw_df: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    df = raw_df.copy()
    non_empty_columns = [column for column in df.columns if df[column].notna().any()]
    if len(non_empty_columns) < 2:
        raise ValueError("Unexpected TANF caseload workbook shape")

    state_column = non_empty_columns[0]
    value_column = non_empty_columns[-1]
    df = df[[state_column, value_column]].rename(
        columns={state_column: "state", value_column: "recipient_families"}
    )
    df["state"] = df["state"].astype(str).str.strip()
    df["recipient_families"] = pd.to_numeric(
        df["recipient_families"],
        errors="coerce",
    )
    df = df.dropna(subset=["recipient_families"])

    national_rows = df["state"].str.contains("U.S.", regex=False, na=False)
    if not national_rows.any():
        raise ValueError("Could not locate U.S. totals row in TANF caseload workbook")
    national_families = float(df.loc[national_rows, "recipient_families"].iloc[0])

    state_df = df.loc[df["state"].isin(STATE_NAME_TO_FIPS.keys())].copy()
    state_df["state_fips"] = state_df["state"].map(STATE_NAME_TO_FIPS).astype(int)
    state_df["ucgid_str"] = state_df["state_fips"].map(lambda fips: f"0400000US{fips:02d}")
    return national_families, state_df[
        ["state", "state_fips", "ucgid_str", "recipient_families"]
    ].sort_values("state_fips")


def extract_tanf_financial_data(
    year: int,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    workbook = _download_acf_excel(
        FINANCIAL_PAGE_URL,
        f"tanf_financial_{year}.xlsx",
        FINANCIAL_URL_PATTERN,
    )
    xls = pd.ExcelFile(io.BytesIO(workbook))
    national_df = pd.read_excel(
        xls,
        sheet_name="A.1 Fed & State by Category",
        header=1,
    )
    state_sheets = {}
    for state_name in STATE_NAME_TO_FIPS:
        sheet_name = "DC" if state_name == "District of Columbia" else state_name
        state_sheets[state_name] = pd.read_excel(
            xls,
            sheet_name=sheet_name,
            header=1,
        )
    return national_df, state_sheets


def _extract_basic_assistance_all_funds(df: pd.DataFrame) -> float:
    normalized = df.copy()
    normalized.columns = [
        re.sub(r"\s+", " ", str(column)).strip() for column in normalized.columns
    ]
    spending_category_column = next(
        (
            column
            for column in normalized.columns
            if column.lower().startswith("spending category")
        ),
        None,
    )
    if spending_category_column is None or "All Funds" not in normalized.columns:
        raise ValueError("Unexpected TANF financial workbook columns")

    mask = (
        normalized[spending_category_column].astype(str).str.strip()
        == "Basic Assistance"
    )
    if not mask.any():
        raise ValueError("Could not locate Basic Assistance row in TANF financial workbook")

    value = (
        normalized.loc[mask, "All Funds"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .pipe(pd.to_numeric, errors="coerce")
        .iloc[0]
    )
    return float(value)


def transform_tanf_financial_data(
    national_df: pd.DataFrame,
    state_sheets: dict[str, pd.DataFrame],
) -> tuple[float, pd.DataFrame]:
    national_spending = _extract_basic_assistance_all_funds(national_df)

    state_rows = []
    for state_name, df in state_sheets.items():
        state_rows.append(
            {
                "state": state_name,
                "state_fips": int(STATE_NAME_TO_FIPS[state_name]),
                "tanf": _extract_basic_assistance_all_funds(df),
            }
        )

    state_df = pd.DataFrame(state_rows).sort_values("state_fips").reset_index(drop=True)
    return national_spending, state_df


def load_tanf_data(
    national_families: float,
    national_spending: float,
    state_caseload_df: pd.DataFrame,
    state_financial_df: pd.DataFrame,
    year: int,
) -> None:
    database_url = f"sqlite:///{STORAGE_FOLDER / 'calibration' / 'policy_data.db'}"
    engine = create_engine(database_url)

    state_df = state_caseload_df.merge(
        state_financial_df,
        on=["state", "state_fips"],
        how="inner",
        validate="one_to_one",
    )
    if len(state_df) != len(STATE_NAME_TO_FIPS):
        raise ValueError(
            "Merged TANF caseload/financial targets do not cover all states: "
            f"{len(state_df)} rows"
        )

    with Session(engine) as session:
        geo_strata = get_geographic_strata(session)

        national_stratum = Stratum(
            parent_stratum_id=geo_strata["national"],
            notes="National TANF recipient families",
        )
        national_stratum.constraints_rel = [
            StratumConstraint(
                constraint_variable="tanf",
                operation=">",
                value="0",
            )
        ]
        national_stratum.targets_rel = [
            Target(
                variable="spm_unit_count",
                period=year,
                value=national_families,
                active=True,
                source="HHS ACF TANF Caseload",
                notes=f"Average monthly TANF recipient families | Source: ACF TFam FY{year}",
            ),
            Target(
                variable="tanf",
                period=year,
                value=national_spending,
                active=True,
                source="HHS ACF TANF Financial",
                notes=(
                    "Basic assistance all funds | "
                    f"Source: ACF TANF & MOE Financial Data FY{year}"
                ),
            ),
        ]
        session.add(national_stratum)
        session.flush()

        for row in state_df.itertuples(index=False):
            parent_stratum_id = geo_strata["state"][int(row.state_fips)]
            state_stratum = Stratum(
                parent_stratum_id=parent_stratum_id,
                notes=f"State FIPS {int(row.state_fips)} TANF recipient families",
            )
            state_stratum.constraints_rel = [
                StratumConstraint(
                    constraint_variable="state_fips",
                    operation="==",
                    value=str(int(row.state_fips)),
                ),
                StratumConstraint(
                    constraint_variable="tanf",
                    operation=">",
                    value="0",
                ),
            ]
            state_stratum.targets_rel = [
                Target(
                    variable="spm_unit_count",
                    period=year,
                    value=float(row.recipient_families),
                    active=True,
                    source="HHS ACF TANF Caseload",
                    notes=f"Average monthly TANF recipient families | Source: ACF TFam FY{year}",
                ),
                Target(
                    variable="tanf",
                    period=year,
                    value=float(row.tanf),
                    active=True,
                    source="HHS ACF TANF Financial",
                    notes=(
                        "Basic assistance all funds | "
                        f"Source: ACF TANF & MOE Financial Data FY{year}"
                    ),
                ),
            ]
            session.add(state_stratum)

        session.commit()


def main():
    _, year = etl_argparser("ETL for TANF administrative calibration targets")
    caseload_raw = extract_tanf_caseload_data(year)
    national_families, state_caseload_df = transform_tanf_caseload_data(caseload_raw)

    financial_national_df, financial_state_sheets = extract_tanf_financial_data(year)
    national_spending, state_financial_df = transform_tanf_financial_data(
        financial_national_df,
        financial_state_sheets,
    )

    load_tanf_data(
        national_families=national_families,
        national_spending=national_spending,
        state_caseload_df=state_caseload_df,
        state_financial_df=state_financial_df,
        year=year,
    )


if __name__ == "__main__":
    main()
