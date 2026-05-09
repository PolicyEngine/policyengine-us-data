import io
import logging
import re

import pandas as pd
import requests
from sqlmodel import Session, create_engine
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

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

ACF_DATA_YEAR = 2024
ACF_REQUEST_TIMEOUT = 60

# Direct URLs for the FY-stamped ACF workbooks. The previous implementation
# scraped the HTML landing page (`acf.gov/ofa/data/tanf-...`) to discover
# these links, but that page is intermittently unreachable on `acf.gov` and
# was the dominant source of `make database` build failures (see #852). The
# workbook URLs themselves on `acf.gov/sites/default/files/documents/ofa/`
# return 200 reliably, so we hit them directly and skip the page entirely.
#
# Update this dict when:
#   - ACF publishes a new fiscal year's workbooks (add a new top-level key
#     and bump `ACF_DATA_YEAR` / `_validate_supported_year`),
#   - or ACF renames an existing FY's workbook on disk. A 404 from
#     `_acf_get` is the early signal — that's an authoritative file
#     rename, not a transient outage, and the new path needs to be
#     copied in by hand from the corresponding ACF page.
TANF_WORKBOOK_URLS: dict[int, dict[str, str]] = {
    2024: {
        "caseload": (
            "https://acf.gov/sites/default/files/documents/ofa/"
            "fy2024_tanf_caseload.xlsx"
        ),
        "financial": (
            "https://acf.gov/sites/default/files/documents/ofa/"
            "fy-2024-tanf-moe-financial-data.xlsx"
        ),
    },
}

TANF_FALLBACK_NATIONAL_FAMILIES_2024 = 841_208.6666666666
TANF_FALLBACK_NATIONAL_SPENDING_2024 = 7_788_317_474.55
TANF_FALLBACK_STATE_TARGETS_2024 = (
    (1, 5_792.91666666667, 32_124_953.24),
    (2, 1_221.91666666667, 18_434_755.86),
    (4, 4_892.75, 14_421_889.43),
    (5, 891.416666666667, 2_434_783.67),
    (6, 290_247.75, 3_742_540_224.36),
    (8, 12_456.0833333333, 69_699_677.55),
    (9, 5_194.83333333333, 42_471_684.66),
    (10, 2_560.75, 5_754_067.35),
    (11, 5_056.25, 45_666_113.5),
    (12, 30_186.8333333333, 100_340_059.95),
    (13, 4_153.25, 10_671_069),
    (15, 2_813.41666666667, 21_359_319),
    (16, 1_482.33333333333, 1_742_622),
    (17, 9_852.83333333333, 57_606_210.68),
    (18, 5_473.41666666667, 19_540_752.5),
    (19, 4_083.33333333333, 16_037_703.13),
    (20, 2_933.08333333333, 9_818_693.75),
    (21, 14_567.6666666667, 73_589_710.67),
    (22, 4_836.66666666667, 33_332_996),
    (23, 3_679.33333333333, 54_834_900.32),
    (24, 12_918.5, 139_982_898.46),
    (25, 36_646.4166666667, 353_175_296.67),
    (26, 7_964.33333333333, 47_781_011.05),
    (27, 13_334.5, 126_664_367),
    (28, 1_463.66666666667, 2_001_203),
    (29, 4_821.66666666667, 14_512_152.22),
    (30, 1_598.41666666667, 11_855_832),
    (31, 2_563.33333333333, 18_973_261.21),
    (32, 5_477.41666666667, 24_607_810),
    (33, 2_499.25, 27_637_906.62),
    (34, 9_513.41666666667, 101_463_207.58),
    (35, 7_008.08333333333, 49_470_156.42),
    (36, 128_334.583333333, 1_498_630_368),
    (37, 7_116.16666666667, 16_855_606),
    (38, 628.75, 4_710_100),
    (39, 39_233.75, 218_767_214.6),
    (40, 3_390.75, 9_712_053),
    (41, 18_196.1666666667, 78_534_753.73),
    (42, 24_170.25, 89_266_129),
    (44, 3_391, 28_690_264.33),
    (45, 5_605.16666666667, 23_327_185.48),
    (46, 2_486.16666666667, 12_891_592.56),
    (47, 12_117.25, 62_576_685.35),
    (48, 9_462.66666666667, 19_825_675.13),
    (49, 1_892.25, 20_275_644),
    (50, 1_654.83333333333, 13_584_363.3),
    (51, 13_187.9166666667, 68_554_925),
    (53, 33_068.25, 221_730_922.64),
    (54, 4_696.75, 41_374_940.75),
    (55, 11_287.6666666667, 62_580_997.86),
    (56, 478.583333333333, 5_880_764.97),
)


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=5, max=60),
    retry=retry_if_exception_type(
        (
            requests.exceptions.Timeout,
            requests.exceptions.ConnectionError,
            requests.exceptions.ChunkedEncodingError,
        )
    ),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def _acf_get(session: requests.Session, url: str) -> requests.Response:
    response = session.get(url, timeout=ACF_REQUEST_TIMEOUT)
    if response.status_code == 202 and response.headers.get("x-amzn-waf-action"):
        raise requests.exceptions.HTTPError(
            "ACF returned an AWS WAF challenge instead of a workbook",
            response=response,
        )
    response.raise_for_status()
    return response


def _validate_supported_year(year: int) -> None:
    if year != ACF_DATA_YEAR:
        raise ValueError(
            "TANF administrative calibration targets are currently available only "
            f"for FY{ACF_DATA_YEAR}; got year={year}"
        )


def _download_acf_excel(workbook_url: str, cache_file: str) -> bytes:
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

    workbook_response = _acf_get(session, workbook_url)
    save_bytes(cache_file, workbook_response.content)
    return workbook_response.content


def extract_tanf_caseload_data(year: int) -> pd.DataFrame:
    _validate_supported_year(year)
    workbook = _download_acf_excel(
        TANF_WORKBOOK_URLS[ACF_DATA_YEAR]["caseload"],
        f"tanf_caseload_{ACF_DATA_YEAR}.xlsx",
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
    state_df["ucgid_str"] = state_df["state_fips"].map(
        lambda fips: f"0400000US{fips:02d}"
    )
    return national_families, state_df[
        ["state", "state_fips", "ucgid_str", "recipient_families"]
    ].sort_values("state_fips")


def extract_tanf_financial_data(
    year: int,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    _validate_supported_year(year)
    workbook = _download_acf_excel(
        TANF_WORKBOOK_URLS[ACF_DATA_YEAR]["financial"],
        f"tanf_financial_{ACF_DATA_YEAR}.xlsx",
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


def _extract_cash_assistance_all_funds(df: pd.DataFrame) -> float:
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

    mask = normalized[spending_category_column].astype(str).str.strip() == (
        "Basic Assistance (excluding Relative Foster Care Maintenance Payments "
        "and Adoption and Guardianship Subsidies)"
    )
    if not mask.any():
        raise ValueError(
            "Could not locate narrow Basic Assistance row in TANF financial workbook"
        )

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
    national_spending = _extract_cash_assistance_all_funds(national_df)

    state_rows = []
    for state_name, df in state_sheets.items():
        state_rows.append(
            {
                "state": state_name,
                "state_fips": int(STATE_NAME_TO_FIPS[state_name]),
                "tanf": _extract_cash_assistance_all_funds(df),
            }
        )

    state_df = pd.DataFrame(state_rows).sort_values("state_fips").reset_index(drop=True)
    return national_spending, state_df


def fallback_tanf_targets_2024() -> tuple[
    float,
    pd.DataFrame,
    float,
    pd.DataFrame,
]:
    """Return FY2024 TANF targets when acf.gov blocks non-browser downloads.

    The values are the ACF FY2024 workbook targets extracted by the prior
    successful data-build checkpoint. They keep Modal builds reproducible when
    ACF's AWS WAF returns a JavaScript challenge to batch downloads.
    """
    state_by_fips = {int(fips): state for state, fips in STATE_NAME_TO_FIPS.items()}
    rows = []
    for state_fips, recipient_families, tanf in TANF_FALLBACK_STATE_TARGETS_2024:
        rows.append(
            {
                "state": state_by_fips[state_fips],
                "state_fips": state_fips,
                "ucgid_str": f"0400000US{state_fips:02d}",
                "recipient_families": recipient_families,
                "tanf": tanf,
            }
        )
    state_df = pd.DataFrame(rows).sort_values("state_fips").reset_index(drop=True)
    return (
        TANF_FALLBACK_NATIONAL_FAMILIES_2024,
        state_df[["state", "state_fips", "ucgid_str", "recipient_families"]],
        TANF_FALLBACK_NATIONAL_SPENDING_2024,
        state_df[["state", "state_fips", "tanf"]],
    )


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
                notes=(
                    "Average monthly TANF recipient families | "
                    f"Source: ACF TFam FY{ACF_DATA_YEAR}"
                ),
            ),
            Target(
                variable="tanf",
                period=year,
                value=national_spending,
                active=True,
                source="HHS ACF TANF Financial",
                notes=(
                    "Basic assistance excluding relative foster care maintenance "
                    "payments and adoption and guardianship subsidies | "
                    f"Source: ACF TANF & MOE Financial Data FY{ACF_DATA_YEAR}"
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
                    notes=(
                        "Average monthly TANF recipient families | "
                        f"Source: ACF TFam FY{ACF_DATA_YEAR}"
                    ),
                ),
                Target(
                    variable="tanf",
                    period=year,
                    value=float(row.tanf),
                    active=True,
                    source="HHS ACF TANF Financial",
                    notes=(
                        "Basic assistance excluding relative foster care maintenance "
                        "payments and adoption and guardianship subsidies | "
                        f"Source: ACF TANF & MOE Financial Data FY{ACF_DATA_YEAR}"
                    ),
                ),
            ]
            session.add(state_stratum)

        session.commit()


def main():
    _, year = etl_argparser("ETL for TANF administrative calibration targets")
    try:
        caseload_raw = extract_tanf_caseload_data(year)
        national_families, state_caseload_df = transform_tanf_caseload_data(
            caseload_raw
        )

        financial_national_df, financial_state_sheets = extract_tanf_financial_data(
            year
        )
        national_spending, state_financial_df = transform_tanf_financial_data(
            financial_national_df,
            financial_state_sheets,
        )
    except requests.exceptions.RequestException as exc:
        if year != ACF_DATA_YEAR:
            raise
        logger.warning(
            "Using bundled FY%s TANF fallback targets because ACF workbook "
            "download failed: %s",
            ACF_DATA_YEAR,
            exc,
        )
        (
            national_families,
            state_caseload_df,
            national_spending,
            state_financial_df,
        ) = fallback_tanf_targets_2024()

    load_tanf_data(
        national_families=national_families,
        national_spending=national_spending,
        state_caseload_df=state_caseload_df,
        state_financial_df=state_financial_df,
        year=year,
    )


if __name__ == "__main__":
    main()
