"""ETL for HUD-assisted household count calibration targets."""

from __future__ import annotations

import io
import logging

import pandas as pd
import requests
from sqlmodel import Session, create_engine

from policyengine_us_data.db.create_database_tables import (
    Stratum,
    StratumConstraint,
    Target,
)
from policyengine_us_data.storage import STORAGE_FOLDER
from policyengine_us_data.utils.census import STATE_ABBREV_TO_FIPS
from policyengine_us_data.utils.db import (
    etl_argparser,
    get_geographic_strata,
    parse_ucgid,
)
from policyengine_us_data.utils.raw_cache import (
    is_cached,
    load_bytes,
    save_bytes,
)

logger = logging.getLogger(__name__)

HUD_PICTURE_SOURCE = "HUD Picture of Subsidized Households"


def _hud_picture_state_url(year: int) -> str:
    return (
        "https://www.huduser.gov/portal/datasets/pictures/files/"
        f"STATE_{year}_2020census.xlsx"
    )


def extract_hud_picture_state_data(year: int) -> bytes:
    """Download HUD Picture of Subsidized Households state extract."""
    cache_file = f"hud_picture_state_{year}_2020census.xlsx"
    if is_cached(cache_file):
        logger.info("Using cached %s", cache_file)
        return load_bytes(cache_file)

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/123.0.0.0 Safari/537.36"
        ),
        "Accept": (
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet,"
            "application/vnd.ms-excel,*/*"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    }
    response = requests.get(
        _hud_picture_state_url(year),
        headers=headers,
        timeout=60,
    )
    response.raise_for_status()
    if not response.content.startswith(b"PK"):
        raise ValueError(
            "HUD Picture state extract did not return an Excel workbook. "
            f"HTTP status={response.status_code}, "
            f"x-amzn-waf-action={response.headers.get('x-amzn-waf-action')!r}"
        )
    save_bytes(cache_file, response.content)
    return response.content


def transform_hud_picture_state_data(workbook_content: bytes) -> pd.DataFrame:
    """Return state assisted-household targets from a HUD Picture workbook."""
    raw_df = pd.read_excel(io.BytesIO(workbook_content))
    summary = raw_df.loc[
        raw_df["program_label"].eq("Summary of All HUD Programs")
        & raw_df["State"].isin(STATE_ABBREV_TO_FIPS)
    ].copy()
    summary["STATE_FIPS"] = summary["State"].map(STATE_ABBREV_TO_FIPS)
    summary["ucgid_str"] = "0400000US" + summary["STATE_FIPS"]
    summary["assisted_households"] = pd.to_numeric(
        summary["number_reported"],
        errors="raise",
    )
    return (
        summary[["ucgid_str", "assisted_households"]]
        .sort_values("ucgid_str")
        .reset_index(drop=True)
    )


def load_housing_assistance_data(state_df: pd.DataFrame, year: int) -> None:
    """Load national and state assisted-household count targets."""
    database_url = f"sqlite:///{STORAGE_FOLDER / 'calibration' / 'policy_data.db'}"
    engine = create_engine(database_url)
    national_households = float(state_df["assisted_households"].sum())

    with Session(engine) as session:
        geo_strata = get_geographic_strata(session)
        if geo_strata["national"] is None:
            raise ValueError(
                "National stratum not found. Run create_initial_strata.py first."
            )

        national_stratum = Stratum(
            parent_stratum_id=geo_strata["national"],
            notes="National HUD-assisted households",
        )
        national_stratum.constraints_rel = [
            StratumConstraint(
                constraint_variable="housing_assistance",
                operation=">",
                value="0",
            )
        ]
        national_stratum.targets_rel.append(
            Target(
                variable="household_count",
                period=year,
                value=national_households,
                active=True,
                source=HUD_PICTURE_SOURCE,
                notes=(
                    "HUD Picture of Subsidized Households state extract, "
                    "Summary of All HUD Programs, number_reported column. "
                    "This is a December point-in-time assisted-household count."
                ),
            )
        )
        session.add(national_stratum)
        session.flush()

        for _, row in state_df.iterrows():
            state_fips = parse_ucgid(row["ucgid_str"])["state_fips"]
            parent_stratum_id = geo_strata["state"][state_fips]
            state_stratum = Stratum(
                parent_stratum_id=parent_stratum_id,
                notes=f"State FIPS {state_fips} HUD-assisted households",
            )
            state_stratum.constraints_rel = [
                StratumConstraint(
                    constraint_variable="state_fips",
                    operation="==",
                    value=str(state_fips),
                ),
                StratumConstraint(
                    constraint_variable="housing_assistance",
                    operation=">",
                    value="0",
                ),
            ]
            state_stratum.targets_rel.append(
                Target(
                    variable="household_count",
                    period=year,
                    value=float(row["assisted_households"]),
                    active=True,
                    source=HUD_PICTURE_SOURCE,
                    notes=(
                        "HUD Picture of Subsidized Households state extract, "
                        "Summary of All HUD Programs, number_reported column. "
                        "This is a December point-in-time assisted-household count."
                    ),
                )
            )
            session.add(state_stratum)

        session.commit()


def main() -> None:
    _, year = etl_argparser("ETL for HUD housing assistance count targets")
    workbook = extract_hud_picture_state_data(year)
    state_df = transform_hud_picture_state_data(workbook)
    load_housing_assistance_data(state_df, year)


if __name__ == "__main__":
    main()
