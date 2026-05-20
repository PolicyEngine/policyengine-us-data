"""ETL for BEA regional state wage calibration targets."""

import logging

import pandas as pd
from sqlmodel import Session, create_engine

from policyengine_us_data.db.etl_national_targets import (
    BEA_NIPA_WAGES_AND_SALARIES_2024,
    _register_target_variable,
    _upsert_baseline_target,
)
from policyengine_us_data.storage import STORAGE_FOLDER
from policyengine_us_data.utils.bea_regional import (
    BEA_STATE_WAGES_SOURCE,
    BEA_STATE_WAGES_SOURCE_URL,
    get_bea_state_wage_targets,
)
from policyengine_us_data.utils.db import etl_argparser, get_geographic_strata

logger = logging.getLogger(__name__)

TARGET_VARIABLE = "employment_income_before_lsr"


def extract_bea_state_wage_targets(year: int) -> tuple[pd.DataFrame, int]:
    """Extract BEA state wage targets scaled to the national NIPA total."""
    return get_bea_state_wage_targets(
        year,
        national_total=BEA_NIPA_WAGES_AND_SALARIES_2024,
    )


def load_bea_state_wage_targets(
    targets: pd.DataFrame,
    *,
    target_year: int,
    source_year: int,
) -> int:
    """Load BEA state wage targets into state geographic strata."""
    if targets.empty:
        return 0

    database_url = f"sqlite:///{STORAGE_FOLDER / 'calibration' / 'policy_data.db'}"
    engine = create_engine(database_url)
    loaded = 0

    with Session(engine) as session:
        _register_target_variable(session, TARGET_VARIABLE)
        geo_strata = get_geographic_strata(session)
        state_strata = geo_strata.get("state", {})

        for row in targets.itertuples(index=False):
            state_fips = int(row.state_fips)
            stratum_id = state_strata.get(state_fips)
            if stratum_id is None:
                logger.warning(
                    "No geographic stratum found for state %s (FIPS %s), skipping",
                    row.state_code,
                    state_fips,
                )
                continue

            _upsert_baseline_target(
                session,
                stratum_id=stratum_id,
                variable=TARGET_VARIABLE,
                period=target_year,
                value=float(row.employment_income_before_lsr),
                source=BEA_STATE_WAGES_SOURCE,
                notes=(
                    "BEA SAINC4 line 50 wages and salaries by state, adjusted "
                    "to a residence basis by allocating line 42's residence "
                    "adjustment to wages in proportion to place-of-work "
                    "net-compensation components, then scaled to the national "
                    "BEA NIPA Table 2.1 wages and salaries target. "
                    f"Source year: {source_year}; state: {row.state_code}; "
                    f"raw residence-adjusted state wages: "
                    f"${row.wages_and_salaries:,.0f}; "
                    f"national scaling factor: {row.scale_factor:.8f}; "
                    f"Source: {BEA_STATE_WAGES_SOURCE_URL}"
                ),
            )
            loaded += 1

        session.commit()

    logger.info("Loaded %s BEA state wage targets", loaded)
    return loaded


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    _, year = etl_argparser(
        "ETL for BEA regional state wage calibration targets",
        allow_year=True,
    )

    targets, source_year = extract_bea_state_wage_targets(year)
    loaded = load_bea_state_wage_targets(
        targets,
        target_year=year,
        source_year=source_year,
    )

    logger.info(
        "BEA State Wage Targets Summary:\n"
        "  Source year: %s\n"
        "  States loaded: %s\n"
        "  Target total: $%.1fT",
        source_year,
        loaded,
        targets["employment_income_before_lsr"].sum() / 1e12,
    )


if __name__ == "__main__":
    main()
