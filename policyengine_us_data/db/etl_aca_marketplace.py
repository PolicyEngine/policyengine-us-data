from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from sqlmodel import Session, create_engine

from policyengine_us_data.calibration.calibration_utils import STATE_CODES
from policyengine_us_data.db.create_database_tables import (
    Stratum,
    StratumConstraint,
    Target,
)
from policyengine_us_data.storage import CALIBRATION_FOLDER, STORAGE_FOLDER
from policyengine_us_data.utils.db import etl_argparser, get_geographic_strata

logger = logging.getLogger(__name__)

# `selected_marketplace_plan_benchmark_ratio == 1.0` represents benchmark
# silver coverage, so bronze plan selections are the subset below this ratio.
BENCHMARK_SILVER_RATIO = 1.0

STATE_METAL_SELECTION_PATH = (
    CALIBRATION_FOLDER / "aca_marketplace_state_metal_selection_2024.csv"
)

STATE_ABBR_TO_FIPS = {abbr: fips for fips, abbr in STATE_CODES.items()}


def _extra_args(parser) -> None:
    parser.add_argument(
        "--state-metal-csv",
        type=Path,
        default=STATE_METAL_SELECTION_PATH,
        help=("State-metal CMS OEP proxy CSV. Default: %(default)s"),
    )


def extract_aca_marketplace_state_metal_data(
    state_metal_csv_path: Path,
) -> pd.DataFrame:
    """Extract CMS marketplace state metal-status inputs from the checked-in CSV.

    This ETL keeps an explicit extract step even though the source file already
    lives in the repository. The original CMS 2024 OEP state metal status PUF
    is not currently pulled from a stable direct-download endpoint in CI, so we
    store the normalized input CSV at
    `policyengine_us_data/storage/calibration_targets/aca_marketplace_state_metal_selection_2024.csv`.

    To reproduce or update that file:
    1. Download the CMS 2024 OEP state metal status public use file.
    2. Preserve one row per state/platform/metal/enrollment-status combination.
    3. Keep the `state_code`, `platform`, `metal_level`,
       `enrollment_status`, `consumers`, and `aptc_consumers` columns.
    4. Save the normalized output back to `state_metal_csv_path`.
    """
    return pd.read_csv(state_metal_csv_path)


def build_state_marketplace_bronze_aptc_targets(
    state_metal_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build HC.gov state bronze-selection targets among APTC consumers.

    The 2024 CMS state-metal-status PUF exposes:
    - metal rows (`B`, `G`, `S`) with enrollment_status=`All`
    - aggregate rows (`All`) broken out by enrollment status (`01-atv`, etc.)

    We use:
    - total APTC consumers = sum of `aptc_consumers` for `metal_level == All`
      across enrollment statuses
    - bronze APTC consumers = `aptc_consumers` on the bronze row
    """
    df = state_metal_df.copy()
    df = df[df["platform"] == "HC.gov"].copy()

    total_rows = df[
        (df["metal_level"] == "All") & (df["aptc_consumers"].notna())
    ].copy()
    bronze_rows = df[
        (df["metal_level"] == "B")
        & (df["enrollment_status"] == "All")
        & (df["aptc_consumers"].notna())
    ].copy()

    total_aptc = total_rows.groupby("state_code", as_index=False).agg(
        marketplace_aptc_consumers=("aptc_consumers", "sum"),
        marketplace_consumers=("consumers", "sum"),
    )
    bronze_aptc = bronze_rows[["state_code", "aptc_consumers", "consumers"]].rename(
        columns={
            "aptc_consumers": "bronze_aptc_consumers",
            "consumers": "bronze_consumers",
        }
    )

    result = total_aptc.merge(bronze_aptc, on="state_code", how="inner")
    result["state_fips"] = result["state_code"].map(STATE_ABBR_TO_FIPS)
    result = result[result["state_fips"].notna()].copy()
    result["state_fips"] = result["state_fips"].astype(int)
    result["bronze_aptc_share"] = (
        result["bronze_aptc_consumers"] / result["marketplace_aptc_consumers"]
    )
    result.insert(0, "year", 2024)
    result.insert(1, "source", "cms_2024_oep_state_metal_status_puf")
    return result.sort_values("state_code").reset_index(drop=True)


def load_state_marketplace_bronze_aptc_targets(
    targets_df: pd.DataFrame,
    year: int,
) -> None:
    db_url = f"sqlite:///{STORAGE_FOLDER / 'calibration' / 'policy_data.db'}"
    engine = create_engine(db_url)

    with Session(engine) as session:
        geo_strata = get_geographic_strata(session)

        for row in targets_df.itertuples(index=False):
            state_fips = int(row.state_fips)
            parent_id = geo_strata["state"].get(state_fips)
            if parent_id is None:
                logger.warning(
                    "No state geographic stratum for FIPS %s, skipping", state_fips
                )
                continue

            # We intentionally do not subset to `tax_unit_is_filer == 1`.
            # These CMS targets describe marketplace coverage groups rather
            # than the IRS filer universe, so the closest calibration entity is
            # a tax unit with positive modeled APTC use.
            aptc_stratum = Stratum(
                parent_stratum_id=parent_id,
                notes=f"State FIPS {state_fips} Marketplace APTC recipients",
            )
            aptc_stratum.constraints_rel = [
                StratumConstraint(
                    constraint_variable="state_fips",
                    operation="==",
                    value=str(state_fips),
                ),
                StratumConstraint(
                    constraint_variable="used_aca_ptc",
                    operation=">",
                    value="0",
                ),
            ]
            aptc_stratum.targets_rel.append(
                Target(
                    # We use `tax_unit_count` rather than household/person
                    # counts because insurance groups map most closely to
                    # PolicyEngine tax units in the current calibration schema.
                    variable="tax_unit_count",
                    period=year,
                    value=float(row.marketplace_aptc_consumers),
                    active=True,
                    source="CMS 2024 OEP state metal status PUF",
                    notes="HC.gov APTC consumers across all enrollment statuses",
                )
            )
            session.add(aptc_stratum)
            session.flush()

            bronze_stratum = Stratum(
                parent_stratum_id=aptc_stratum.stratum_id,
                notes=f"State FIPS {state_fips} Marketplace bronze APTC recipients",
            )
            bronze_stratum.constraints_rel = [
                StratumConstraint(
                    constraint_variable="state_fips",
                    operation="==",
                    value=str(state_fips),
                ),
                StratumConstraint(
                    constraint_variable="used_aca_ptc",
                    operation=">",
                    value="0",
                ),
                StratumConstraint(
                    constraint_variable="selected_marketplace_plan_benchmark_ratio",
                    operation="<",
                    value=str(BENCHMARK_SILVER_RATIO),
                ),
            ]
            bronze_stratum.targets_rel.append(
                Target(
                    variable="tax_unit_count",
                    period=year,
                    value=float(row.bronze_aptc_consumers),
                    active=True,
                    source="CMS 2024 OEP state metal status PUF",
                    notes="HC.gov bronze plan selections among APTC consumers",
                )
            )
            session.add(bronze_stratum)
            session.flush()

        session.commit()


def main() -> None:
    args, year = etl_argparser(
        "ETL for ACA marketplace bronze-selection calibration targets",
        extra_args_fn=_extra_args,
    )

    state_metal = extract_aca_marketplace_state_metal_data(args.state_metal_csv)
    targets_df = build_state_marketplace_bronze_aptc_targets(state_metal)
    if targets_df.empty:
        raise RuntimeError("No HC.gov marketplace bronze/APTC targets were generated.")

    print(
        "Loading ACA marketplace bronze/APTC state targets for "
        f"{len(targets_df)} states from {args.state_metal_csv}"
    )
    load_state_marketplace_bronze_aptc_targets(targets_df, year)
    print("ACA marketplace bronze/APTC targets loaded.")


if __name__ == "__main__":
    main()
