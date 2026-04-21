"""ETL for ACA spending/enrollment and AGI state targets into policy_data.db."""

from __future__ import annotations

import logging
import hashlib

import pandas as pd
from sqlmodel import Session, create_engine, select

from policyengine_us_data.db.create_database_tables import (
    Stratum,
    StratumConstraint,
    Target,
)
from policyengine_us_data.storage import STORAGE_FOLDER
from policyengine_us_data.utils.census import STATE_ABBREV_TO_FIPS
from policyengine_us_data.utils.db import etl_argparser, get_geographic_strata

logger = logging.getLogger(__name__)

ACA_SPENDING_2024 = 9.8e10


def _definition_hash(
    parent_stratum_id: int, constraints: list[StratumConstraint]
) -> str:
    constraint_strings = [
        f"{c.constraint_variable}|{c.operation}|{c.value}" for c in constraints
    ]
    constraint_strings.sort()
    fingerprint_text = f"{parent_stratum_id}\n" + "\n".join(constraint_strings)
    return hashlib.sha256(fingerprint_text.encode("utf-8")).hexdigest()


def _get_or_create_stratum(
    session: Session,
    parent_stratum_id: int,
    note: str,
    constraints: list[StratumConstraint],
) -> Stratum:
    definition_hash = _definition_hash(parent_stratum_id, constraints)
    existing = session.exec(
        select(Stratum).where(Stratum.definition_hash == definition_hash)
    ).first()
    if existing is not None:
        return existing

    stratum = Stratum(
        parent_stratum_id=parent_stratum_id,
        notes=note,
    )
    stratum.constraints_rel = constraints
    session.add(stratum)
    return stratum


def _upsert_target(
    session: Session,
    stratum: Stratum,
    *,
    variable: str,
    period: int,
    value: float,
    source: str,
    notes: str | None = None,
) -> None:
    if stratum.stratum_id is None:
        stratum.targets_rel.append(
            Target(
                variable=variable,
                period=period,
                value=value,
                active=True,
                source=source,
                notes=notes,
            )
        )
        return

    existing = session.exec(
        select(Target).where(
            Target.stratum_id == stratum.stratum_id,
            Target.variable == variable,
            Target.period == period,
            Target.reform_id == 0,
        )
    ).first()
    if existing is None:
        session.add(
            Target(
                variable=variable,
                period=period,
                value=value,
                active=True,
                source=source,
                notes=notes,
                stratum_id=stratum.stratum_id,
            )
        )
        return

    existing.value = value
    existing.active = True
    existing.source = source
    if notes is not None:
        existing.notes = notes


def _load_aca_targets(session: Session, year: int, geo_strata: dict) -> None:
    data = pd.read_csv(
        STORAGE_FOLDER / "calibration_targets" / "aca_spending_and_enrollment_2024.csv"
    )

    # Monthly to yearly and normalize to national target to match loss.py.
    data["spending"] = data["spending"] * 12
    data["spending"] = data["spending"] * (ACA_SPENDING_2024 / data["spending"].sum())

    for _, row in data.iterrows():
        state = str(row["state"]).strip()
        state_fips = STATE_ABBREV_TO_FIPS.get(state)
        if state_fips is None:
            logger.warning("Skipping ACA target for unknown state %s", state)
            continue
        state_fips = int(state_fips)

        parent_stratum_id = geo_strata["state"].get(state_fips)
        if parent_stratum_id is None:
            logger.warning("No geo stratum for state %s (%s)", state, state_fips)
            continue

        spending_note = f"State FIPS {state_fips} ACA PTC spending"
        enrollment_note = f"State FIPS {state_fips} ACA PTC enrollment"

        spending_constraints = [
            StratumConstraint(
                constraint_variable="state_fips",
                operation="==",
                value=str(state_fips),
            ),
        ]
        spending_stratum = _get_or_create_stratum(
            session,
            parent_stratum_id,
            spending_note,
            spending_constraints,
        )
        _upsert_target(
            session,
            spending_stratum,
            variable="aca_ptc",
            period=year,
            value=float(row["spending"]),
            source="CMS Marketplace",
            notes="Annualized state ACA PTC spending scaled to national total",
        )

        enrollment_constraints = [
            StratumConstraint(
                constraint_variable="state_fips",
                operation="==",
                value=str(state_fips),
            ),
            StratumConstraint(
                constraint_variable="aca_ptc",
                operation=">",
                value="0",
            ),
            StratumConstraint(
                constraint_variable="is_aca_ptc_eligible",
                operation="==",
                value="True",
            ),
        ]
        enrollment_stratum = _get_or_create_stratum(
            session,
            parent_stratum_id,
            enrollment_note,
            enrollment_constraints,
        )
        _upsert_target(
            session,
            enrollment_stratum,
            variable="person_count",
            period=year,
            value=float(row["enrollment"]),
            source="CMS Marketplace",
            notes="State ACA enrollment (eligible with positive PTC)",
        )


def _load_agi_state_targets(session: Session, year: int, geo_strata: dict) -> None:
    soi_targets = pd.read_csv(STORAGE_FOLDER / "calibration_targets" / "agi_state.csv")

    for _, row in soi_targets.iterrows():
        state = str(row["GEO_NAME"]).strip()
        state_fips = STATE_ABBREV_TO_FIPS.get(state)
        if state_fips is None:
            logger.warning("Skipping AGI target for unknown state %s", state)
            continue
        state_fips = int(state_fips)

        parent_stratum_id = geo_strata["state"].get(state_fips)
        if parent_stratum_id is None:
            logger.warning("No geo stratum for state %s (%s)", state, state_fips)
            continue

        lower = float(row["AGI_LOWER_BOUND"])
        upper = float(row["AGI_UPPER_BOUND"])
        is_count = bool(row["IS_COUNT"])
        if is_count:
            target_variable = "tax_unit_count"
            note = (
                f"State FIPS {state_fips} AGI tax-unit count ({lower} <= AGI < {upper})"
            )
        else:
            target_variable = "adjusted_gross_income"
            note = f"State FIPS {state_fips} AGI total ({lower} <= AGI < {upper})"

        constraints = [
            StratumConstraint(
                constraint_variable="state_fips",
                operation="==",
                value=str(state_fips),
            ),
            StratumConstraint(
                constraint_variable="adjusted_gross_income",
                operation="<=",
                value=str(upper),
            ),
        ]
        if is_count:
            if lower > 0:
                constraints.append(
                    StratumConstraint(
                        constraint_variable="adjusted_gross_income",
                        operation=">=",
                        value=str(lower),
                    )
                )
            else:
                constraints.append(
                    StratumConstraint(
                        constraint_variable="adjusted_gross_income",
                        operation=">",
                        value="0",
                    )
                )
        else:
            constraints.append(
                StratumConstraint(
                    constraint_variable="adjusted_gross_income",
                    operation=">=",
                    value=str(lower),
                )
            )
        stratum = _get_or_create_stratum(
            session,
            parent_stratum_id,
            note,
            constraints,
        )
        _upsert_target(
            session,
            stratum,
            variable=target_variable,
            period=year,
            value=float(row["VALUE"]),
            source="IRS SOI",
        )


def main() -> int:
    _, year = etl_argparser(
        "ETL for ACA spending/enrollment and AGI state targets",
        allow_year=True,
    )

    database_url = f"sqlite:///{STORAGE_FOLDER / 'calibration' / 'policy_data.db'}"
    engine = create_engine(database_url)

    with Session(engine) as session:
        geo_strata = get_geographic_strata(session)
        _load_aca_targets(session, year, geo_strata)
        _load_agi_state_targets(session, year, geo_strata)
        session.commit()

    logger.info("Loaded ACA and AGI state targets for %s", year)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
