"""Tests for CBO-based scaling of IRS SOI database targets."""

import pytest
from sqlmodel import Session, select

from policyengine_us_data.db.create_database_tables import (
    Stratum,
    StratumConstraint,
    Target,
    create_database,
)
from policyengine_us_data.db.scale_irs_soi_to_cbo import (
    SOI_SOURCE_ID,
    TARGET_YEAR,
    _compute_state_aggregate,
    _get_cbo_targets,
    _scale_targets,
    scale_soi_to_cbo,
)


@pytest.fixture
def engine(tmp_path):
    db_uri = f"sqlite:///{tmp_path / 'test.db'}"
    return create_database(db_uri)


def _make_stratum(session, ucgid):
    """Helper: create a stratum with a ucgid_str constraint."""
    s = Stratum(stratum_group_id=0, notes=f"Geo: {ucgid}")
    s.constraints_rel = [
        StratumConstraint(
            constraint_variable="ucgid_str",
            operation="in",
            value=ucgid,
        )
    ]
    session.add(s)
    session.flush()
    return s


CBO_VARIABLES = [
    "income_tax",
    "unemployment_compensation",
    "eitc",
    "adjusted_gross_income",
    "taxable_social_security",
    "taxable_pension_income",
    "net_capital_gain",
    "qualified_dividend_income",
    "taxable_interest_income",
    "tax_exempt_interest_income",
    "tax_unit_partnership_s_corp_income",
    "dividend_income",
    "person_count",
]


def test_cbo_targets_are_positive():
    """All CBO targets return positive 2024 values."""
    targets = _get_cbo_targets(TARGET_YEAR)
    for name in CBO_VARIABLES:
        assert name in targets, f"'{name}' missing"
        assert targets[name] > 0, f"{name} = {targets[name]}"


def test_only_soi_targets_are_scaled(engine):
    """Scaling only affects source_id=5, not other sources."""
    with Session(engine) as session:
        st = _make_stratum(session, "0400000US06")
        st.targets_rel = [
            Target(
                variable="income_tax",
                period=2022,
                value=300e9,
                source_id=SOI_SOURCE_ID,
                active=True,
            ),
        ]
        st2 = _make_stratum(session, "0400000US48")
        st2.targets_rel = [
            Target(
                variable="person_count",
                period=2023,
                value=30_000_000,
                source_id=1,  # Census age, not SOI
                active=True,
            ),
        ]
        session.commit()

        _scale_targets(session, "income_tax", 1.5, 2024)
        session.commit()

        soi_t = session.exec(
            select(Target)
            .where(Target.variable == "income_tax")
            .where(Target.source_id == SOI_SOURCE_ID)
        ).one()
        assert soi_t.value == pytest.approx(450e9)
        assert soi_t.period == 2024

        census_t = session.exec(
            select(Target)
            .where(Target.variable == "person_count")
            .where(Target.source_id == 1)
        ).one()
        assert census_t.value == pytest.approx(30_000_000)
        assert census_t.period == 2023


def test_end_to_end_scaling(engine):
    """After scaling, state aggregate matches CBO target."""
    cbo = _get_cbo_targets(TARGET_YEAR)
    cbo_income_tax = cbo["income_tax"]

    with Session(engine) as session:
        for fips, share in [("01", 0.02), ("06", 0.15), ("48", 0.10)]:
            val = cbo_income_tax * share * 0.5  # intentionally stale
            st = _make_stratum(session, f"0400000US{fips}")
            st.targets_rel = [
                Target(
                    variable="income_tax",
                    period=2022,
                    value=val,
                    source_id=SOI_SOURCE_ID,
                    active=True,
                )
            ]
        session.commit()

        factors = scale_soi_to_cbo(session, TARGET_YEAR)

        assert "income_tax" in factors
        new_total, _ = _compute_state_aggregate(session, "income_tax")
        assert new_total == pytest.approx(cbo_income_tax, rel=1e-6)
