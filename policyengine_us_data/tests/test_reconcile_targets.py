"""Tests for the database target reconciliation step."""

import pytest
from sqlmodel import Session

from policyengine_us_data.db.create_database_tables import (
    Stratum,
    StratumConstraint,
    Target,
    create_database,
)
from policyengine_us_data.db.reconcile_targets import (
    _compute_state_aggregate,
    _get_authoritative_targets,
    _scale_targets,
    reconcile_targets,
    TARGET_YEAR,
)


@pytest.fixture
def engine(tmp_path):
    db_uri = f"sqlite:///{tmp_path / 'test.db'}"
    return create_database(db_uri)


def _make_stratum(session, ucgid, extra_constraints=None):
    """Helper: create a stratum with a ucgid_str constraint."""
    s = Stratum(stratum_group_id=0, notes=f"Geo: {ucgid}")
    constraints = [
        StratumConstraint(
            constraint_variable="ucgid_str",
            operation="in",
            value=ucgid,
        )
    ]
    if extra_constraints:
        constraints.extend(extra_constraints)
    s.constraints_rel = constraints
    session.add(s)
    session.flush()
    return s


# Expected (variable, source_id) pairs in the authoritative map.
EXPECTED_TARGETS = [
    ("income_tax", 5),
    ("snap", 3),
    ("unemployment_compensation", 5),
    ("eitc", 5),
    ("adjusted_gross_income", 5),
    ("taxable_social_security", 5),
    ("taxable_pension_income", 5),
    ("net_capital_gain", 5),
    ("qualified_dividend_income", 5),
    ("taxable_interest_income", 5),
    ("tax_exempt_interest_income", 5),
    ("tax_unit_partnership_s_corp_income", 5),
    ("dividend_income", 5),
    ("person_count", 1),
    ("person_count", 2),
    ("person_count", 5),
]


def test_authoritative_targets_are_positive():
    """All mapped (variable, source_id) pairs return positive values."""
    targets = _get_authoritative_targets(TARGET_YEAR)
    for key in EXPECTED_TARGETS:
        assert key in targets, f"{key} missing from target map"
        assert (
            targets[key] > 0
        ), f"{key} should be positive, got {targets[key]}"


def test_compute_state_aggregate(engine):
    """State-level targets are summed; national/district excluded."""
    with Session(engine) as session:
        # National target (should NOT count)
        nat = _make_stratum(session, "0100000US")
        nat.targets_rel = [
            Target(
                variable="income_tax",
                period=2022,
                value=999e9,
                source_id=5,
                active=True,
            )
        ]

        # Two state targets (should count)
        for fips, val in [("01", 50e9), ("06", 150e9)]:
            st = _make_stratum(session, f"0400000US{fips}")
            st.targets_rel = [
                Target(
                    variable="income_tax",
                    period=2022,
                    value=val,
                    source_id=5,
                    active=True,
                )
            ]

        # District target (should NOT count)
        dist = _make_stratum(session, "5001800US0601")
        dist.targets_rel = [
            Target(
                variable="income_tax",
                period=2022,
                value=30e9,
                source_id=5,
                active=True,
            )
        ]

        session.commit()

        total, count = _compute_state_aggregate(
            session, "income_tax", source_id=5
        )
        assert count == 2
        assert total == pytest.approx(200e9)


def test_compute_state_aggregate_filters_by_source(engine):
    """Same variable with different source_ids are counted separately."""
    with Session(engine) as session:
        # Use different states to avoid definition_hash collision.
        # person_count from census age ETL (source_id=1)
        st1 = _make_stratum(session, "0400000US01")
        st1.targets_rel = [
            Target(
                variable="person_count",
                period=2023,
                value=5_000_000,
                source_id=1,
                active=True,
            )
        ]

        # person_count from medicaid ETL (source_id=2)
        st2 = _make_stratum(session, "0400000US06")
        st2.targets_rel = [
            Target(
                variable="person_count",
                period=2023,
                value=1_200_000,
                source_id=2,
                active=True,
            )
        ]

        # person_count from IRS SOI ETL (source_id=5)
        st3 = _make_stratum(session, "0400000US48")
        st3.targets_rel = [
            Target(
                variable="person_count",
                period=2022,
                value=2_500_000,
                source_id=5,
                active=True,
            )
        ]

        session.commit()

        total_age, count_age = _compute_state_aggregate(
            session, "person_count", source_id=1
        )
        total_med, count_med = _compute_state_aggregate(
            session, "person_count", source_id=2
        )
        total_soi, count_soi = _compute_state_aggregate(
            session, "person_count", source_id=5
        )

        assert count_age == 1
        assert total_age == pytest.approx(5_000_000)
        assert count_med == 1
        assert total_med == pytest.approx(1_200_000)
        assert count_soi == 1
        assert total_soi == pytest.approx(2_500_000)


def test_scale_targets(engine):
    """Targets for a variable+source are scaled; period updated."""
    with Session(engine) as session:
        nat = _make_stratum(session, "0100000US")
        nat.targets_rel = [
            Target(
                variable="eitc",
                period=2022,
                value=100e9,
                source_id=5,
                active=True,
            )
        ]
        st = _make_stratum(session, "0400000US01")
        st.targets_rel = [
            Target(
                variable="eitc",
                period=2022,
                value=40e9,
                source_id=5,
                active=True,
            )
        ]
        session.commit()

        n = _scale_targets(
            session,
            "eitc",
            source_id=5,
            scale_factor=1.5,
            target_year=2024,
        )
        session.commit()

        assert n == 2
        for t in session.exec(
            __import__("sqlmodel")
            .select(Target)
            .where(Target.variable == "eitc")
        ).all():
            assert t.period == 2024
            assert t.value in [
                pytest.approx(150e9),
                pytest.approx(60e9),
            ]


def test_scale_targets_isolates_sources(engine):
    """Scaling one source_id does not touch another source_id."""
    with Session(engine) as session:
        # Use separate strata to avoid unique-constraint collision
        # on (variable, period, stratum_id, reform_id).
        st1 = _make_stratum(session, "0400000US06")
        st1.targets_rel = [
            Target(
                variable="person_count",
                period=2023,
                value=39_000_000,
                source_id=1,
                active=True,
            ),
        ]
        st2 = _make_stratum(session, "0400000US48")
        st2.targets_rel = [
            Target(
                variable="person_count",
                period=2023,
                value=14_000_000,
                source_id=2,
                active=True,
            ),
        ]
        session.commit()

        # Scale only census age targets (source_id=1)
        _scale_targets(
            session,
            "person_count",
            source_id=1,
            scale_factor=1.1,
            target_year=2024,
        )
        session.commit()

        targets = session.exec(
            __import__("sqlmodel")
            .select(Target)
            .where(Target.variable == "person_count")
        ).all()

        by_source = {t.source_id: t for t in targets}
        # source_id=1 was scaled
        assert by_source[1].value == pytest.approx(39_000_000 * 1.1)
        assert by_source[1].period == 2024
        # source_id=2 was NOT touched
        assert by_source[2].value == pytest.approx(14_000_000)
        assert by_source[2].period == 2023


def test_reconcile_targets_scales_correctly(engine):
    """End-to-end: reconciliation scales DB to match parameters."""
    auth = _get_authoritative_targets(TARGET_YEAR)
    income_tax_target = auth[("income_tax", 5)]

    with Session(engine) as session:
        # Seed with a known state aggregate that differs from target
        state_total = 0.0
        for fips, share in [
            ("01", 0.02),
            ("06", 0.15),
            ("48", 0.10),
        ]:
            # Use an intentionally stale value (half of target)
            val = income_tax_target * share * 0.5
            state_total += val
            st = _make_stratum(session, f"0400000US{fips}")
            st.targets_rel = [
                Target(
                    variable="income_tax",
                    period=2022,
                    value=val,
                    source_id=5,
                    active=True,
                )
            ]
        session.commit()

        factors = reconcile_targets(session, TARGET_YEAR)

        assert ("income_tax", 5) in factors
        # After reconciliation state sum should match the target
        new_total, _ = _compute_state_aggregate(
            session, "income_tax", source_id=5
        )
        assert new_total == pytest.approx(income_tax_target, rel=1e-6)
