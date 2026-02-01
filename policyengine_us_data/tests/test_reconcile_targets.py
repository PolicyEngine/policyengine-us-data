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


def test_authoritative_targets_are_positive():
    """Smoke test: policyengine-us parameters return positive values."""
    targets = _get_authoritative_targets(TARGET_YEAR)
    assert len(targets) >= 4
    for name, value in targets.items():
        assert value > 0, f"{name} should be positive, got {value}"


def test_compute_state_aggregate(engine):
    """State-level targets are summed; national/district are excluded."""
    with Session(engine) as session:
        # National target (should NOT count)
        nat = _make_stratum(session, "0100000US")
        nat.targets_rel = [
            Target(
                variable="income_tax",
                period=2022,
                value=999e9,
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
                active=True,
            )
        ]

        session.commit()

        total, count = _compute_state_aggregate(session, "income_tax")
        assert count == 2
        assert total == pytest.approx(200e9)


def test_scale_targets(engine):
    """All targets for a variable are scaled and period updated."""
    with Session(engine) as session:
        nat = _make_stratum(session, "0100000US")
        nat.targets_rel = [
            Target(
                variable="eitc",
                period=2022,
                value=100e9,
                active=True,
            )
        ]
        st = _make_stratum(session, "0400000US01")
        st.targets_rel = [
            Target(
                variable="eitc",
                period=2022,
                value=40e9,
                active=True,
            )
        ]
        session.commit()

        n = _scale_targets(session, "eitc", 1.5, 2024)
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


def test_reconcile_targets_scales_correctly(engine):
    """End-to-end: reconciliation scales DB to match parameters."""
    auth = _get_authoritative_targets(TARGET_YEAR)
    income_tax_target = auth["income_tax"]

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
                    active=True,
                )
            ]
        session.commit()

        factors = reconcile_targets(session, TARGET_YEAR)

        assert "income_tax" in factors
        # After reconciliation state sum should match the target
        new_total, _ = _compute_state_aggregate(session, "income_tax")
        assert new_total == pytest.approx(income_tax_target, rel=1e-6)
