"""Tests for the two-pass geographic target reconciliation."""

import logging

import pytest
from sqlmodel import Session, SQLModel, create_engine, select

from policyengine_us_data.db.create_database_tables import (
    Stratum,
    StratumConstraint,
    Target,
)
from policyengine_us_data.db.reconcile_targets import reconcile_targets


# ------------------------------------------------------------------ #
#  Fixtures                                                           #
# ------------------------------------------------------------------ #


@pytest.fixture
def engine():
    """Create an in-memory SQLite engine with all tables."""
    eng = create_engine("sqlite:///:memory:")
    SQLModel.metadata.create_all(eng)
    return eng


@pytest.fixture
def session(engine):
    """Provide a session."""
    with Session(engine) as sess:
        yield sess


def _make_geo_hierarchy(session):
    """Build a small national -> 3 states -> CDs hierarchy.

    Returns dict with stratum IDs.
    """
    # National
    national = Stratum(
        parent_stratum_id=None,
        stratum_group_id=1,
        notes="United States",
    )
    national.constraints_rel = []
    session.add(national)
    session.flush()

    # States
    state_ids = {}
    for fips in [1, 2, 3]:
        s = Stratum(
            parent_stratum_id=national.stratum_id,
            stratum_group_id=1,
            notes=f"State {fips}",
        )
        s.constraints_rel = [
            StratumConstraint(
                constraint_variable="state_fips",
                operation="==",
                value=str(fips),
            )
        ]
        session.add(s)
        session.flush()
        state_ids[fips] = s.stratum_id

    # CDs under state 1
    cd_ids = {}
    for cd_geoid in [101, 102]:
        cd = Stratum(
            parent_stratum_id=state_ids[1],
            stratum_group_id=1,
            notes=f"CD {cd_geoid}",
        )
        cd.constraints_rel = [
            StratumConstraint(
                constraint_variable="congressional_district_geoid",
                operation="==",
                value=str(cd_geoid),
            )
        ]
        session.add(cd)
        session.flush()
        cd_ids[cd_geoid] = cd.stratum_id

    session.commit()
    return {
        "national": national.stratum_id,
        "states": state_ids,
        "cds": cd_ids,
    }


def _add_target(session, stratum_id, variable, period, value):
    """Helper to add a target."""
    t = Target(
        stratum_id=stratum_id,
        variable=variable,
        period=period,
        value=value,
        active=True,
    )
    session.add(t)
    session.flush()
    return t


# ------------------------------------------------------------------ #
#  Tests                                                              #
# ------------------------------------------------------------------ #


class TestPassOneStateToNational:
    def test_states_scaled_to_national(self, session):
        """National=100, states=[30,40,50] -> scaled to sum=100."""
        ids = _make_geo_hierarchy(session)

        _add_target(session, ids["national"], "real_estate_taxes", 2022, 100)
        _add_target(session, ids["states"][1], "real_estate_taxes", 2022, 30)
        _add_target(session, ids["states"][2], "real_estate_taxes", 2022, 40)
        _add_target(session, ids["states"][3], "real_estate_taxes", 2022, 50)
        session.commit()

        stats = reconcile_targets(session)

        # Verify state targets scaled
        assert stats["scaled_state"] == 3

        state_targets = session.exec(
            select(Target).where(
                Target.variable == "real_estate_taxes",
                Target.stratum_id.in_(list(ids["states"].values())),
            )
        ).all()
        total = sum(t.value for t in state_targets)
        assert abs(total - 100) < 1e-6

    def test_raw_value_preserved(self, session):
        """raw_value should hold the original source value."""
        ids = _make_geo_hierarchy(session)

        _add_target(session, ids["national"], "real_estate_taxes", 2022, 100)
        _add_target(session, ids["states"][1], "real_estate_taxes", 2022, 30)
        _add_target(session, ids["states"][2], "real_estate_taxes", 2022, 40)
        _add_target(session, ids["states"][3], "real_estate_taxes", 2022, 50)
        session.commit()

        reconcile_targets(session)

        state_targets = session.exec(
            select(Target).where(
                Target.variable == "real_estate_taxes",
                Target.stratum_id.in_(list(ids["states"].values())),
            )
        ).all()
        raw_values = sorted(t.raw_value for t in state_targets)
        assert raw_values == [30, 40, 50]

    def test_proportions_preserved(self, session):
        """Relative proportions across states should be unchanged."""
        ids = _make_geo_hierarchy(session)

        _add_target(session, ids["national"], "real_estate_taxes", 2022, 100)
        _add_target(session, ids["states"][1], "real_estate_taxes", 2022, 30)
        _add_target(session, ids["states"][2], "real_estate_taxes", 2022, 40)
        _add_target(session, ids["states"][3], "real_estate_taxes", 2022, 50)
        session.commit()

        reconcile_targets(session)

        state_targets = session.exec(
            select(Target).where(
                Target.variable == "real_estate_taxes",
                Target.stratum_id.in_(list(ids["states"].values())),
            )
        ).all()
        values = {t.stratum_id: t.value for t in state_targets}
        # 30/40 == 3/4 should be preserved
        ratio = values[ids["states"][1]] / values[ids["states"][2]]
        assert abs(ratio - 30 / 40) < 1e-9


class TestPassTwoCdToState:
    def test_two_pass_reconciliation(self, session):
        """Full two-pass: national -> states -> CDs."""
        ids = _make_geo_hierarchy(session)

        _add_target(session, ids["national"], "income_tax", 2022, 1000)
        # States sum to 500 (not 1000)
        _add_target(session, ids["states"][1], "income_tax", 2022, 200)
        _add_target(session, ids["states"][2], "income_tax", 2022, 150)
        _add_target(session, ids["states"][3], "income_tax", 2022, 150)
        # CDs under state 1 sum to 100 (not 200)
        _add_target(session, ids["cds"][101], "income_tax", 2022, 60)
        _add_target(session, ids["cds"][102], "income_tax", 2022, 40)
        session.commit()

        stats = reconcile_targets(session)

        # States should sum to national (1000)
        state_targets = session.exec(
            select(Target).where(
                Target.variable == "income_tax",
                Target.stratum_id.in_(list(ids["states"].values())),
            )
        ).all()
        state_sum = sum(t.value for t in state_targets)
        assert abs(state_sum - 1000) < 1e-6

        # State 1 should be 200 * (1000/500) = 400
        state1_val = next(
            t.value for t in state_targets if t.stratum_id == ids["states"][1]
        )
        assert abs(state1_val - 400) < 1e-6

        # CDs should sum to corrected state 1 (400)
        cd_targets = session.exec(
            select(Target).where(
                Target.variable == "income_tax",
                Target.stratum_id.in_(list(ids["cds"].values())),
            )
        ).all()
        cd_sum = sum(t.value for t in cd_targets)
        assert abs(cd_sum - 400) < 1e-6

        assert stats["scaled_state"] == 3
        assert stats["scaled_cd"] == 2


class TestNoNationalTarget:
    def test_states_unchanged_without_national(self, session):
        """Without a national target, state values stay the same."""
        ids = _make_geo_hierarchy(session)

        _add_target(session, ids["states"][1], "real_estate_taxes", 2022, 30)
        _add_target(session, ids["states"][2], "real_estate_taxes", 2022, 40)
        session.commit()

        stats = reconcile_targets(session)

        assert stats["scaled_state"] == 0
        state_targets = session.exec(
            select(Target).where(
                Target.variable == "real_estate_taxes",
            )
        ).all()
        values = sorted(t.value for t in state_targets)
        assert values == [30, 40]


class TestZeroChildSum:
    def test_zero_state_sum_skipped(self, session, caplog):
        """Zero state sum should log warning, not divide by zero."""
        ids = _make_geo_hierarchy(session)

        _add_target(session, ids["national"], "income_tax", 2022, 100)
        _add_target(session, ids["states"][1], "income_tax", 2022, 0)
        _add_target(session, ids["states"][2], "income_tax", 2022, 0)
        _add_target(session, ids["states"][3], "income_tax", 2022, 0)
        session.commit()

        with caplog.at_level(logging.WARNING):
            stats = reconcile_targets(session)

        assert stats["skipped_zero_sum"] >= 1
        assert "zero" in caplog.text.lower()


class TestIdempotency:
    def test_running_twice_same_result(self, session):
        """Running reconciliation twice should produce same result."""
        ids = _make_geo_hierarchy(session)

        _add_target(session, ids["national"], "real_estate_taxes", 2022, 100)
        _add_target(session, ids["states"][1], "real_estate_taxes", 2022, 30)
        _add_target(session, ids["states"][2], "real_estate_taxes", 2022, 40)
        _add_target(session, ids["states"][3], "real_estate_taxes", 2022, 50)
        session.commit()

        reconcile_targets(session)

        # Capture values after first run
        first_run = {
            t.target_id: (t.value, t.raw_value)
            for t in session.exec(select(Target)).all()
        }

        reconcile_targets(session)

        # Values should be unchanged
        second_run = {
            t.target_id: (t.value, t.raw_value)
            for t in session.exec(select(Target)).all()
        }

        for tid in first_run:
            v1, r1 = first_run[tid]
            v2, r2 = second_run[tid]
            if v1 is not None:
                assert abs(v1 - v2) < 1e-9
            if r1 is not None:
                assert abs(r1 - r2) < 1e-9


class TestRawValuePreservation:
    def test_re_run_uses_raw_value_as_base(self, session):
        """Re-running should use raw_value, not already-scaled value."""
        ids = _make_geo_hierarchy(session)

        _add_target(session, ids["national"], "salt", 2022, 200)
        _add_target(session, ids["states"][1], "salt", 2022, 50)
        _add_target(session, ids["states"][2], "salt", 2022, 50)
        session.commit()

        # First run: scale factor = 200/100 = 2
        reconcile_targets(session)

        state_targets = session.exec(
            select(Target).where(
                Target.variable == "salt",
                Target.stratum_id.in_(list(ids["states"].values())),
            )
        ).all()
        for t in state_targets:
            assert t.raw_value == 50
            assert abs(t.value - 100) < 1e-6

        # Second run: same raw_value * same scale = same result
        reconcile_targets(session)

        state_targets = session.exec(
            select(Target).where(
                Target.variable == "salt",
                Target.stratum_id.in_(list(ids["states"].values())),
            )
        ).all()
        for t in state_targets:
            assert t.raw_value == 50  # raw_value unchanged
            assert abs(t.value - 100) < 1e-6  # not compounded


class TestNonGeographicSubStrata:
    def test_filer_substrata_grouped_by_geo_ancestor(self, session):
        """Targets on non-geo strata resolve to their geo ancestor."""
        ids = _make_geo_hierarchy(session)

        # Create filer sub-strata under geo strata (group_id=2)
        nat_filer = Stratum(
            parent_stratum_id=ids["national"],
            stratum_group_id=2,
            notes="National filers",
        )
        nat_filer.constraints_rel = [
            StratumConstraint(
                constraint_variable="tax_unit_is_filer",
                operation="==",
                value="1",
            )
        ]
        session.add(nat_filer)
        session.flush()

        state_filers = {}
        for fips in [1, 2, 3]:
            sf = Stratum(
                parent_stratum_id=ids["states"][fips],
                stratum_group_id=2,
                notes=f"State {fips} filers",
            )
            sf.constraints_rel = [
                StratumConstraint(
                    constraint_variable="tax_unit_is_filer",
                    operation="==",
                    value="1",
                ),
                StratumConstraint(
                    constraint_variable="state_fips",
                    operation="==",
                    value=str(fips),
                ),
            ]
            session.add(sf)
            session.flush()
            state_filers[fips] = sf.stratum_id

        session.commit()

        # Add targets on the filer strata
        _add_target(session, nat_filer.stratum_id, "income_tax", 2022, 500)
        _add_target(session, state_filers[1], "income_tax", 2022, 100)
        _add_target(session, state_filers[2], "income_tax", 2022, 200)
        _add_target(session, state_filers[3], "income_tax", 2022, 300)
        session.commit()

        stats = reconcile_targets(session)

        # States sum was 600, national is 500 -> scale = 5/6
        assert stats["scaled_state"] == 3

        state_targets = session.exec(
            select(Target).where(
                Target.variable == "income_tax",
                Target.stratum_id.in_(list(state_filers.values())),
            )
        ).all()
        total = sum(t.value for t in state_targets)
        assert abs(total - 500) < 1e-6


class TestAlreadyReconciled:
    def test_no_scaling_when_already_matching(self, session):
        """When states already sum to national, no scaling occurs."""
        ids = _make_geo_hierarchy(session)

        _add_target(session, ids["national"], "income_tax", 2022, 120)
        _add_target(session, ids["states"][1], "income_tax", 2022, 30)
        _add_target(session, ids["states"][2], "income_tax", 2022, 40)
        _add_target(session, ids["states"][3], "income_tax", 2022, 50)
        session.commit()

        stats = reconcile_targets(session)

        # Scale is 1.0, so no targets are actually modified
        assert stats["scaled_state"] == 0

        state_targets = session.exec(
            select(Target).where(
                Target.variable == "income_tax",
                Target.stratum_id.in_(list(ids["states"].values())),
            )
        ).all()
        # Values should be unchanged
        values = sorted(t.value for t in state_targets)
        assert values == [30, 40, 50]
        # raw_value should still be None (no scaling happened)
        for t in state_targets:
            assert t.raw_value is None
