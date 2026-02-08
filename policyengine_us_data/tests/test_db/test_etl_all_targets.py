"""Tests for the comprehensive ETL that migrates all legacy loss.py
targets into the database."""

import numpy as np
import pandas as pd
import pytest
from sqlmodel import Session, SQLModel, create_engine, select

from policyengine_us_data.db.create_database_tables import (
    Source,
    SourceType,
    Stratum,
    StratumConstraint,
    Target,
)
from policyengine_us_data.db.etl_all_targets import (
    extract_census_age_populations,
    extract_eitc_by_child_count,
    extract_healthcare_by_age,
    extract_soi_filer_counts,
    extract_spm_threshold_agi,
    extract_negative_market_income,
    extract_infant_count,
    extract_net_worth,
    extract_state_population,
    extract_tax_expenditure_targets,
    extract_state_real_estate_taxes,
    extract_state_aca,
    extract_state_medicaid_enrollment,
    extract_state_10yr_age,
    extract_state_agi,
    extract_soi_filing_status_targets,
    load_all_targets,
)
from policyengine_us_data.storage import CALIBRATION_FOLDER


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
    """Provide a session that auto-rolls back."""
    with Session(engine) as sess:
        yield sess


@pytest.fixture
def root_stratum(session):
    """Create the root (national) stratum required by the ETL."""
    root = Stratum(
        definition_hash="root_national",
        parent_stratum_id=None,
        stratum_group_id=1,
        notes="United States",
    )
    session.add(root)
    session.commit()
    session.refresh(root)
    return root


# ------------------------------------------------------------------ #
#  Extract-layer tests                                                #
# ------------------------------------------------------------------ #


class TestExtractCensusAgePopulations:
    def test_returns_86_bins(self):
        records = extract_census_age_populations(time_period=2024)
        assert len(records) == 86

    def test_values_are_positive(self):
        records = extract_census_age_populations(time_period=2024)
        for r in records:
            assert r["value"] > 0
            assert r["age"] == r["age"]  # not NaN

    def test_first_bin_is_age_0(self):
        records = extract_census_age_populations(time_period=2024)
        assert records[0]["age"] == 0


class TestExtractEitcByChildCount:
    def test_returns_4_child_buckets(self):
        records = extract_eitc_by_child_count()
        child_counts = {r["count_children"] for r in records}
        assert child_counts == {0, 1, 2, 3}

    def test_has_returns_and_total(self):
        records = extract_eitc_by_child_count()
        for r in records:
            assert "eitc_returns" in r
            assert "eitc_total" in r
            assert r["eitc_returns"] > 0
            assert r["eitc_total"] > 0


class TestExtractSoiFilerCounts:
    def test_returns_7_bands(self):
        records = extract_soi_filer_counts()
        assert len(records) == 7

    def test_bands_cover_full_range(self):
        records = extract_soi_filer_counts()
        lowers = [r["agi_lower"] for r in records]
        uppers = [r["agi_upper"] for r in records]
        assert -np.inf in lowers
        assert np.inf in uppers


class TestExtractHealthcareByAge:
    def test_returns_9_age_bands(self):
        records = extract_healthcare_by_age()
        assert len(records) == 9

    def test_has_4_expense_types(self):
        records = extract_healthcare_by_age()
        for r in records:
            assert (
                "health_insurance_premiums_without_medicare_part_b"
                in r["expenses"]
            )
            assert "over_the_counter_health_expenses" in r["expenses"]
            assert "other_medical_expenses" in r["expenses"]
            assert "medicare_part_b_premiums" in r["expenses"]


class TestExtractSpmThresholdAgi:
    def test_returns_10_deciles(self):
        records = extract_spm_threshold_agi()
        assert len(records) == 10

    def test_has_agi_and_count(self):
        records = extract_spm_threshold_agi()
        for r in records:
            assert "adjusted_gross_income" in r
            assert "count" in r
            assert r["decile"] >= 1


class TestExtractNegativeMarketIncome:
    def test_has_total_and_count(self):
        result = extract_negative_market_income()
        assert result["total"] == -138e9
        assert result["count"] == 3e6


class TestExtractInfantCount:
    def test_returns_positive(self):
        result = extract_infant_count()
        assert result > 3e6


class TestExtractNetWorth:
    def test_returns_160_trillion(self):
        result = extract_net_worth()
        assert result == 160e12


class TestExtractStatePopulation:
    def test_returns_51_or_52_rows(self):
        records = extract_state_population()
        assert len(records) in (51, 52)  # 50 states + DC (+ PR)

    def test_has_under_5(self):
        records = extract_state_population()
        for r in records:
            assert "population_under_5" in r


class TestExtractTaxExpenditureTargets:
    def test_returns_5_deductions(self):
        records = extract_tax_expenditure_targets()
        assert len(records) == 5
        names = {r["variable"] for r in records}
        assert "salt_deduction" in names
        assert "qualified_business_income_deduction" in names


class TestExtractStateRealEstateTaxes:
    def test_returns_51_states(self):
        records = extract_state_real_estate_taxes()
        assert len(records) == 51

    def test_sums_to_national_target(self):
        records = extract_state_real_estate_taxes()
        total = sum(r["value"] for r in records)
        assert abs(total - 500e9) < 1e6  # within $1M


class TestExtractStateAca:
    def test_returns_spending_and_enrollment(self):
        records = extract_state_aca()
        assert len(records) > 0
        first = records[0]
        assert "spending" in first
        assert "enrollment" in first


class TestExtractStateMedicaidEnrollment:
    def test_returns_51_rows(self):
        records = extract_state_medicaid_enrollment()
        assert len(records) == 51


class TestExtractState10yrAge:
    def test_returns_50_states(self):
        records = extract_state_10yr_age()
        states = {r["state"] for r in records}
        assert len(states) == 50

    def test_has_18_age_ranges(self):
        records = extract_state_10yr_age()
        first_state = records[0]["state"]
        state_records = [r for r in records if r["state"] == first_state]
        assert len(state_records) == 18


class TestExtractStateAgi:
    def test_returns_918_rows(self):
        records = extract_state_agi()
        assert len(records) == 918


class TestExtractSoiFilingStatusTargets:
    def test_returns_filtered_rows(self):
        records = extract_soi_filing_status_targets()
        # Only PE-valid variables are kept after filtering
        assert len(records) == 532
        for r in records:
            assert r["taxable_only"] is True
            assert r["agi_upper"] > 10_000


# ------------------------------------------------------------------ #
#  Load-layer integration test                                        #
# ------------------------------------------------------------------ #


class TestLoadAllTargets:
    def test_load_creates_targets(self, engine, root_stratum):
        """Run the full load and verify target counts."""
        load_all_targets(
            engine=engine,
            time_period=2024,
            root_stratum_id=root_stratum.stratum_id,
        )

        with Session(engine) as sess:
            total_targets = sess.exec(select(Target)).all()
            total_strata = sess.exec(select(Stratum)).unique().all()
            total_sources = sess.exec(select(Source)).all()

            # Must have created at least 1 source
            assert len(total_sources) >= 1

            # Must have created many strata beyond the root
            assert len(total_strata) > 10

            # Must have created many targets
            assert len(total_targets) > 100

    def test_census_age_targets_present(self, engine, root_stratum):
        load_all_targets(
            engine=engine,
            time_period=2024,
            root_stratum_id=root_stratum.stratum_id,
        )
        with Session(engine) as sess:
            age_targets = sess.exec(
                select(Target).where(
                    Target.variable == "person_count",
                    Target.notes.contains("Census age"),
                )
            ).all()
            assert len(age_targets) == 86

    def test_eitc_targets_present(self, engine, root_stratum):
        load_all_targets(
            engine=engine,
            time_period=2024,
            root_stratum_id=root_stratum.stratum_id,
        )
        with Session(engine) as sess:
            # 4 child counts x 2 (returns + spending) = 8
            eitc_targets = sess.exec(
                select(Target).where(
                    Target.notes.contains("EITC"),
                )
            ).all()
            assert len(eitc_targets) == 8

    def test_soi_filer_count_targets(self, engine, root_stratum):
        load_all_targets(
            engine=engine,
            time_period=2024,
            root_stratum_id=root_stratum.stratum_id,
        )
        with Session(engine) as sess:
            filer_targets = sess.exec(
                select(Target).where(
                    Target.notes.contains("SOI filer count"),
                )
            ).all()
            assert len(filer_targets) == 7

    def test_healthcare_targets(self, engine, root_stratum):
        load_all_targets(
            engine=engine,
            time_period=2024,
            root_stratum_id=root_stratum.stratum_id,
        )
        with Session(engine) as sess:
            hc_targets = sess.exec(
                select(Target).where(
                    Target.notes.contains("Healthcare"),
                )
            ).all()
            # 9 age bands x 4 expense types = 36
            assert len(hc_targets) == 36

    def test_spm_targets(self, engine, root_stratum):
        load_all_targets(
            engine=engine,
            time_period=2024,
            root_stratum_id=root_stratum.stratum_id,
        )
        with Session(engine) as sess:
            spm_targets = sess.exec(
                select(Target).where(
                    Target.notes.contains("SPM threshold"),
                )
            ).all()
            # 10 deciles x 2 (agi + count) = 20
            assert len(spm_targets) == 20

    def test_negative_market_income_targets(self, engine, root_stratum):
        load_all_targets(
            engine=engine,
            time_period=2024,
            root_stratum_id=root_stratum.stratum_id,
        )
        with Session(engine) as sess:
            nmi_targets = sess.exec(
                select(Target).where(
                    Target.notes.contains("Negative household market"),
                )
            ).all()
            assert len(nmi_targets) == 2

    def test_state_real_estate_tax_targets(self, engine, root_stratum):
        load_all_targets(
            engine=engine,
            time_period=2024,
            root_stratum_id=root_stratum.stratum_id,
        )
        with Session(engine) as sess:
            ret_targets = sess.exec(
                select(Target).where(
                    Target.notes.contains("State real estate"),
                )
            ).all()
            assert len(ret_targets) == 51

    def test_idempotent(self, engine, root_stratum):
        """Running twice should not duplicate targets."""
        load_all_targets(
            engine=engine,
            time_period=2024,
            root_stratum_id=root_stratum.stratum_id,
        )
        with Session(engine) as sess:
            count_1 = len(sess.exec(select(Target)).all())

        load_all_targets(
            engine=engine,
            time_period=2024,
            root_stratum_id=root_stratum.stratum_id,
        )
        with Session(engine) as sess:
            count_2 = len(sess.exec(select(Target)).all())

        assert count_1 == count_2

    def test_soi_filing_status_targets(self, engine, root_stratum):
        load_all_targets(
            engine=engine,
            time_period=2024,
            root_stratum_id=root_stratum.stratum_id,
        )
        with Session(engine) as sess:
            soi_targets = sess.exec(
                select(Target).where(
                    Target.notes.contains("SOI filing-status"),
                )
            ).all()
            # Only PE-valid variables are loaded
            # PE-valid vars, deduplicated by (variable, stratum)
            assert len(soi_targets) == 289

    def test_state_aca_targets(self, engine, root_stratum):
        load_all_targets(
            engine=engine,
            time_period=2024,
            root_stratum_id=root_stratum.stratum_id,
        )
        with Session(engine) as sess:
            # 51 states x 2 (spending + enrollment) = 102
            # but actually stored as separate targets
            spending = sess.exec(
                select(Target).where(
                    Target.notes.contains("ACA spending"),
                )
            ).all()
            enrollment = sess.exec(
                select(Target).where(
                    Target.notes.contains("ACA enrollment"),
                )
            ).all()
            assert len(spending) >= 50
            assert len(enrollment) >= 50

    def test_state_medicaid_targets(self, engine, root_stratum):
        load_all_targets(
            engine=engine,
            time_period=2024,
            root_stratum_id=root_stratum.stratum_id,
        )
        with Session(engine) as sess:
            med = sess.exec(
                select(Target).where(
                    Target.notes.contains("State Medicaid"),
                )
            ).all()
            assert len(med) == 51

    def test_state_age_targets(self, engine, root_stratum):
        load_all_targets(
            engine=engine,
            time_period=2024,
            root_stratum_id=root_stratum.stratum_id,
        )
        with Session(engine) as sess:
            age = sess.exec(
                select(Target).where(
                    Target.notes.contains("State 10yr age"),
                )
            ).all()
            # 50 states x 18 age ranges = 900
            assert len(age) == 900

    def test_state_agi_targets(self, engine, root_stratum):
        load_all_targets(
            engine=engine,
            time_period=2024,
            root_stratum_id=root_stratum.stratum_id,
        )
        with Session(engine) as sess:
            agi = sess.exec(
                select(Target).where(
                    Target.notes.contains("State AGI"),
                )
            ).all()
            assert len(agi) == 918
