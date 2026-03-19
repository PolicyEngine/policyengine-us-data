"""Tests for disaggregate_puf module.

Uses a synthetic mini-PUF to validate the disaggregation logic
without requiring private IRS data.
"""

import numpy as np
import pandas as pd
import pytest


def _make_mini_puf() -> pd.DataFrame:
    """Create a minimal PUF with 4 aggregate + 20 regular records."""
    rng = np.random.default_rng(123)

    regular_rows = []
    for i in range(20):
        agi = rng.uniform(2e6, 50e6)
        regular_rows.append(
            {
                "RECID": i + 1,
                "S006": rng.uniform(50, 500),
                "MARS": rng.choice([1, 2, 3, 4]),
                "E00100": agi,
                "E00200": agi * rng.uniform(0, 0.6),
                "P23250": agi * rng.uniform(0, 0.5),
                "P22250": agi * rng.uniform(-0.1, 0.2),
                "E00650": agi * rng.uniform(0, 0.15),
                "E00300": agi * rng.uniform(0, 0.1),
                "E26270": agi * rng.uniform(0, 0.4),
                "E00900": agi * rng.uniform(-0.05, 0.1),
                "E02100": agi * rng.uniform(-0.02, 0.02),
                "E00600": agi * rng.uniform(0, 0.2),
                "E18400": agi * rng.uniform(0, 0.05),
                "E19800": agi * rng.uniform(0, 0.1),
            }
        )

    aggregate_rows = [
        {
            "RECID": 999996,
            "S006": 179 * 100,
            "MARS": 0,
            "E00100": -5_000_000,
            "E00200": 100_000,
            "P23250": -3_000_000,
            "P22250": -1_000_000,
            "E00650": 50_000,
            "E00300": 200_000,
            "E26270": -1_500_000,
            "E00900": 50_000,
            "E02100": -20_000,
            "E00600": 80_000,
            "E18400": 100_000,
            "E19800": 500_000,
        },
        {
            "RECID": 999997,
            "S006": 324 * 100,
            "MARS": 0,
            "E00100": 5_000_000,
            "E00200": 1_000_000,
            "P23250": 2_000_000,
            "P22250": 500_000,
            "E00650": 400_000,
            "E00300": 300_000,
            "E26270": 800_000,
            "E00900": 100_000,
            "E02100": 10_000,
            "E00600": 600_000,
            "E18400": 200_000,
            "E19800": 300_000,
        },
        {
            "RECID": 999998,
            "S006": 448 * 100,
            "MARS": 0,
            "E00100": 30_000_000,
            "E00200": 3_000_000,
            "P23250": 15_000_000,
            "P22250": 2_000_000,
            "E00650": 3_000_000,
            "E00300": 1_500_000,
            "E26270": 5_000_000,
            "E00900": 200_000,
            "E02100": 50_000,
            "E00600": 4_000_000,
            "E18400": 800_000,
            "E19800": 2_000_000,
        },
        {
            "RECID": 999999,
            "S006": 349 * 100,
            "MARS": 0,
            "E00100": 300_000_000,
            "E00200": 10_000_000,
            "P23250": 200_000_000,
            "P22250": 20_000_000,
            "E00650": 30_000_000,
            "E00300": 5_000_000,
            "E26270": 40_000_000,
            "E00900": 1_000_000,
            "E02100": 500_000,
            "E00600": 40_000_000,
            "E18400": 3_000_000,
            "E19800": 15_000_000,
        },
    ]

    return pd.DataFrame(regular_rows + aggregate_rows)


@pytest.fixture
def mini_puf():
    return _make_mini_puf()


@pytest.fixture
def result(mini_puf):
    from policyengine_us_data.datasets.puf.disaggregate_puf import (
        disaggregate_aggregate_records,
    )

    return disaggregate_aggregate_records(mini_puf, seed=42)


class TestBasics:
    def test_aggregate_records_removed(self, result):
        agg = result.RECID.isin([999996, 999997, 999998, 999999])
        assert agg.sum() == 0

    def test_regular_records_preserved(self, mini_puf, result):
        regular = set(mini_puf[mini_puf.MARS != 0].RECID)
        kept = set(result[result.RECID < 999996].RECID)
        assert regular == kept

    def test_synthetic_count_reasonable(self, result):
        """Should produce ~120 records (20-40 per bucket), not 1,214."""
        syn = result[result.RECID >= 1_000_000]
        assert 80 <= len(syn) <= 160

    def test_synthetic_recids_unique(self, result):
        syn = result[result.RECID >= 1_000_000]
        assert syn.RECID.nunique() == len(syn)

    def test_synthetic_mars_valid(self, result):
        syn = result[result.RECID >= 1_000_000]
        assert syn.MARS.isin([1, 2, 3, 4]).all()

    def test_synthetic_mars_mostly_joint(self, result):
        syn = result[result.RECID >= 1_000_000]
        assert (syn.MARS == 2).mean() > 0.5

    def test_columns_match(self, mini_puf, result):
        assert set(result.columns) == set(mini_puf.columns)


class TestWeights:
    def test_weights_are_variable(self, result):
        """Weights should NOT all be 100 (weight=1). They vary."""
        syn = result[result.RECID >= 1_000_000]
        assert syn.S006.nunique() > 1

    def test_weights_minimum(self, result):
        """All weights should be >= 300 (i.e. >= 3 after /100)."""
        syn = result[result.RECID >= 1_000_000]
        assert (syn.S006 >= 300).all()

    def test_weights_sum_per_bucket(self, mini_puf, result):
        """Weights should sum to original S006/100 per bucket."""
        from policyengine_us_data.datasets.puf.disaggregate_puf import (
            _choose_n_synthetic,
            SYNTHETIC_RECID_START,
        )

        offset = 0
        for recid in [999996, 999997, 999998, 999999]:
            orig = mini_puf[mini_puf.RECID == recid].iloc[0]
            expected_weight = orig.S006 / 100
            n_syn = _choose_n_synthetic(expected_weight)

            start = SYNTHETIC_RECID_START + offset
            end = start + n_syn
            offset += n_syn

            bucket = result[(result.RECID >= start) & (result.RECID < end)]
            actual_weight = (bucket.S006 / 100).sum()
            # Allow small rounding error from integer weights
            assert abs(actual_weight - expected_weight) <= n_syn


class TestBucketBounds:
    def test_negative_agi_bucket(self, result):
        """Negative AGI bucket records should have AGI <= 0."""
        from policyengine_us_data.datasets.puf.disaggregate_puf import (
            _choose_n_synthetic,
            SYNTHETIC_RECID_START,
        )

        n = _choose_n_synthetic(179)
        neg = result[
            (result.RECID >= SYNTHETIC_RECID_START)
            & (result.RECID < SYNTHETIC_RECID_START + n)
        ]
        assert (neg.E00100 <= 0).all()

    def test_under_10m_bucket_bounded(self, mini_puf, result):
        """<$10M bucket should have AGI within [0, $10M]."""
        from policyengine_us_data.datasets.puf.disaggregate_puf import (
            _choose_n_synthetic,
            SYNTHETIC_RECID_START,
        )

        n_neg = _choose_n_synthetic(179)
        n_10m = _choose_n_synthetic(324)
        start = SYNTHETIC_RECID_START + n_neg
        bucket = result[(result.RECID >= start) & (result.RECID < start + n_10m)]
        assert (bucket.E00100 >= 0).all()
        assert (bucket.E00100 <= 10_000_000).all()

    def test_100m_plus_capped(self, result):
        """$100M+ bucket should have AGI <= $1.25B."""
        syn = result[result.RECID >= 1_000_000]
        # The last bucket has the highest AGIs
        assert (syn.E00100 <= 1_250_000_000).all()

    def test_no_record_dominates(self, mini_puf, result):
        """No single record should carry > 25% of bucket AGI."""
        from policyengine_us_data.datasets.puf.disaggregate_puf import (
            _choose_n_synthetic,
            SYNTHETIC_RECID_START,
        )

        offset = 0
        for recid in [999996, 999997, 999998, 999999]:
            orig = mini_puf[mini_puf.RECID == recid].iloc[0]
            pop_weight = orig.S006 / 100
            total = pop_weight * orig.E00100
            n_syn = _choose_n_synthetic(pop_weight)

            start = SYNTHETIC_RECID_START + offset
            end = start + n_syn
            offset += n_syn

            bucket = result[(result.RECID >= start) & (result.RECID < end)]
            if abs(total) > 0:
                weighted = (bucket.E00100 * bucket.S006 / 100).abs()
                # Allow 25% (slightly more than the 20% target
                # due to rescaling)
                assert (weighted <= 0.25 * abs(total)).all(), (
                    f"RECID {recid}: record dominance exceeded 25%"
                )


class TestCalibration:
    def _weighted_total(self, df, col):
        return (df[col] * df.S006 / 100).sum()

    def test_agi_total_preserved(self, mini_puf, result):
        """Weighted AGI total should match per bucket."""
        from policyengine_us_data.datasets.puf.disaggregate_puf import (
            _choose_n_synthetic,
            SYNTHETIC_RECID_START,
        )

        offset = 0
        for recid in [999996, 999997, 999998, 999999]:
            orig = mini_puf[mini_puf.RECID == recid].iloc[0]
            pop_weight = orig.S006 / 100
            target = pop_weight * orig.E00100
            n_syn = _choose_n_synthetic(pop_weight)

            start = SYNTHETIC_RECID_START + offset
            end = start + n_syn
            offset += n_syn

            bucket = result[(result.RECID >= start) & (result.RECID < end)]
            actual = self._weighted_total(bucket, "E00100")
            if abs(target) > 1:
                rel_err = abs(actual - target) / abs(target)
                assert rel_err < 0.05, (
                    f"RECID {recid} AGI: target={target:.0f}, "
                    f"actual={actual:.0f}, err={rel_err:.4f}"
                )


class TestReproducibility:
    def test_same_seed_same_result(self):
        from policyengine_us_data.datasets.puf.disaggregate_puf import (
            disaggregate_aggregate_records,
        )

        puf = _make_mini_puf()
        r1 = disaggregate_aggregate_records(puf.copy(), seed=42)
        r2 = disaggregate_aggregate_records(puf.copy(), seed=42)
        pd.testing.assert_frame_equal(r1, r2)

    def test_different_seed_different_result(self):
        from policyengine_us_data.datasets.puf.disaggregate_puf import (
            disaggregate_aggregate_records,
        )

        puf = _make_mini_puf()
        r1 = disaggregate_aggregate_records(puf.copy(), seed=42)
        r2 = disaggregate_aggregate_records(puf.copy(), seed=99)
        syn1 = r1[r1.RECID >= 1_000_000].E00100.values
        syn2 = r2[r2.RECID >= 1_000_000].E00100.values
        assert not np.allclose(syn1, syn2)


class TestEdgeCases:
    def test_no_aggregate_records(self):
        from policyengine_us_data.datasets.puf.disaggregate_puf import (
            disaggregate_aggregate_records,
        )

        puf = _make_mini_puf()
        puf = puf[puf.MARS != 0]
        result = disaggregate_aggregate_records(puf, seed=42)
        pd.testing.assert_frame_equal(result, puf)
