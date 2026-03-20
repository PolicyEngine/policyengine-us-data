"""Contract tests for aggregate-record disaggregation."""

import numpy as np
import pandas as pd
import pytest


AGGREGATE_RECIDS = [999996, 999997, 999998, 999999]


def _make_regular_rows() -> list[dict]:
    rng = np.random.default_rng(123)
    rows = []
    next_recid = 1

    bucket_specs = [
        (999996, -18_000_000, -400_000, 14),
        (999997, 150_000, 9_500_000, 18),
        (999998, 12_000_000, 85_000_000, 16),
        (999999, 120_000_000, 480_000_000, 14),
    ]

    for bucket_recid, low, high, count in bucket_specs:
        for i in range(count):
            agi = rng.uniform(low, high)
            mars = rng.choice([1, 2, 3, 4], p=[0.18, 0.72, 0.05, 0.05])
            xtot_lower = 2 if mars == 2 else 0
            xtot = int(rng.integers(xtot_lower, 6))
            dsi = int(rng.binomial(1, 0.03))
            eic = int(rng.integers(0, 4))

            # Every fourth donor is intentionally more tail-heavy so the
            # selector has a visible extremeness margin to learn from.
            tail_boost = rng.uniform(2.0, 5.0) if i % 4 == 0 else 1.0

            abs_agi = abs(agi)
            wages = max(0.0, abs_agi * rng.uniform(0.02, 0.18) * tail_boost)
            ltcg = abs_agi * rng.uniform(0.05, 0.55) * tail_boost
            stcg = abs_agi * rng.uniform(0.0, 0.12) * tail_boost
            if bucket_recid == 999996:
                stcg *= -1
            qdiv = abs_agi * rng.uniform(0.01, 0.10) * tail_boost
            interest = abs_agi * rng.uniform(0.01, 0.08) * tail_boost
            passthrough = abs_agi * rng.uniform(0.02, 0.18) * tail_boost
            business = abs_agi * rng.uniform(0.0, 0.10) * tail_boost
            farm = abs_agi * rng.uniform(0.0, 0.03) * tail_boost

            if bucket_recid == 999996:
                ltcg *= -1 if i % 2 == 0 else 1
                passthrough *= -1 if i % 3 == 0 else 1
                business *= -1 if i % 2 == 0 else 1
                farm *= -1 if i % 2 == 0 else 1

            ordinary_dividends = qdiv + abs_agi * rng.uniform(0.0, 0.04)
            tax_exempt_interest = abs_agi * rng.uniform(0.0, 0.03) * tail_boost

            rows.append(
                {
                    "RECID": next_recid,
                    "S006": int(rng.integers(300, 1800)),
                    "AGIR1": 19 if bucket_recid in (999998, 999999) else 11,
                    "F6251": int(rng.binomial(1, 0.2)),
                    "MARS": int(mars),
                    "XTOT": int(xtot),
                    "DSI": int(dsi),
                    "EIC": int(eic),
                    "E00100": float(agi),
                    "E00200": float(wages),
                    "P23250": float(ltcg),
                    "P22250": float(stcg),
                    "E00650": float(qdiv),
                    "E00300": float(interest),
                    "E26270": float(passthrough),
                    "E00900": float(business),
                    "E02100": float(farm),
                    "E00400": float(tax_exempt_interest),
                    "E00600": float(ordinary_dividends),
                    "E18400": float(abs_agi * rng.uniform(0.0, 0.02)),
                    "E19800": float(abs_agi * rng.uniform(0.0, 0.04)),
                    "T27800": float(farm),
                }
            )
            next_recid += 1

    return rows


def _make_aggregate_rows() -> list[dict]:
    return [
        {
            "RECID": 999996,
            "S006": 179 * 100,
            "AGIR1": 0,
            "F6251": 0,
            "MARS": 0,
            "XTOT": 1,
            "DSI": 0,
            "EIC": 0,
            "E00100": -5_000_000,
            "E00200": 100_000,
            "P23250": -3_000_000,
            "P22250": -1_000_000,
            "E00650": 50_000,
            "E00300": 200_000,
            "E26270": -1_500_000,
            "E00900": -200_000,
            "E02100": -20_000,
            "E00400": 150_000,
            "E00600": 120_000,
            "E18400": 100_000,
            "E19800": 500_000,
            "T27800": -20_000,
        },
        {
            "RECID": 999997,
            "S006": 324 * 100,
            "AGIR1": 0,
            "F6251": 0,
            "MARS": 0,
            "XTOT": 1,
            "DSI": 0,
            "EIC": 0,
            "E00100": 5_000_000,
            "E00200": 1_000_000,
            "P23250": 2_000_000,
            "P22250": 500_000,
            "E00650": 400_000,
            "E00300": 300_000,
            "E26270": 800_000,
            "E00900": 100_000,
            "E02100": 10_000,
            "E00400": 200_000,
            "E00600": 600_000,
            "E18400": 200_000,
            "E19800": 300_000,
            "T27800": 10_000,
        },
        {
            "RECID": 999998,
            "S006": 448 * 100,
            "AGIR1": 0,
            "F6251": 0,
            "MARS": 0,
            "XTOT": 1,
            "DSI": 0,
            "EIC": 0,
            "E00100": 30_000_000,
            "E00200": 3_000_000,
            "P23250": 15_000_000,
            "P22250": 2_000_000,
            "E00650": 3_000_000,
            "E00300": 1_500_000,
            "E26270": 5_000_000,
            "E00900": 200_000,
            "E02100": 50_000,
            "E00400": 700_000,
            "E00600": 4_000_000,
            "E18400": 800_000,
            "E19800": 2_000_000,
            "T27800": 50_000,
        },
        {
            "RECID": 999999,
            "S006": 349 * 100,
            "AGIR1": 0,
            "F6251": 0,
            "MARS": 0,
            "XTOT": 1,
            "DSI": 0,
            "EIC": 0,
            "E00100": 300_000_000,
            "E00200": 10_000_000,
            "P23250": 200_000_000,
            "P22250": 20_000_000,
            "E00650": 30_000_000,
            "E00300": 5_000_000,
            "E26270": 40_000_000,
            "E00900": 1_000_000,
            "E02100": 500_000,
            "E00400": 3_000_000,
            "E00600": 40_000_000,
            "E18400": 3_000_000,
            "E19800": 15_000_000,
            "T27800": 500_000,
        },
    ]


def _make_mini_puf() -> pd.DataFrame:
    return pd.DataFrame(_make_regular_rows() + _make_aggregate_rows())


def _bucket_mask(df: pd.DataFrame, recid: int) -> pd.Series:
    if recid == 999996:
        return df.E00100 < 0
    if recid == 999997:
        return (df.E00100 >= 0) & (df.E00100 < 10_000_000)
    if recid == 999998:
        return (df.E00100 >= 10_000_000) & (df.E00100 < 100_000_000)
    if recid == 999999:
        return df.E00100 >= 100_000_000
    raise ValueError(recid)


def _weighted_total(df: pd.DataFrame, col: str) -> float:
    return float((df[col] * df.S006 / 100).sum())


def _synthetic_bucket(
    result: pd.DataFrame, mini_puf: pd.DataFrame, recid: int
) -> pd.DataFrame:
    from policyengine_us_data.datasets.puf.disaggregate_puf import (
        AGGREGATE_RECIDS,
        SYNTHETIC_RECID_START,
        _choose_n_synthetic,
    )

    offset = 0
    for bucket_recid in AGGREGATE_RECIDS:
        pop_weight = mini_puf.loc[mini_puf.RECID == bucket_recid, "S006"].iloc[0] / 100
        n_syn = _choose_n_synthetic(pop_weight)
        start = SYNTHETIC_RECID_START + offset
        end = start + n_syn
        if bucket_recid == recid:
            return result[(result.RECID >= start) & (result.RECID < end)]
        offset += n_syn

    raise AssertionError(f"Unknown aggregate recid {recid}")


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
        assert not result.RECID.isin(AGGREGATE_RECIDS).any()

    def test_regular_records_preserved(self, mini_puf, result):
        regular = set(mini_puf[~mini_puf.RECID.isin(AGGREGATE_RECIDS)].RECID)
        kept = set(result[result.RECID < 999996].RECID)
        assert regular == kept

    def test_synthetic_count_reasonable(self, result):
        synthetic = result[result.RECID >= 1_000_000]
        assert 80 <= len(synthetic) <= 160

    def test_columns_match(self, mini_puf, result):
        assert list(result.columns) == list(mini_puf.columns)


class TestStructure:
    def test_structural_fields_stay_valid(self, result):
        synthetic = result[result.RECID >= 1_000_000]

        assert synthetic.MARS.isin([1, 2, 3, 4]).all()
        assert synthetic.XTOT.isin([0, 1, 2, 3, 4, 5]).all()
        assert synthetic.DSI.isin([0, 1]).all()
        assert synthetic.EIC.isin([0, 1, 2, 3]).all()
        assert np.allclose(synthetic.AGIR1, np.round(synthetic.AGIR1))
        assert np.allclose(synthetic.F6251, np.round(synthetic.F6251))

    def test_joint_returns_imply_at_least_two_people(self, result):
        synthetic = result[result.RECID >= 1_000_000]
        joint = synthetic[synthetic.MARS == 2]
        assert (joint.XTOT >= 2).all()


class TestWeights:
    def test_weights_are_variable(self, result):
        synthetic = result[result.RECID >= 1_000_000]
        assert synthetic.S006.nunique() > 1

    def test_weights_sum_per_bucket(self, mini_puf, result):
        for recid in AGGREGATE_RECIDS:
            expected = mini_puf.loc[mini_puf.RECID == recid, "S006"].iloc[0] / 100
            bucket = _synthetic_bucket(result, mini_puf, recid)
            assert (bucket.S006 / 100).sum() == pytest.approx(expected)


class TestCalibration:
    def test_bucket_agi_bounds(self, mini_puf, result):
        bounds = {
            999996: (None, 0),
            999997: (0, 10_000_000),
            999998: (10_000_000, 100_000_000),
            999999: (100_000_000, 1_250_000_000),
        }

        for recid, (lower, upper) in bounds.items():
            bucket = _synthetic_bucket(result, mini_puf, recid)
            if lower is not None:
                assert (bucket.E00100 >= lower).all()
            if upper is not None:
                assert (bucket.E00100 <= upper).all()

    def test_exact_weighted_totals_for_amount_columns(self, mini_puf, result):
        from policyengine_us_data.datasets.puf.disaggregate_puf import (
            _get_amount_columns,
        )

        amount_columns = _get_amount_columns(mini_puf.columns)

        for recid in AGGREGATE_RECIDS:
            bucket = _synthetic_bucket(result, mini_puf, recid)
            target_row = mini_puf.loc[mini_puf.RECID == recid].iloc[0]
            pop_weight = target_row.S006 / 100

            for column in amount_columns:
                target = pop_weight * target_row[column]
                actual = _weighted_total(bucket, column)
                assert actual == pytest.approx(target, rel=1e-9, abs=1e-6), (
                    f"{recid=} {column=} {actual=} {target=}"
                )


class TestSelection:
    def test_selected_records_are_more_extreme_than_bucket_baseline(
        self, mini_puf, result
    ):
        from policyengine_us_data.datasets.puf.disaggregate_puf import (
            _choose_n_synthetic,
            _sample_bucket_donors,
            compute_aggregate_eligibility_scores,
        )

        rng = np.random.default_rng(0)
        regular = mini_puf[~mini_puf.RECID.isin(AGGREGATE_RECIDS)]
        regular_scores = compute_aggregate_eligibility_scores(regular)

        for recid in AGGREGATE_RECIDS:
            donor_bucket = regular[_bucket_mask(regular, recid)]
            donor_scores = compute_aggregate_eligibility_scores(
                donor_bucket,
                reference_df=regular,
            )
            pop_weight = mini_puf.loc[mini_puf.RECID == recid, "S006"].iloc[0] / 100
            n_syn = _choose_n_synthetic(pop_weight)

            sampled_donors = _sample_bucket_donors(
                donor_bucket=donor_bucket,
                donor_scores=regular_scores,
                target_mean_agi=mini_puf.loc[mini_puf.RECID == recid, "E00100"].iloc[0],
                n_syn=n_syn,
                rng=rng,
            )
            selected_scores = compute_aggregate_eligibility_scores(
                sampled_donors,
                reference_df=regular,
            )

            random_means = []
            sample_size = len(sampled_donors)
            for _ in range(200):
                sampled_indices = rng.choice(
                    donor_bucket.index.to_numpy(),
                    size=sample_size,
                    replace=len(donor_bucket) < sample_size,
                )
                random_scores = donor_scores.loc[sampled_indices]
                random_means.append(random_scores.mean())

            assert selected_scores.mean() > np.mean(random_means)


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
        syn1 = r1[r1.RECID >= 1_000_000].E00100.to_numpy()
        syn2 = r2[r2.RECID >= 1_000_000].E00100.to_numpy()
        assert not np.allclose(syn1, syn2)


class TestEdgeCases:
    def test_no_aggregate_records(self):
        from policyengine_us_data.datasets.puf.disaggregate_puf import (
            disaggregate_aggregate_records,
        )

        puf = _make_mini_puf()
        puf = puf[~puf.RECID.isin(AGGREGATE_RECIDS)]
        result = disaggregate_aggregate_records(puf, seed=42)
        pd.testing.assert_frame_equal(result, puf)
