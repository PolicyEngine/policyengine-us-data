"""Contract tests for aggregate-record disaggregation."""

import json

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
            pensions = abs_agi * rng.uniform(0.0, 0.025)
            taxable_pensions = pensions * rng.uniform(0.55, 0.95)
            social_security = abs_agi * rng.uniform(0.0, 0.006)

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
                    "E01500": float(pensions),
                    "E01700": float(taxable_pensions),
                    "E02400": float(social_security),
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
            "E01500": 80_000,
            "E01700": 60_000,
            "E02400": 40_000,
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
            "E01500": 250_000,
            "E01700": 200_000,
            "E02400": 100_000,
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
            "E01500": 1_000_000,
            "E01700": 800_000,
            "E02400": 300_000,
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
            "E01500": 6_000_000,
            "E01700": 5_000_000,
            "E02400": 2_000_000,
            "E18400": 3_000_000,
            "E19800": 15_000_000,
            "T27800": 500_000,
        },
    ]


def _make_mini_puf() -> pd.DataFrame:
    return pd.DataFrame(_make_regular_rows() + _make_aggregate_rows())


def _make_mock_forbes_records(n: int = 400) -> pd.DataFrame:
    industries = [
        "technology",
        "finance-investments",
        "real-estate",
        "diversified",
    ]
    records = []
    for i in range(n):
        records.append(
            {
                "alias": f"mock-{i}",
                "rank": i + 1,
                "snapshot_date": "2026-03-31",
                "name": f"Mock Forbes {i}",
                "age": 45 + (i % 25),
                "birth_date": None,
                "networth_millions": float(40_000 - i * 60),
                "citizenship": "us",
                "industry": industries[i % len(industries)],
                "source": "Mock",
                "residence_country": "us",
                "residence_state": "California",
                "marital_status": "Married" if i % 2 == 0 else "Single",
                "children": i % 3,
                "deceased": False,
                "family": False,
                "self_made": bool(i % 3),
                "self_made_type": "self-made" if i % 3 else "inherited",
            }
        )
    return pd.DataFrame(records)


def _make_mock_scf_forbes_donors() -> pd.DataFrame:
    rows = [
        {
            "age": 54,
            "is_married": True,
            "wgt": 3.0,
            "net_worth": 800_000_000.0,
            "wageinc": 8_000_000.0,
            "kginc": 150_000_000.0,
            "intdivinc": 20_000_000.0,
            "bussefarminc": 15_000_000.0,
            "ssretinc": 0.0,
            "houses": 40_000_000.0,
            "mrthel": 5_000_000.0,
            "ccbal": 0.0,
            "edn_inst": 0.0,
            "archetype": "ltcg",
        },
        {
            "age": 67,
            "is_married": False,
            "wgt": 2.5,
            "net_worth": 500_000_000.0,
            "wageinc": 2_000_000.0,
            "kginc": 30_000_000.0,
            "intdivinc": 45_000_000.0,
            "bussefarminc": 5_000_000.0,
            "ssretinc": 500_000.0,
            "houses": 20_000_000.0,
            "mrthel": 2_000_000.0,
            "ccbal": 0.0,
            "edn_inst": 0.0,
            "archetype": "dividend",
        },
        {
            "age": 61,
            "is_married": True,
            "wgt": 4.0,
            "net_worth": 650_000_000.0,
            "wageinc": 5_000_000.0,
            "kginc": 40_000_000.0,
            "intdivinc": 15_000_000.0,
            "bussefarminc": 110_000_000.0,
            "ssretinc": 0.0,
            "houses": 180_000_000.0,
            "mrthel": 15_000_000.0,
            "ccbal": 0.0,
            "edn_inst": 0.0,
            "archetype": "partnership",
        },
        {
            "age": 49,
            "is_married": False,
            "wgt": 2.0,
            "net_worth": 350_000_000.0,
            "wageinc": 60_000_000.0,
            "kginc": 12_000_000.0,
            "intdivinc": 5_000_000.0,
            "bussefarminc": 3_000_000.0,
            "ssretinc": 0.0,
            "houses": 12_000_000.0,
            "mrthel": 1_000_000.0,
            "ccbal": 0.0,
            "edn_inst": 0.0,
            "archetype": "mixed",
        },
    ]
    donor = pd.DataFrame(rows)
    wealth = donor["net_worth"].to_numpy(dtype=float)
    for column in [
        "wageinc",
        "kginc",
        "intdivinc",
        "bussefarminc",
        "ssretinc",
        "houses",
        "mrthel",
        "ccbal",
        "edn_inst",
    ]:
        donor[f"{column}_ratio"] = donor[column].to_numpy(dtype=float) / wealth
    donor["wealth_score"] = np.log1p(wealth)
    donor["major_income_total"] = (
        donor["wageinc"]
        + donor["kginc"]
        + donor["intdivinc"]
        + donor["bussefarminc"]
    )
    return donor


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
    result: pd.DataFrame,
    mini_puf: pd.DataFrame,
    recid: int,
    use_forbes_top_tail: bool = False,
) -> pd.DataFrame:
    from policyengine_us_data.datasets.puf.disaggregate_puf import (
        AGGREGATE_RECIDS,
        SYNTHETIC_RECID_START,
        _choose_n_synthetic,
    )
    from policyengine_us_data.datasets.puf.forbes_backbone import (
        FORBES_DEFAULT_REPLICATES,
    )

    offset = 0
    for bucket_recid in AGGREGATE_RECIDS:
        pop_weight = mini_puf.loc[mini_puf.RECID == bucket_recid, "S006"].iloc[0] / 100
        if use_forbes_top_tail and bucket_recid == 999999:
            n_syn = int(round(pop_weight)) * FORBES_DEFAULT_REPLICATES
        else:
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

    return disaggregate_aggregate_records(mini_puf, seed=42, use_forbes_top_tail=False)


@pytest.fixture
def forbes_result(mini_puf, monkeypatch):
    from policyengine_us_data.datasets.puf import forbes_backbone
    from policyengine_us_data.datasets.puf.disaggregate_puf import (
        disaggregate_aggregate_records,
    )

    monkeypatch.setattr(
        forbes_backbone,
        "load_forbes_us_top_400",
        lambda *args, **kwargs: _make_mock_forbes_records(),
    )
    monkeypatch.setattr(
        forbes_backbone,
        "load_scf_forbes_donor_pool",
        lambda *args, **kwargs: _make_mock_scf_forbes_donors(),
    )
    return disaggregate_aggregate_records(mini_puf, seed=42, use_forbes_top_tail=True)


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
        r1 = disaggregate_aggregate_records(
            puf.copy(), seed=42, use_forbes_top_tail=False
        )
        r2 = disaggregate_aggregate_records(
            puf.copy(), seed=42, use_forbes_top_tail=False
        )
        pd.testing.assert_frame_equal(r1, r2)

    def test_different_seed_different_result(self):
        from policyengine_us_data.datasets.puf.disaggregate_puf import (
            disaggregate_aggregate_records,
        )

        puf = _make_mini_puf()
        r1 = disaggregate_aggregate_records(
            puf.copy(), seed=42, use_forbes_top_tail=False
        )
        r2 = disaggregate_aggregate_records(
            puf.copy(), seed=99, use_forbes_top_tail=False
        )
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
        result = disaggregate_aggregate_records(puf, seed=42, use_forbes_top_tail=False)
        pd.testing.assert_frame_equal(result, puf)

    def test_forbes_failure_falls_back_to_legacy_top_tail(self, mini_puf, monkeypatch):
        from policyengine_us_data.datasets.puf import forbes_backbone
        from policyengine_us_data.datasets.puf.disaggregate_puf import (
            _choose_n_synthetic,
            disaggregate_aggregate_records,
        )

        def fail(*args, **kwargs):
            raise RuntimeError("forbes unavailable")

        monkeypatch.setattr(forbes_backbone, "load_forbes_us_top_400", fail)
        result = disaggregate_aggregate_records(
            mini_puf, seed=42, use_forbes_top_tail=True
        )
        bucket = _synthetic_bucket(
            result,
            mini_puf,
            999999,
            use_forbes_top_tail=False,
        )
        expected = _choose_n_synthetic(
            mini_puf.loc[mini_puf.RECID == 999999, "S006"].iloc[0] / 100
        )
        assert len(bucket) == expected

    def test_forbes_downstream_exception_falls_back_to_legacy_top_tail(
        self, mini_puf, monkeypatch
    ):
        from policyengine_us_data.datasets.puf import disaggregate_puf
        from policyengine_us_data.datasets.puf.disaggregate_puf import (
            _choose_n_synthetic,
            disaggregate_aggregate_records,
        )

        def fail(*args, **kwargs):
            raise RuntimeError("downstream failure")

        monkeypatch.setattr(disaggregate_puf, "build_forbes_top_tail_bucket", fail)
        result = disaggregate_aggregate_records(
            mini_puf, seed=42, use_forbes_top_tail=True
        )
        bucket = _synthetic_bucket(
            result,
            mini_puf,
            999999,
            use_forbes_top_tail=False,
        )
        expected = _choose_n_synthetic(
            mini_puf.loc[mini_puf.RECID == 999999, "S006"].iloc[0] / 100
        )
        assert len(bucket) == expected


class TestForbesBackbone:
    def test_forbes_artifact_exposes_staged_outputs(self, mini_puf, monkeypatch):
        from policyengine_us_data.datasets.puf import forbes_backbone
        from policyengine_us_data.datasets.puf.disaggregate_puf import (
            _get_amount_columns,
            compute_aggregate_eligibility_scores,
        )

        monkeypatch.setattr(
            forbes_backbone,
            "load_forbes_us_top_400",
            lambda *args, **kwargs: _make_mock_forbes_records(),
        )
        monkeypatch.setattr(
            forbes_backbone,
            "load_scf_forbes_donor_pool",
            lambda *args, **kwargs: _make_mock_scf_forbes_donors(),
        )

        regular = mini_puf[~mini_puf.RECID.isin(AGGREGATE_RECIDS)].copy()
        row = mini_puf.loc[mini_puf.RECID == 999999].iloc[0]
        config = forbes_backbone.ForbesTopTailConfig(replicate_count=10)
        artifact = forbes_backbone.build_forbes_top_tail_artifact(
            row=row,
            regular=regular,
            amount_columns=_get_amount_columns(mini_puf.columns),
            donor_scores=compute_aggregate_eligibility_scores(regular),
            next_recid=2_000_000,
            rng=np.random.default_rng(42),
            config=config,
        )

        assert len(artifact.source_forbes) == 400
        assert len(artifact.selected_forbes) == 349
        assert len(artifact.scf_donors) == 4
        assert len(artifact.scf_draws) == 349 * config.replicate_count
        assert len(artifact.puf_templates) == len(artifact.scf_draws)
        assert len(artifact.puf_priors) == len(artifact.scf_draws)
        assert len(artifact.synthetic) == len(artifact.scf_draws)
        assert artifact.diagnostics["selected_forbes_units"] == 349
        assert artifact.diagnostics["synthetic_rows"] == len(artifact.synthetic)
        assert artifact.diagnostics["source_ref"] == forbes_backbone.FORBES_RTB_API_REF

    def test_forbes_diagnostics_are_pr_ready(self, mini_puf, monkeypatch):
        from policyengine_us_data.datasets.puf import forbes_backbone
        from policyengine_us_data.datasets.puf.disaggregate_puf import (
            _get_amount_columns,
            compute_aggregate_eligibility_scores,
        )

        monkeypatch.setattr(
            forbes_backbone,
            "load_forbes_us_top_400",
            lambda *args, **kwargs: _make_mock_forbes_records(),
        )
        monkeypatch.setattr(
            forbes_backbone,
            "load_scf_forbes_donor_pool",
            lambda *args, **kwargs: _make_mock_scf_forbes_donors(),
        )

        regular = mini_puf[~mini_puf.RECID.isin(AGGREGATE_RECIDS)].copy()
        row = mini_puf.loc[mini_puf.RECID == 999999].iloc[0]
        amount_columns = _get_amount_columns(mini_puf.columns)
        artifact = forbes_backbone.build_forbes_top_tail_artifact(
            row=row,
            regular=regular,
            amount_columns=amount_columns,
            donor_scores=compute_aggregate_eligibility_scores(regular),
            next_recid=2_000_000,
            rng=np.random.default_rng(42),
            config=forbes_backbone.ForbesTopTailConfig(replicate_count=10),
        )

        tables = forbes_backbone.build_forbes_top_tail_diagnostic_tables(
            artifact=artifact,
            row=row,
            amount_columns=amount_columns,
        )

        assert set(tables) == {
            "summary",
            "calibration",
            "composition",
            "selection",
            "draws",
        }
        summary = tables["summary"].iloc[0]
        assert summary["max_calibration_abs_error"] <= 1e-4
        assert summary["synthetic_total_agi"] == pytest.approx(
            summary["target_total_agi"],
            rel=1e-9,
            abs=1e-4,
        )
        assert summary["total_calibration_loss"] <= 1e-24
        composition = tables["composition"].set_index("component")
        assert "capital_gains" in composition.index
        assert "pension_social_security" in composition.index
        assert composition.loc["pension_social_security", "absolute_error"] == (
            pytest.approx(0.0, abs=1e-4)
        )
        assert composition.loc["capital_gains", "puf_prior_loss"] >= 0
        assert composition.loc["capital_gains", "synthetic_loss"] <= 1e-24
        assert tables["selection"]["alias"].iloc[0] == "mock-0"
        assert tables["draws"]["weighted_units"].sum() == pytest.approx(349)

        summary_text = forbes_backbone.format_forbes_top_tail_diagnostics(tables)
        assert "Forbes top-tail diagnostics" in summary_text
        assert "loss=" in summary_text
        assert "capital_gains" in summary_text
        assert "pension_social_security" in summary_text
        assert "mock-0" in summary_text
        assert "#" in summary_text

    def test_forbes_artifact_validation_rejects_bad_weights(
        self, mini_puf, monkeypatch
    ):
        from policyengine_us_data.datasets.puf import forbes_backbone
        from policyengine_us_data.datasets.puf.disaggregate_puf import (
            _get_amount_columns,
            compute_aggregate_eligibility_scores,
        )

        monkeypatch.setattr(
            forbes_backbone,
            "load_forbes_us_top_400",
            lambda *args, **kwargs: _make_mock_forbes_records(),
        )
        monkeypatch.setattr(
            forbes_backbone,
            "load_scf_forbes_donor_pool",
            lambda *args, **kwargs: _make_mock_scf_forbes_donors(),
        )

        regular = mini_puf[~mini_puf.RECID.isin(AGGREGATE_RECIDS)].copy()
        row = mini_puf.loc[mini_puf.RECID == 999999].iloc[0]
        amount_columns = _get_amount_columns(mini_puf.columns)
        config = forbes_backbone.ForbesTopTailConfig(replicate_count=10)
        artifact = forbes_backbone.build_forbes_top_tail_artifact(
            row=row,
            regular=regular,
            amount_columns=amount_columns,
            donor_scores=compute_aggregate_eligibility_scores(regular),
            next_recid=2_000_000,
            rng=np.random.default_rng(42),
            config=config,
        )
        artifact.synthetic.loc[artifact.synthetic.index[0], "S006"] += 1

        with pytest.raises(ValueError, match="weights sum"):
            forbes_backbone.validate_forbes_top_tail_artifact(
                artifact=artifact,
                config=config,
                row=row,
                pop_weight=row.S006 / 100,
                amount_columns=amount_columns,
            )

    def test_forbes_artifact_validation_rejects_bad_calibrated_totals(
        self, mini_puf, monkeypatch
    ):
        from policyengine_us_data.datasets.puf import forbes_backbone
        from policyengine_us_data.datasets.puf.disaggregate_puf import (
            _get_amount_columns,
            compute_aggregate_eligibility_scores,
        )

        monkeypatch.setattr(
            forbes_backbone,
            "load_forbes_us_top_400",
            lambda *args, **kwargs: _make_mock_forbes_records(),
        )
        monkeypatch.setattr(
            forbes_backbone,
            "load_scf_forbes_donor_pool",
            lambda *args, **kwargs: _make_mock_scf_forbes_donors(),
        )

        regular = mini_puf[~mini_puf.RECID.isin(AGGREGATE_RECIDS)].copy()
        row = mini_puf.loc[mini_puf.RECID == 999999].iloc[0]
        amount_columns = _get_amount_columns(mini_puf.columns)
        config = forbes_backbone.ForbesTopTailConfig(replicate_count=10)
        artifact = forbes_backbone.build_forbes_top_tail_artifact(
            row=row,
            regular=regular,
            amount_columns=amount_columns,
            donor_scores=compute_aggregate_eligibility_scores(regular),
            next_recid=2_000_000,
            rng=np.random.default_rng(42),
            config=config,
        )
        artifact.synthetic.loc[artifact.synthetic.index[0], "E00100"] += 1_000_000

        with pytest.raises(ValueError, match="calibrated total mismatch"):
            forbes_backbone.validate_forbes_top_tail_artifact(
                artifact=artifact,
                config=config,
                row=row,
                pop_weight=row.S006 / 100,
                amount_columns=amount_columns,
            )

    def test_forbes_bucket_uses_replicate_weights(self, mini_puf, forbes_result):
        from policyengine_us_data.datasets.puf.forbes_backbone import (
            FORBES_DEFAULT_REPLICATES,
        )

        bucket = _synthetic_bucket(
            forbes_result, mini_puf, 999999, use_forbes_top_tail=True
        )
        assert len(bucket) == 349 * FORBES_DEFAULT_REPLICATES
        assert (bucket.S006 == int(100 / FORBES_DEFAULT_REPLICATES)).all()

    def test_forbes_bucket_still_hits_exact_weighted_totals(
        self, mini_puf, forbes_result
    ):
        from policyengine_us_data.datasets.puf.disaggregate_puf import (
            _get_amount_columns,
        )

        amount_columns = _get_amount_columns(mini_puf.columns)
        bucket = _synthetic_bucket(
            forbes_result, mini_puf, 999999, use_forbes_top_tail=True
        )
        target_row = mini_puf.loc[mini_puf.RECID == 999999].iloc[0]
        pop_weight = target_row.S006 / 100

        for column in amount_columns:
            target = pop_weight * target_row[column]
            actual = _weighted_total(bucket, column)
            assert actual == pytest.approx(target, rel=1e-9, abs=1e-6), (
                f"{column=} {actual=} {target=}"
            )

    def test_forbes_bucket_uses_marital_metadata(self, mini_puf, forbes_result):
        bucket = _synthetic_bucket(
            forbes_result, mini_puf, 999999, use_forbes_top_tail=True
        )
        assert set(bucket.MARS.unique()) == {1, 2}
        assert (bucket.DSI == 0).all()
        assert (bucket.EIC == 0).all()

    def test_scf_joint_profiles_scale_ratios_to_forbes_wealth(self):
        from policyengine_us_data.datasets.puf.forbes_backbone import (
            sample_scf_joint_profiles,
        )

        forbes_draws = pd.DataFrame(
            [
                {
                    "alias": "mock-0",
                    "rank": 1,
                    "industry_key": "technology",
                    "archetype": "ltcg",
                    "age": 55,
                    "is_married": True,
                    "self_made_flag": True,
                    "children": 2,
                    "networth_dollars": 1_000_000_000.0,
                    "forbes_unit_id": 0,
                    "replicate_id": 0,
                }
            ]
        )
        scf_donors = _make_mock_scf_forbes_donors().iloc[[0]].copy()

        result = sample_scf_joint_profiles(
            forbes_draws=forbes_draws,
            scf_donors=scf_donors,
            rng=np.random.default_rng(0),
        )

        assert result["employment_income"].iloc[0] == pytest.approx(10_000_000.0)
        assert result["capital_gains"].iloc[0] == pytest.approx(187_500_000.0)
        assert result["interest_dividend_income"].iloc[0] == pytest.approx(
            25_000_000.0
        )
        assert result["business_farm_income"].iloc[0] == pytest.approx(18_750_000.0)

    def test_forbes_selection_uses_scf_membership_scores(self, monkeypatch):
        from policyengine_us_data.datasets.puf import forbes_backbone

        forbes = pd.DataFrame(
            [
                {
                    "alias": "heuristic-winner",
                    "rank": 1,
                    "networth_millions": 1_100.0,
                    "citizenship": "us",
                    "residence_country": "us",
                    "industry": "technology",
                    "self_made": True,
                    "deceased": False,
                    "age": 55,
                    "marital_status": "Single",
                    "children": 0,
                },
                {
                    "alias": "scf-winner",
                    "rank": 2,
                    "networth_millions": 1_000.0,
                    "citizenship": "us",
                    "residence_country": "us",
                    "industry": "technology",
                    "self_made": True,
                    "deceased": False,
                    "age": 55,
                    "marital_status": "Single",
                    "children": 0,
                },
            ]
        )
        scf_donors = pd.DataFrame(
            {
                "archetype": ["ltcg", "ltcg", "ltcg", "ltcg"],
                "wageinc_ratio": [0.00, 0.00, 0.01, 0.01],
                "kginc_ratio": [0.04, 0.04, 0.18, 0.18],
                "intdivinc_ratio": [0.01, 0.01, 0.02, 0.02],
                "bussefarminc_ratio": [0.00, 0.00, 0.01, 0.01],
                "ssretinc_ratio": [0.00, 0.00, 0.00, 0.00],
                "houses_ratio": [0.00, 0.00, 0.00, 0.00],
                "mrthel_ratio": [0.00, 0.00, 0.00, 0.00],
                "age": [55, 55, 55, 55],
                "is_married": [False, False, False, False],
                "wgt": [1.0, 1.0, 1.0, 1.0],
                "wealth_score": [1.0, 1.0, 1.0, 1.0],
            }
        )

        def fake_match_probabilities(candidates, receiver):
            if receiver.alias == "scf-winner":
                return np.array([0.0, 0.0, 0.5, 0.5])
            return np.array([0.5, 0.5, 0.0, 0.0])

        monkeypatch.setattr(
            forbes_backbone,
            "scf_match_probabilities",
            fake_match_probabilities,
        )

        selected = forbes_backbone.select_forbes_extreme_tail(
            forbes=forbes,
            target_n=1,
            scf_donors=scf_donors,
        )

        assert selected["alias"].iloc[0] == "scf-winner"
        assert selected["scf_tail_probability"].iloc[0] == pytest.approx(1.0)

    def test_scf_pension_signal_maps_to_puf_pension_lines(self):
        from policyengine_us_data.datasets.puf.forbes_backbone import (
            apply_forbes_joint_amount_bases,
        )

        donor_templates = pd.DataFrame(
            {
                "E00100": [20_000_000.0],
                "E00200": [1_000_000.0],
                "P23250": [10_000_000.0],
                "P22250": [1_000_000.0],
                "E00300": [500_000.0],
                "E00400": [100_000.0],
                "E00600": [900_000.0],
                "E00650": [700_000.0],
                "E26270": [2_000_000.0],
                "E00900": [300_000.0],
                "E02100": [100_000.0],
                "E01500": [800_000.0],
                "E01700": [600_000.0],
                "E02400": [200_000.0],
            }
        )
        selected = donor_templates.copy()
        forbes_draws = pd.DataFrame(
            {
                "employment_income": [2_000_000.0],
                "capital_gains": [12_000_000.0],
                "interest_dividend_income": [3_000_000.0],
                "business_farm_income": [4_000_000.0],
                "pension_income": [1_000_000.0],
            }
        )

        apply_forbes_joint_amount_bases(
            selected=selected,
            forbes_draws=forbes_draws,
            donor_templates=donor_templates,
        )

        assert selected["E01500"].iloc[0] > 0
        assert selected["E01700"].iloc[0] > 0
        assert selected["E02400"].iloc[0] > 0
        assert selected["E01500"].iloc[0] + selected["E02400"].iloc[0] == (
            pytest.approx(1_000_000.0)
        )
        assert selected["E01700"].iloc[0] == pytest.approx(600_000.0)


class TestForbesCache:
    def test_default_load_uses_packaged_snapshot_without_network(self, monkeypatch):
        from policyengine_us_data.datasets.puf import forbes_backbone

        def fail(*args, **kwargs):
            raise AssertionError("default Forbes load should not hit the network")

        monkeypatch.setattr(forbes_backbone, "fetch_forbes_us_top_400", fail)

        result = forbes_backbone.load_forbes_us_top_400()

        assert len(result) == forbes_backbone.FORBES_TOP_400_SIZE
        assert set(result["snapshot_date"]) == {
            forbes_backbone.FORBES_DEFAULT_SNAPSHOT_DATE
        }
        assert result["alias"].iloc[0] == "elon-musk"

    def test_default_scf_load_uses_packaged_donors_without_raw_scf(
        self, monkeypatch
    ):
        from policyengine_us_data.datasets.puf import forbes_backbone

        forbes_backbone._SCF_FORBES_DONOR_CACHE.clear()

        def fail(*args, **kwargs):
            raise AssertionError("default SCF donor load should not use raw SCF")

        monkeypatch.setattr(forbes_backbone, "prepare_scf_forbes_donor_pool", fail)

        result = forbes_backbone.load_scf_forbes_donor_pool()

        assert len(result) == 6092
        assert "wageinc_ratio" in result.columns
        assert result["archetype"].isin(["dividend", "ltcg", "mixed", "partnership"]).all()

    def test_corrupt_stale_cache_refreshes(self, tmp_path, monkeypatch):
        from policyengine_us_data.datasets.puf import forbes_backbone

        cache_path = tmp_path / "forbes.json"
        cache_path.write_text("{not json", encoding="utf-8")

        monkeypatch.setattr(forbes_backbone, "cache_is_fresh", lambda path: False)
        monkeypatch.setattr(
            forbes_backbone,
            "fetch_forbes_us_top_400",
            lambda *args, **kwargs: _make_mock_forbes_records(2).to_dict("records"),
        )

        result = forbes_backbone.load_forbes_us_top_400(cache_path=cache_path)

        assert len(result) == 2
        assert (
            json.loads(cache_path.read_text(encoding="utf-8"))[0]["alias"] == "mock-0"
        )

    def test_cache_path_is_versioned_by_pinned_source_ref(self):
        from policyengine_us_data.datasets.puf import forbes_backbone

        cache_path = forbes_backbone.forbes_cache_path(
            forbes_backbone.FORBES_DEFAULT_SNAPSHOT_DATE
        )

        assert forbes_backbone.FORBES_RTB_API_REF[:12] in cache_path.name

    def test_existing_cache_is_immutable_by_default(self, tmp_path):
        from policyengine_us_data.datasets.puf import forbes_backbone

        cache_path = tmp_path / "forbes.json"
        cache_path.write_text("[]", encoding="utf-8")

        assert forbes_backbone.cache_is_fresh(cache_path)

    def test_profile_fetch_failure_raises(self, monkeypatch):
        from policyengine_us_data.datasets.puf import forbes_backbone

        def fail(*args, **kwargs):
            raise RuntimeError("profile unavailable")

        monkeypatch.setattr(forbes_backbone.requests, "get", fail)

        with pytest.raises(RuntimeError, match="profile metadata"):
            forbes_backbone.fetch_profile_info_batch(["mock-0"], max_workers=1)


class TestForbesSnapshot:
    def test_fetch_forbes_uses_explicit_2024_snapshot(self, monkeypatch):
        from policyengine_us_data.datasets.puf import forbes_backbone

        requested_urls = []

        def fake_fetch_json(session, url):
            requested_urls.append(url)
            if url == forbes_backbone.FORBES_US_ALIASES_URL:
                return ["mock-0"]
            if url == forbes_backbone.FORBES_LIST_URL_TEMPLATE.format(
                snapshot_date=forbes_backbone.FORBES_DEFAULT_SNAPSHOT_DATE
            ):
                return {
                    "date": forbes_backbone.FORBES_DEFAULT_SNAPSHOT_DATE,
                    "list": [
                        {
                            "rank": 1,
                            "uri": "mock-0",
                            "name": "Mock Forbes 0",
                            "age": 55,
                            "networth": 10_000.0,
                            "citizenship": "us",
                            "industry": ["technology"],
                            "source": ["Mock"],
                        }
                    ],
                }
            raise AssertionError(f"Unexpected URL: {url}")

        monkeypatch.setattr(forbes_backbone, "fetch_json", fake_fetch_json)
        monkeypatch.setattr(
            forbes_backbone,
            "fetch_profile_info_batch",
            lambda *args, **kwargs: {
                "mock-0": {
                    "citizenship": "us",
                    "residence": {"country": "us", "state": "California"},
                    "industry": ["technology"],
                    "source": ["Mock"],
                    "selfMade": {"_is": True, "type": "self-made"},
                    "family": False,
                    "deceased": False,
                }
            },
        )

        result = forbes_backbone.fetch_forbes_us_top_400()

        assert len(result) == 1
        assert (
            result[0]["snapshot_date"] == forbes_backbone.FORBES_DEFAULT_SNAPSHOT_DATE
        )
        assert (
            forbes_backbone.FORBES_LIST_URL_TEMPLATE.format(
                snapshot_date=forbes_backbone.FORBES_DEFAULT_SNAPSHOT_DATE
            )
            in requested_urls
        )
        assert not any(url.endswith("/latest") for url in requested_urls)
        assert not any("/main/" in url for url in requested_urls)
        assert all(forbes_backbone.FORBES_RTB_API_REF in url for url in requested_urls)
