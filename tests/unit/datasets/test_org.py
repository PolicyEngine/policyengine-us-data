import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import time

from policyengine_us_data.datasets.cps import cps as cps_module
from policyengine_us_data.datasets.org import (
    apply_org_domain_constraints,
    build_org_receiver_frame,
)
from policyengine_us_data.datasets.org.org import (
    CPS_BASIC_MONTHLY_ORG_COLUMNS,
    _build_union_priority_weights,
    _load_cps_basic_org_month,
    load_org_training_data,
    _predict_union_coverage_from_bls_tables,
    _select_cps_basic_org_columns,
    _transform_cps_basic_org_month,
)


def test_build_org_receiver_frame_derives_wbho_groups():
    receiver = build_org_receiver_frame(
        age=np.array([30, 40, 35, 50]),
        is_female=np.array([0, 1, 1, 0]),
        is_hispanic=np.array([0, 0, 1, 0]),
        cps_race=np.array([1, 2, 1, 4]),
        state_fips=np.array([6, 36, 48, 12]),
        employment_income=np.array([50_000, 60_000, 45_000, 70_000]),
        weekly_hours_worked=np.array([40, 40, 35, 45]),
    )

    np.testing.assert_array_equal(
        receiver["race_wbho"].values,
        np.array([1, 2, 3, 4], dtype=np.float32),
    )


def test_apply_org_domain_constraints_zeroes_inactive_workers():
    predictions = pd.DataFrame(
        {
            "hourly_wage": [-5.0, 22.0, 30.0],
            "is_paid_hourly": [0.8, 0.7, 0.2],
            "is_union_member_or_covered": [0.9, 0.6, 0.7],
        }
    )
    receiver = pd.DataFrame(
        {
            "employment_income": [0.0, 40_000.0, 55_000.0],
            "weekly_hours_worked": [10.0, 0.0, 40.0],
        }
    )

    result = apply_org_domain_constraints(predictions, receiver)

    assert result["hourly_wage"].tolist() == [0.0, 0.0, 30.0]
    assert result["is_paid_hourly"].tolist() == [False, False, False]
    assert result["is_union_member_or_covered"].tolist() == [False, False, True]


def test_apply_org_domain_constraints_zeroes_self_employed_nonwage_workers():
    predictions = pd.DataFrame(
        {
            "hourly_wage": [35.0],
            "is_paid_hourly": [0.9],
            "is_union_member_or_covered": [0.8],
        }
    )
    receiver = pd.DataFrame(
        {
            "employment_income": [0.0],
            "weekly_hours_worked": [40.0],
        }
    )

    result = apply_org_domain_constraints(
        predictions,
        receiver,
        self_employment_income=np.array([20_000.0]),
    )

    assert result["hourly_wage"].iloc[0] == 0.0
    assert not bool(result["is_paid_hourly"].iloc[0])
    assert not bool(result["is_union_member_or_covered"].iloc[0])


def test_transform_cps_basic_org_month_uses_primary_cps_fields():
    raw = pd.DataFrame(
        {
            "HRMIS": [4, 8, 4, 3],
            "gestfips": [6, 36, 12, 6],
            "prtage": [30, 45, 28, 32],
            "pesex": [1, 2, 2, 1],
            "ptdtrace": [1, 2, 3, 1],
            "pehspnon": [2, 2, 1, 2],
            "pworwgt": [100.0, 200.0, 150.0, 0.0],
            "pternwa": [100000.0, 80000.0, 120000.0, 90000.0],
            "pternhly": [2500.0, -1.0, 3000.0, 2000.0],
            "peernhry": [1, 2, 1, 1],
            "pehruslt": [40.0, 40.0, 50.0, 40.0],
            "prerelg": [1, 1, 1, 1],
            "pemlr": [1, 1, 2, 1],
            "peio1cow": [1, 4, 2, 1],
        }
    )

    transformed = _transform_cps_basic_org_month(raw)

    assert len(transformed) == 3
    assert transformed["hourly_wage"].tolist() == [25.0, 20.0, 30.0]
    assert transformed["is_paid_hourly"].tolist() == [1.0, 0.0, 1.0]
    assert "is_union_member_or_covered" not in transformed.columns


def test_select_cps_basic_org_columns_normalizes_case_and_order():
    month_df = pd.DataFrame(
        {
            "hrmis": [4],
            "GESTFIPS": [6],
            "PRTAGE": [30],
            "PESEX": [2],
            "PTDTRACE": [1],
            "PEHSPNON": [2],
            "PWORWGT": [100.0],
            "PTERNWA": [100000.0],
            "PTERNHLY": [2500.0],
            "PEERNHRY": [1],
            "PEHRUSLT": [40.0],
            "PRERELG": [1],
            "PEMLR": [1],
            "PEIO1COW": [1],
        }
    )

    selected = _select_cps_basic_org_columns(month_df)

    assert selected.columns.tolist() == CPS_BASIC_MONTHLY_ORG_COLUMNS
    assert selected.iloc[0].to_dict() == {
        "HRMIS": 4,
        "gestfips": 6,
        "prtage": 30,
        "pesex": 2,
        "ptdtrace": 1,
        "pehspnon": 2,
        "pworwgt": 100.0,
        "pternwa": 100000.0,
        "pternhly": 2500.0,
        "peernhry": 1,
        "pehruslt": 40.0,
        "prerelg": 1,
        "pemlr": 1,
        "peio1cow": 1,
    }


def test_load_cps_basic_org_month_retries_after_transient_parser_failure(
    monkeypatch,
):
    calls = []
    csv_text = (
        "hrmis,GESTFIPS,PRTAGE,PESEX,PTDTRACE,PEHSPNON,PWORWGT,"
        "PTERNWA,PTERNHLY,PEERNHRY,PEHRUSLT,PRERELG,PEMLR,PEIO1COW\n"
        "4,6,30,2,1,2,100.0,100000.0,2500.0,1,40.0,1,1,1\n"
    )

    class FakeResponse:
        def __init__(self, text: str, status_code: int = 200):
            self.content = text.encode("utf-8")
            self.status_code = status_code

        def raise_for_status(self):
            if self.status_code >= 400:
                raise ValueError("bad status")

    responses = [
        FakeResponse("<html>temporary error</html>"),
        FakeResponse(csv_text),
    ]

    def fake_get(*args, **kwargs):
        calls.append(kwargs)
        return responses.pop(0)

    monkeypatch.setattr("policyengine_us_data.datasets.org.org.requests.get", fake_get)

    loaded = _load_cps_basic_org_month(2024, "may", max_attempts=2)

    assert len(calls) == 2
    assert loaded.columns.tolist() == CPS_BASIC_MONTHLY_ORG_COLUMNS


def test_load_org_training_data_serializes_first_cache_build(monkeypatch, tmp_path):
    raw_month = pd.DataFrame(
        {
            "HRMIS": [4],
            "gestfips": [6],
            "prtage": [30],
            "pesex": [2],
            "ptdtrace": [1],
            "pehspnon": [2],
            "pworwgt": [100.0],
            "pternwa": [100000.0],
            "pternhly": [2500.0],
            "peernhry": [1],
            "pehruslt": [40.0],
            "prerelg": [1],
            "pemlr": [1],
            "peio1cow": [1],
        }
    )
    call_count = {"value": 0}

    monkeypatch.setattr(
        "policyengine_us_data.datasets.org.org.STORAGE_FOLDER", tmp_path
    )
    monkeypatch.setattr(
        "policyengine_us_data.datasets.org.org.ORG_MONTHS",
        ("may",),
    )

    def fake_load_month(year, month):
        call_count["value"] += 1
        time.sleep(0.2)
        return raw_month.copy()

    monkeypatch.setattr(
        "policyengine_us_data.datasets.org.org._load_cps_basic_org_month",
        fake_load_month,
    )

    load_org_training_data.cache_clear()
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            left = executor.submit(load_org_training_data)
            right = executor.submit(load_org_training_data)
            left_result = left.result()
            right_result = right.result()
    finally:
        load_org_training_data.cache_clear()

    assert call_count["value"] == 1
    pd.testing.assert_frame_equal(left_result, right_result)


def test_load_org_training_data_rebuilds_invalid_cached_file(monkeypatch, tmp_path):
    raw_month = pd.DataFrame(
        {
            "HRMIS": [4],
            "gestfips": [6],
            "prtage": [30],
            "pesex": [2],
            "ptdtrace": [1],
            "pehspnon": [2],
            "pworwgt": [100.0],
            "pternwa": [100000.0],
            "pternhly": [2500.0],
            "peernhry": [1],
            "pehruslt": [40.0],
            "prerelg": [1],
            "pemlr": [1],
            "peio1cow": [1],
        }
    )
    cache_path = tmp_path / "census_cps_org_2024_wages.csv.gz"
    pd.DataFrame(columns=["employment_income", "weekly_hours_worked"]).to_csv(
        cache_path,
        index=False,
        compression="gzip",
    )
    call_count = {"value": 0}

    monkeypatch.setattr(
        "policyengine_us_data.datasets.org.org.STORAGE_FOLDER", tmp_path
    )
    monkeypatch.setattr(
        "policyengine_us_data.datasets.org.org.ORG_MONTHS",
        ("may",),
    )

    def fake_load_month(year, month):
        call_count["value"] += 1
        return raw_month.copy()

    monkeypatch.setattr(
        "policyengine_us_data.datasets.org.org._load_cps_basic_org_month",
        fake_load_month,
    )

    load_org_training_data.cache_clear()
    try:
        rebuilt = load_org_training_data()
    finally:
        load_org_training_data.cache_clear()

    assert call_count["value"] == 1
    assert not rebuilt.empty
    assert set(
        [
            "employment_income",
            "weekly_hours_worked",
            "age",
            "is_female",
            "is_hispanic",
            "race_wbho",
            "state_fips",
            "hourly_wage",
            "is_paid_hourly",
            "sample_weight",
        ]
    ).issubset(rebuilt.columns)


def test_build_union_priority_weights_reflect_bls_demographics():
    receiver = pd.DataFrame(
        {
            "age": [22, 52],
            "is_female": [1.0, 0.0],
            "race_wbho": [1.0, 2.0],
            "weekly_hours_worked": [20.0, 40.0],
        }
    )

    weights = _build_union_priority_weights(receiver)

    assert weights[1] > weights[0]


def test_predict_union_coverage_from_bls_tables_matches_state_targets():
    n_california = 20
    n_north_carolina = 20
    receiver = pd.DataFrame(
        {
            "employment_income": np.concatenate(
                [
                    np.linspace(40_000, 78_000, n_california),
                    np.linspace(30_000, 68_000, n_north_carolina),
                    [0.0],
                ]
            ),
            "weekly_hours_worked": np.concatenate(
                [
                    np.tile([40.0, 35.0, 20.0, 45.0], 10)[:n_california],
                    np.tile([40.0, 25.0, 30.0, 38.0], 10)[:n_north_carolina],
                    [40.0],
                ]
            ),
            "age": np.concatenate(
                [
                    np.tile([24.0, 32.0, 41.0, 53.0, 61.0], 4),
                    np.tile([23.0, 31.0, 43.0, 51.0, 67.0], 4),
                    [45.0],
                ]
            ),
            "is_female": np.concatenate(
                [
                    np.tile([0.0, 1.0], n_california // 2),
                    np.tile([1.0, 0.0], n_north_carolina // 2),
                    [0.0],
                ]
            ),
            "is_hispanic": np.concatenate(
                [
                    np.tile([0.0, 1.0, 0.0, 0.0, 0.0], 4),
                    np.tile([1.0, 0.0, 0.0, 0.0, 0.0], 4),
                    [0.0],
                ]
            ),
            "race_wbho": np.concatenate(
                [
                    np.tile([1.0, 3.0, 2.0, 4.0, 1.0], 4),
                    np.tile([3.0, 1.0, 2.0, 4.0, 1.0], 4),
                    [1.0],
                ]
            ),
            "state_fips": np.concatenate(
                [
                    np.full(n_california, 6.0),
                    np.full(n_north_carolina, 37.0),
                    [6.0],
                ]
            ),
        }
    )
    self_employment_income = np.concatenate(
        [
            np.zeros(n_california + n_north_carolina),
            [20_000.0],
        ]
    )

    first = _predict_union_coverage_from_bls_tables(
        receiver,
        self_employment_income=self_employment_income,
    )
    second = _predict_union_coverage_from_bls_tables(
        receiver,
        self_employment_income=self_employment_income,
    )

    np.testing.assert_array_equal(first, second)
    assert int(first[:n_california].sum()) == 3
    assert int(first[n_california : n_california + n_north_carolina].sum()) == 1
    assert first[-1] == 0


def test_add_org_labor_market_inputs_handles_nonsequential_household_index(
    monkeypatch,
):
    cps = {
        "household_id": np.array([10, 20], dtype=np.int64),
        "person_household_id": np.array([20, 10], dtype=np.int64),
        "state_fips": pd.Series([6.0, 36.0], index=[100, 200]),
        "age": np.array([30.0, 40.0]),
        "is_female": np.array([0.0, 1.0]),
        "is_hispanic": np.array([0.0, 0.0]),
        "cps_race": np.array([1.0, 2.0]),
        "employment_income": np.array([50_000.0, 60_000.0]),
        "weekly_hours_worked": np.array([40.0, 40.0]),
    }
    captured_state_fips = {}

    def fake_build_org_receiver_frame(**kwargs):
        captured_state_fips["value"] = kwargs["state_fips"]
        return pd.DataFrame(kwargs)

    def fake_predict_org_features(receiver, self_employment_income):
        assert np.array_equal(self_employment_income, np.zeros(len(receiver)))
        return pd.DataFrame(
            {
                "hourly_wage": np.array([20.0, 30.0]),
                "is_paid_hourly": np.array([1.0, 0.0]),
                "is_union_member_or_covered": np.array([0.0, 1.0]),
            }
        )

    monkeypatch.setattr(
        cps_module,
        "build_org_receiver_frame",
        fake_build_org_receiver_frame,
    )
    monkeypatch.setattr(
        cps_module,
        "predict_org_features",
        fake_predict_org_features,
    )

    cps_module.add_org_labor_market_inputs(cps)

    np.testing.assert_array_equal(
        captured_state_fips["value"],
        np.array([36.0, 6.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(cps["hourly_wage"], np.array([20.0, 30.0]))
    np.testing.assert_array_equal(cps["is_paid_hourly"], np.array([True, False]))
    np.testing.assert_array_equal(
        cps["is_union_member_or_covered"],
        np.array([False, True]),
    )
