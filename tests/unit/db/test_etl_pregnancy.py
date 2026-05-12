from policyengine_us_data.db import etl_pregnancy


class DummyResponse:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


class BadJsonResponse:
    def json(self):
        raise ValueError("not json")


def b01001_payload():
    return [
        [
            "B01001_030E",
            "B01001_031E",
            "B01001_032E",
            "B01001_033E",
            "B01001_034E",
            "B01001_035E",
            "B01001_036E",
            "B01001_037E",
            "B01001_038E",
            "state",
        ],
        ["1", "2", "3", "4", "5", "6", "7", "8", "9", "01"],
    ]


def test_extract_female_population_uses_retrying_http_helper(monkeypatch):
    monkeypatch.delenv("CENSUS_API_KEY", raising=False)
    monkeypatch.setattr(etl_pregnancy, "is_cached", lambda filename: False)
    monkeypatch.setattr(etl_pregnancy, "save_json", lambda filename, data: None)
    calls = []

    def fake_get(url, **kwargs):
        calls.append((url, kwargs))
        return DummyResponse(b01001_payload())

    monkeypatch.setattr(etl_pregnancy, "get_with_exponential_backoff", fake_get)

    result = etl_pregnancy.extract_female_population(2023)

    assert result.to_dict("records") == [{"state_abbrev": "AL", "female_15_44": 45}]
    assert calls == [
        (
            "https://api.census.gov/data/2023/acs/acs1?get="
            "B01001_030E,B01001_031E,B01001_032E,B01001_033E,"
            "B01001_034E,B01001_035E,B01001_036E,B01001_037E,"
            "B01001_038E&for=state:*",
            {"timeout": 30},
        )
    ]


def test_extract_female_population_prefers_cached_s0101(monkeypatch):
    calls = []

    def fake_is_cached(filename):
        return filename == "acs_S0101_state_2024.json"

    def fake_load_json(filename):
        assert filename == "acs_S0101_state_2024.json"
        return [
            ["GEO_ID", "NAME", "S0101_C05_024E"],
            ["0400000US01", "Alabama", "1015655"],
        ]

    def fake_get(url, **kwargs):
        calls.append((url, kwargs))
        raise AssertionError("S0101 cache should avoid Census API calls")

    monkeypatch.setattr(etl_pregnancy, "is_cached", fake_is_cached)
    monkeypatch.setattr(etl_pregnancy, "load_json", fake_load_json)
    monkeypatch.setattr(etl_pregnancy, "get_with_exponential_backoff", fake_get)

    result = etl_pregnancy.extract_female_population(2024)

    assert result.to_dict("records") == [
        {"state_abbrev": "AL", "female_15_44": 1015655}
    ]
    assert calls == []


def test_extract_female_population_falls_back_to_acs5(monkeypatch):
    monkeypatch.delenv("CENSUS_API_KEY", raising=False)
    monkeypatch.setattr(etl_pregnancy, "is_cached", lambda filename: False)
    calls = []
    saved = []

    def fake_get(url, **kwargs):
        calls.append((url, kwargs))
        if "/acs/acs1?" in url:
            return BadJsonResponse()
        return DummyResponse(b01001_payload())

    monkeypatch.setattr(etl_pregnancy, "get_with_exponential_backoff", fake_get)
    monkeypatch.setattr(
        etl_pregnancy,
        "save_json",
        lambda filename, data: saved.append((filename, data)),
    )

    result = etl_pregnancy.extract_female_population(2023)

    assert result.to_dict("records") == [{"state_abbrev": "AL", "female_15_44": 45}]
    assert [call[0] for call in calls] == [
        (
            "https://api.census.gov/data/2023/acs/acs1?get="
            "B01001_030E,B01001_031E,B01001_032E,B01001_033E,"
            "B01001_034E,B01001_035E,B01001_036E,B01001_037E,"
            "B01001_038E&for=state:*"
        ),
        (
            "https://api.census.gov/data/2023/acs/acs5?get="
            "B01001_030E,B01001_031E,B01001_032E,B01001_033E,"
            "B01001_034E,B01001_035E,B01001_036E,B01001_037E,"
            "B01001_038E&for=state:*"
        ),
    ]
    assert saved == [("census_b01001_female_15_44_2023.json", b01001_payload())]
