from policyengine_us_data.db import etl_pregnancy


class DummyResponse:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


def test_extract_female_population_uses_retrying_http_helper(monkeypatch):
    monkeypatch.setattr(etl_pregnancy, "is_cached", lambda filename: False)
    monkeypatch.setattr(etl_pregnancy, "save_json", lambda filename, data: None)
    calls = []

    def fake_get(url, **kwargs):
        calls.append((url, kwargs))
        return DummyResponse(
            [
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
        )

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
