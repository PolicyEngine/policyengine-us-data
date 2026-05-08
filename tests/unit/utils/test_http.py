import requests
import pytest

from policyengine_us_data.utils.http import get_with_exponential_backoff


class DummyResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(response=self)

    def json(self):
        return self._payload


class DummySession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def test_get_with_exponential_backoff_retries_transient_statuses():
    session = DummySession(
        [
            DummyResponse(503),
            DummyResponse(502),
            DummyResponse(200, {"ok": True}),
        ]
    )
    sleeps = []

    response = get_with_exponential_backoff(
        "https://example.test/data",
        session=session,
        sleep=sleeps.append,
    )

    assert response.json() == {"ok": True}
    assert len(session.calls) == 3
    assert sleeps == [30, 60]


def test_get_with_exponential_backoff_caps_wait_at_four_minutes():
    session = DummySession([DummyResponse(503)] * 5)
    sleeps = []

    with pytest.raises(requests.HTTPError):
        get_with_exponential_backoff(
            "https://example.test/data",
            session=session,
            sleep=sleeps.append,
        )

    assert len(session.calls) == 5
    assert sleeps == [30, 60, 120, 240]


def test_get_with_exponential_backoff_does_not_retry_non_transient_statuses():
    session = DummySession([DummyResponse(404)])
    sleeps = []

    with pytest.raises(requests.HTTPError):
        get_with_exponential_backoff(
            "https://example.test/missing",
            session=session,
            sleep=sleeps.append,
        )

    assert len(session.calls) == 1
    assert sleeps == []
