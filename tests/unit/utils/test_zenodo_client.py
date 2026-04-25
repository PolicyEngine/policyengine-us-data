"""Unit tests for the Zenodo deposit client.

HTTP is fully mocked via ``requests.Session`` so these run without
credentials and without reaching out to any network endpoint.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from policyengine_us_data.utils.zenodo_client import (
    DEFAULT_BASE_URL,
    SANDBOX_BASE_URL,
    DepositedFile,
    ZenodoDeposit,
    ZenodoDepositError,
    ZenodoMetadata,
    ZenodoNotConfigured,
    create_and_publish_deposit,
    resolve_base_url,
    zenodo_md5_to_hex,
)


class _MockResponse:
    def __init__(self, status_code: int, body: dict, text: str = "") -> None:
        self.status_code = status_code
        self._body = body
        self.text = text or str(body)

    def json(self) -> dict:
        return self._body


def _client_with_queued_responses(responses: list[_MockResponse]):
    session = MagicMock()
    session.post.side_effect = [r for r in responses if r is not None and _is_post(r)]
    session.put.side_effect = [r for r in responses if r is not None and _is_put(r)]
    return session


def _is_post(resp) -> bool:
    return getattr(resp, "_method", "") == "POST"


def _is_put(resp) -> bool:
    return getattr(resp, "_method", "") == "PUT"


def _tag(resp: _MockResponse, method: str) -> _MockResponse:
    resp._method = method
    return resp


@pytest.fixture
def enable_token(monkeypatch):
    monkeypatch.setenv("ZENODO_ACCESS_TOKEN", "fake-token-for-tests")


@pytest.fixture
def metadata() -> ZenodoMetadata:
    return ZenodoMetadata(
        title="PolicyEngine US calibrated microdata — enhanced CPS 2024 release 1.85.2",
        description="Calibrated enhanced Current Population Survey for PolicyEngine US.",
        creators=(
            {
                "name": "PolicyEngine",
                "affiliation": "PolicyEngine",
            },
        ),
        version="1.85.2",
        keywords=("microsimulation", "calibration", "enhanced CPS"),
    )


@pytest.fixture
def h5_file(tmp_path: Path) -> Path:
    target = tmp_path / "enhanced_cps_2024.h5"
    target.write_bytes(b"not a real h5 but fine for upload-path tests")
    return target


class TestCreateAndPublishDeposit:
    def test_happy_path_returns_version_and_concept_dois(
        self, enable_token, metadata, h5_file, tmp_path, monkeypatch
    ):
        bucket_url = "https://sandbox.zenodo.org/api/files/bucket-abc123"
        session = MagicMock()
        session.post.side_effect = [
            _MockResponse(
                201,
                {"id": 424242, "links": {"bucket": bucket_url}},
            ),
            _MockResponse(
                202,
                {
                    "conceptdoi": "10.5072/zenodo.424240",
                    "doi": "10.5072/zenodo.424242",
                    "links": {"html": "https://sandbox.zenodo.org/records/424242"},
                },
            ),
        ]
        session.put.side_effect = [
            _MockResponse(
                201,
                {
                    "links": {
                        "self": f"{bucket_url}/enhanced_cps_2024.h5",
                    },
                    "size": 42,
                    "checksum": "md5:d41d8cd98f00b204e9800998ecf8427e",
                },
            ),
            _MockResponse(200, {"id": 424242}),
        ]

        deposit = create_and_publish_deposit(
            files=[(h5_file, "enhanced_cps_2024.h5")],
            metadata=metadata,
            session=session,
            base_url=SANDBOX_BASE_URL,
        )

        assert isinstance(deposit, ZenodoDeposit)
        assert deposit.deposit_id == 424242
        assert deposit.version_doi == "10.5072/zenodo.424242"
        assert deposit.concept_doi == "10.5072/zenodo.424240"
        assert deposit.landing_page == "https://sandbox.zenodo.org/records/424242"
        assert len(deposit.files) == 1
        assert isinstance(deposit.files[0], DepositedFile)
        assert deposit.files[0].path_in_deposit == "enhanced_cps_2024.h5"
        assert deposit.files[0].size_bytes == 42
        assert deposit.files[0].checksum.startswith("md5:")

    def test_raises_when_token_is_unset(self, metadata, h5_file, monkeypatch):
        monkeypatch.delenv("ZENODO_ACCESS_TOKEN", raising=False)
        with pytest.raises(ZenodoNotConfigured):
            create_and_publish_deposit(
                files=[(h5_file, "enhanced_cps_2024.h5")],
                metadata=metadata,
            )

    def test_raises_on_missing_source_file(self, enable_token, metadata, tmp_path):
        session = MagicMock()
        session.post.side_effect = [
            _MockResponse(
                201,
                {"id": 1, "links": {"bucket": "https://example.com/bucket"}},
            ),
        ]
        missing = tmp_path / "does-not-exist.h5"
        with pytest.raises(FileNotFoundError):
            create_and_publish_deposit(
                files=[(missing, "does-not-exist.h5")],
                metadata=metadata,
                session=session,
                base_url=SANDBOX_BASE_URL,
            )

    def test_wraps_zenodo_error_responses(self, enable_token, metadata, h5_file):
        session = MagicMock()
        session.post.side_effect = [
            _MockResponse(403, {"message": "forbidden"}, text="forbidden"),
        ]
        with pytest.raises(ZenodoDepositError, match="create-deposit"):
            create_and_publish_deposit(
                files=[(h5_file, "enhanced_cps_2024.h5")],
                metadata=metadata,
                session=session,
                base_url=SANDBOX_BASE_URL,
            )


class TestResolveBaseUrl:
    def test_defaults_to_production(self, monkeypatch):
        monkeypatch.delenv("ZENODO_BASE_URL", raising=False)
        assert resolve_base_url() == DEFAULT_BASE_URL

    def test_honors_env_override(self, monkeypatch):
        monkeypatch.setenv("ZENODO_BASE_URL", SANDBOX_BASE_URL)
        assert resolve_base_url() == SANDBOX_BASE_URL


class TestZenodoMetadataPayload:
    def test_minimal_payload_omits_optional_fields(self):
        metadata = ZenodoMetadata(
            title="t",
            description="d",
            creators=({"name": "PolicyEngine"},),
        )
        payload = metadata.as_zenodo_payload()
        assert payload["title"] == "t"
        assert payload["description"] == "d"
        assert payload["creators"] == [{"name": "PolicyEngine"}]
        assert "keywords" not in payload
        assert "version" not in payload
        assert "related_identifiers" not in payload

    def test_full_payload_serializes_all_fields(self):
        metadata = ZenodoMetadata(
            title="t",
            description="d",
            creators=({"name": "PolicyEngine"},),
            keywords=("a", "b"),
            version="1.2.3",
            related_identifiers=(
                {"relation": "isSupplementTo", "identifier": "example"},
            ),
        )
        payload = metadata.as_zenodo_payload()
        assert payload["keywords"] == ["a", "b"]
        assert payload["version"] == "1.2.3"
        assert payload["related_identifiers"] == [
            {"relation": "isSupplementTo", "identifier": "example"}
        ]


class TestZenodoMd5ToHex:
    def test_strips_md5_prefix(self):
        assert (
            zenodo_md5_to_hex("md5:d41d8cd98f00b204e9800998ecf8427e")
            == "d41d8cd98f00b204e9800998ecf8427e"
        )

    def test_passes_bare_hex_through(self):
        assert (
            zenodo_md5_to_hex("d41d8cd98f00b204e9800998ecf8427e")
            == "d41d8cd98f00b204e9800998ecf8427e"
        )

    def test_rejects_other_algorithms(self):
        with pytest.raises(ValueError, match="Unsupported"):
            zenodo_md5_to_hex("sha256:abc")
