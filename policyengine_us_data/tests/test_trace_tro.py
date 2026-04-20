import hashlib
import json
from pathlib import Path

import pytest

from policyengine_us_data.utils.release_manifest import build_release_manifest
from policyengine_us_data.utils.trace_tro import (
    POLICYENGINE_TRACE_NAMESPACE,
    TRACE_TRO_FILENAME,
    TRACE_TROV_NAMESPACE,
    build_trace_tro_from_release_manifest,
    canonical_json_bytes,
    compute_trace_composition_fingerprint,
    serialize_trace_tro,
)


def _write_file(path: Path, content: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _build_manifest(tmp_path: Path, **overrides):
    national_path = _write_file(tmp_path / "enhanced_cps_2024.h5", b"national")
    state_path = _write_file(tmp_path / "AL.h5", b"state")

    defaults = dict(
        files_with_repo_paths=[
            (national_path, "enhanced_cps_2024.h5"),
            (state_path, "states/AL.h5"),
        ],
        version="1.73.0",
        repo_id="policyengine/policyengine-us-data",
        model_package_version="1.634.4",
        model_package_git_sha="deadbeef",
        model_package_data_build_fingerprint="sha256:fingerprint",
        created_at="2026-04-10T12:00:00Z",
    )
    defaults.update(overrides)
    return build_release_manifest(**defaults)


def test_compute_trace_composition_fingerprint_is_order_independent():
    a = "b" * 64
    b = "a" * 64

    assert compute_trace_composition_fingerprint([a, b]) == (
        compute_trace_composition_fingerprint([b, a])
    )


def test_compute_trace_composition_fingerprint_uses_newline_separator():
    """Matches policyengine.py's canonical fingerprint algorithm."""
    a = "a" * 64
    b = "b" * 64

    expected = hashlib.sha256(f"{a}\n{b}".encode("utf-8")).hexdigest()
    assert compute_trace_composition_fingerprint([b, a]) == expected


def test_build_trace_tro_uses_canonical_trov_namespace(tmp_path):
    manifest = _build_manifest(tmp_path)

    tro = build_trace_tro_from_release_manifest(manifest)

    assert tro["@context"][0]["trov"] == TRACE_TROV_NAMESPACE
    assert tro["@context"][0]["pe"] == POLICYENGINE_TRACE_NAMESPACE


def test_build_trace_tro_from_release_manifest_tracks_artifacts(tmp_path):
    manifest = _build_manifest(tmp_path)

    tro = build_trace_tro_from_release_manifest(manifest)
    root = tro["@graph"][0]
    composition = root["trov:hasComposition"]
    artifacts = composition["trov:hasArtifact"]
    arrangement = root["trov:hasArrangement"][0]

    assert root["schema:name"] == "policyengine-us-data 1.73.0 release TRO"
    assert root["trov:wasAssembledBy"]["schema:name"] == (
        "PolicyEngine US data release pipeline"
    )
    assert len(artifacts) == 3
    assert (
        arrangement["trov:hasArtifactLocation"][-1]["trov:path"]
        == "release_manifest.json"
    )

    manifest_hash = hashlib.sha256(
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    ).hexdigest()
    assert artifacts[-1]["trov:hash"]["trov:hashValue"] == manifest_hash

    expected_fingerprint = compute_trace_composition_fingerprint(
        [artifact["trov:hash"]["trov:hashValue"] for artifact in artifacts]
    )
    assert (
        composition["trov:hasFingerprint"]["trov:hash"]["trov:hashValue"]
        == expected_fingerprint
    )


def test_performance_links_arrangement_directly(tmp_path):
    """TROv 0.1 has no ArrangementBinding; contributedToArrangement is a direct link."""
    manifest = _build_manifest(tmp_path)

    tro = build_trace_tro_from_release_manifest(manifest)
    performance = tro["@graph"][0]["trov:hasPerformance"][0]

    assert performance["trov:contributedToArrangement"] == {"@id": "arrangement/1"}
    assert "trov:arrangement" not in performance["trov:contributedToArrangement"]


def test_performance_exposes_structured_build_metadata(tmp_path):
    manifest = _build_manifest(tmp_path)

    tro = build_trace_tro_from_release_manifest(manifest)
    performance = tro["@graph"][0]["trov:hasPerformance"][0]

    assert performance["pe:builtWithModelPackageName"] == "policyengine-us"
    assert performance["pe:builtWithModelVersion"] == "1.634.4"
    assert performance["pe:builtWithModelGitSha"] == "deadbeef"
    assert performance["pe:dataBuildFingerprint"] == "sha256:fingerprint"
    assert performance["pe:dataBuildId"] == "policyengine-us-data-1.73.0"


def test_performance_omits_missing_build_metadata(tmp_path):
    manifest = _build_manifest(
        tmp_path,
        model_package_version=None,
        model_package_git_sha=None,
        model_package_data_build_fingerprint=None,
    )

    tro = build_trace_tro_from_release_manifest(manifest)
    performance = tro["@graph"][0]["trov:hasPerformance"][0]

    assert "pe:builtWithModelVersion" not in performance
    assert "pe:builtWithModelGitSha" not in performance
    assert "pe:dataBuildFingerprint" not in performance


def test_performance_records_local_emission(tmp_path, monkeypatch):
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    manifest = _build_manifest(tmp_path)

    tro = build_trace_tro_from_release_manifest(manifest)
    performance = tro["@graph"][0]["trov:hasPerformance"][0]

    assert performance["pe:emittedIn"] == "local"
    assert "pe:ciRunUrl" not in performance


def test_performance_records_github_actions_emission(tmp_path, monkeypatch):
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv("GITHUB_SERVER_URL", "https://github.com")
    monkeypatch.setenv("GITHUB_REPOSITORY", "PolicyEngine/policyengine-us-data")
    monkeypatch.setenv("GITHUB_RUN_ID", "1234567890")
    monkeypatch.setenv("GITHUB_SHA", "abc123")
    monkeypatch.setenv("GITHUB_REF", "refs/heads/main")
    manifest = _build_manifest(tmp_path)

    tro = build_trace_tro_from_release_manifest(manifest)
    performance = tro["@graph"][0]["trov:hasPerformance"][0]

    assert performance["pe:emittedIn"] == "github-actions"
    assert performance["pe:ciRunUrl"] == (
        "https://github.com/PolicyEngine/policyengine-us-data/actions/runs/1234567890"
    )
    assert performance["pe:ciGitSha"] == "abc123"
    assert performance["pe:ciGitRef"] == "refs/heads/main"


def test_index_numbering_is_consistent(tmp_path):
    manifest = _build_manifest(tmp_path)

    tro = build_trace_tro_from_release_manifest(manifest)
    root = tro["@graph"][0]
    arrangement = root["trov:hasArrangement"][0]
    performance = root["trov:hasPerformance"][0]

    assert root["trov:hasComposition"]["@id"] == "composition/1"
    assert arrangement["@id"] == "arrangement/1"
    assert performance["@id"] == "trp/1"
    assert arrangement["trov:hasArtifactLocation"][0]["@id"].startswith(
        "arrangement/1/location/"
    )


def test_build_trace_tro_is_deterministic(tmp_path, monkeypatch):
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    manifest = _build_manifest(tmp_path)

    tro_a = build_trace_tro_from_release_manifest(manifest)
    tro_b = build_trace_tro_from_release_manifest(manifest)

    assert serialize_trace_tro(tro_a) == serialize_trace_tro(tro_b)


def test_canonical_json_bytes_match_serialize(tmp_path):
    """A caller recomputing the TRO hash from dict form must get the same bytes."""
    manifest = _build_manifest(tmp_path)

    tro = build_trace_tro_from_release_manifest(manifest)

    assert canonical_json_bytes(tro) == serialize_trace_tro(tro)


def test_trace_tro_filename_is_stable():
    assert TRACE_TRO_FILENAME == "trace.tro.jsonld"


def test_tro_validates_against_shipped_schema(tmp_path):
    """The emitted TRO must pass the packaged JSON schema."""
    jsonschema = pytest.importorskip("jsonschema")
    manifest = _build_manifest(tmp_path)

    tro = build_trace_tro_from_release_manifest(manifest)

    schema_path = (
        Path(__file__).resolve().parents[1] / "schemas" / "trace_tro.schema.json"
    )
    schema = json.loads(schema_path.read_text())
    jsonschema.validate(instance=tro, schema=schema)
