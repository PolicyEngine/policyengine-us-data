"""TRACE Transparent Research Object (TRO) export for US data releases.

Emits JSON-LD conforming to the canonical TROv 0.1 vocabulary at
``https://w3id.org/trace/trov/0.1#``. The TRO pins every artifact in the
release (plus the release manifest itself) by sha256 and records
PolicyEngine-specific build provenance under the ``pe:`` extension
namespace so a verifier can cross-check against the certified-bundle TRO
emitted by ``policyengine.py`` without parsing prose.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Iterable, Mapping, Optional

TRACE_TRO_FILENAME = "trace.tro.jsonld"
TRACE_TROV_VERSION = "0.1"

TRACE_TROV_NAMESPACE = "https://w3id.org/trace/trov/0.1#"
POLICYENGINE_TRACE_NAMESPACE = "https://policyengine.org/trace/0.1#"

TRACE_CONTEXT: list[dict[str, str]] = [
    {
        "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
        "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
        "trov": TRACE_TROV_NAMESPACE,
        "schema": "https://schema.org/",
        "pe": POLICYENGINE_TRACE_NAMESPACE,
    }
]

_COMPOSITION_ID = "composition/1"
_ARRANGEMENT_ID = "arrangement/1"
_PERFORMANCE_ID = "trp/1"

_MIME_TYPES = {
    "h5": "application/x-hdf5",
    "db": "application/vnd.sqlite3",
    "json": "application/json",
    "jsonld": "application/ld+json",
    "npy": "application/octet-stream",
    "npz": "application/octet-stream",
    "csv": "text/csv",
}


def _hash_object(value: str) -> dict[str, str]:
    return {
        "trov:hashAlgorithm": "sha256",
        "trov:hashValue": value,
    }


def _artifact_mime_type(path_in_repo: str) -> Optional[str]:
    suffix = path_in_repo.rsplit(".", 1)[-1].lower() if "." in path_in_repo else ""
    return _MIME_TYPES.get(suffix)


def canonical_json_bytes(value: Mapping) -> bytes:
    """Canonical JSON serialization used for every content hash.

    Kept public so a third-party verifier can reproduce these bytes
    exactly to recompute artifact hashes and the composition fingerprint.
    """
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def compute_trace_composition_fingerprint(
    artifact_hashes: Iterable[str],
) -> str:
    """Fingerprint a composition by the sorted set of its artifact hashes.

    Joins with ``\\n`` so the boundary between hashes is unambiguous even
    if a future hash algorithm produces variable-length values. This
    matches ``policyengine.py``'s fingerprint algorithm so a verifier can
    recompute fingerprints across data-side and bundle-side TROs.
    """
    sorted_hashes = sorted(artifact_hashes)
    digest = hashlib.sha256()
    digest.update("\n".join(sorted_hashes).encode("utf-8"))
    return digest.hexdigest()


def _emission_context() -> dict[str, str]:
    """Attestation metadata about where and how the TRO was emitted.

    Always includes ``pe:emittedIn`` so a verifier can distinguish a CI
    build from a laptop build without inferring from the absence of
    optional fields.
    """
    context: dict[str, str] = {}
    if os.environ.get("GITHUB_ACTIONS") == "true":
        context["pe:emittedIn"] = "github-actions"
        server = os.environ.get("GITHUB_SERVER_URL")
        repo = os.environ.get("GITHUB_REPOSITORY")
        run_id = os.environ.get("GITHUB_RUN_ID")
        if server and repo and run_id:
            context["pe:ciRunUrl"] = f"{server}/{repo}/actions/runs/{run_id}"
        sha = os.environ.get("GITHUB_SHA")
        if sha:
            context["pe:ciGitSha"] = sha
        ref = os.environ.get("GITHUB_REF")
        if ref:
            context["pe:ciGitRef"] = ref
    else:
        context["pe:emittedIn"] = "local"
    return context


def _build_artifact_entry(
    artifact_id: str,
    sha256: str,
    *,
    mime_type: Optional[str] = None,
    name: Optional[str] = None,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "@id": artifact_id,
        "@type": "trov:ResearchArtifact",
        "trov:hash": _hash_object(sha256),
    }
    if mime_type is not None:
        entry["trov:mimeType"] = mime_type
    if name is not None:
        entry["schema:name"] = name
    return entry


def _build_location_entry(
    location_id: str, artifact_id: str, path: str
) -> dict[str, Any]:
    return {
        "@id": location_id,
        "@type": "trov:ArtifactLocation",
        "trov:artifact": {"@id": artifact_id},
        "trov:path": path,
    }


def _build_performance(
    *,
    build_id: str,
    data_package: Mapping[str, str],
    built_with_model: Mapping[str, Any],
    started_at: Optional[str],
    ended_at: Optional[str],
) -> dict[str, Any]:
    performance: dict[str, Any] = {
        "@id": _PERFORMANCE_ID,
        "@type": "trov:TrustedResearchPerformance",
        "rdfs:comment": (
            f"Publication of release build {build_id} for "
            f"{data_package['name']} {data_package['version']}"
        ),
        "trov:wasConductedBy": {"@id": "trs"},
        "trov:contributedToArrangement": {"@id": _ARRANGEMENT_ID},
        "pe:dataBuildId": build_id,
    }
    if started_at is not None:
        performance["trov:startedAtTime"] = started_at
    if ended_at is not None:
        performance["trov:endedAtTime"] = ended_at

    model_name = built_with_model.get("name")
    if model_name:
        performance["pe:builtWithModelPackageName"] = model_name
    model_version = built_with_model.get("version")
    if model_version:
        performance["pe:builtWithModelVersion"] = model_version
    model_git_sha = built_with_model.get("git_sha")
    if model_git_sha:
        performance["pe:builtWithModelGitSha"] = model_git_sha
    model_fingerprint = built_with_model.get("data_build_fingerprint")
    if model_fingerprint:
        performance["pe:dataBuildFingerprint"] = model_fingerprint

    performance.update(_emission_context())
    return performance


def build_trace_tro_from_release_manifest(
    manifest: Mapping,
    *,
    release_manifest_path: str = "release_manifest.json",
) -> dict:
    """Build a TRACE TRO from a PolicyEngine US data release manifest.

    Artifacts in the composition: every file listed in the release
    manifest's ``artifacts`` block, plus the canonical release manifest
    bytes themselves. Every artifact hash feeds into the composition
    fingerprint, and build-time ``policyengine-us`` provenance is
    surfaced as structured ``pe:*`` fields on the performance node.
    """
    data_package = manifest["data_package"]
    created_at = manifest.get("created_at") or manifest.get("build", {}).get("built_at")
    build = manifest.get("build", {})
    build_id = (
        build.get("build_id") or f"{data_package['name']}-{data_package['version']}"
    )
    built_with_model = build.get("built_with_model_package") or {}
    artifact_items = sorted(manifest.get("artifacts", {}).items())

    composition_artifacts: list[dict[str, Any]] = []
    arrangement_locations: list[dict[str, Any]] = []
    artifact_hashes: list[str] = []

    for index, (artifact_key, artifact) in enumerate(artifact_items):
        artifact_id = f"{_COMPOSITION_ID}/artifact/{index}"
        artifact_hashes.append(artifact["sha256"])
        composition_artifacts.append(
            _build_artifact_entry(
                artifact_id,
                artifact["sha256"],
                mime_type=_artifact_mime_type(artifact["path"]),
                name=artifact_key,
            )
        )
        arrangement_locations.append(
            _build_location_entry(
                f"{_ARRANGEMENT_ID}/location/{index}",
                artifact_id,
                artifact["path"],
            )
        )

    manifest_bytes = canonical_json_bytes(manifest)
    release_manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    manifest_index = len(composition_artifacts)
    manifest_artifact_id = f"{_COMPOSITION_ID}/artifact/{manifest_index}"
    artifact_hashes.append(release_manifest_hash)
    composition_artifacts.append(
        _build_artifact_entry(
            manifest_artifact_id,
            release_manifest_hash,
            mime_type="application/json",
            name="release_manifest.json",
        )
    )
    arrangement_locations.append(
        _build_location_entry(
            f"{_ARRANGEMENT_ID}/location/{manifest_index}",
            manifest_artifact_id,
            release_manifest_path,
        )
    )

    trs_description = {
        "@id": "trs",
        "@type": ["trov:TrustedResearchSystem", "schema:Organization"],
        "schema:name": "PolicyEngine US data release pipeline",
        "schema:description": (
            "PolicyEngine build and release workflow for versioned US microdata artifacts."
        ),
    }

    description = (
        f"TRACE TRO for {data_package['name']} {data_package['version']} "
        f"covering immutable release artifacts and the accompanying release manifest."
    )

    composition = {
        "@id": _COMPOSITION_ID,
        "@type": "trov:ArtifactComposition",
        "trov:hasFingerprint": {
            "@id": f"{_COMPOSITION_ID}/fingerprint",
            "@type": "trov:CompositionFingerprint",
            "trov:hash": _hash_object(
                compute_trace_composition_fingerprint(artifact_hashes)
            ),
        },
        "trov:hasArtifact": composition_artifacts,
    }

    arrangement = {
        "@id": _ARRANGEMENT_ID,
        "@type": "trov:ArtifactArrangement",
        "rdfs:comment": (
            f"Immutable release artifact arrangement for build {build_id}"
        ),
        "trov:hasArtifactLocation": arrangement_locations,
    }

    performance = _build_performance(
        build_id=build_id,
        data_package=data_package,
        built_with_model=built_with_model,
        started_at=build.get("built_at") or created_at,
        ended_at=created_at,
    )

    tro_node: dict[str, Any] = {
        "@id": "tro",
        "@type": ["trov:TransparentResearchObject", "schema:CreativeWork"],
        "trov:vocabularyVersion": TRACE_TROV_VERSION,
        "schema:creator": data_package["name"],
        "schema:name": f"{data_package['name']} {data_package['version']} release TRO",
        "schema:description": description,
        "trov:wasAssembledBy": trs_description,
        "trov:createdWith": {
            "@type": "schema:SoftwareApplication",
            "schema:name": data_package["name"],
            "schema:softwareVersion": data_package["version"],
        },
        "trov:hasComposition": composition,
        "trov:hasArrangement": [arrangement],
        "trov:hasPerformance": [performance],
    }
    if created_at is not None:
        tro_node["schema:dateCreated"] = created_at

    return {"@context": TRACE_CONTEXT, "@graph": [tro_node]}


def serialize_trace_tro(tro: Mapping) -> bytes:
    """Serialize a TRO with the same canonical JSON used for hashing."""
    return canonical_json_bytes(tro)
