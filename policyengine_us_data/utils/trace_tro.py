from __future__ import annotations

import hashlib
import json
from typing import Iterable, Mapping

TRACE_TRO_FILENAME = "trace.tro.jsonld"
TRACE_TROV_VERSION = "0.1"
TRACE_CONTEXT = [
    {
        "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
        "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
        "trov": "https://w3id.org/trace/trov/0.1#",
        "schema": "https://schema.org/",
    }
]


def _hash_object(value: str) -> dict[str, str]:
    return {
        "trov:hashAlgorithm": "sha256",
        "trov:hashValue": value,
    }


def _artifact_mime_type(path_in_repo: str) -> str | None:
    suffix = path_in_repo.rsplit(".", 1)[-1].lower() if "." in path_in_repo else ""
    return {
        "h5": "application/x-hdf5",
        "db": "application/vnd.sqlite3",
        "json": "application/json",
        "jsonld": "application/ld+json",
        "npy": "application/octet-stream",
        "npz": "application/octet-stream",
        "csv": "text/csv",
    }.get(suffix)


def compute_trace_composition_fingerprint(
    artifact_hashes: Iterable[str],
) -> str:
    digest = hashlib.sha256()
    digest.update("".join(sorted(artifact_hashes)).encode("utf-8"))
    return digest.hexdigest()


def build_trace_tro_from_release_manifest(
    manifest: Mapping,
    *,
    release_manifest_path: str = "release_manifest.json",
) -> dict:
    data_package = manifest["data_package"]
    created_at = manifest.get("created_at") or manifest.get("build", {}).get("built_at")
    build = manifest.get("build", {})
    build_id = (
        build.get("build_id") or f"{data_package['name']}-{data_package['version']}"
    )
    built_with_model = build.get("built_with_model_package") or {}
    artifact_items = sorted(manifest.get("artifacts", {}).items())

    composition_artifacts = []
    arrangement_locations = []
    artifact_hashes = []

    for index, (_, artifact) in enumerate(artifact_items):
        artifact_id = f"composition/1/artifact/{index}"
        artifact_hashes.append(artifact["sha256"])
        artifact_entry = {
            "@id": artifact_id,
            "@type": "trov:ResearchArtifact",
            "trov:hash": _hash_object(artifact["sha256"]),
        }
        mime_type = _artifact_mime_type(artifact["path"])
        if mime_type is not None:
            artifact_entry["trov:mimeType"] = mime_type
        composition_artifacts.append(artifact_entry)
        arrangement_locations.append(
            {
                "@id": f"arrangement/0/location/{index}",
                "@type": "trov:ArtifactLocation",
                "trov:artifact": {"@id": artifact_id},
                "trov:path": artifact["path"],
            }
        )

    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    release_manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    manifest_artifact_id = f"composition/1/artifact/{len(composition_artifacts)}"
    artifact_hashes.append(release_manifest_hash)
    composition_artifacts.append(
        {
            "@id": manifest_artifact_id,
            "@type": "trov:ResearchArtifact",
            "trov:hash": _hash_object(release_manifest_hash),
            "trov:mimeType": "application/json",
        }
    )
    arrangement_locations.append(
        {
            "@id": f"arrangement/0/location/{len(arrangement_locations)}",
            "@type": "trov:ArtifactLocation",
            "trov:artifact": {"@id": manifest_artifact_id},
            "trov:path": release_manifest_path,
        }
    )

    trs_description = {
        "@id": "trs",
        "@type": ["trov:TrustedResearchSystem", "schema:Organization"],
        "schema:name": "PolicyEngine US data release pipeline",
        "schema:description": (
            "PolicyEngine build and release workflow for versioned US microdata artifacts."
        ),
    }

    model_version = built_with_model.get("version")
    model_name = built_with_model.get("name", "policyengine-us")
    model_fingerprint = built_with_model.get("data_build_fingerprint")
    description = (
        f"TRACE TRO for {data_package['name']} {data_package['version']} "
        f"covering immutable release artifacts and the accompanying release manifest."
    )
    if model_version:
        description += f" Built with {model_name} {model_version}."
    if model_fingerprint:
        description += f" Data-build fingerprint: {model_fingerprint}."

    return {
        "@context": TRACE_CONTEXT,
        "@graph": [
            {
                "@id": "tro",
                "@type": ["trov:TransparentResearchObject", "schema:CreativeWork"],
                "trov:vocabularyVersion": TRACE_TROV_VERSION,
                "schema:creator": data_package["name"],
                "schema:name": f"{data_package['name']} {data_package['version']} release TRO",
                "schema:description": description,
                "schema:dateCreated": created_at,
                "trov:wasAssembledBy": trs_description,
                "trov:createdWith": {
                    "@type": "schema:SoftwareApplication",
                    "schema:name": data_package["name"],
                    "schema:softwareVersion": data_package["version"],
                },
                "trov:hasComposition": {
                    "@id": "composition/1",
                    "@type": "trov:ArtifactComposition",
                    "trov:hasFingerprint": {
                        "@id": "fingerprint",
                        "@type": "trov:CompositionFingerprint",
                        "trov:hash": _hash_object(
                            compute_trace_composition_fingerprint(artifact_hashes)
                        ),
                    },
                    "trov:hasArtifact": composition_artifacts,
                },
                "trov:hasArrangement": [
                    {
                        "@id": "arrangement/0",
                        "@type": "trov:ArtifactArrangement",
                        "rdfs:comment": (
                            f"Immutable release artifact arrangement for build {build_id}"
                        ),
                        "trov:hasArtifactLocation": arrangement_locations,
                    }
                ],
                "trov:hasPerformance": [
                    {
                        "@id": "trp/0",
                        "@type": "trov:TrustedResearchPerformance",
                        "rdfs:comment": (
                            f"Publication of release build {build_id} for "
                            f"{data_package['name']} {data_package['version']}"
                        ),
                        "trov:wasConductedBy": {"@id": "trs"},
                        "trov:startedAtTime": build.get("built_at") or created_at,
                        "trov:endedAtTime": created_at,
                        "trov:contributedToArrangement": {
                            "@id": "trp/0/binding/0",
                            "@type": "trov:ArrangementBinding",
                            "trov:arrangement": {"@id": "arrangement/0"},
                        },
                    }
                ],
            }
        ],
    }


def serialize_trace_tro(tro: Mapping) -> bytes:
    return (json.dumps(tro, indent=2, sort_keys=True) + "\n").encode("utf-8")
