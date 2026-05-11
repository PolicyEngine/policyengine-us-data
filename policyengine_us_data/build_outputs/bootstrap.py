"""Persisted worker-bootstrap artifacts for local H5 publication.

This module defines the artifact boundary introduced before worker sessions
become canonical. Bootstrap bundles capture deterministic worker setup facts
once per run, but current workers may still derive those facts from raw inputs.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Mapping

import numpy as np

from policyengine_us_data.pipeline_metadata import pipeline_node

from .fingerprinting import (
    ArtifactIdentity,
    FingerprintingService,
    PublishingInputBundle,
    TraceabilityBundle,
)
from .geography_loader import CalibrationGeographyLoader
from .source_dataset import EntityGraph, PolicyEngineDatasetReader
from .weights import CloneWeightMatrix

BootstrapScope = Literal["regional", "national"]

BOOTSTRAP_SCHEMA_VERSION = 1
BOOTSTRAP_DIR_NAME = "bootstrap"
BOOTSTRAP_MANIFEST_FILENAME = "worker_bootstrap.json"
ENTITY_GRAPH_FILENAME = "entity_graph.npz"

__all__ = [
    "BOOTSTRAP_DIR_NAME",
    "BOOTSTRAP_MANIFEST_FILENAME",
    "BOOTSTRAP_SCHEMA_VERSION",
    "ENTITY_GRAPH_FILENAME",
    "BootstrapScope",
    "WorkerBootstrapBuilder",
    "WorkerBootstrapBundle",
    "WorkerBootstrapStore",
    "load_entity_graph",
    "save_entity_graph",
]


@pipeline_node(
    id="local_h5_worker_bootstrap_bundle",
    label="WorkerBootstrapBundle",
    node_type="library",
    description="Persisted deterministic worker setup facts for one local H5 scope.",
    source_file="policyengine_us_data/build_outputs/bootstrap.py",
    status="current",
    stability="moving",
    pathways=["local_h5"],
    artifacts_in=[
        "calibration_weights.npy",
        "source_imputed_stratified_extended_cps.h5",
        "geography_assignment.npz",
        "policy_data.db",
    ],
    artifacts_out=[
        "worker_bootstrap.json",
        "entity_graph.npz",
    ],
    validation_commands=["uv run pytest tests/unit/build_outputs/test_bootstrap.py"],
)
@dataclass(frozen=True)
class WorkerBootstrapBundle:
    """Manifest-backed bootstrap bundle for one worker setup scope.

    The bundle is an introduced artifact contract only. It records the
    deterministic source, weight, geography, and traceability facts that later
    worker-session code can load instead of rebuilding inside every worker.

    Attributes:
        run_id: GitHub-created pipeline run identifier.
        scope: Bootstrap scope, currently ``"regional"`` or ``"national"``.
        root_dir: Directory containing this scope's bootstrap artifacts.
        manifest_path: Path to ``worker_bootstrap.json``.
        entity_graph_path: Path to ``entity_graph.npz``.
        inputs: Manifest artifact identity map.
        source_dataset: Source dataset summary payload.
        weights: Clone-weight summary payload.
        geography: Geography summary payload.
        traceability: Scope fingerprint and resumability payload.
        artifacts: Relative bootstrap artifact filenames.
        schema_version: Bootstrap schema version.
    """

    run_id: str
    scope: BootstrapScope
    root_dir: Path
    manifest_path: Path
    entity_graph_path: Path
    inputs: Mapping[str, Mapping[str, Any]]
    source_dataset: Mapping[str, Any]
    weights: Mapping[str, Any]
    geography: Mapping[str, Any]
    traceability: Mapping[str, Any]
    artifacts: Mapping[str, str]
    schema_version: int = BOOTSTRAP_SCHEMA_VERSION
    created_by: str = "WorkerBootstrapBuilder"

    def __post_init__(self) -> None:
        _validate_scope(self.scope)
        if self.schema_version != BOOTSTRAP_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported worker bootstrap schema version {self.schema_version}"
            )
        object.__setattr__(self, "run_id", str(self.run_id))
        object.__setattr__(self, "root_dir", Path(self.root_dir))
        object.__setattr__(self, "manifest_path", Path(self.manifest_path))
        object.__setattr__(self, "entity_graph_path", Path(self.entity_graph_path))
        object.__setattr__(self, "inputs", _readonly_mapping(self.inputs))
        object.__setattr__(
            self,
            "source_dataset",
            _readonly_mapping(self.source_dataset),
        )
        object.__setattr__(self, "weights", _readonly_mapping(self.weights))
        object.__setattr__(self, "geography", _readonly_mapping(self.geography))
        object.__setattr__(
            self,
            "traceability",
            _readonly_mapping(self.traceability),
        )
        object.__setattr__(self, "artifacts", MappingProxyType(dict(self.artifacts)))

    def to_manifest(self) -> dict[str, Any]:
        """Return the JSON-serializable bootstrap manifest payload."""

        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "scope": self.scope,
            "created_by": self.created_by,
            "inputs": _plain_mapping(self.inputs),
            "source_dataset": _plain_mapping(self.source_dataset),
            "weights": _plain_mapping(self.weights),
            "geography": _plain_mapping(self.geography),
            "traceability": _plain_mapping(self.traceability),
            "artifacts": dict(self.artifacts),
        }

    @classmethod
    def from_manifest(
        cls,
        *,
        root_dir: Path,
        manifest: Mapping[str, Any],
    ) -> "WorkerBootstrapBundle":
        """Build a bundle object from an on-disk manifest payload."""

        _require_manifest_fields(
            manifest,
            {
                "schema_version",
                "run_id",
                "scope",
                "created_by",
                "inputs",
                "source_dataset",
                "weights",
                "geography",
                "traceability",
                "artifacts",
            },
        )
        root = Path(root_dir)
        artifacts = dict(manifest["artifacts"])
        entity_graph_name = artifacts.get("entity_graph")
        if not entity_graph_name:
            raise ValueError("Bootstrap manifest is missing artifacts.entity_graph")

        return cls(
            schema_version=int(manifest["schema_version"]),
            run_id=str(manifest["run_id"]),
            scope=manifest["scope"],
            root_dir=root,
            manifest_path=root / BOOTSTRAP_MANIFEST_FILENAME,
            entity_graph_path=root / entity_graph_name,
            created_by=str(manifest["created_by"]),
            inputs=manifest["inputs"],
            source_dataset=manifest["source_dataset"],
            weights=manifest["weights"],
            geography=manifest["geography"],
            traceability=manifest["traceability"],
            artifacts=artifacts,
        )


@pipeline_node(
    id="local_h5_worker_bootstrap_store",
    label="WorkerBootstrapStore",
    node_type="library",
    description="Filesystem path adapter for run-scoped local H5 bootstrap artifacts.",
    source_file="policyengine_us_data/build_outputs/bootstrap.py",
    status="current",
    stability="moving",
    pathways=["local_h5"],
    artifacts_out=[
        "bootstrap/{scope}/worker_bootstrap.json",
        "bootstrap/{scope}/entity_graph.npz",
    ],
    validation_commands=["uv run pytest tests/unit/build_outputs/test_bootstrap.py"],
)
@dataclass(frozen=True)
class WorkerBootstrapStore:
    """Filesystem adapter for scope-specific bootstrap bundle paths."""

    artifacts_dir: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "artifacts_dir", Path(self.artifacts_dir))

    def scope_dir(self, scope: BootstrapScope) -> Path:
        """Return the directory for one bootstrap scope."""

        _validate_scope(scope)
        return self.artifacts_dir / BOOTSTRAP_DIR_NAME / scope

    def manifest_path(self, scope: BootstrapScope) -> Path:
        """Return the manifest path for one bootstrap scope."""

        return self.scope_dir(scope) / BOOTSTRAP_MANIFEST_FILENAME

    def entity_graph_path(self, scope: BootstrapScope) -> Path:
        """Return the entity graph artifact path for one bootstrap scope."""

        return self.scope_dir(scope) / ENTITY_GRAPH_FILENAME

    def load(self, scope: BootstrapScope) -> WorkerBootstrapBundle:
        """Load one persisted bootstrap bundle from disk."""

        manifest_path = self.manifest_path(scope)
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Missing worker bootstrap manifest: {manifest_path}"
            )
        with open(manifest_path) as handle:
            manifest = json.load(handle)
        bundle = WorkerBootstrapBundle.from_manifest(
            root_dir=self.scope_dir(scope),
            manifest=manifest,
        )
        if not bundle.entity_graph_path.exists():
            raise FileNotFoundError(
                f"Missing worker bootstrap entity graph: {bundle.entity_graph_path}"
            )
        return bundle


@pipeline_node(
    id="local_h5_worker_bootstrap_builder",
    label="WorkerBootstrapBuilder",
    node_type="library",
    description="Materialize deterministic local H5 worker bootstrap artifacts.",
    source_file="policyengine_us_data/build_outputs/bootstrap.py",
    status="current",
    stability="moving",
    pathways=["local_h5"],
    artifacts_in=[
        "calibration_weights.npy",
        "source_imputed_stratified_extended_cps.h5",
        "geography_assignment.npz",
        "policy_data.db",
    ],
    artifacts_out=[
        "bootstrap/{scope}/worker_bootstrap.json",
        "bootstrap/{scope}/entity_graph.npz",
    ],
    validation_commands=["uv run pytest tests/unit/build_outputs/test_bootstrap.py"],
)
class WorkerBootstrapBuilder:
    """Build and persist one scope's local H5 worker bootstrap artifacts."""

    def __init__(
        self,
        *,
        dataset_reader: PolicyEngineDatasetReader | None = None,
        geography_loader: CalibrationGeographyLoader | None = None,
        fingerprinting_service: FingerprintingService | None = None,
    ) -> None:
        """Create a bootstrap builder with injectable seams for tests."""

        self._dataset_reader = dataset_reader or PolicyEngineDatasetReader()
        self._geography_loader = geography_loader or CalibrationGeographyLoader()
        self._fingerprinting_service = fingerprinting_service or FingerprintingService(
            geography_loader=self._geography_loader
        )

    def build(
        self,
        *,
        inputs: PublishingInputBundle,
        scope: BootstrapScope,
        artifacts_dir: Path,
        scope_fingerprint: str | None = None,
    ) -> WorkerBootstrapBundle:
        """Build and persist one bootstrap bundle.

        Args:
            inputs: Normalized local H5 publishing input bundle.
            scope: Bootstrap scope, currently ``"regional"`` or ``"national"``.
            artifacts_dir: Run-scoped pipeline artifact directory.
            scope_fingerprint: Already-resolved scope fingerprint. When omitted,
                the builder computes it from traceability material.

        Returns:
            Persisted bootstrap bundle metadata.

        Raises:
            FileNotFoundError: If required input or geography artifacts are missing.
            ValueError: If weights, source data, or geography dimensions disagree.
        """

        _validate_scope(scope)
        store = WorkerBootstrapStore(artifacts_dir)
        root_dir = store.scope_dir(scope)
        root_dir.mkdir(parents=True, exist_ok=True)

        snapshot = self._dataset_reader.load(inputs.source_dataset_path)
        weights_array = np.load(inputs.weights_path, mmap_mode="r")
        weights = CloneWeightMatrix.from_vector(
            weights_array,
            n_records=snapshot.n_households,
        )
        if inputs.n_clones is not None and weights.n_clones != int(inputs.n_clones):
            raise ValueError(
                f"Weight vector implies n_clones={weights.n_clones}, "
                f"expected {inputs.n_clones}"
            )

        geography_source = self._geography_loader.resolve_source(
            weights_path=inputs.weights_path,
            geography_path=inputs.exact_geography_path,
            blocks_path=inputs.legacy_blocks_path,
            calibration_package_path=inputs.calibration_package_path,
        )
        if geography_source is None:
            raise FileNotFoundError(
                "No geography artifact available for worker bootstrap"
            )
        geography = self._geography_loader.load(
            weights_path=inputs.weights_path,
            n_records=weights.n_records,
            n_clones=weights.n_clones,
            geography_path=inputs.exact_geography_path,
            blocks_path=inputs.legacy_blocks_path,
            calibration_package_path=inputs.calibration_package_path,
        )
        canonical_geography_sha256 = self._geography_loader.compute_canonical_checksum(
            weights_path=inputs.weights_path,
            n_records=weights.n_records,
            n_clones=weights.n_clones,
            geography_path=inputs.exact_geography_path,
            blocks_path=inputs.legacy_blocks_path,
            calibration_package_path=inputs.calibration_package_path,
        )

        traceability = self._fingerprinting_service.build_traceability(
            inputs=inputs,
            scope=scope,
        )
        computed_scope_fingerprint = (
            self._fingerprinting_service.compute_scope_fingerprint(traceability)
        )
        if (
            scope_fingerprint is not None
            and scope_fingerprint != computed_scope_fingerprint
        ):
            raise ValueError(
                f"Bootstrap fingerprint {scope_fingerprint!r} does not match "
                f"computed {scope} fingerprint {computed_scope_fingerprint!r}"
            )
        scope_fingerprint = computed_scope_fingerprint

        entity_graph_path = store.entity_graph_path(scope)
        save_entity_graph(snapshot.entity_graph, entity_graph_path)

        bundle = WorkerBootstrapBundle(
            run_id=inputs.run_id,
            scope=scope,
            root_dir=root_dir,
            manifest_path=store.manifest_path(scope),
            entity_graph_path=entity_graph_path,
            inputs=_traceability_inputs(traceability),
            source_dataset={
                "path": str(Path(inputs.source_dataset_path)),
                "sha256": _sha256_file(inputs.source_dataset_path),
                "time_period": snapshot.time_period,
                "n_households": snapshot.n_households,
                "input_variables": sorted(snapshot.input_variables),
                "entity_graph_artifact": ENTITY_GRAPH_FILENAME,
            },
            weights={
                "path": str(Path(inputs.weights_path)),
                "sha256": _sha256_file(inputs.weights_path),
                "n_records": weights.n_records,
                "n_clones": weights.n_clones,
                "dtype": str(weights.values.dtype),
            },
            geography={
                "source_kind": geography_source.kind,
                "source_path": str(geography_source.path),
                "sha256": _sha256_file(geography_source.path),
                "canonical_sha256": canonical_geography_sha256,
                "n_records": int(geography.n_records),
                "n_clones": int(geography.n_clones),
            },
            traceability={
                "scope_fingerprint": scope_fingerprint,
                "resumability_material": traceability.resumability_material(),
            },
            artifacts={
                "manifest": BOOTSTRAP_MANIFEST_FILENAME,
                "entity_graph": ENTITY_GRAPH_FILENAME,
            },
        )

        _write_json(bundle.manifest_path, bundle.to_manifest())
        return bundle


def save_entity_graph(entity_graph: EntityGraph, path: Path) -> Path:
    """Persist an `EntityGraph` to an NPZ structural artifact."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    subentity_keys = tuple(sorted(entity_graph.subentity_ids))
    arrays: dict[str, np.ndarray] = {
        "household_ids": entity_graph.household_ids,
        "person_household_ids": entity_graph.person_household_ids,
        "subentity_keys": np.asarray(subentity_keys, dtype="U"),
    }
    for key in subentity_keys:
        arrays[f"subentity_ids__{key}"] = entity_graph.subentity_ids[key]
        arrays[f"person_subentity_ids__{key}"] = entity_graph.person_subentity_ids[key]
    np.savez(destination, **arrays)
    return destination


def load_entity_graph(path: Path) -> EntityGraph:
    """Load an `EntityGraph` from an NPZ structural artifact."""

    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Missing entity graph artifact: {source}")
    with np.load(source, allow_pickle=False) as data:
        required = {"household_ids", "person_household_ids", "subentity_keys"}
        missing = required - set(data.files)
        if missing:
            raise ValueError(
                f"Entity graph artifact {source} is missing fields: {sorted(missing)}"
            )
        subentity_keys = tuple(str(key) for key in data["subentity_keys"].tolist())
        subentity_ids = {}
        person_subentity_ids = {}
        for key in subentity_keys:
            entity_field = f"subentity_ids__{key}"
            person_field = f"person_subentity_ids__{key}"
            if entity_field not in data.files or person_field not in data.files:
                raise ValueError(
                    f"Entity graph artifact {source} is missing fields for {key!r}"
                )
            subentity_ids[key] = data[entity_field]
            person_subentity_ids[key] = data[person_field]
        return EntityGraph(
            household_ids=data["household_ids"],
            person_household_ids=data["person_household_ids"],
            subentity_ids=subentity_ids,
            person_subentity_ids=person_subentity_ids,
        )


def _validate_scope(scope: str) -> None:
    if scope not in {"regional", "national"}:
        raise ValueError("Worker bootstrap scope must be 'regional' or 'national'")


def _traceability_inputs(
    traceability: TraceabilityBundle,
) -> dict[str, dict[str, Any] | None]:
    return {
        "weights": _artifact_identity_manifest(traceability.weights),
        "source_dataset": _artifact_identity_manifest(traceability.source_dataset),
        "exact_geography": _artifact_identity_manifest(traceability.exact_geography),
        "target_db": _artifact_identity_manifest(traceability.target_db),
        "calibration_package": _artifact_identity_manifest(
            traceability.calibration_package
        ),
        "run_config": _artifact_identity_manifest(traceability.run_config),
    }


def _artifact_identity_manifest(
    identity: ArtifactIdentity | None,
) -> dict[str, Any] | None:
    if identity is None:
        return None
    return {
        "logical_name": identity.logical_name,
        "path": str(identity.path) if identity.path is not None else None,
        "sha256": identity.sha256,
        "size_bytes": identity.size_bytes,
        "metadata": dict(identity.metadata),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _require_manifest_fields(
    manifest: Mapping[str, Any],
    required_fields: set[str],
) -> None:
    missing = required_fields - set(manifest)
    if missing:
        raise ValueError(
            f"Bootstrap manifest is missing required fields: {sorted(missing)}"
        )


def _readonly_mapping(values: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(
        {
            str(key): _readonly_mapping(value)
            if isinstance(value, Mapping)
            else tuple(value)
            if isinstance(value, list)
            else value
            for key, value in values.items()
        }
    )


def _plain_mapping(values: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): _plain_mapping(value)
        if isinstance(value, Mapping)
        else list(value)
        if isinstance(value, tuple)
        else value
        for key, value in values.items()
    }
