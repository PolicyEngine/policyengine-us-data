"""Worker-scoped local H5 setup contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

from policyengine_us_data.pipeline_metadata import pipeline_node

from .bootstrap import (
    BootstrapScope,
    WorkerBootstrapBundle,
    WorkerBootstrapStore,
    load_entity_graph,
)
from .fingerprinting import PublishingInputBundle
from .geography_loader import CalibrationGeographyLoader
from .source_dataset import PolicyEngineDatasetReader, SourceDatasetSnapshot
from .validation import AreaValidationService, ValidationContext, ValidationPolicy
from .weights import CloneWeightMatrix

BootstrapStatus = Literal["used", "fallback", "unavailable"]

__all__ = [
    "BootstrapStatus",
    "WorkerSession",
    "WorkerSessionFactory",
]


@pipeline_node(
    id="local_h5_worker_session",
    label="WorkerSession",
    node_type="library",
    description="Worker-scoped local H5 setup state reused across queued requests.",
    source_file="policyengine_us_data/build_outputs/worker_session.py",
    status="current",
    stability="moving",
    pathways=["local_h5"],
    artifacts_in=[
        "calibration_weights.npy",
        "source_imputed_stratified_extended_cps.h5",
        "geography_assignment.npz",
        "policy_data.db",
        "worker_bootstrap.json",
    ],
    validation_commands=[
        "uv run pytest tests/unit/build_outputs/test_worker_session.py"
    ],
)
@dataclass
class WorkerSession:
    """Prepared local H5 state for one worker process."""

    inputs: PublishingInputBundle
    scope: BootstrapScope
    source: SourceDatasetSnapshot
    weights: CloneWeightMatrix
    geography: Any
    validation_context: ValidationContext | None = None
    bootstrap_bundle: WorkerBootstrapBundle | None = None
    bootstrap_status: BootstrapStatus = "unavailable"
    caches: dict[str, Any] = field(default_factory=dict)


@pipeline_node(
    id="local_h5_worker_session_factory",
    label="WorkerSessionFactory",
    node_type="library",
    description="Load local H5 source, weights, geography, and validation context once per worker.",
    source_file="policyengine_us_data/build_outputs/worker_session.py",
    status="current",
    stability="moving",
    pathways=["local_h5"],
    artifacts_in=[
        "calibration_weights.npy",
        "source_imputed_stratified_extended_cps.h5",
        "geography_assignment.npz",
        "policy_data.db",
        "bootstrap/{scope}/worker_bootstrap.json",
        "bootstrap/{scope}/entity_graph.npz",
    ],
    validation_commands=[
        "uv run pytest tests/unit/build_outputs/test_worker_session.py",
        "uv run pytest tests/integration/build_outputs/h5_worker_runtime/test_worker_script_tiny_fixture.py",
    ],
)
class WorkerSessionFactory:
    """Build worker-scoped setup from raw inputs or persisted bootstrap facts."""

    def __init__(
        self,
        *,
        dataset_reader: PolicyEngineDatasetReader | None = None,
        geography_loader: CalibrationGeographyLoader | None = None,
        validation_service: AreaValidationService | None = None,
        bootstrap_store: WorkerBootstrapStore | None = None,
    ) -> None:
        """Create a session factory with injectable seams for tests."""

        self._dataset_reader = dataset_reader or PolicyEngineDatasetReader()
        self._geography_loader = geography_loader or CalibrationGeographyLoader()
        self._validation_service = validation_service or AreaValidationService()
        self._bootstrap_store = bootstrap_store

    def create(
        self,
        *,
        inputs: PublishingInputBundle,
        scope: BootstrapScope,
        validation_policy: ValidationPolicy | None = None,
        period: int = 2024,
        target_config_path: Path | None = None,
        validation_config_path: Path | None = None,
        artifacts_dir: Path | None = None,
    ) -> WorkerSession:
        """Create a worker session for one local H5 scope.

        Bootstrap artifacts are preferred when present. If they are missing,
        stale, or unreadable, the factory falls back to raw source loaders so
        rollout can remain dual-path until the bootstrap contract is mandatory.
        """

        bootstrap_store = self._bootstrap_store
        if bootstrap_store is None and artifacts_dir is not None:
            bootstrap_store = WorkerBootstrapStore(artifacts_dir)

        bundle, bootstrap_error = self._load_bootstrap(
            bootstrap_store=bootstrap_store,
            scope=scope,
        )
        if bundle is not None:
            bootstrap_error = self._validate_bootstrap_bundle(
                bundle=bundle,
                inputs=inputs,
                scope=scope,
            )
            if bootstrap_error is not None:
                bundle = None
        source, bootstrap_status, source_error = self._load_source(
            inputs=inputs,
            bundle=bundle,
        )
        if bootstrap_error is not None and bootstrap_status == "unavailable":
            bootstrap_status = "fallback"
        fallback_error = source_error or bootstrap_error

        weights = self._load_weights(inputs=inputs, source=source)
        geography = self._geography_loader.load(
            weights_path=inputs.weights_path,
            n_records=weights.n_records,
            n_clones=weights.n_clones,
            geography_path=inputs.exact_geography_path,
            blocks_path=inputs.legacy_blocks_path,
            calibration_package_path=inputs.calibration_package_path,
        )

        policy = validation_policy or ValidationPolicy()
        validation_context = self._validation_service.prepare_context(
            inputs=inputs,
            policy=policy,
            period=period,
            target_config_path=target_config_path,
            validation_config_path=validation_config_path,
        )

        caches: dict[str, Any] = {}
        if fallback_error is not None:
            caches["bootstrap_error"] = str(fallback_error)

        return WorkerSession(
            inputs=inputs,
            scope=scope,
            source=source,
            weights=weights,
            geography=geography,
            validation_context=validation_context,
            bootstrap_bundle=bundle if bootstrap_status == "used" else None,
            bootstrap_status=bootstrap_status,
            caches=caches,
        )

    def _load_bootstrap(
        self,
        *,
        bootstrap_store: WorkerBootstrapStore | None,
        scope: BootstrapScope,
    ) -> tuple[WorkerBootstrapBundle | None, Exception | None]:
        if bootstrap_store is None:
            return None, None

        manifest_path = getattr(bootstrap_store, "manifest_path", None)
        manifest_exists = False
        if callable(manifest_path):
            manifest_exists = Path(manifest_path(scope)).exists()
            if not manifest_exists:
                return None, None

        try:
            return bootstrap_store.load(scope), None
        except FileNotFoundError as exc:
            return None, exc if manifest_exists else None
        except Exception as exc:
            return None, exc

    def _validate_bootstrap_bundle(
        self,
        *,
        bundle: WorkerBootstrapBundle,
        inputs: PublishingInputBundle,
        scope: BootstrapScope,
    ) -> Exception | None:
        try:
            self._raise_for_bootstrap_mismatch(
                bundle=bundle,
                inputs=inputs,
                scope=scope,
            )
        except Exception as exc:
            return exc
        return None

    def _raise_for_bootstrap_mismatch(
        self,
        *,
        bundle: WorkerBootstrapBundle,
        inputs: PublishingInputBundle,
        scope: BootstrapScope,
    ) -> None:
        if bundle.run_id != inputs.run_id:
            raise ValueError(
                f"Bootstrap run_id {bundle.run_id!r} does not match "
                f"worker run_id {inputs.run_id!r}"
            )
        if bundle.scope != scope:
            raise ValueError(
                f"Bootstrap scope {bundle.scope!r} does not match "
                f"worker scope {scope!r}"
            )

        expected_paths = {
            "weights": inputs.weights_path,
            "source_dataset": inputs.source_dataset_path,
            "exact_geography": inputs.exact_geography_path,
            "target_db": inputs.target_db_path,
            "calibration_package": inputs.calibration_package_path,
            "run_config": inputs.run_config_path,
        }
        for logical_name, expected_path in expected_paths.items():
            _assert_manifest_path_matches(
                logical_name=logical_name,
                expected_path=expected_path,
                manifest_identity=bundle.inputs.get(logical_name),
            )

        _assert_summary_field_matches(
            section="weights",
            field="n_clones",
            expected=inputs.n_clones,
            actual=bundle.weights.get("n_clones"),
        )

    def _load_source(
        self,
        *,
        inputs: PublishingInputBundle,
        bundle: WorkerBootstrapBundle | None,
    ) -> tuple[SourceDatasetSnapshot, BootstrapStatus, Exception | None]:
        if bundle is not None:
            try:
                entity_graph = load_entity_graph(bundle.entity_graph_path)
                load_with_entity_graph = getattr(
                    self._dataset_reader,
                    "load_with_entity_graph",
                )
                return (
                    load_with_entity_graph(
                        inputs.source_dataset_path,
                        entity_graph,
                    ),
                    "used",
                    None,
                )
            except Exception as exc:
                source = self._dataset_reader.load(inputs.source_dataset_path)
                return source, "fallback", exc

        source = self._dataset_reader.load(inputs.source_dataset_path)
        return source, "unavailable", None

    def _load_weights(
        self,
        *,
        inputs: PublishingInputBundle,
        source: SourceDatasetSnapshot,
    ) -> CloneWeightMatrix:
        weights_array = np.load(inputs.weights_path)
        weights = CloneWeightMatrix.from_vector(
            weights_array,
            n_records=source.n_households,
        )
        if inputs.n_clones is not None and weights.n_clones != int(inputs.n_clones):
            raise ValueError(
                f"Weight vector implies n_clones={weights.n_clones}, "
                f"expected {inputs.n_clones}"
            )
        return weights


def _assert_manifest_path_matches(
    *,
    logical_name: str,
    expected_path: Path | None,
    manifest_identity,
) -> None:
    if expected_path is None:
        if manifest_identity is not None:
            raise ValueError(
                f"Bootstrap {logical_name} identity presence does not match "
                "current inputs"
            )
        return

    if manifest_identity is None:
        raise ValueError(
            f"Bootstrap {logical_name} identity presence does not match current inputs"
        )

    actual_path = manifest_identity.get("path")
    if actual_path is None or Path(actual_path) != Path(expected_path):
        raise ValueError(f"Bootstrap {logical_name} path does not match current inputs")


def _assert_summary_field_matches(
    *,
    section: str,
    field: str,
    expected,
    actual,
) -> None:
    if expected is None:
        return
    if actual != expected:
        raise ValueError(
            f"Bootstrap {section}.{field} {actual!r} does not match "
            f"current value {expected!r}"
        )
