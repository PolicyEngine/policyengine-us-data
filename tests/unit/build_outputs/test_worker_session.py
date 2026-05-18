from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from policyengine_us_data.build_outputs.bootstrap import (
    WorkerBootstrapBuilder,
    WorkerBootstrapStore,
)
from policyengine_us_data.build_outputs.validation import (
    ValidationContext,
    ValidationPolicy,
)
from policyengine_us_data.build_outputs.worker_session import (
    WorkerSession,
    WorkerSessionFactory,
)
from tests.support.build_outputs.bootstrap import (
    FakeDatasetReader,
    FakeFingerprintingService,
    FakeGeographyLoader,
    make_bootstrap_test_artifacts,
)


class SessionDatasetReader(FakeDatasetReader):
    """Dataset reader fake that records raw and bootstrap source loads."""

    def __init__(self, snapshot, *, fail_with_entity_graph: bool = False):
        super().__init__(snapshot)
        self.loaded_with_entity_graph: list[tuple[Path, object]] = []
        self.fail_with_entity_graph = fail_with_entity_graph

    def load_with_entity_graph(self, dataset_path: Path, entity_graph):
        self.loaded_with_entity_graph.append((Path(dataset_path), entity_graph))
        if self.fail_with_entity_graph:
            raise RuntimeError("entity graph load failed")
        return self.snapshot


class SequenceDatasetReader(SessionDatasetReader):
    """Dataset reader fake that returns a new snapshot for each load."""

    def __init__(self, snapshots):
        super().__init__(snapshots[0])
        self.snapshots = list(snapshots)
        self.load_index = 0

    def load(self, dataset_path: Path):
        self.loaded_paths.append(Path(dataset_path))
        snapshot = self.snapshots[self.load_index]
        self.load_index += 1
        return snapshot


class SessionGeographyLoader(FakeGeographyLoader):
    """Geography loader fake that records load calls."""

    def __init__(self, artifacts):
        super().__init__(artifacts)
        self.load_calls = []

    def load(self, **kwargs):
        self.load_calls.append(kwargs)
        return super().load(**kwargs)


class FakeValidationService:
    """Validation service fake returning a prepared context."""

    def __init__(self):
        self.calls = []

    def prepare_context(self, **kwargs):
        self.calls.append(kwargs)
        policy = kwargs["policy"]
        if not policy.enabled:
            return None
        return ValidationContext(
            policy=policy,
            target_db_path=kwargs["inputs"].target_db_path,
            period=kwargs["period"],
            validation_targets=SimpleNamespace(name="targets"),
            training_mask=np.array([True]),
            constraints_map={1: []},
        )


def test_worker_session_caches_are_per_session(tmp_path):
    artifacts = make_bootstrap_test_artifacts(tmp_path / "inputs")

    first = WorkerSession(
        inputs=artifacts.inputs,
        scope="regional",
        source=artifacts.snapshot,
        weights=SimpleNamespace(values=np.ones(2), n_records=1, n_clones=2),
        geography=SimpleNamespace(n_records=1, n_clones=2),
    )
    second = WorkerSession(
        inputs=artifacts.inputs,
        scope="regional",
        source=artifacts.snapshot,
        weights=SimpleNamespace(values=np.ones(2), n_records=1, n_clones=2),
        geography=SimpleNamespace(n_records=1, n_clones=2),
    )

    first.caches["marker"] = "first"

    assert second.caches == {}


def test_worker_session_factory_uses_raw_loaders_without_bootstrap(tmp_path):
    artifacts = make_bootstrap_test_artifacts(tmp_path / "inputs")
    dataset_reader = SessionDatasetReader(artifacts.snapshot)
    geography_loader = SessionGeographyLoader(artifacts)
    validation_service = FakeValidationService()

    session = WorkerSessionFactory(
        dataset_reader=dataset_reader,
        geography_loader=geography_loader,
        validation_service=validation_service,
    ).create(
        inputs=artifacts.inputs,
        scope="regional",
        validation_policy=ValidationPolicy(),
        period=2024,
    )

    assert session.bootstrap_status == "unavailable"
    assert session.bootstrap_bundle is None
    assert dataset_reader.loaded_paths == [artifacts.inputs.source_dataset_path]
    assert dataset_reader.loaded_with_entity_graph == []
    assert session.weights.n_records == artifacts.n_records
    assert session.weights.n_clones == artifacts.n_clones
    assert geography_loader.load_calls[0]["n_records"] == artifacts.n_records
    assert validation_service.calls[0]["inputs"] == artifacts.inputs


def test_worker_session_factory_raw_source_loader_returns_fresh_snapshots(tmp_path):
    first = make_bootstrap_test_artifacts(tmp_path / "first")
    second = make_bootstrap_test_artifacts(tmp_path / "second")
    third = make_bootstrap_test_artifacts(tmp_path / "third")
    dataset_reader = SequenceDatasetReader(
        (first.snapshot, second.snapshot, third.snapshot)
    )

    session = WorkerSessionFactory(
        dataset_reader=dataset_reader,
        geography_loader=SessionGeographyLoader(first),
        validation_service=FakeValidationService(),
    ).create(
        inputs=first.inputs,
        scope="regional",
        validation_policy=ValidationPolicy(enabled=False),
        period=2024,
    )

    assert session.source is first.snapshot
    assert session.load_source() is second.snapshot
    assert session.load_source() is third.snapshot
    assert dataset_reader.loaded_paths == [
        first.inputs.source_dataset_path,
        first.inputs.source_dataset_path,
        first.inputs.source_dataset_path,
    ]


def test_worker_session_factory_prefers_bootstrap_entity_graph(tmp_path):
    artifacts = make_bootstrap_test_artifacts(tmp_path / "inputs")
    store = WorkerBootstrapStore(tmp_path / "artifacts")
    WorkerBootstrapBuilder(
        dataset_reader=FakeDatasetReader(artifacts.snapshot),
        geography_loader=FakeGeographyLoader(artifacts),
        fingerprinting_service=FakeFingerprintingService(),
    ).build(
        inputs=artifacts.inputs,
        scope="regional",
        artifacts_dir=store.artifacts_dir,
    )
    dataset_reader = SessionDatasetReader(artifacts.snapshot)

    session = WorkerSessionFactory(
        dataset_reader=dataset_reader,
        geography_loader=SessionGeographyLoader(artifacts),
        validation_service=FakeValidationService(),
        bootstrap_store=store,
    ).create(
        inputs=artifacts.inputs,
        scope="regional",
        validation_policy=ValidationPolicy(),
        period=2024,
        expected_scope_fingerprint="regional-fingerprint",
    )

    assert session.bootstrap_status == "used"
    assert session.bootstrap_bundle is not None
    assert dataset_reader.loaded_paths == []
    assert dataset_reader.loaded_with_entity_graph[0][0] == (
        artifacts.inputs.source_dataset_path
    )


def test_worker_session_factory_requires_expected_fingerprint_for_bootstrap(
    tmp_path,
):
    artifacts = make_bootstrap_test_artifacts(tmp_path / "inputs")
    store = WorkerBootstrapStore(tmp_path / "artifacts")
    WorkerBootstrapBuilder(
        dataset_reader=FakeDatasetReader(artifacts.snapshot),
        geography_loader=FakeGeographyLoader(artifacts),
        fingerprinting_service=FakeFingerprintingService(),
    ).build(
        inputs=artifacts.inputs,
        scope="regional",
        artifacts_dir=store.artifacts_dir,
    )
    dataset_reader = SessionDatasetReader(artifacts.snapshot)

    session = WorkerSessionFactory(
        dataset_reader=dataset_reader,
        geography_loader=SessionGeographyLoader(artifacts),
        validation_service=FakeValidationService(),
        bootstrap_store=store,
    ).create(
        inputs=artifacts.inputs,
        scope="regional",
        validation_policy=ValidationPolicy(),
        period=2024,
    )

    assert session.bootstrap_status == "fallback"
    assert session.bootstrap_bundle is None
    assert dataset_reader.loaded_paths == [artifacts.inputs.source_dataset_path]
    assert "expected scope fingerprint" in session.caches["bootstrap_error"]


def test_worker_session_factory_falls_back_when_bootstrap_source_load_fails(tmp_path):
    artifacts = make_bootstrap_test_artifacts(tmp_path / "inputs")
    store = WorkerBootstrapStore(tmp_path / "artifacts")
    WorkerBootstrapBuilder(
        dataset_reader=FakeDatasetReader(artifacts.snapshot),
        geography_loader=FakeGeographyLoader(artifacts),
        fingerprinting_service=FakeFingerprintingService(),
    ).build(
        inputs=artifacts.inputs,
        scope="regional",
        artifacts_dir=store.artifacts_dir,
    )
    dataset_reader = SessionDatasetReader(
        artifacts.snapshot,
        fail_with_entity_graph=True,
    )

    session = WorkerSessionFactory(
        dataset_reader=dataset_reader,
        geography_loader=SessionGeographyLoader(artifacts),
        validation_service=FakeValidationService(),
        bootstrap_store=store,
    ).create(
        inputs=artifacts.inputs,
        scope="regional",
        validation_policy=ValidationPolicy(),
        period=2024,
        expected_scope_fingerprint="regional-fingerprint",
    )

    assert session.bootstrap_status == "fallback"
    assert session.bootstrap_bundle is None
    assert dataset_reader.loaded_paths == [artifacts.inputs.source_dataset_path]
    assert "entity graph load failed" in session.caches["bootstrap_error"]


def test_worker_session_factory_falls_back_when_bootstrap_inputs_mismatch(tmp_path):
    artifacts = make_bootstrap_test_artifacts(tmp_path / "inputs")
    store = WorkerBootstrapStore(tmp_path / "artifacts")
    WorkerBootstrapBuilder(
        dataset_reader=FakeDatasetReader(artifacts.snapshot),
        geography_loader=FakeGeographyLoader(artifacts),
        fingerprinting_service=FakeFingerprintingService(),
    ).build(
        inputs=artifacts.inputs,
        scope="regional",
        artifacts_dir=store.artifacts_dir,
    )
    changed_inputs = type(artifacts.inputs)(
        weights_path=artifacts.inputs.weights_path,
        source_dataset_path=artifacts.inputs.source_dataset_path,
        target_db_path=artifacts.inputs.target_db_path,
        exact_geography_path=artifacts.inputs.exact_geography_path,
        calibration_package_path=artifacts.inputs.calibration_package_path,
        run_config_path=artifacts.inputs.run_config_path,
        run_id="different-run",
        version=artifacts.inputs.version,
        n_clones=artifacts.inputs.n_clones,
        seed=artifacts.inputs.seed,
        legacy_blocks_path=artifacts.inputs.legacy_blocks_path,
    )
    dataset_reader = SessionDatasetReader(artifacts.snapshot)

    session = WorkerSessionFactory(
        dataset_reader=dataset_reader,
        geography_loader=SessionGeographyLoader(artifacts),
        validation_service=FakeValidationService(),
        bootstrap_store=store,
    ).create(
        inputs=changed_inputs,
        scope="regional",
        validation_policy=ValidationPolicy(),
        period=2024,
    )

    assert session.bootstrap_status == "fallback"
    assert session.bootstrap_bundle is None
    assert dataset_reader.loaded_paths == [artifacts.inputs.source_dataset_path]
    assert "does not match worker run_id" in session.caches["bootstrap_error"]


def test_worker_session_factory_falls_back_when_bootstrap_fingerprint_mismatch(
    tmp_path,
):
    artifacts = make_bootstrap_test_artifacts(tmp_path / "inputs")
    store = WorkerBootstrapStore(tmp_path / "artifacts")
    WorkerBootstrapBuilder(
        dataset_reader=FakeDatasetReader(artifacts.snapshot),
        geography_loader=FakeGeographyLoader(artifacts),
        fingerprinting_service=FakeFingerprintingService(),
    ).build(
        inputs=artifacts.inputs,
        scope="regional",
        artifacts_dir=store.artifacts_dir,
    )
    dataset_reader = SessionDatasetReader(artifacts.snapshot)

    session = WorkerSessionFactory(
        dataset_reader=dataset_reader,
        geography_loader=SessionGeographyLoader(artifacts),
        validation_service=FakeValidationService(),
        bootstrap_store=store,
    ).create(
        inputs=artifacts.inputs,
        scope="regional",
        validation_policy=ValidationPolicy(),
        period=2024,
        expected_scope_fingerprint="changed-fingerprint",
    )

    assert session.bootstrap_status == "fallback"
    assert session.bootstrap_bundle is None
    assert dataset_reader.loaded_paths == [artifacts.inputs.source_dataset_path]
    assert "does not match expected fingerprint" in session.caches["bootstrap_error"]


def test_worker_session_factory_marks_corrupt_bootstrap_as_fallback(tmp_path):
    artifacts = make_bootstrap_test_artifacts(tmp_path / "inputs")
    store = WorkerBootstrapStore(tmp_path / "artifacts")
    store.scope_dir("regional").mkdir(parents=True)
    store.manifest_path("regional").write_text("{not-json")
    dataset_reader = SessionDatasetReader(artifacts.snapshot)

    session = WorkerSessionFactory(
        dataset_reader=dataset_reader,
        geography_loader=SessionGeographyLoader(artifacts),
        validation_service=FakeValidationService(),
        bootstrap_store=store,
    ).create(
        inputs=artifacts.inputs,
        scope="regional",
        validation_policy=ValidationPolicy(),
        period=2024,
    )

    assert session.bootstrap_status == "fallback"
    assert session.bootstrap_bundle is None
    assert dataset_reader.loaded_paths == [artifacts.inputs.source_dataset_path]
    assert "Expecting property name" in session.caches["bootstrap_error"]


def test_worker_session_factory_rejects_weight_clone_mismatch(tmp_path):
    artifacts = make_bootstrap_test_artifacts(tmp_path / "inputs", n_clones=2)
    bad_inputs = type(artifacts.inputs)(
        weights_path=artifacts.inputs.weights_path,
        source_dataset_path=artifacts.inputs.source_dataset_path,
        target_db_path=artifacts.inputs.target_db_path,
        exact_geography_path=artifacts.inputs.exact_geography_path,
        calibration_package_path=artifacts.inputs.calibration_package_path,
        run_config_path=artifacts.inputs.run_config_path,
        run_id=artifacts.inputs.run_id,
        version=artifacts.inputs.version,
        n_clones=3,
        seed=artifacts.inputs.seed,
        legacy_blocks_path=artifacts.inputs.legacy_blocks_path,
    )

    with pytest.raises(ValueError, match="expected 3"):
        WorkerSessionFactory(
            dataset_reader=SessionDatasetReader(artifacts.snapshot),
            geography_loader=SessionGeographyLoader(artifacts),
            validation_service=FakeValidationService(),
        ).create(
            inputs=bad_inputs,
            scope="regional",
            validation_policy=ValidationPolicy(enabled=False),
            period=2024,
        )
