import json

import numpy as np
import pytest

from policyengine_us_data.build_outputs.bootstrap import (
    BOOTSTRAP_MANIFEST_FILENAME,
    ENTITY_GRAPH_FILENAME,
    WorkerBootstrapBuilder,
    WorkerBootstrapBundle,
    WorkerBootstrapStore,
    load_entity_graph,
    save_entity_graph,
)
from policyengine_us_data.build_outputs.source_dataset import EntityGraph
from tests.support.build_outputs.bootstrap import (
    FakeDatasetReader,
    FakeFingerprintingService,
    FakeGeographyLoader,
    make_bootstrap_test_artifacts,
)


def test_entity_graph_round_trips_through_npz(tmp_path):
    artifacts = make_bootstrap_test_artifacts(tmp_path)
    source_graph = artifacts.snapshot.entity_graph
    path = tmp_path / "entity_graph.npz"

    save_entity_graph(source_graph, path)
    loaded = load_entity_graph(path)

    assert np.array_equal(loaded.household_ids, source_graph.household_ids)
    assert np.array_equal(
        loaded.person_household_ids,
        source_graph.person_household_ids,
    )
    assert set(loaded.subentity_ids) == set(source_graph.subentity_ids)
    for entity_key in source_graph.subentity_ids:
        assert np.array_equal(
            loaded.subentity_ids[entity_key],
            source_graph.subentity_ids[entity_key],
        )
        assert np.array_equal(
            loaded.person_subentity_ids[entity_key],
            source_graph.person_subentity_ids[entity_key],
        )


def test_load_entity_graph_rejects_missing_structural_fields(tmp_path):
    path = tmp_path / "bad_entity_graph.npz"
    np.savez(path, household_ids=np.array([1]))

    with pytest.raises(ValueError, match="missing fields"):
        load_entity_graph(path)


def test_worker_bootstrap_builder_persists_manifest_and_entity_graph(tmp_path):
    artifacts = make_bootstrap_test_artifacts(tmp_path / "inputs")
    store = WorkerBootstrapStore(tmp_path / "artifacts")
    reader = FakeDatasetReader(artifacts.snapshot)

    bundle = WorkerBootstrapBuilder(
        dataset_reader=reader,
        geography_loader=FakeGeographyLoader(artifacts),
        fingerprinting_service=FakeFingerprintingService(),
    ).build(
        inputs=artifacts.inputs,
        scope="regional",
        artifacts_dir=store.artifacts_dir,
    )

    assert reader.loaded_paths == [artifacts.inputs.source_dataset_path]
    assert bundle.run_id == "run-123"
    assert bundle.scope == "regional"
    assert bundle.manifest_path == store.manifest_path("regional")
    assert bundle.entity_graph_path == store.entity_graph_path("regional")
    assert bundle.manifest_path.exists()
    assert bundle.entity_graph_path.exists()

    manifest = json.loads(bundle.manifest_path.read_text())
    assert manifest["schema_version"] == 1
    assert manifest["created_by"] == "WorkerBootstrapBuilder"
    assert manifest["artifacts"] == {
        "manifest": BOOTSTRAP_MANIFEST_FILENAME,
        "entity_graph": ENTITY_GRAPH_FILENAME,
    }
    assert manifest["source_dataset"]["n_households"] == artifacts.n_records
    assert manifest["source_dataset"]["entity_graph_artifact"] == ENTITY_GRAPH_FILENAME
    assert manifest["weights"]["n_records"] == artifacts.n_records
    assert manifest["weights"]["n_clones"] == artifacts.n_clones
    assert manifest["weights"]["dtype"] == "float32"
    assert manifest["geography"]["source_kind"] == "saved_geography"
    assert manifest["geography"]["canonical_sha256"] == "sha256:canonical-geography"
    assert manifest["traceability"]["scope_fingerprint"] == "regional-fingerprint"
    assert manifest["inputs"]["weights"]["sha256"] == "sha256:weights"


def test_worker_bootstrap_store_loads_persisted_bundle(tmp_path):
    artifacts = make_bootstrap_test_artifacts(tmp_path / "inputs")
    store = WorkerBootstrapStore(tmp_path / "artifacts")
    WorkerBootstrapBuilder(
        dataset_reader=FakeDatasetReader(artifacts.snapshot),
        geography_loader=FakeGeographyLoader(artifacts),
        fingerprinting_service=FakeFingerprintingService(),
    ).build(
        inputs=artifacts.inputs,
        scope="national",
        artifacts_dir=store.artifacts_dir,
    )

    loaded = store.load("national")

    assert loaded.scope == "national"
    assert loaded.run_id == "run-123"
    assert loaded.source_dataset["n_households"] == artifacts.n_records
    assert loaded.traceability["scope_fingerprint"] == "national-fingerprint"
    roundtrip_graph = load_entity_graph(loaded.entity_graph_path)
    assert isinstance(roundtrip_graph, EntityGraph)


def test_worker_bootstrap_bundle_rejects_missing_manifest_fields(tmp_path):
    with pytest.raises(ValueError, match="missing required fields"):
        WorkerBootstrapBundle.from_manifest(
            root_dir=tmp_path,
            manifest={
                "schema_version": 1,
                "run_id": "run-123",
            },
        )


def test_worker_bootstrap_builder_rejects_weight_clone_mismatch(tmp_path):
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
        WorkerBootstrapBuilder(
            dataset_reader=FakeDatasetReader(artifacts.snapshot),
            geography_loader=FakeGeographyLoader(artifacts),
            fingerprinting_service=FakeFingerprintingService(),
        ).build(
            inputs=bad_inputs,
            scope="regional",
            artifacts_dir=tmp_path / "artifacts",
        )
