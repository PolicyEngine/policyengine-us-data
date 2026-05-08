import json

import pytest

from policyengine_us_data.build_outputs.bootstrap import (
    WorkerBootstrapBuilder,
    load_entity_graph,
)
from policyengine_us_data.build_outputs.fingerprinting import PublishingInputBundle
from tests.integration.local_h5.fixtures import (
    SEED,
    VERSION,
    seed_local_h5_artifacts,
)

pytestmark = pytest.mark.integration


def test_worker_bootstrap_builder_composes_real_local_h5_seams(tmp_path):
    pytest.importorskip("policyengine_us")

    artifacts = seed_local_h5_artifacts(tmp_path / "local-h5")
    inputs = PublishingInputBundle(
        weights_path=artifacts.weights_path,
        source_dataset_path=artifacts.dataset_path,
        target_db_path=artifacts.db_path,
        exact_geography_path=artifacts.geography_path,
        calibration_package_path=artifacts.calibration_package_path,
        run_config_path=artifacts.run_config_path,
        run_id="run-123",
        version=VERSION,
        n_clones=artifacts.n_clones,
        seed=SEED,
    )

    bundle = WorkerBootstrapBuilder().build(
        inputs=inputs,
        scope="regional",
        artifacts_dir=tmp_path / "pipeline-artifacts",
    )

    manifest = json.loads(bundle.manifest_path.read_text())
    graph = load_entity_graph(bundle.entity_graph_path)
    assert manifest["source_dataset"]["n_households"] == artifacts.n_records
    assert manifest["weights"]["n_records"] == artifacts.n_records
    assert manifest["weights"]["n_clones"] == artifacts.n_clones
    assert manifest["geography"]["source_kind"] == "saved_geography"
    assert manifest["geography"]["canonical_sha256"].startswith("sha256:")
    assert len(manifest["traceability"]["scope_fingerprint"]) == 16
    assert len(graph.household_ids) == artifacts.n_records
