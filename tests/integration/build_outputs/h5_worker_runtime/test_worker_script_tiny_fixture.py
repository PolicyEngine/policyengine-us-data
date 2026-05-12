from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from policyengine_us_data.build_outputs.bootstrap import WorkerBootstrapBuilder
from policyengine_us_data.build_outputs.fingerprinting import PublishingInputBundle
from policyengine_us_data.build_outputs.source_dataset import (
    DEFAULT_SUBENTITIES,
    PolicyEngineDatasetReader,
)
from policyengine_us_data.build_outputs.weights import CloneWeightMatrix
from tests.integration.build_outputs.fixtures import (
    build_request,
    seed_local_h5_artifacts,
)

pytestmark = pytest.mark.integration

pytest.importorskip("policyengine_us")


def _require_worker_dependencies() -> None:
    pytest.importorskip("scipy")
    pytest.importorskip("spm_calculator")


def _run_worker(
    *,
    requests,
    artifacts,
    output_dir: Path,
    use_saved_geography: bool = False,
    use_package_geography: bool = False,
    validate: bool = False,
    target_config: Path | None = None,
    validation_config: Path | None = None,
    run_id: str = "tiny-worker-run",
    scope: str = "regional",
    artifacts_dir: Path | None = None,
    return_process: bool = False,
) -> dict:
    _require_worker_dependencies()
    if not isinstance(requests, (list, tuple)):
        requests = (requests,)
    cmd = [
        sys.executable,
        "-m",
        "modal_app.worker_script",
        "--requests-json",
        json.dumps([request.to_dict() for request in requests]),
        "--weights-path",
        str(artifacts.weights_path),
        "--dataset-path",
        str(artifacts.dataset_path),
        "--db-path",
        str(artifacts.db_path),
        "--output-dir",
        str(output_dir),
        "--scope",
        scope,
        "--run-id",
        run_id,
        "--run-config-path",
        str(artifacts.run_config_path),
        "--n-clones",
        str(artifacts.n_clones),
    ]
    if artifacts_dir is not None:
        cmd.extend(["--artifacts-dir", str(artifacts_dir)])
    if not validate:
        cmd.append("--no-validate")
    if target_config is not None:
        cmd.extend(["--target-config", str(target_config)])
    if validation_config is not None:
        cmd.extend(["--validation-config", str(validation_config)])
    if use_saved_geography:
        cmd.extend(["--geography-path", str(artifacts.geography_path)])
    if use_package_geography:
        cmd.extend(
            [
                "--calibration-package-path",
                str(artifacts.calibration_package_path),
            ]
        )

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=True,
    )
    if return_process:
        return result
    return json.loads(result.stdout)


def test_tiny_fixture_source_snapshot_matches_worker_artifacts(tmp_path):
    artifacts = seed_local_h5_artifacts(tmp_path / "source-snapshot")

    snapshot = PolicyEngineDatasetReader().load(artifacts.dataset_path)
    weights = CloneWeightMatrix.from_vector(
        np.load(artifacts.weights_path),
        n_records=snapshot.n_households,
    )

    assert snapshot.n_households == artifacts.n_records
    assert snapshot.variable_provider.get_array(
        "household_id",
        snapshot.time_period,
    ).shape == (artifacts.n_records,)
    assert weights.n_records == snapshot.n_households
    assert weights.n_clones == artifacts.n_clones

    assert set(snapshot.entity_graph.subentity_ids) == set(DEFAULT_SUBENTITIES)
    assert len(snapshot.entity_graph.household_to_person_indices) == (
        artifacts.n_records
    )
    for entity_key in DEFAULT_SUBENTITIES:
        assert (
            len(snapshot.entity_graph.household_to_subentity_indices[entity_key])
            == artifacts.n_records
        )


def test_worker_builds_district_h5_from_saved_geography(tmp_path):
    artifacts = seed_local_h5_artifacts(tmp_path / "district")
    request = build_request("district", geography=artifacts.geography)
    output_dir = tmp_path / "district-out"

    result = _run_worker(
        requests=request,
        artifacts=artifacts,
        output_dir=output_dir,
        use_saved_geography=True,
    )

    assert result["failed"] == []
    assert result["errors"] == []
    assert result["completed"] == [f"district:{request.area_id}"]
    assert (output_dir / request.output_relative_path).exists()


def test_worker_builds_state_h5_from_package_geography(tmp_path):
    artifacts = seed_local_h5_artifacts(tmp_path / "state")
    request = build_request("state", geography=artifacts.geography)
    output_dir = tmp_path / "state-out"

    result = _run_worker(
        requests=request,
        artifacts=artifacts,
        output_dir=output_dir,
        use_package_geography=True,
    )

    assert result["failed"] == []
    assert result["errors"] == []
    assert result["completed"] == [f"state:{request.area_id}"]
    assert (output_dir / request.output_relative_path).exists()


def test_worker_builds_national_h5_from_package_geography(tmp_path):
    artifacts = seed_local_h5_artifacts(tmp_path / "national")
    request = build_request("national", geography=artifacts.geography)
    output_dir = tmp_path / "national-out"

    result = _run_worker(
        requests=request,
        artifacts=artifacts,
        output_dir=output_dir,
        use_package_geography=True,
        scope="national",
    )

    assert result["failed"] == []
    assert result["errors"] == []
    assert result["completed"] == ["national:US"]
    assert (output_dir / request.output_relative_path).exists()


def test_worker_validation_runs_for_tiny_district_state_and_national_h5s(tmp_path):
    artifacts = seed_local_h5_artifacts(tmp_path / "validated")
    requests = (
        build_request("district", geography=artifacts.geography),
        build_request("state", geography=artifacts.geography),
        build_request("national", geography=artifacts.geography),
    )
    output_dir = tmp_path / "validated-out"
    target_config = tmp_path / "target_config.yaml"
    validation_config = tmp_path / "validation_config.yaml"
    config = """
include:
  - variable: household_count
""".strip()
    target_config.write_text(config)
    validation_config.write_text(config)

    result = _run_worker(
        requests=requests,
        artifacts=artifacts,
        output_dir=output_dir,
        use_saved_geography=True,
        validate=True,
        target_config=target_config,
        validation_config=validation_config,
        return_process=True,
    )
    parsed = json.loads(result.stdout)

    assert result.stderr.count("Worker session ready:") == 1
    assert parsed["failed"] == []
    assert parsed["errors"] == []
    assert parsed["completed"] == ["district:NC-01", "state:NC", "national:US"]
    assert len(parsed["validation_rows"]) == 3
    assert set(parsed["validation_summary"]) == {
        "district:NC-01",
        "state:NC",
        "national:US",
    }
    for summary in parsed["validation_summary"].values():
        assert summary["n_targets"] == 1
        assert summary["n_sanity_fail"] == 0
    for row in parsed["validation_rows"]:
        assert row["variable"] == "household_count"
        assert row["sanity_check"] == "PASS"
        assert row["in_training"] is True


def test_worker_consumes_scope_bootstrap_when_available(tmp_path):
    artifacts = seed_local_h5_artifacts(tmp_path / "bootstrap")
    request = build_request("district", geography=artifacts.geography)
    output_dir = tmp_path / "bootstrap-out"
    artifacts_dir = tmp_path / "pipeline-artifacts" / "run-123"
    inputs = PublishingInputBundle(
        weights_path=artifacts.weights_path,
        source_dataset_path=artifacts.dataset_path,
        target_db_path=artifacts.db_path,
        exact_geography_path=artifacts.geography_path,
        calibration_package_path=artifacts.calibration_package_path,
        run_config_path=artifacts.run_config_path,
        run_id="run-123",
        version="0.0.0",
        n_clones=artifacts.n_clones,
        seed=42,
    )
    WorkerBootstrapBuilder().build(
        inputs=inputs,
        scope="regional",
        artifacts_dir=artifacts_dir,
    )

    result = _run_worker(
        requests=request,
        artifacts=artifacts,
        output_dir=output_dir,
        use_saved_geography=True,
        use_package_geography=True,
        run_id="run-123",
        artifacts_dir=artifacts_dir,
        return_process=True,
    )
    parsed = json.loads(result.stdout)

    assert "Worker session ready: scope=regional, bootstrap=used" in result.stderr
    assert parsed["failed"] == []
    assert parsed["errors"] == []
    assert parsed["completed"] == [f"district:{request.area_id}"]
