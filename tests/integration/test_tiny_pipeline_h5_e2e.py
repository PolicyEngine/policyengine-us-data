from __future__ import annotations

from pathlib import Path

import h5py
import pytest

from policyengine_us_data.calibration.local_h5.fingerprinting import (
    FingerprintingService,
)
from policyengine_us_data.utils.manifest import generate_manifest, verify_manifest
from tests.integration.support.pipeline_workspace import TinyPipelineWorkspace
from tests.integration.support.tiny_h5 import (
    build_h5_request,
    build_publishing_input_bundle,
    create_tiny_h5_artifacts,
    run_local_h5_worker,
)
from tests.integration.support.tiny_pipeline import create_tiny_pipeline_artifacts

pytestmark = pytest.mark.integration

pytest.importorskip("scipy")
pytest.importorskip("spm_calculator")


def test_tiny_pipeline_stage_5_outputs_continue_into_local_h5s(tmp_path):
    workspace = TinyPipelineWorkspace.create(tmp_path / "tiny-pipeline")
    pipeline_artifacts = create_tiny_pipeline_artifacts(workspace)
    h5_artifacts = create_tiny_h5_artifacts(workspace, pipeline_artifacts)
    run_id = "tiny-run-001"
    run_dir = workspace.h5_staging / run_id

    requests = (
        build_h5_request("district"),
        build_h5_request("state"),
        build_h5_request("national"),
    )

    result = run_local_h5_worker(
        requests=requests,
        artifacts=h5_artifacts,
        output_dir=run_dir,
        use_saved_geography=True,
        use_package_geography=False,
    )

    assert result["failed"] == []
    assert result["errors"] == []
    assert result["completed"] == ["district:NC-01", "state:NC", "national:US"]

    for request in requests:
        h5_path = run_dir / request.output_relative_path
        assert h5_path.exists()
        _assert_h5_contract(h5_path)

    manifest = generate_manifest(
        workspace.h5_staging,
        run_id,
        version="0.0.0",
        categories=["states", "districts", "national"],
    )
    verification = verify_manifest(workspace.h5_staging, manifest, subdir=run_id)

    assert sorted(manifest["files"]) == [
        "districts/NC-01.h5",
        "national/US.h5",
        "states/NC.h5",
    ]
    assert manifest["totals"]["states"] == 1
    assert manifest["totals"]["districts"] == 1
    assert manifest["totals"]["national"] == 1
    assert verification == {
        "valid": True,
        "missing": [],
        "checksum_mismatch": [],
        "verified": 3,
    }

    fingerprints = _scope_fingerprints(h5_artifacts, run_id=run_id)
    assert set(fingerprints) == {"regional", "national"}
    assert fingerprints["regional"] != fingerprints["national"]
    assert all(len(value) == 16 for value in fingerprints.values())


def _assert_h5_contract(path: Path) -> None:
    with h5py.File(path, mode="r") as h5:
        for variable in (
            "household_id",
            "person_id",
            "household_weight",
            "state_fips",
            "congressional_district_geoid",
        ):
            assert variable in h5
            assert "2024" in h5[variable]
            assert len(h5[variable]["2024"]) > 0


def _scope_fingerprints(h5_artifacts, *, run_id: str) -> dict[str, str]:
    service = FingerprintingService()
    fingerprints = {}
    for scope in ("regional", "national"):
        inputs = build_publishing_input_bundle(
            h5_artifacts,
            run_id=run_id,
            scope=scope,
        )
        traceability = service.build_traceability(inputs=inputs, scope=scope)
        assert traceability.metadata["run_id"] == run_id
        assert traceability.weights.path == h5_artifacts.weights_path
        assert traceability.source_dataset.path == h5_artifacts.dataset_path
        fingerprints[scope] = service.compute_scope_fingerprint(traceability)
    return fingerprints
