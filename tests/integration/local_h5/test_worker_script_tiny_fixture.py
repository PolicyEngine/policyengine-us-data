from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tests.integration.local_h5.fixtures import (
    build_request,
    seed_local_h5_artifacts,
)

pytest.importorskip("scipy")
pytest.importorskip("spm_calculator")


def _run_worker(
    *,
    request,
    artifacts,
    output_dir: Path,
    use_saved_geography: bool = False,
    use_package_geography: bool = False,
) -> dict:
    cmd = [
        sys.executable,
        "-m",
        "modal_app.worker_script",
        "--requests-json",
        json.dumps([request.to_dict()]),
        "--weights-path",
        str(artifacts.weights_path),
        "--dataset-path",
        str(artifacts.dataset_path),
        "--db-path",
        str(artifacts.db_path),
        "--output-dir",
        str(output_dir),
        "--n-clones",
        str(artifacts.n_clones),
        "--no-validate",
    ]
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
    return json.loads(result.stdout)


def test_worker_builds_district_h5_from_saved_geography(tmp_path):
    artifacts = seed_local_h5_artifacts(tmp_path / "district")
    request = build_request("district", geography=artifacts.geography)
    output_dir = tmp_path / "district-out"

    result = _run_worker(
        request=request,
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
        request=request,
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
        request=request,
        artifacts=artifacts,
        output_dir=output_dir,
        use_package_geography=True,
    )

    assert result["failed"] == []
    assert result["errors"] == []
    assert result["completed"] == ["national:US"]
    assert (output_dir / request.output_relative_path).exists()
