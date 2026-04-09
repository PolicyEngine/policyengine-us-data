from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.io import mmwrite
from scipy.sparse import csr_matrix


REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = REPO_ROOT / "paper-l0" / "benchmarking"
BENCHMARK_CLI_PATH = BENCHMARK_DIR / "benchmark_cli.py"


def _r_package_available(package: str) -> bool:
    proc = subprocess.run(
        [
            "Rscript",
            "-e",
            f"quit(status = if (requireNamespace('{package}', quietly = TRUE)) 0 else 1)",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    return proc.returncode == 0


def _load_benchmark_cli_module():
    benchmark_dir_str = str(BENCHMARK_DIR)
    if benchmark_dir_str not in sys.path:
        sys.path.insert(0, benchmark_dir_str)
    spec = importlib.util.spec_from_file_location(
        "benchmark_cli_for_tests", BENCHMARK_CLI_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_common_inputs(
    run_dir: Path,
    matrix,
    target_values: list[float],
    variables: list[str],
    geo_levels: list[str] | None = None,
    target_names: list[str] | None = None,
    initial_weights: np.ndarray | None = None,
    method_options: dict | None = None,
) -> Path:
    inputs = run_dir / "inputs"
    outputs = run_dir / "outputs"
    inputs.mkdir(parents=True, exist_ok=True)
    outputs.mkdir(parents=True, exist_ok=True)

    mmwrite(str(inputs / "X_targets_by_units.mtx"), matrix)
    if initial_weights is None:
        initial_weights = np.ones(matrix.shape[1], dtype=np.float64)
    np.save(
        inputs / "initial_weights.npy", np.asarray(initial_weights, dtype=np.float64)
    )

    if geo_levels is None:
        geo_levels = ["national"] * len(target_values)
    if target_names is None:
        target_names = [f"target_{idx}" for idx in range(len(target_values))]

    target_metadata = pd.DataFrame(
        {
            "value": np.asarray(target_values, dtype=np.float64),
            "variable": variables,
            "geo_level": geo_levels,
            "target_name": target_names,
        }
    )
    target_metadata.to_csv(inputs / "target_metadata.csv", index=False)

    manifest = {
        "method_options": method_options or {},
    }
    with open(inputs / "benchmark_manifest.json", "w") as f:
        json.dump(manifest, f)

    return inputs


@pytest.fixture
def benchmark_cli_module(monkeypatch, tmp_path_factory):
    cache_root = tmp_path_factory.mktemp("benchmarking-cache")
    monkeypatch.setenv("MPLCONFIGDIR", str(cache_root / "mpl"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(cache_root / "xdg"))
    return _load_benchmark_cli_module()


@pytest.fixture(autouse=True)
def _require_rscript():
    if shutil.which("Rscript") is None:
        pytest.skip("Rscript is required for benchmarking runner tests")


def test_greg_runner_end_to_end_exact_fit(benchmark_cli_module, tmp_path):
    if not _r_package_available("survey"):
        pytest.skip("R package 'survey' is required for this test")

    run_dir = tmp_path / "greg-run"
    matrix = csr_matrix(np.eye(2, dtype=np.float64))
    _write_common_inputs(
        run_dir=run_dir,
        matrix=matrix,
        target_values=[2.0, 3.0],
        variables=["household_count", "person_count"],
        method_options={"greg": {"maxit": 50, "epsilon": 1e-10}},
    )

    weights_path, _ = benchmark_cli_module._run_greg(run_dir)
    fitted_weights = np.load(weights_path)

    np.testing.assert_allclose(
        fitted_weights, np.array([2.0, 3.0]), atol=1e-8, rtol=1e-8
    )
    np.testing.assert_allclose(
        matrix.dot(fitted_weights), np.array([2.0, 3.0]), atol=1e-8
    )


def test_ipf_runner_end_to_end_numeric_total_person_scope(
    benchmark_cli_module, tmp_path
):
    if not _r_package_available("surveysd"):
        pytest.skip("R package 'surveysd' is required for this test")

    run_dir = tmp_path / "ipf-numeric-run"
    matrix = csr_matrix(np.array([[1.0, 0.0]], dtype=np.float64))
    inputs = _write_common_inputs(
        run_dir=run_dir,
        matrix=matrix,
        target_values=[1.0],
        variables=["person_count"],
        method_options={
            "ipf": {"max_iter": 500, "bound": 20.0, "epsP": 1e-4, "epsH": 1e-4}
        },
    )

    unit_metadata = pd.DataFrame(
        {
            "unit_index": [0, 0, 1],
            "household_id": [0, 0, 1],
            "benchmark_all": ["all", "all", "all"],
            "ipf_indicator_00000": [1, 0, 0],
        }
    )
    unit_metadata.to_csv(inputs / "unit_metadata.csv", index=False)

    ipf_target_metadata = pd.DataFrame(
        {
            "scope": ["person"],
            "target_type": ["numeric_total"],
            "value_column": ["ipf_indicator_00000"],
            "variables": ["benchmark_all"],
            "cell": ["benchmark_all=all"],
            "target_value": [1.0],
            "target_name": ["under_5_people"],
            "source_variable": ["person_count"],
            "stratum_id": [1],
        }
    )
    ipf_target_metadata.to_csv(inputs / "ipf_target_metadata.csv", index=False)

    weights_path, _ = benchmark_cli_module._run_ipf(run_dir)
    fitted_weights = np.load(weights_path)

    np.testing.assert_allclose(
        fitted_weights, np.array([1.0, 1.0]), atol=1e-8, rtol=1e-8
    )
    np.testing.assert_allclose(
        matrix.dot(fitted_weights), np.array([1.0]), atol=1e-8, rtol=1e-8
    )


def test_ipf_runner_end_to_end_categorical_margin_household_scope(
    benchmark_cli_module, tmp_path
):
    if not _r_package_available("surveysd"):
        pytest.skip("R package 'surveysd' is required for this test")

    run_dir = tmp_path / "ipf-margin-run"
    matrix = csr_matrix(np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64))
    inputs = _write_common_inputs(
        run_dir=run_dir,
        matrix=matrix,
        target_values=[2.0, 2.0],
        variables=["household_count", "household_count"],
        method_options={
            "ipf": {"max_iter": 50, "bound": 10.0, "epsP": 1e-9, "epsH": 1e-9}
        },
    )

    unit_metadata = pd.DataFrame(
        {
            "unit_index": [0, 1],
            "household_id": [0, 1],
            "snap": ["yes", "no"],
        }
    )
    unit_metadata.to_csv(inputs / "unit_metadata.csv", index=False)

    ipf_target_metadata = pd.DataFrame(
        {
            "scope": ["household", "household"],
            "target_type": ["categorical_margin", "categorical_margin"],
            "margin_id": ["snap_margin", "snap_margin"],
            "variables": ["snap", "snap"],
            "cell": ["snap=yes", "snap=no"],
            "target_value": [2.0, 2.0],
        }
    )
    ipf_target_metadata.to_csv(inputs / "ipf_target_metadata.csv", index=False)

    weights_path, _ = benchmark_cli_module._run_ipf(run_dir)
    fitted_weights = np.load(weights_path)

    np.testing.assert_allclose(
        fitted_weights, np.array([2.0, 2.0]), atol=1e-8, rtol=1e-8
    )
    np.testing.assert_allclose(
        matrix.dot(fitted_weights), np.array([2.0, 2.0]), atol=1e-8
    )
