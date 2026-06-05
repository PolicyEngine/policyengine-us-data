"""Engine-independent CLI wiring tests for the benchmark harness.

These cover `cmd_run` scoring/training-subset selection, the `--train-on
ipf_retained_authored` matched-summary path, and `_run_l0` seed forwarding.
They monkeypatch the method runners, so they need neither R nor the svy extra
(ported out of the former R-gated `test_benchmarking_runners.py`).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.io import mmwrite
from scipy.sparse import csr_matrix

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = REPO_ROOT / "paper-l0" / "benchmarking"
BENCHMARK_CLI_PATH = BENCHMARK_DIR / "benchmark_cli.py"


def _load_benchmark_cli_module():
    if str(BENCHMARK_DIR) not in sys.path:
        sys.path.insert(0, str(BENCHMARK_DIR))
    spec = importlib.util.spec_from_file_location(
        "benchmark_cli_for_cli_tests", BENCHMARK_CLI_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_common_inputs(
    run_dir: Path,
    matrix,
    target_values,
    variables,
    geo_levels=None,
    target_names=None,
    initial_weights=None,
    method_options=None,
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

    pd.DataFrame(
        {
            "value": np.asarray(target_values, dtype=np.float64),
            "variable": variables,
            "geo_level": geo_levels,
            "target_name": target_names,
        }
    ).to_csv(inputs / "target_metadata.csv", index=False)

    with open(inputs / "benchmark_manifest.json", "w") as f:
        json.dump({"method_options": method_options or {}}, f)

    return inputs


@pytest.fixture
def benchmark_cli_module(monkeypatch, tmp_path_factory):
    cache_root = tmp_path_factory.mktemp("benchmarking-cli-cache")
    monkeypatch.setenv("MPLCONFIGDIR", str(cache_root / "mpl"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(cache_root / "xdg"))
    return _load_benchmark_cli_module()


def _write_scoring_subset(inputs: Path):
    subset_matrix = csr_matrix(np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float))
    mmwrite(str(inputs / "ipf_scoring_X_targets_by_units.mtx"), subset_matrix)
    pd.DataFrame(
        {
            "value": [2.0, 3.0],
            "variable": ["household_count", "household_count"],
            "geo_level": ["national", "national"],
            "target_name": ["retained_a", "retained_b"],
        }
    ).to_csv(inputs / "ipf_scoring_target_metadata.csv", index=False)


def test_cmd_run_ipf_uses_retained_authored_scoring_subset(
    benchmark_cli_module, tmp_path
):
    run_dir = tmp_path / "ipf-summary-run"
    inputs = _write_common_inputs(
        run_dir=run_dir,
        matrix=csr_matrix(np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=float)),
        target_values=[2.0, 3.0, 5.0],
        variables=["household_count"] * 3,
        target_names=["requested_a", "requested_b", "requested_c"],
    )
    _write_scoring_subset(inputs)
    weights_path = run_dir / "outputs" / "fitted_weights.npy"
    np.save(weights_path, np.array([2.0, 3.0], dtype=np.float64))

    benchmark_cli_module._run_ipf = lambda _run_dir, **_kw: (weights_path, 0.0)
    exit_code = benchmark_cli_module.cmd_run(
        SimpleNamespace(method="ipf", run_dir=str(run_dir))
    )

    assert exit_code == 0
    summary = json.loads((run_dir / "outputs" / "ipf_summary.json").read_text())
    assert summary["n_targets"] == 2
    assert summary["scoring_target_set"] == "ipf_retained_authored"


def test_run_l0_passes_seed_from_manifest_to_fit(
    benchmark_cli_module, tmp_path, monkeypatch
):
    matrix = csr_matrix(np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]], dtype=np.float64))
    run_dir = tmp_path / "seed-wireup"
    _write_common_inputs(
        run_dir=run_dir,
        matrix=matrix,
        target_values=[2.0, 2.0],
        variables=["household_count", "household_count"],
        method_options={"l0": {"seed": 42}},
    )

    captured: dict = {}

    def _fake_fit_l0_weights(**kwargs):
        captured.update(kwargs)
        return np.ones(matrix.shape[1], dtype=np.float64)

    import policyengine_us_data.calibration.unified_calibration as uc

    monkeypatch.setattr(uc, "fit_l0_weights", _fake_fit_l0_weights)
    benchmark_cli_module._run_l0(run_dir)
    assert captured.get("seed") == 42

    captured.clear()
    run_dir_no_seed = tmp_path / "seed-wireup-none"
    _write_common_inputs(
        run_dir=run_dir_no_seed,
        matrix=matrix,
        target_values=[2.0, 2.0],
        variables=["household_count", "household_count"],
        method_options={"l0": {}},
    )
    benchmark_cli_module._run_l0(run_dir_no_seed)
    assert captured.get("seed") is None


def test_cmd_run_train_on_retained_subset_uses_subset_inputs(
    benchmark_cli_module, tmp_path
):
    run_dir = tmp_path / "matched-run"
    inputs = _write_common_inputs(
        run_dir=run_dir,
        matrix=csr_matrix(np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=float)),
        target_values=[2.0, 3.0, 5.0],
        variables=["household_count"] * 3,
        target_names=["requested_a", "requested_b", "requested_c"],
    )
    _write_scoring_subset(inputs)
    weights_path = run_dir / "outputs" / "fitted_weights.npy"
    np.save(weights_path, np.array([2.0, 3.0], dtype=np.float64))

    captured = {}

    def _fake_run_l0(run_dir, train_on="shared_requested"):
        targets_path, matrix_path, label = benchmark_cli_module._select_training_inputs(
            run_dir, train_on
        )
        captured.update(targets_path=targets_path, matrix_path=matrix_path, label=label)
        return weights_path

    benchmark_cli_module._run_l0 = _fake_run_l0
    exit_code = benchmark_cli_module.cmd_run(
        SimpleNamespace(
            method="l0",
            run_dir=str(run_dir),
            train_on="ipf_retained_authored",
            score_on="ipf_retained_authored",
        )
    )

    assert exit_code == 0
    assert captured["targets_path"].name == "ipf_scoring_target_metadata.csv"
    assert captured["matrix_path"].name == "ipf_scoring_X_targets_by_units.mtx"
    assert captured["label"] == "ipf_retained_authored"

    matched = run_dir / "outputs" / "l0_matched_summary.json"
    assert matched.exists()
    assert not (run_dir / "outputs" / "l0_summary.json").exists()
    summary = json.loads(matched.read_text())
    assert summary["training_target_set"] == "ipf_retained_authored"
    assert summary["scoring_target_set"] == "ipf_retained_authored"


def test_cmd_run_train_on_retained_subset_fails_when_subset_missing(
    benchmark_cli_module, tmp_path
):
    run_dir = tmp_path / "matched-missing-subset"
    _write_common_inputs(
        run_dir=run_dir,
        matrix=csr_matrix(np.array([[1.0, 0.0]], dtype=float)),
        target_values=[2.0],
        variables=["household_count"],
        target_names=["requested_a"],
    )
    with pytest.raises(FileNotFoundError, match="ipf_scoring_"):
        benchmark_cli_module._run_l0(run_dir, train_on="ipf_retained_authored")


def test_cmd_run_l0_can_opt_into_retained_authored_scoring_subset(
    benchmark_cli_module, tmp_path
):
    run_dir = tmp_path / "l0-summary-run"
    inputs = _write_common_inputs(
        run_dir=run_dir,
        matrix=csr_matrix(np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=float)),
        target_values=[2.0, 3.0, 5.0],
        variables=["household_count"] * 3,
        target_names=["requested_a", "requested_b", "requested_c"],
    )
    _write_scoring_subset(inputs)
    weights_path = run_dir / "outputs" / "fitted_weights.npy"
    np.save(weights_path, np.array([2.0, 3.0], dtype=np.float64))

    benchmark_cli_module._run_l0 = lambda _run_dir, **_kw: weights_path
    exit_code = benchmark_cli_module.cmd_run(
        SimpleNamespace(
            method="l0", run_dir=str(run_dir), score_on="ipf_retained_authored"
        )
    )

    assert exit_code == 0
    summary = json.loads((run_dir / "outputs" / "l0_summary.json").read_text())
    assert summary["n_targets"] == 2
    assert summary["scoring_target_set"] == "ipf_retained_authored"
