from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from types import SimpleNamespace
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


def test_ipf_runner_end_to_end_categorical_margin_person_scope(
    benchmark_cli_module, tmp_path
):
    if not _r_package_available("surveysd"):
        pytest.skip("R package 'surveysd' is required for this test")

    run_dir = tmp_path / "ipf-person-margin-run"
    # Two household units, each with two person rows (same unit_index). One unit's
    # persons are age_bracket="0-4", the other's are "5-9". Person-scope margin
    # targets 4 people total in each bucket -> each unit's weight doubles.
    matrix = csr_matrix(np.array([[2.0, 0.0], [0.0, 2.0]], dtype=np.float64))
    inputs = _write_common_inputs(
        run_dir=run_dir,
        matrix=matrix,
        target_values=[4.0, 4.0],
        variables=["person_count", "person_count"],
        method_options={
            "ipf": {"max_iter": 50, "bound": 10.0, "epsP": 1e-9, "epsH": 1e-9}
        },
    )

    unit_metadata = pd.DataFrame(
        {
            "unit_index": [0, 0, 1, 1],
            "household_id": [0, 0, 1, 1],
            "age_bracket": ["0-4", "0-4", "5-9", "5-9"],
        }
    )
    unit_metadata.to_csv(inputs / "unit_metadata.csv", index=False)

    ipf_target_metadata = pd.DataFrame(
        {
            "scope": ["person", "person"],
            "target_type": ["categorical_margin", "categorical_margin"],
            "margin_id": ["age_margin", "age_margin"],
            "variables": ["age_bracket", "age_bracket"],
            "cell": ["age_bracket=0-4", "age_bracket=5-9"],
            "target_value": [4.0, 4.0],
        }
    )
    ipf_target_metadata.to_csv(inputs / "ipf_target_metadata.csv", index=False)

    weights_path, _ = benchmark_cli_module._run_ipf(run_dir)
    fitted_weights = np.load(weights_path)

    np.testing.assert_allclose(
        fitted_weights, np.array([2.0, 2.0]), atol=1e-8, rtol=1e-8
    )
    np.testing.assert_allclose(
        matrix.dot(fitted_weights), np.array([4.0, 4.0]), atol=1e-8
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


def test_ipf_runner_end_to_end_single_cell_margin_leaves_complement_untouched(
    benchmark_cli_module, tmp_path
):
    """A 1-cell categorical_margin must rake the authored cell and leave
    units outside that cell at their base weights. This is the semantic the
    converter now relies on instead of synthesizing a baseline complement.
    """
    if not _r_package_available("surveysd"):
        pytest.skip("R package 'surveysd' is required for this test")

    run_dir = tmp_path / "ipf-single-cell-run"
    # Two households: unit 0 is snap=yes, unit 1 is snap=no. Target: 1 yes-hh
    # (authored), no complement constraint. Expect calib[0]=2.0 (rescaled from
    # base 4.0 -> 2.0 so 1 weighted hh = 1), calib[1]=4.0 (untouched base).
    matrix = csr_matrix(np.array([[1.0, 0.0]], dtype=np.float64))
    inputs = _write_common_inputs(
        run_dir=run_dir,
        matrix=matrix,
        target_values=[2.0],
        variables=["household_count"],
        initial_weights=np.array([4.0, 4.0], dtype=np.float64),
        method_options={
            "ipf": {"max_iter": 100, "bound": 10.0, "epsP": 1e-9, "epsH": 1e-9}
        },
    )

    unit_metadata = pd.DataFrame(
        {
            "unit_index": [0, 1],
            "household_id": [0, 1],
            "snap": ["yes", "no"],
            "base_weight": [4.0, 4.0],
        }
    )
    unit_metadata.to_csv(inputs / "unit_metadata.csv", index=False)

    # 1-cell authored margin: only snap=yes has a target, no complement row.
    ipf_target_metadata = pd.DataFrame(
        {
            "scope": ["household"],
            "target_type": ["categorical_margin"],
            "margin_id": ["snap_yes_only"],
            "variables": ["snap"],
            "cell": ["snap=yes"],
            "target_value": [2.0],
        }
    )
    ipf_target_metadata.to_csv(inputs / "ipf_target_metadata.csv", index=False)

    weights_path, _ = benchmark_cli_module._run_ipf(run_dir)
    fitted_weights = np.load(weights_path)

    # Authored cell (snap=yes, unit 0): rescaled to hit target = 2.0.
    np.testing.assert_allclose(fitted_weights[0], 2.0, atol=1e-8, rtol=1e-8)
    # Complement unit (snap=no, unit 1): untouched, equals its base weight.
    np.testing.assert_allclose(fitted_weights[1], 4.0, atol=1e-8, rtol=1e-8)


def test_ipf_runner_rejects_numeric_total_target_type(benchmark_cli_module, tmp_path):
    """The runner supports only `categorical_margin`. A CSV with
    `target_type='numeric_total'` must fail fast with a clear error so old
    external pipelines cannot silently fall back to the (removed)
    `benchmark_all` raking path.
    """
    if not _r_package_available("surveysd"):
        pytest.skip("R package 'surveysd' is required for this test")

    run_dir = tmp_path / "ipf-numeric-rejected"
    matrix = csr_matrix(np.array([[1.0, 0.0]], dtype=np.float64))
    inputs = _write_common_inputs(
        run_dir=run_dir,
        matrix=matrix,
        target_values=[1.0],
        variables=["person_count"],
        method_options={
            "ipf": {"max_iter": 10, "bound": 10.0, "epsP": 1e-4, "epsH": 1e-4}
        },
    )
    pd.DataFrame(
        {
            "unit_index": [0, 1],
            "household_id": [0, 1],
            "snap": ["yes", "no"],
            "base_weight": [1.0, 1.0],
        }
    ).to_csv(inputs / "unit_metadata.csv", index=False)
    pd.DataFrame(
        {
            "scope": ["person"],
            "target_type": ["numeric_total"],
            "margin_id": ["_numeric"],
            "value_column": ["snap"],
            "variables": ["snap"],
            "cell": ["snap=yes"],
            "target_value": [1.0],
        }
    ).to_csv(inputs / "ipf_target_metadata.csv", index=False)

    with pytest.raises(RuntimeError, match="IPF runner failed"):
        benchmark_cli_module._run_ipf(run_dir)


def test_ipf_runner_single_call_multi_margin_exact_fit(benchmark_cli_module, tmp_path):
    """Compatible closed margins should run in one `surveysd::ipf` call."""
    if not _r_package_available("surveysd"):
        pytest.skip("R package 'surveysd' is required for this test")

    run_dir = tmp_path / "ipf-joint-run"
    matrix = csr_matrix(
        np.array(
            [
                [1.0, 1.0, 0.0, 0.0],  # district A count
                [0.0, 0.0, 1.0, 1.0],  # district B count
                [1.0, 0.0, 0.0, 0.0],  # district A snap=yes
                [0.0, 1.0, 0.0, 0.0],  # district A snap=no
                [0.0, 0.0, 1.0, 0.0],  # district B snap=yes
                [0.0, 0.0, 0.0, 1.0],  # district B snap=no
            ],
            dtype=np.float64,
        )
    )
    inputs = _write_common_inputs(
        run_dir=run_dir,
        matrix=matrix,
        target_values=[4.0, 4.0, 2.0, 2.0, 2.0, 2.0],
        variables=["household_count"] * 6,
        initial_weights=np.ones(4, dtype=np.float64),
        method_options={
            "ipf": {"max_iter": 200, "bound": 10.0, "epsP": 1e-9, "epsH": 1e-9}
        },
    )
    pd.DataFrame(
        {
            "unit_index": [0, 1, 2, 3],
            "household_id": [0, 1, 2, 3],
            "district": ["A", "A", "B", "B"],
            "snap": ["yes", "no", "yes", "no"],
            "base_weight": [1.0, 1.0, 1.0, 1.0],
        }
    ).to_csv(inputs / "unit_metadata.csv", index=False)

    pd.DataFrame(
        {
            "scope": ["household"] * 6,
            "target_type": ["categorical_margin"] * 6,
            "margin_id": [
                "district_margin",
                "district_margin",
                "district_snap_margin",
                "district_snap_margin",
                "district_snap_margin",
                "district_snap_margin",
            ],
            "variables": [
                "district",
                "district",
                "district|snap",
                "district|snap",
                "district|snap",
                "district|snap",
            ],
            "cell": [
                "district=A",
                "district=B",
                "district=A|snap=yes",
                "district=A|snap=no",
                "district=B|snap=yes",
                "district=B|snap=no",
            ],
            "target_value": [4.0, 4.0, 2.0, 2.0, 2.0, 2.0],
        }
    ).to_csv(inputs / "ipf_target_metadata.csv", index=False)

    weights_path, _ = benchmark_cli_module._run_ipf(run_dir)
    fitted_weights = np.load(weights_path)

    np.testing.assert_allclose(
        matrix.dot(fitted_weights),
        np.array([4.0, 4.0, 2.0, 2.0, 2.0, 2.0]),
        atol=1e-6,
    )
    assert not (run_dir / "outputs" / "_ipf_blocks").exists()


def test_cmd_run_ipf_uses_retained_authored_scoring_subset(
    benchmark_cli_module, tmp_path
):
    run_dir = tmp_path / "ipf-summary-run"
    matrix = csr_matrix(np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=float))
    inputs = _write_common_inputs(
        run_dir=run_dir,
        matrix=matrix,
        target_values=[2.0, 3.0, 5.0],
        variables=["household_count", "household_count", "household_count"],
        target_names=["requested_a", "requested_b", "requested_c"],
    )
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

    weights_path = run_dir / "outputs" / "fitted_weights.npy"
    np.save(weights_path, np.array([2.0, 3.0], dtype=np.float64))

    def _fake_run_ipf(_run_dir):
        return weights_path, 0.0

    benchmark_cli_module._run_ipf = _fake_run_ipf
    exit_code = benchmark_cli_module.cmd_run(
        SimpleNamespace(method="ipf", run_dir=str(run_dir))
    )

    assert exit_code == 0
    summary = json.loads((run_dir / "outputs" / "ipf_summary.json").read_text())
    assert summary["n_targets"] == 2
    assert summary["scoring_target_set"] == "ipf_retained_authored"


def test_cmd_run_l0_can_opt_into_retained_authored_scoring_subset(
    benchmark_cli_module, tmp_path
):
    run_dir = tmp_path / "l0-summary-run"
    matrix = csr_matrix(np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=float))
    inputs = _write_common_inputs(
        run_dir=run_dir,
        matrix=matrix,
        target_values=[2.0, 3.0, 5.0],
        variables=["household_count", "household_count", "household_count"],
        target_names=["requested_a", "requested_b", "requested_c"],
    )
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

    weights_path = run_dir / "outputs" / "fitted_weights.npy"
    np.save(weights_path, np.array([2.0, 3.0], dtype=np.float64))

    def _fake_run_l0(_run_dir):
        return weights_path

    benchmark_cli_module._run_l0 = _fake_run_l0
    exit_code = benchmark_cli_module.cmd_run(
        SimpleNamespace(
            method="l0",
            run_dir=str(run_dir),
            score_on="ipf_retained_authored",
        )
    )

    assert exit_code == 0
    summary = json.loads((run_dir / "outputs" / "l0_summary.json").read_text())
    assert summary["n_targets"] == 2
    assert summary["scoring_target_set"] == "ipf_retained_authored"
