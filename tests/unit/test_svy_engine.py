"""Always-on tests for the in-process `svy` benchmark engines.

These replace the R-skipping coverage in ``test_benchmarking_runners.py``:
they exercise ``paper-l0/benchmarking/svy_engine.py`` directly and require no
R. They skip only if the ``svy`` calibration extra is not installed.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.io import mmwrite
from scipy.sparse import csr_matrix

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = REPO_ROOT / "paper-l0" / "benchmarking"

svy = pytest.importorskip("svy", reason="svy calibration extra not installed")


def _load_svy_engine():
    if str(BENCHMARK_DIR) not in sys.path:
        sys.path.insert(0, str(BENCHMARK_DIR))
    spec = importlib.util.spec_from_file_location(
        "svy_engine_for_tests", BENCHMARK_DIR / "svy_engine.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_greg_inputs(inputs: Path, matrix, target_values, initial_weights=None):
    inputs.mkdir(parents=True, exist_ok=True)
    mmwrite(str(inputs / "X.mtx"), csr_matrix(matrix))
    pd.DataFrame(
        {
            "value": list(target_values),
            "target_name": [f"target_{i}" for i in range(len(target_values))],
        }
    ).to_csv(inputs / "t.csv", index=False)
    if initial_weights is None:
        initial_weights = np.ones(matrix.shape[1], dtype=np.float64)
    np.save(inputs / "w.npy", np.asarray(initial_weights, dtype=np.float64))
    return inputs


@pytest.fixture(scope="module")
def svy_engine():
    return _load_svy_engine()


def test_greg_svy_exact_fit_identity(svy_engine, tmp_path):
    """Identity design: calibrated weights equal the targets exactly."""
    inputs = _write_greg_inputs(
        tmp_path / "inputs", np.eye(2, dtype=np.float64), [2.0, 3.0]
    )
    weights, diag = svy_engine.fit_greg_svy(
        matrix_path=inputs / "X.mtx",
        targets_path=inputs / "t.csv",
        initial_weights_path=inputs / "w.npy",
        options={"maxit": 50, "epsilon": 1e-10},
    )
    np.testing.assert_allclose(weights, [2.0, 3.0], atol=1e-8, rtol=1e-8)
    assert diag["greg_engine"] == "svy"
    # maxit/epsilon are inert under svy's closed-form linear solve.
    assert set(diag["greg_ignored_options"]) == {"maxit", "epsilon"}


def test_greg_svy_overdetermined_hits_all_targets(svy_engine, tmp_path):
    """Classical linear GREG reproduces all control totals on an
    over-determined problem with non-uniform base weights."""
    X = np.array(
        [
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
            [2.0, 1.0, 0.0, 3.0, 1.0, 2.0],
        ]
    )
    targets = [70.0, 30.0, 95.0]
    w0 = np.array([10.0, 12.0, 8.0, 15.0, 9.0, 11.0])
    inputs = _write_greg_inputs(tmp_path / "inputs", X, targets, initial_weights=w0)
    weights, _ = svy_engine.fit_greg_svy(
        matrix_path=inputs / "X.mtx",
        targets_path=inputs / "t.csv",
        initial_weights_path=inputs / "w.npy",
        options={},
    )
    np.testing.assert_allclose(X @ weights, targets, atol=1e-6)


def test_greg_svy_target_count_mismatch_raises(svy_engine, tmp_path):
    inputs = tmp_path / "inputs"
    inputs.mkdir(parents=True)
    mmwrite(str(inputs / "X.mtx"), csr_matrix(np.eye(2)))
    pd.DataFrame({"value": [1.0], "target_name": ["only_one"]}).to_csv(
        inputs / "t.csv", index=False
    )
    np.save(inputs / "w.npy", np.ones(2))
    with pytest.raises(RuntimeError, match="target count mismatch"):
        svy_engine.fit_greg_svy(
            matrix_path=inputs / "X.mtx",
            targets_path=inputs / "t.csv",
            initial_weights_path=inputs / "w.npy",
            options={},
        )


def test_greg_svy_max_aux_bytes_guard_raises(svy_engine, tmp_path):
    inputs = _write_greg_inputs(
        tmp_path / "inputs", np.eye(2, dtype=np.float64), [2.0, 3.0]
    )
    with pytest.raises(RuntimeError, match="aux matrix too large"):
        svy_engine.fit_greg_svy(
            matrix_path=inputs / "X.mtx",
            targets_path=inputs / "t.csv",
            initial_weights_path=inputs / "w.npy",
            options={"max_aux_bytes": 1},
        )


# --- IPF -----------------------------------------------------------------
# Expected weights below are surveysd::ipf fixed points (verified equal to the
# R runner in tests/unit/test_svy_ipf_parity.py); these run without R.


def _ipf_targets(rows):
    return pd.DataFrame(
        [
            {
                "margin_id": m,
                "scope": s,
                "target_type": "categorical_margin",
                "variables": v,
                "cell": c,
                "target_value": val,
            }
            for m, s, v, c, val in rows
        ]
    )


def _write_ipf_inputs(inputs, unit_df, targets_df, initial_weights):
    inputs.mkdir(parents=True, exist_ok=True)
    unit_df.to_csv(inputs / "unit_metadata.csv", index=False)
    targets_df.to_csv(inputs / "ipf_target_metadata.csv", index=False)
    np.save(inputs / "w.npy", np.asarray(initial_weights, dtype=np.float64))
    return inputs


def _run_ipf(svy_engine, inputs, options=None):
    weights, diag = svy_engine.fit_ipf_svy(
        unit_metadata_path=inputs / "unit_metadata.csv",
        ipf_target_metadata_path=inputs / "ipf_target_metadata.csv",
        initial_weights_path=inputs / "w.npy",
        options=options
        or {"max_iter": 1000, "bound": 1e6, "epsP": 1e-10, "epsH": 1e-10},
    )
    return weights, diag


def test_ipf_svy_household_full_partition(svy_engine, tmp_path):
    inputs = _write_ipf_inputs(
        tmp_path / "inputs",
        pd.DataFrame(
            {"unit_index": [0, 1], "household_id": [0, 1], "snap": ["yes", "no"]}
        ),
        _ipf_targets(
            [
                ("m", "household", "snap", "snap=yes", 2.0),
                ("m", "household", "snap", "snap=no", 2.0),
            ]
        ),
        np.ones(2),
    )
    weights, _ = _run_ipf(svy_engine, inputs)
    np.testing.assert_allclose(weights, [2.0, 2.0], atol=1e-6)


def test_ipf_svy_person_scope_meanhh(svy_engine, tmp_path):
    # Two households of two persons each; person-scope age margin doubles each.
    inputs = _write_ipf_inputs(
        tmp_path / "inputs",
        pd.DataFrame(
            {
                "unit_index": [0, 0, 1, 1],
                "household_id": [0, 0, 1, 1],
                "age": ["0-4", "0-4", "5-9", "5-9"],
            }
        ),
        _ipf_targets(
            [
                ("m", "person", "age", "age=0-4", 4.0),
                ("m", "person", "age", "age=5-9", 4.0),
            ]
        ),
        np.ones(2),
    )
    weights, _ = _run_ipf(svy_engine, inputs)
    np.testing.assert_allclose(weights, [2.0, 2.0], atol=1e-6)


def test_ipf_svy_single_cell_pads_untargeted_complement(svy_engine, tmp_path):
    # Only snap=yes targeted; svy pads snap=no at its base total so it is
    # left untouched — reproducing surveysd's leave-complement behavior.
    inputs = _write_ipf_inputs(
        tmp_path / "inputs",
        pd.DataFrame(
            {"unit_index": [0, 1], "household_id": [0, 1], "snap": ["yes", "no"]}
        ),
        _ipf_targets([("m", "household", "snap", "snap=yes", 2.0)]),
        np.array([4.0, 4.0]),
    )
    weights, diag = _run_ipf(svy_engine, inputs)
    np.testing.assert_allclose(weights, [2.0, 4.0], atol=1e-6)
    assert diag["ipf_padded_uncovered_cells"] == 1


def test_ipf_svy_geo_partial_leaves_untargeted_state(svy_engine, tmp_path):
    # State margin targets states 6 and 12; the state-36 unit is untargeted
    # and must keep its base weight (svy pads it). Matches surveysd [2,2,3,1].
    inputs = _write_ipf_inputs(
        tmp_path / "inputs",
        pd.DataFrame(
            {
                "unit_index": [0, 1, 2, 3],
                "household_id": [0, 1, 2, 3],
                "state_fips": [6, 6, 12, 36],
            }
        ),
        _ipf_targets(
            [
                ("m", "household", "state_fips", "state_fips=6", 4.0),
                ("m", "household", "state_fips", "state_fips=12", 3.0),
            ]
        ),
        np.ones(4),
    )
    weights, _ = _run_ipf(svy_engine, inputs)
    np.testing.assert_allclose(weights, [2.0, 2.0, 3.0, 1.0], atol=1e-6)


def test_ipf_svy_mixed_scope_raises(svy_engine, tmp_path):
    inputs = _write_ipf_inputs(
        tmp_path / "inputs",
        pd.DataFrame(
            {
                "unit_index": [0, 0, 1, 1],
                "household_id": [0, 0, 1, 1],
                "age": ["child", "adult", "child", "adult"],
                "snap": ["yes", "yes", "no", "no"],
            }
        ),
        _ipf_targets(
            [
                ("a", "person", "age", "age=child", 4.0),
                ("a", "person", "age", "age=adult", 4.0),
                ("s", "household", "snap", "snap=yes", 1.0),
                ("s", "household", "snap", "snap=no", 3.0),
            ]
        ),
        np.ones(2),
    )
    with pytest.raises(RuntimeError, match="single-scope"):
        _run_ipf(svy_engine, inputs)
