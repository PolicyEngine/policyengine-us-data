"""Synthetic fixtures for the paper-l0 IPF benchmark conversion + end-to-end tests.

These let `build_ipf_inputs` (and the full export -> run -> metrics chain) run
with no Modal, no network, and no real `policyengine_us` microsimulation:

- `write_synthetic_policy_db` writes the one table `ipf_conversion` reads
  (`stratum_constraints`).
- `FakeMicrosimulation` / `install_fake_microsimulation` satisfy the local
  `from policyengine_us import Microsimulation` import inside `build_ipf_inputs`.
- `build_benchmark_package` returns an export-ready calibration package dict
  whose targets mix `household_count` and `person_count` across states, with
  one unit in an untargeted state (so the geo-padding path is exercised).
"""

from __future__ import annotations

import sqlite3
import sys
import types
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


class _Column:
    def __init__(self, array):
        self.values = np.asarray(array)


class FakeMicrosimulation:
    """Minimal stand-in for `policyengine_us.Microsimulation`.

    Only implements what `ipf_conversion.build_ipf_inputs` calls:
    `dataset.load_dataset()` and `calculate(var, map_to=...)`.
    """

    def __init__(self, dataset=None, n_households: int = 4, period: int = 2024):
        self._n = n_households
        self._period = period
        self.dataset = types.SimpleNamespace(
            load_dataset=lambda: {"household_id": {period: np.arange(n_households)}}
        )

    def calculate(self, variable, map_to=None, **_kwargs):
        # household_id at either level; one person per household keeps the
        # person/household mapping trivial for single-scope runs.
        return _Column(np.arange(self._n))


def install_fake_microsimulation(monkeypatch=None, n_households: int = 4):
    """Inject a fake `policyengine_us` module exposing `Microsimulation`.

    With `monkeypatch`, registers via `monkeypatch.setitem` (auto-undone).
    Otherwise mutates `sys.modules` directly (for ad-hoc scripts).
    """
    module = types.ModuleType("policyengine_us")
    module.Microsimulation = lambda dataset=None: FakeMicrosimulation(
        dataset=dataset, n_households=n_households
    )
    if monkeypatch is not None:
        monkeypatch.setitem(sys.modules, "policyengine_us", module)
    else:
        sys.modules["policyengine_us"] = module
    return module


def write_synthetic_policy_db(path: Path, constraints) -> Path:
    """Write a minimal `policy_data.db` with just the `stratum_constraints` table.

    `constraints` is an iterable of (stratum_id, constraint_variable, operation,
    value) rows — the exact columns `ipf_conversion._load_stratum_constraints`
    selects.
    """
    con = sqlite3.connect(str(path))
    try:
        con.execute(
            "CREATE TABLE stratum_constraints "
            "(stratum_id INT, constraint_variable TEXT, operation TEXT, value TEXT)"
        )
        con.executemany(
            "INSERT INTO stratum_constraints VALUES (?,?,?,?)",
            [(int(s), str(v), str(op), str(val)) for s, v, op, val in constraints],
        )
        con.commit()
    finally:
        con.close()
    return path


# Four single-person households across states 6, 6, 12, 36 (36 untargeted).
BENCHMARK_BLOCK_GEOIDS = (
    "060010000000001",
    "060010000000002",
    "120010000000001",
    "360010000000001",
)
BENCHMARK_CD_GEOIDS = ("601", "602", "1201", "3601")


def build_benchmark_package(tmp_path: Path) -> Dict:
    """Build an export-ready synthetic calibration package + its DB/dataset.

    Returns a dict with `package` (the calibration-package payload), `db_path`,
    and `dataset_path`. The targets mix household_count (states 6, 12) and
    person_count (state 6); the IPF single-count-family filter keeps only the
    configured family.
    """
    db_path = write_synthetic_policy_db(
        tmp_path / "policy_data.db",
        constraints=[
            (100, "state_fips", "==", "6"),
            (101, "state_fips", "==", "12"),
            (200, "state_fips", "==", "6"),
        ],
    )
    dataset_path = tmp_path / "dataset.h5"
    dataset_path.write_bytes(b"stub")  # existence is all build_ipf_inputs checks

    # Target/unit design matrix (targets x units), 3 targets x 4 units.
    # rows: hh_count s6 [1,1,0,0]; hh_count s12 [0,0,1,0]; person_count s6 [1,1,0,0]
    x_sparse = csr_matrix(
        np.array(
            [
                [1.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0, 0.0],
            ]
        )
    )
    targets_df = pd.DataFrame(
        {
            "target_id": [100, 101, 200],
            "stratum_id": [100, 101, 200],
            "variable": ["household_count", "household_count", "person_count"],
            "value": [4.0, 3.0, 4.0],
            "period": [2024, 2024, 2024],
            "geo_level": ["state", "state", "state"],
            "geographic_id": ["6", "12", "6"],
            "domain_variable": ["", "", ""],
        }
    )
    package = {
        "X_sparse": x_sparse,
        "targets_df": targets_df,
        "target_names": ["hh_count_s6", "hh_count_s12", "person_count_s6"],
        "metadata": {
            "dataset_path": str(dataset_path),
            "db_path": str(db_path),
            "n_clones": 1,
        },
        "initial_weights": np.array([1.0, 1.0, 1.0, 1.0]),
        "cd_geoid": np.array(BENCHMARK_CD_GEOIDS),
        "block_geoid": np.array(BENCHMARK_BLOCK_GEOIDS),
    }
    return {"package": package, "db_path": db_path, "dataset_path": dataset_path}
