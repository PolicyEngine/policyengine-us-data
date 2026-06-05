"""Fully in-process end-to-end test of the paper-l0 benchmark spine.

synthetic package -> export_bundle (real IPF conversion via stubbed
Microsimulation) -> cmd_run for l0/greg/ipf -> metrics. No R, no Modal, no
network. GREG and IPF run through the real svy engine; L0 is stubbed (its
solver is a property of the L0 package, not this harness).
"""

from __future__ import annotations

import importlib
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tests.unit.fixtures.benchmark_ipf_inputs import (
    build_benchmark_package,
    install_fake_microsimulation,
)

pytest.importorskip("svy", reason="svy calibration extra not installed")

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = REPO_ROOT / "paper-l0" / "benchmarking"


def _load(name: str):
    # Normal import (registered in sys.modules) so dataclass annotation
    # resolution in benchmark_manifest works.
    if str(BENCHMARK_DIR) not in sys.path:
        sys.path.insert(0, str(BENCHMARK_DIR))
    return importlib.import_module(name)


def test_export_run_metrics_all_methods_in_process(tmp_path, monkeypatch):
    install_fake_microsimulation(monkeypatch, n_households=4)
    benchmark_manifest = _load("benchmark_manifest")
    benchmark_export = _load("benchmark_export")
    benchmark_cli = _load("benchmark_cli")

    bundle = build_benchmark_package(tmp_path)
    package = bundle["package"]
    package_path = tmp_path / "calibration_package.pkl"
    with open(package_path, "wb") as f:
        pickle.dump(package, f)

    manifest = benchmark_manifest.BenchmarkManifest(
        name="e2e",
        tier="t",
        description="",
        package_path=str(package_path),
        methods=["l0", "greg", "ipf"],
        method_options=benchmark_manifest.MethodOptions.from_dict(
            {
                "ipf": {
                    "count_variable": "household_count",
                    "max_iter": 1000,
                    "bound": 1e6,
                    "epsP": 1e-10,
                    "epsH": 1e-10,
                }
            }
        ),
    )

    run_dir = tmp_path / "bundle"
    benchmark_export.export_bundle(manifest=manifest, output_dir=run_dir)
    assert (run_dir / "inputs" / "ipf_target_metadata.csv").exists()

    # Stub L0 (its solver belongs to the L0 package, not this harness).
    def _fake_run_l0(rd, train_on="shared_requested"):
        weights_path = rd / "outputs" / "fitted_weights.npy"
        np.save(weights_path, np.ones(4, dtype=np.float64))
        return weights_path

    monkeypatch.setattr(benchmark_cli, "_run_l0", _fake_run_l0)

    for method in ("l0", "greg", "ipf"):
        exit_code = benchmark_cli.cmd_run(
            SimpleNamespace(method=method, run_dir=str(run_dir))
        )
        assert exit_code == 0
        summary_path = run_dir / "outputs" / f"{method}_summary.json"
        assert summary_path.exists(), f"missing summary for {method}"

    # GREG (real svy) must reproduce the achievable shared targets.
    import json

    greg = json.loads((run_dir / "outputs" / "greg_summary.json").read_text())
    assert greg["method"] == "greg"
    assert greg["mean_abs_rel_error"] < 1e-6
