"""Integration test for ipf_conversion.build_ipf_inputs on synthetic data.

Drives the real converter (resolve -> assemble -> close -> emit) against a
synthetic policy_data.db + a stubbed Microsimulation, with no Modal/network.
Verifies the single-count-family filter keeps the IPF problem single-scope and
that geo coverage is as expected (untargeted state present in the unit table).
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

from tests.unit.fixtures.benchmark_ipf_inputs import (
    build_benchmark_package,
    install_fake_microsimulation,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = REPO_ROOT / "paper-l0" / "benchmarking"


def _load(name: str):
    # Normal import (registered in sys.modules) so dataclass annotation
    # resolution in benchmark_manifest works.
    if str(BENCHMARK_DIR) not in sys.path:
        sys.path.insert(0, str(BENCHMARK_DIR))
    return importlib.import_module(name)


def _filtered_targets(targets_df, target_names):
    df = targets_df.copy()
    df["target_name"] = list(target_names)
    return df


def test_build_ipf_inputs_single_family_filter_keeps_household_scope(
    tmp_path, monkeypatch
):
    install_fake_microsimulation(monkeypatch, n_households=4)
    ipf_conversion = _load("ipf_conversion")
    benchmark_manifest = _load("benchmark_manifest")

    bundle = build_benchmark_package(tmp_path)
    package = bundle["package"]
    filtered = _filtered_targets(package["targets_df"], package["target_names"])

    manifest = benchmark_manifest.BenchmarkManifest(
        name="probe",
        tier="t",
        description="",
        package_path="x",
        methods=["ipf"],
        method_options=benchmark_manifest.MethodOptions.from_dict(
            {"ipf": {"count_variable": "household_count"}}
        ),
    )

    unit_meta, target_meta = ipf_conversion.build_ipf_inputs(
        package=package, manifest=manifest, filtered_targets=filtered
    )

    # person_count was dropped -> only household scope remains (single-scope).
    assert set(target_meta["scope"]) == {"household"}
    assert set(target_meta["variables"]) == {"state_fips"}
    # Cells cover only the targeted states 6 and 12...
    assert set(target_meta["cell"]) == {"state_fips=6", "state_fips=12"}
    # ...while the unit table still contains the untargeted state 36.
    assert 36 in set(unit_meta["state_fips"])


def test_build_ipf_inputs_missing_count_family_raises(tmp_path, monkeypatch):
    install_fake_microsimulation(monkeypatch, n_households=4)
    ipf_conversion = _load("ipf_conversion")
    benchmark_manifest = _load("benchmark_manifest")

    bundle = build_benchmark_package(tmp_path)
    package = bundle["package"]
    # Keep only person_count targets, then ask IPF for household_count.
    targets = package["targets_df"]
    names = package["target_names"]
    mask = targets["variable"] == "person_count"
    package["targets_df"] = targets[mask].reset_index(drop=True)
    filtered = _filtered_targets(
        package["targets_df"], [n for n, m in zip(names, mask) if m]
    )

    manifest = benchmark_manifest.BenchmarkManifest(
        name="probe",
        tier="t",
        description="",
        package_path="x",
        methods=["ipf"],
        method_options=benchmark_manifest.MethodOptions.from_dict(
            {"ipf": {"count_variable": "household_count"}}
        ),
    )

    with pytest.raises(ipf_conversion.IPFConversionError):
        ipf_conversion.build_ipf_inputs(
            package=package, manifest=manifest, filtered_targets=filtered
        )
