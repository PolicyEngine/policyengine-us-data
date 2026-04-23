from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.io import mmwrite
from scipy.sparse import csr_matrix


REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = REPO_ROOT / "paper-l0" / "benchmarking"
BENCHMARK_EXPORT_PATH = BENCHMARK_DIR / "benchmark_export.py"


def _load_module(name: str, path: Path):
    benchmark_dir_str = str(BENCHMARK_DIR)
    if benchmark_dir_str not in sys.path:
        sys.path.insert(0, benchmark_dir_str)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_export_bundle_writes_ipf_scoring_subset(tmp_path, monkeypatch):
    benchmark_export = _load_module(
        "benchmark_export_for_tests", BENCHMARK_EXPORT_PATH
    )
    benchmark_manifest = _load_module(
        "benchmark_manifest_for_tests", BENCHMARK_DIR / "benchmark_manifest.py"
    )

    package = {
        "targets_df": pd.DataFrame(
            {
                "target_id": [1, 2, 3],
                "value": [2.0, 3.0, 5.0],
                "variable": [
                    "household_count",
                    "household_count",
                    "household_count",
                ],
                "geo_level": ["national", "national", "national"],
                "geographic_id": ["0", "0", "0"],
            }
        ),
        "target_names": ["requested_a", "requested_b", "requested_c"],
        "X_sparse": csr_matrix(
            np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float64)
        ),
        "initial_weights": np.array([1.0, 1.0], dtype=np.float64),
        "metadata": {},
    }

    monkeypatch.setattr(
        benchmark_export, "load_calibration_package_raw", lambda _path: package
    )

    def _fake_build_ipf_inputs(package, manifest, filtered_targets):
        unit_metadata = pd.DataFrame(
            {
                "unit_index": [0, 1],
                "household_id": [0, 1],
                "base_weight": [1.0, 1.0],
            }
        )
        ipf_target_metadata = pd.DataFrame(
            {
                "margin_id": ["m0", "m1", "m1"],
                "scope": ["household", "household", "household"],
                "target_type": ["categorical_margin"] * 3,
                "variables": ["district", "district|snap", "district|snap"],
                "cell": ["district=A", "district=A|snap=yes", "district=A|snap=no"],
                "target_value": [2.0, 1.0, 1.0],
                "is_authored": [True, True, False],
            }
        )
        ipf_target_metadata.attrs["retained_authored_target_ids"] = [1, 2]
        ipf_target_metadata.attrs["requested_target_count"] = 3
        ipf_target_metadata.attrs["retained_authored_target_count"] = 2
        ipf_target_metadata.attrs["derived_complement_count"] = 1
        ipf_target_metadata.attrs["dropped_targets"] = {"unsupported_partial_margin": 1}
        ipf_target_metadata.attrs["dropped_target_details"] = [
            {"reason": "unsupported_partial_margin", "target_ids": [3]}
        ]
        ipf_target_metadata.attrs["margin_consistency_issues"] = []
        ipf_target_metadata.attrs["derived_complement_rows"] = [
            {"cell": "district=A|snap=no", "target_value": 1.0}
        ]
        return unit_metadata, ipf_target_metadata

    monkeypatch.setattr(benchmark_export, "build_ipf_inputs", _fake_build_ipf_inputs)

    manifest = benchmark_manifest.BenchmarkManifest(
        name="ipf-export-test",
        tier="unit",
        description="",
        package_path="/tmp/fake-package.pkl",
        methods=["ipf"],
    )

    output_dir, info = benchmark_export.export_bundle(
        manifest=manifest,
        output_dir=tmp_path / "bundle",
    )

    scoring_targets = pd.read_csv(
        output_dir / "inputs" / "ipf_scoring_target_metadata.csv"
    )
    diagnostics = json.loads(
        (output_dir / "inputs" / "ipf_conversion_diagnostics.json").read_text()
    )

    assert len(scoring_targets) == 2
    assert scoring_targets["target_id"].tolist() == [1, 2]
    assert diagnostics["retained_authored_target_count"] == 2
    assert diagnostics["derived_complement_count"] == 1
    assert info["ipf_retained_authored_target_count"] == 2


def test_export_bundle_requires_external_ipf_scoring_artifacts(
    tmp_path, monkeypatch
):
    benchmark_export = _load_module(
        "benchmark_export_for_external_contract", BENCHMARK_EXPORT_PATH
    )
    benchmark_manifest = _load_module(
        "benchmark_manifest_for_external_contract",
        BENCHMARK_DIR / "benchmark_manifest.py",
    )

    package = {
        "targets_df": pd.DataFrame(
            {
                "target_id": [1],
                "value": [2.0],
                "variable": ["household_count"],
                "geo_level": ["national"],
                "geographic_id": ["0"],
            }
        ),
        "target_names": ["requested_a"],
        "X_sparse": csr_matrix(np.array([[1.0, 0.0]], dtype=np.float64)),
        "initial_weights": np.array([1.0, 1.0], dtype=np.float64),
        "metadata": {},
    }

    monkeypatch.setattr(
        benchmark_export, "load_calibration_package_raw", lambda _path: package
    )

    unit_csv = tmp_path / "external_unit.csv"
    target_csv = tmp_path / "external_targets.csv"
    pd.DataFrame(
        {"unit_index": [0, 1], "household_id": [0, 1], "base_weight": [1.0, 1.0]}
    ).to_csv(unit_csv, index=False)
    pd.DataFrame(
        {
            "margin_id": ["m0"],
            "scope": ["household"],
            "target_type": ["categorical_margin"],
            "variables": ["district"],
            "cell": ["district=A"],
            "target_value": [2.0],
        }
    ).to_csv(target_csv, index=False)

    manifest = benchmark_manifest.BenchmarkManifest(
        name="ipf-external-contract",
        tier="unit",
        description="",
        package_path="/tmp/fake-package.pkl",
        methods=["ipf"],
        external_inputs=benchmark_manifest.ExternalInputs(
            ipf_unit_metadata_csv=str(unit_csv),
            ipf_target_metadata_csv=str(target_csv),
        ),
    )

    with pytest.raises(ValueError, match="must provide all of"):
        benchmark_export.export_bundle(manifest=manifest, output_dir=tmp_path / "bundle")


def test_export_bundle_accepts_fully_specified_external_ipf_inputs(
    tmp_path, monkeypatch
):
    benchmark_export = _load_module(
        "benchmark_export_for_external_copy", BENCHMARK_EXPORT_PATH
    )
    benchmark_manifest = _load_module(
        "benchmark_manifest_for_external_copy", BENCHMARK_DIR / "benchmark_manifest.py"
    )

    package = {
        "targets_df": pd.DataFrame(
            {
                "target_id": [1, 2],
                "value": [2.0, 3.0],
                "variable": ["household_count", "household_count"],
                "geo_level": ["national", "national"],
                "geographic_id": ["0", "0"],
            }
        ),
        "target_names": ["requested_a", "requested_b"],
        "X_sparse": csr_matrix(np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)),
        "initial_weights": np.array([1.0, 1.0], dtype=np.float64),
        "metadata": {},
    }

    monkeypatch.setattr(
        benchmark_export, "load_calibration_package_raw", lambda _path: package
    )

    unit_csv = tmp_path / "external_unit.csv"
    target_csv = tmp_path / "external_targets.csv"
    scoring_csv = tmp_path / "external_scoring.csv"
    scoring_mtx = tmp_path / "external_scoring.mtx"
    diagnostics_json = tmp_path / "external_diag.json"

    pd.DataFrame(
        {"unit_index": [0, 1], "household_id": [0, 1], "base_weight": [1.0, 1.0]}
    ).to_csv(unit_csv, index=False)
    pd.DataFrame(
        {
            "margin_id": ["m0", "m0"],
            "scope": ["household", "household"],
            "target_type": ["categorical_margin", "categorical_margin"],
            "variables": ["district", "district"],
            "cell": ["district=A", "district=B"],
            "target_value": [2.0, 3.0],
        }
    ).to_csv(target_csv, index=False)
    pd.DataFrame(
        {
            "value": [2.0],
            "variable": ["household_count"],
            "geo_level": ["national"],
            "target_name": ["retained_a"],
        }
    ).to_csv(scoring_csv, index=False)
    mmwrite(str(scoring_mtx), csr_matrix(np.array([[1.0, 0.0]], dtype=np.float64)))
    diagnostics_json.write_text(
        json.dumps(
            {
                "requested_target_count": 2,
                "retained_authored_target_count": 1,
                "derived_complement_count": 0,
                "dropped_targets": {"missing_parent_total": 1},
            }
        )
    )

    manifest = benchmark_manifest.BenchmarkManifest(
        name="ipf-external-complete",
        tier="unit",
        description="",
        package_path="/tmp/fake-package.pkl",
        methods=["ipf"],
        external_inputs=benchmark_manifest.ExternalInputs(
            ipf_unit_metadata_csv=str(unit_csv),
            ipf_target_metadata_csv=str(target_csv),
            ipf_scoring_target_metadata_csv=str(scoring_csv),
            ipf_scoring_matrix_mtx=str(scoring_mtx),
            ipf_conversion_diagnostics_json=str(diagnostics_json),
        ),
    )

    output_dir, info = benchmark_export.export_bundle(
        manifest=manifest,
        output_dir=tmp_path / "bundle",
    )

    assert (output_dir / "inputs" / "ipf_target_metadata.csv").exists()
    assert (output_dir / "inputs" / "ipf_scoring_target_metadata.csv").exists()
    copied_diag = json.loads(
        (output_dir / "inputs" / "ipf_conversion_diagnostics.json").read_text()
    )
    assert copied_diag["retained_authored_target_count"] == 1
    assert info["ipf_retained_authored_target_count"] == 1
