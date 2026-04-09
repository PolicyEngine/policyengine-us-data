from __future__ import annotations

import json
import pickle
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.io import mmwrite

from benchmark_manifest import BenchmarkManifest, filter_targets
from ipf_conversion import build_ipf_inputs


def load_calibration_package_raw(path: str | Path) -> Dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def build_shared_unit_metadata(package: Dict) -> pd.DataFrame:
    initial_weights = package.get("initial_weights")
    n_units = (
        int(initial_weights.shape[0])
        if initial_weights is not None
        else int(package["X_sparse"].shape[1])
    )
    data = {"unit_index": np.arange(n_units, dtype=np.int64)}

    if initial_weights is not None:
        data["base_weight"] = np.asarray(initial_weights, dtype=np.float64)

    if package.get("cd_geoid") is not None:
        data["cd_geoid"] = np.asarray(package["cd_geoid"]).astype(str)

    if package.get("block_geoid") is not None:
        data["block_geoid"] = np.asarray(package["block_geoid"]).astype(str)

    return pd.DataFrame(data)


def export_bundle(
    manifest: BenchmarkManifest, output_dir: str | Path
) -> Tuple[Path, Dict]:
    output_dir = Path(output_dir)
    inputs_dir = output_dir / "inputs"
    outputs_dir = output_dir / "outputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    package = load_calibration_package_raw(manifest.package_path)
    targets_df = package["targets_df"].copy()
    target_names = list(package["target_names"])
    X_sparse = package["X_sparse"]

    filtered_targets, filtered_names, filtered_matrix, kept_indices = filter_targets(
        targets_df=targets_df,
        target_names=target_names,
        X_sparse=X_sparse,
        filters=manifest.target_filters,
    )

    filtered_targets.to_csv(inputs_dir / "target_metadata.csv", index=False)
    mmwrite(str(inputs_dir / "X_targets_by_units.mtx"), filtered_matrix)

    initial_weights = np.asarray(package["initial_weights"], dtype=np.float64)
    np.save(inputs_dir / "initial_weights.npy", initial_weights)

    unit_metadata = build_shared_unit_metadata(package)
    has_external_ipf_inputs = bool(
        manifest.external_inputs.ipf_unit_metadata_csv
        or manifest.external_inputs.ipf_target_metadata_csv
    )
    has_partial_external_ipf_inputs = bool(
        manifest.external_inputs.ipf_unit_metadata_csv
    ) != bool(manifest.external_inputs.ipf_target_metadata_csv)
    if has_partial_external_ipf_inputs:
        raise ValueError(
            "IPF external input overrides must provide both "
            "ipf_unit_metadata_csv and ipf_target_metadata_csv"
        )

    if manifest.external_inputs.ipf_unit_metadata_csv:
        shutil.copyfile(
            manifest.external_inputs.ipf_unit_metadata_csv,
            inputs_dir / "unit_metadata.csv",
        )
    elif "ipf" in manifest.methods and not has_external_ipf_inputs:
        ipf_unit_metadata, ipf_target_metadata = build_ipf_inputs(
            package=package,
            manifest=manifest,
            filtered_targets=filtered_targets,
        )
        ipf_unit_metadata.to_csv(inputs_dir / "unit_metadata.csv", index=False)
        ipf_target_metadata.to_csv(inputs_dir / "ipf_target_metadata.csv", index=False)
    else:
        unit_metadata.to_csv(inputs_dir / "unit_metadata.csv", index=False)

    if manifest.external_inputs.ipf_target_metadata_csv:
        shutil.copyfile(
            manifest.external_inputs.ipf_target_metadata_csv,
            inputs_dir / "ipf_target_metadata.csv",
        )

    runtime_manifest = manifest.to_dict()
    runtime_manifest["resolved"] = {
        "output_dir": str(output_dir.resolve()),
        "inputs_dir": str(inputs_dir.resolve()),
        "outputs_dir": str(outputs_dir.resolve()),
        "n_targets": int(filtered_matrix.shape[0]),
        "n_units": int(filtered_matrix.shape[1]),
        "kept_target_indices": [int(i) for i in kept_indices.tolist()],
        "target_names": filtered_names,
        "package_metadata": package.get("metadata", {}),
    }

    with open(inputs_dir / "benchmark_manifest.json", "w") as f:
        json.dump(runtime_manifest, f, indent=2, sort_keys=True)

    export_info = {
        "n_targets": int(filtered_matrix.shape[0]),
        "n_units": int(filtered_matrix.shape[1]),
        "inputs_dir": str(inputs_dir),
        "outputs_dir": str(outputs_dir),
    }
    return output_dir, export_info
