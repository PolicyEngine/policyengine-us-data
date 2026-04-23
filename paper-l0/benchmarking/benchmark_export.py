from __future__ import annotations

import json
import pickle
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.io import mmread, mmwrite

from benchmark_manifest import BenchmarkManifest, filter_targets
from ipf_conversion import IPFConversionError, build_ipf_inputs


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


def _validate_external_ipf_inputs(
    manifest: BenchmarkManifest,
    *,
    n_units: int,
) -> None:
    ext = manifest.external_inputs
    provided = {
        "ipf_unit_metadata_csv": ext.ipf_unit_metadata_csv,
        "ipf_target_metadata_csv": ext.ipf_target_metadata_csv,
        "ipf_scoring_target_metadata_csv": ext.ipf_scoring_target_metadata_csv,
        "ipf_scoring_matrix_mtx": ext.ipf_scoring_matrix_mtx,
    }
    if not any(provided.values()):
        return

    missing = [name for name, path in provided.items() if not path]
    if missing:
        raise ValueError(
            "External IPF overrides must provide all of: "
            "ipf_unit_metadata_csv, ipf_target_metadata_csv, "
            "ipf_scoring_target_metadata_csv, ipf_scoring_matrix_mtx. "
            f"Missing: {missing}"
        )

    target_meta = pd.read_csv(ext.ipf_target_metadata_csv)
    required_target_cols = {
        "margin_id",
        "scope",
        "target_type",
        "variables",
        "cell",
        "target_value",
    }
    missing_target_cols = sorted(required_target_cols - set(target_meta.columns))
    if missing_target_cols:
        raise ValueError(
            "External ipf_target_metadata_csv is missing required columns: "
            f"{missing_target_cols}"
        )
    unsupported_types = set(target_meta["target_type"].astype(str)) - {
        "categorical_margin"
    }
    if unsupported_types:
        raise ValueError(
            "External ipf_target_metadata_csv contains unsupported target_type "
            f"values: {sorted(unsupported_types)}"
        )
    unsupported_scopes = set(target_meta["scope"].astype(str)) - {"person", "household"}
    if unsupported_scopes:
        raise ValueError(
            "External ipf_target_metadata_csv contains unsupported scope values: "
            f"{sorted(unsupported_scopes)}"
        )

    unit_meta = pd.read_csv(ext.ipf_unit_metadata_csv)
    weight_col = str(manifest.method_options.ipf.get("weight_col", "base_weight"))
    household_id_col = str(
        manifest.method_options.ipf.get("household_id_col", "household_id")
    )
    if "unit_index" not in unit_meta.columns and weight_col not in unit_meta.columns:
        raise ValueError(
            "External ipf_unit_metadata_csv must include either unit_index or the "
            f"configured weight column '{weight_col}'."
        )
    if "household" in set(target_meta["scope"].astype(str)) and (
        household_id_col not in unit_meta.columns
    ):
        raise ValueError(
            "External ipf_unit_metadata_csv must include the configured household "
            f"id column '{household_id_col}' when household-scope margins are present."
        )

    scoring_meta = pd.read_csv(ext.ipf_scoring_target_metadata_csv)
    if "value" not in scoring_meta.columns:
        raise ValueError(
            "External ipf_scoring_target_metadata_csv must include a 'value' column."
        )
    scoring_matrix = mmread(str(ext.ipf_scoring_matrix_mtx)).tocsr()
    if scoring_matrix.shape[0] != len(scoring_meta):
        raise ValueError(
            "External ipf_scoring_matrix_mtx row count does not match "
            "ipf_scoring_target_metadata_csv."
        )
    if scoring_matrix.shape[1] != n_units:
        raise ValueError(
            "External ipf_scoring_matrix_mtx column count does not match the "
            "shared benchmark unit count."
        )


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
    _validate_external_ipf_inputs(
        manifest,
        n_units=int(filtered_matrix.shape[1]),
    )
    has_external_ipf_inputs = bool(
        manifest.external_inputs.ipf_unit_metadata_csv
        or manifest.external_inputs.ipf_target_metadata_csv
        or manifest.external_inputs.ipf_scoring_target_metadata_csv
        or manifest.external_inputs.ipf_scoring_matrix_mtx
    )

    ipf_diagnostics: Dict = {}
    if manifest.external_inputs.ipf_unit_metadata_csv:
        shutil.copyfile(
            manifest.external_inputs.ipf_unit_metadata_csv,
            inputs_dir / "unit_metadata.csv",
        )
        shutil.copyfile(
            manifest.external_inputs.ipf_target_metadata_csv,
            inputs_dir / "ipf_target_metadata.csv",
        )
        shutil.copyfile(
            manifest.external_inputs.ipf_scoring_target_metadata_csv,
            inputs_dir / "ipf_scoring_target_metadata.csv",
        )
        shutil.copyfile(
            manifest.external_inputs.ipf_scoring_matrix_mtx,
            inputs_dir / "ipf_scoring_X_targets_by_units.mtx",
        )
        if manifest.external_inputs.ipf_conversion_diagnostics_json:
            shutil.copyfile(
                manifest.external_inputs.ipf_conversion_diagnostics_json,
                inputs_dir / "ipf_conversion_diagnostics.json",
            )
            with open(inputs_dir / "ipf_conversion_diagnostics.json") as f:
                ipf_diagnostics = json.load(f)
        else:
            ipf_diagnostics = {
                "requested_target_count": int(filtered_matrix.shape[0]),
                "retained_authored_target_count": int(
                    len(pd.read_csv(inputs_dir / "ipf_scoring_target_metadata.csv"))
                ),
                "derived_complement_count": None,
                "dropped_targets": {},
                "dropped_target_details": [],
                "margin_consistency_issues": [],
                "derived_complement_rows": [],
                "converted_target_rows": int(
                    len(pd.read_csv(inputs_dir / "ipf_target_metadata.csv"))
                ),
                "source": "external_ipf_override",
            }
    elif "ipf" in manifest.methods and not has_external_ipf_inputs:
        try:
            ipf_unit_metadata, ipf_target_metadata = build_ipf_inputs(
                package=package,
                manifest=manifest,
                filtered_targets=filtered_targets,
            )
        except IPFConversionError as exc:
            if exc.diagnostics:
                with open(inputs_dir / "ipf_conversion_diagnostics.json", "w") as f:
                    json.dump(exc.diagnostics, f, indent=2, sort_keys=True, default=str)
            raise
        ipf_unit_metadata.to_csv(inputs_dir / "unit_metadata.csv", index=False)
        ipf_target_metadata.to_csv(inputs_dir / "ipf_target_metadata.csv", index=False)
        retained_target_ids = list(
            ipf_target_metadata.attrs.get("retained_authored_target_ids", [])
        )
        if retained_target_ids and "target_id" in filtered_targets.columns:
            retained_mask = filtered_targets["target_id"].isin(retained_target_ids)
            ipf_scoring_targets = filtered_targets.loc[retained_mask].reset_index(
                drop=True
            )
            ipf_scoring_matrix = filtered_matrix[retained_mask.to_numpy(), :]
            ipf_scoring_targets.to_csv(
                inputs_dir / "ipf_scoring_target_metadata.csv", index=False
            )
            mmwrite(
                str(inputs_dir / "ipf_scoring_X_targets_by_units.mtx"),
                ipf_scoring_matrix,
            )
        ipf_diagnostics = {
            "requested_target_count": int(
                ipf_target_metadata.attrs.get("requested_target_count", len(filtered_targets))
            ),
            "retained_authored_target_count": int(
                ipf_target_metadata.attrs.get(
                    "retained_authored_target_count", 0
                )
            ),
            "derived_complement_count": int(
                ipf_target_metadata.attrs.get("derived_complement_count", 0)
            ),
            "dropped_targets": dict(
                ipf_target_metadata.attrs.get("dropped_targets", {})
            ),
            "dropped_target_details": list(
                ipf_target_metadata.attrs.get("dropped_target_details", [])
            ),
            "margin_consistency_issues": list(
                ipf_target_metadata.attrs.get("margin_consistency_issues", [])
            ),
            "derived_complement_rows": list(
                ipf_target_metadata.attrs.get("derived_complement_rows", [])
            ),
            "converted_target_rows": int(len(ipf_target_metadata)),
        }
        with open(inputs_dir / "ipf_conversion_diagnostics.json", "w") as f:
            json.dump(ipf_diagnostics, f, indent=2, sort_keys=True, default=str)
    else:
        unit_metadata.to_csv(inputs_dir / "unit_metadata.csv", index=False)

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
    if ipf_diagnostics:
        dropped = ipf_diagnostics.get("dropped_targets", {})
        export_info["ipf_requested_target_count"] = int(
            ipf_diagnostics.get("requested_target_count", len(filtered_targets))
        )
        export_info["ipf_retained_authored_target_count"] = int(
            ipf_diagnostics.get("retained_authored_target_count", 0)
        )
        derived_complement_count = ipf_diagnostics.get("derived_complement_count", 0)
        export_info["ipf_derived_complement_count"] = (
            int(derived_complement_count)
            if derived_complement_count is not None
            else 0
        )
        export_info["ipf_converted_target_rows"] = int(
            ipf_diagnostics.get("converted_target_rows", 0)
        )
        export_info["ipf_dropped_non_count_count"] = int(
            dropped.get("non_count_style", 0)
        )
        export_info["ipf_dropped_unresolvable_count"] = int(
            dropped.get("unresolvable_constraints", 0)
        )
        export_info["ipf_margin_consistency_issue_count"] = int(
            len(ipf_diagnostics.get("margin_consistency_issues", []))
        )
    return output_dir, export_info
