from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


COUNT_LIKE_VARIABLES = {
    "person_count",
    "household_count",
    "tax_unit_count",
    "spm_unit_count",
    "family_count",
    "marital_unit_count",
}


def _normalize_string_list(values: Optional[List[str]]) -> Optional[List[str]]:
    if values is None:
        return None
    return [str(v) for v in values]


@dataclass
class TargetFilters:
    include_geo_levels: Optional[List[str]] = None
    include_national: bool = True
    state_ids: Optional[List[str]] = None
    district_ids: Optional[List[str]] = None
    variables: Optional[List[str]] = None
    domain_variables: Optional[List[str]] = None
    count_like_only: bool = False
    max_targets: Optional[int] = None

    @classmethod
    def from_dict(cls, raw: Optional[Dict[str, Any]]) -> "TargetFilters":
        raw = raw or {}
        return cls(
            include_geo_levels=_normalize_string_list(raw.get("include_geo_levels")),
            include_national=bool(raw.get("include_national", True)),
            state_ids=_normalize_string_list(raw.get("state_ids")),
            district_ids=_normalize_string_list(raw.get("district_ids")),
            variables=_normalize_string_list(raw.get("variables")),
            domain_variables=_normalize_string_list(raw.get("domain_variables")),
            count_like_only=bool(raw.get("count_like_only", False)),
            max_targets=raw.get("max_targets"),
        )


@dataclass
class ExternalInputs:
    ipf_unit_metadata_csv: Optional[str] = None
    ipf_target_metadata_csv: Optional[str] = None

    @classmethod
    def from_dict(cls, raw: Optional[Dict[str, Any]]) -> "ExternalInputs":
        raw = raw or {}
        return cls(
            ipf_unit_metadata_csv=raw.get("ipf_unit_metadata_csv"),
            ipf_target_metadata_csv=raw.get("ipf_target_metadata_csv"),
        )


@dataclass
class MethodOptions:
    l0: Dict[str, Any] = field(default_factory=dict)
    greg: Dict[str, Any] = field(default_factory=dict)
    ipf: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: Optional[Dict[str, Any]]) -> "MethodOptions":
        raw = raw or {}
        return cls(
            l0=dict(raw.get("l0", {})),
            greg=dict(raw.get("greg", {})),
            ipf=dict(raw.get("ipf", {})),
        )


@dataclass
class BenchmarkManifest:
    name: str
    tier: str
    description: str
    package_path: str
    methods: List[str]
    target_filters: TargetFilters = field(default_factory=TargetFilters)
    external_inputs: ExternalInputs = field(default_factory=ExternalInputs)
    method_options: MethodOptions = field(default_factory=MethodOptions)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "BenchmarkManifest":
        return cls(
            name=str(raw["name"]),
            tier=str(raw["tier"]),
            description=str(raw.get("description", "")),
            package_path=str(raw["package_path"]),
            methods=[str(m) for m in raw.get("methods", [])],
            target_filters=TargetFilters.from_dict(raw.get("target_filters")),
            external_inputs=ExternalInputs.from_dict(raw.get("external_inputs")),
            method_options=MethodOptions.from_dict(raw.get("method_options")),
        )

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["package_path"] = str(self.package_path)
        return payload


def load_manifest(path: str | Path) -> BenchmarkManifest:
    with open(path) as f:
        return BenchmarkManifest.from_dict(json.load(f))


def save_manifest(manifest: BenchmarkManifest, path: str | Path) -> None:
    with open(path, "w") as f:
        json.dump(manifest.to_dict(), f, indent=2, sort_keys=True)


def is_count_like_variable(variable: str) -> bool:
    return variable in COUNT_LIKE_VARIABLES or variable.endswith("_count")


def _build_geo_mask(targets_df: pd.DataFrame, filters: TargetFilters) -> np.ndarray:
    geo_level = targets_df["geo_level"].astype(str)
    geographic_id = targets_df["geographic_id"].astype(str)
    mask = np.ones(len(targets_df), dtype=bool)

    if filters.include_geo_levels:
        mask &= geo_level.isin(filters.include_geo_levels).to_numpy()

    geo_keep = np.zeros(len(targets_df), dtype=bool)
    national_mask = geo_level.eq("national").to_numpy()
    state_mask = geo_level.eq("state").to_numpy()
    district_mask = geo_level.eq("district").to_numpy()

    if filters.include_national:
        geo_keep |= national_mask

    if filters.state_ids:
        geo_keep |= state_mask & geographic_id.isin(filters.state_ids).to_numpy()
    else:
        geo_keep |= state_mask

    if filters.district_ids:
        geo_keep |= district_mask & geographic_id.isin(filters.district_ids).to_numpy()
    else:
        geo_keep |= district_mask

    other_mask = ~(national_mask | state_mask | district_mask)
    geo_keep |= other_mask
    return mask & geo_keep


def filter_targets(
    targets_df: pd.DataFrame,
    target_names: List[str],
    X_sparse,
    filters: TargetFilters,
):
    mask = _build_geo_mask(targets_df, filters)

    if filters.variables:
        mask &= targets_df["variable"].astype(str).isin(filters.variables).to_numpy()

    if filters.domain_variables:
        domain_series = targets_df.get(
            "domain_variable", pd.Series("", index=targets_df.index)
        )
        mask &= (
            domain_series.fillna("")
            .astype(str)
            .isin(filters.domain_variables)
            .to_numpy()
        )

    if filters.count_like_only:
        mask &= (
            targets_df["variable"].astype(str).map(is_count_like_variable).to_numpy()
        )

    indices = np.where(mask)[0]
    if filters.max_targets is not None:
        indices = indices[: int(filters.max_targets)]

    filtered_targets = targets_df.iloc[indices].reset_index(drop=True).copy()
    filtered_targets["target_name"] = [target_names[i] for i in indices]
    filtered_names = [target_names[i] for i in indices]
    filtered_matrix = X_sparse[indices, :]
    return filtered_targets, filtered_names, filtered_matrix, indices
