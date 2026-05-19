"""Calibration target tolerance policy resolution and artifacts."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

TARGET_POLICY_SCHEMA_VERSION = "1"
TARGET_POLICY_ARTIFACT = "calibration_target_policy.jsonl"
TARGET_POLICY_SUMMARY_ARTIFACT = "calibration_target_policy_summary.json"
DEFAULT_TARGET_POLICY_PATH = Path(__file__).resolve().parent / "target_policy.yaml"
VALID_ENFORCEMENT = frozenset({"fail", "warn", "diagnostic_only"})

COUNT_VARIABLES = frozenset(
    {
        "household_count",
        "person_count",
        "spm_unit_count",
        "tax_unit_count",
    }
)
ACA_TOKENS = ("aca", "marketplace", "selected_marketplace_plan")


def load_target_policy_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load target policy YAML, defaulting to the bundled policy."""

    import yaml

    policy_path = Path(path) if path is not None else DEFAULT_TARGET_POLICY_PATH
    with policy_path.open() as file:
        config = yaml.safe_load(file) or {}
    if not isinstance(config, dict):
        raise ValueError("target policy config must be a mapping")
    return config


def build_target_policy(
    targets_df: pd.DataFrame,
    *,
    target_names: Sequence[str] | None = None,
    config: Mapping[str, Any] | None = None,
    row_sums: np.ndarray | None = None,
) -> pd.DataFrame:
    """Return one resolved tolerance-policy row per calibration target."""

    config = dict(config or {})
    defaults = _policy_defaults(config.get("defaults", {}))
    rules = list(config.get("rules", []))
    if not isinstance(rules, list):
        raise ValueError("target policy rules must be a list")

    rows: list[dict[str, Any]] = []
    for index, target in targets_df.reset_index(drop=True).iterrows():
        rule_id = "default"
        policy = _default_policy_for_target(target, defaults)
        for raw_rule in rules:
            if not isinstance(raw_rule, Mapping):
                raise ValueError("target policy rule must be a mapping")
            if _rule_matches(target, raw_rule.get("match", {})):
                policy.update(_rule_updates(raw_rule))
                rule_id = str(raw_rule.get("id", rule_id))

        if row_sums is not None and float(row_sums[index]) <= 0:
            policy["enforcement"] = "diagnostic_only"
            policy["loss_weight"] = 0.0

        enforcement = str(policy["enforcement"])
        if enforcement not in VALID_ENFORCEMENT:
            raise ValueError(
                f"target policy enforcement must be one of {sorted(VALID_ENFORCEMENT)}"
            )
        tolerance_pct = float(policy["tolerance_pct"])
        loss_weight = float(policy["loss_weight"])
        scale_floor = float(policy["scale_floor"])
        if tolerance_pct < 0:
            raise ValueError("target policy tolerance_pct must be non-negative")
        if loss_weight < 0:
            raise ValueError("target policy loss_weight must be non-negative")
        if scale_floor <= 0:
            raise ValueError("target policy scale_floor must be positive")

        row = {
            "target_index": int(index),
            "target": (
                str(target_names[index])
                if target_names is not None and index < len(target_names)
                else None
            ),
            "variable": _target_string(target, "variable"),
            "geo_level": _target_string(target, "geo_level"),
            "geographic_id": _target_string(target, "geographic_id"),
            "domain_variable": _target_string(target, "domain_variable"),
            "enforcement": enforcement,
            "priority": str(policy["priority"]),
            "tolerance_pct": tolerance_pct,
            "tolerance": tolerance_pct / 100.0,
            "scale_floor": scale_floor,
            "loss_weight": 0.0 if enforcement == "diagnostic_only" else loss_weight,
            "policy_rule_id": rule_id,
            "policy_group_key": _policy_group_key(target, policy),
            "loss_enabled": enforcement != "diagnostic_only" and loss_weight > 0,
            "schema_version": TARGET_POLICY_SCHEMA_VERSION,
        }
        rows.append(row)

    return pd.DataFrame(rows)


def target_policy_arrays(
    target_policy: pd.DataFrame,
    targets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return L0-ready target weights, tolerances, and relative-error scales."""

    if len(target_policy) != len(targets):
        raise ValueError(
            "target policy length must match target vector length: "
            f"{len(target_policy)} != {len(targets)}"
        )
    targets_arr = np.asarray(targets, dtype=np.float64)
    target_weights = target_policy["loss_weight"].to_numpy(dtype=np.float64)
    tolerances = target_policy["tolerance"].to_numpy(dtype=np.float64)
    scale_floors = target_policy["scale_floor"].to_numpy(dtype=np.float64)
    scales = np.maximum(np.abs(targets_arr), scale_floors)
    return target_weights, tolerances, scales


def annotate_diagnostics_with_policy(
    diagnostics: pd.DataFrame,
    target_policy: pd.DataFrame | None,
) -> pd.DataFrame:
    """Attach policy columns and final validation status to diagnostics."""

    annotated = diagnostics.copy()
    if target_policy is None:
        annotated["enforcement"] = "warn"
        annotated["priority"] = "P3"
        annotated["tolerance_pct"] = np.inf
        annotated["tolerance"] = np.inf
        annotated["scale_floor"] = 1.0
        annotated["loss_weight"] = 1.0
        annotated["policy_rule_id"] = "legacy"
        annotated["policy_group_key"] = "legacy"
        annotated["loss_enabled"] = True
    else:
        if len(target_policy) != len(annotated):
            raise ValueError(
                "target policy length must match diagnostics length: "
                f"{len(target_policy)} != {len(annotated)}"
            )
        policy_columns = [
            "enforcement",
            "priority",
            "tolerance_pct",
            "tolerance",
            "scale_floor",
            "loss_weight",
            "policy_rule_id",
            "policy_group_key",
            "loss_enabled",
        ]
        for column in policy_columns:
            annotated[column] = target_policy[column].to_numpy()

    annotated["excess_abs_rel_error"] = np.maximum(
        annotated["abs_rel_error"].to_numpy(dtype=np.float64)
        - annotated["tolerance"].to_numpy(dtype=np.float64),
        0.0,
    )
    annotated["within_tolerance"] = annotated["excess_abs_rel_error"] <= 0
    annotated["validation_status"] = np.select(
        [
            ~annotated["achievable"].astype(bool),
            annotated["within_tolerance"],
            annotated["enforcement"] == "fail",
            annotated["enforcement"] == "warn",
        ],
        ["diagnostic_only", "pass", "fail", "warn"],
        default="diagnostic_only",
    )
    return annotated


def enforce_target_tolerances(diagnostics: pd.DataFrame) -> None:
    """Raise if any achievable hard-fail target exceeds its tolerance."""

    failed = diagnostics[
        (diagnostics["validation_status"] == "fail")
        & diagnostics["achievable"].astype(bool)
    ]
    if failed.empty:
        return
    preview = failed.sort_values("excess_abs_rel_error", ascending=False).head(10)
    rows = [
        f"{row.target}: abs_rel_error={row.abs_rel_error:.4%}, "
        f"tolerance={row.tolerance:.4%}"
        for row in preview.itertuples()
    ]
    raise ValueError(
        "Calibration hard-fail targets exceeded tolerance: " + "; ".join(rows)
    )


def summarize_target_policy(policy: pd.DataFrame) -> dict[str, Any]:
    """Return a compact JSON summary of a resolved target policy table."""

    enforcement_counts = (
        policy["enforcement"].value_counts().sort_index().astype(int).to_dict()
    )
    priority_counts = (
        policy["priority"].value_counts().sort_index().astype(int).to_dict()
    )
    tolerance_rows = (
        policy[
            [
                "priority",
                "enforcement",
                "tolerance_pct",
                "scale_floor",
                "loss_weight",
                "policy_rule_id",
            ]
        ]
        .drop_duplicates()
        .sort_values(["priority", "enforcement", "policy_rule_id"])
        .to_dict(orient="records")
    )
    return {
        "schema_version": TARGET_POLICY_SCHEMA_VERSION,
        "n_targets": int(len(policy)),
        "enforcement_counts": enforcement_counts,
        "priority_counts": priority_counts,
        "tolerances": tolerance_rows,
    }


def write_target_policy_artifacts(
    policy: pd.DataFrame,
    output_dir: str | Path,
    *,
    prefix: str = "",
) -> tuple[Path, Path]:
    """Write target policy JSONL and summary JSON artifacts."""

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    jsonl_path = directory / f"{prefix}{TARGET_POLICY_ARTIFACT}"
    summary_path = directory / f"{prefix}{TARGET_POLICY_SUMMARY_ARTIFACT}"
    records = policy.where(pd.notnull(policy), None).to_dict(orient="records")
    with jsonl_path.open("w") as file:
        for record in records:
            file.write(json.dumps(record, sort_keys=True) + "\n")
    summary_path.write_text(
        json.dumps(summarize_target_policy(policy), indent=2, sort_keys=True) + "\n"
    )
    return jsonl_path, summary_path


def _policy_defaults(raw_defaults: Mapping[str, Any]) -> dict[str, Any]:
    defaults = dict(raw_defaults or {})
    return {
        "count_scale_floors": {
            "national": float(defaults.get("count_scale_floor_national", 100_000.0)),
            "state": float(defaults.get("count_scale_floor_state", 10_000.0)),
            "district": float(defaults.get("count_scale_floor_district", 1_000.0)),
        },
        "dollar_scale_floors": {
            "national": float(
                defaults.get("dollar_scale_floor_national", 250_000_000.0)
            ),
            "state": float(defaults.get("dollar_scale_floor_state", 25_000_000.0)),
            "district": float(defaults.get("dollar_scale_floor_district", 1_000_000.0)),
        },
        "net_worth_scale_floor": float(
            defaults.get("net_worth_scale_floor", 1_000_000_000.0)
        ),
    }


def _default_policy_for_target(
    target: pd.Series,
    defaults: Mapping[str, Any],
) -> dict[str, Any]:
    variable = _target_string(target, "variable")
    geo_level = _target_string(target, "geo_level") or "national"
    domain_variable = _target_string(target, "domain_variable")

    if variable == "household_count" and not domain_variable:
        return _policy(
            priority="P0",
            enforcement="fail",
            tolerance_pct=_geo_tolerance(geo_level, 0.25, 0.5, 1.0),
            scale_floor=_scale_floor(variable, geo_level, defaults),
            loss_weight=40.0,
        )
    if variable == "person_count" and not domain_variable:
        return _policy(
            priority="P0",
            enforcement="fail",
            tolerance_pct=_geo_tolerance(geo_level, 0.25, 0.5, 1.0),
            scale_floor=_scale_floor(variable, geo_level, defaults),
            loss_weight=40.0,
        )
    if variable == "person_count" and domain_variable == "age":
        return _policy(
            priority="P1",
            enforcement="warn",
            tolerance_pct=_geo_tolerance(geo_level, 1.0, 2.0, 3.0),
            scale_floor=_scale_floor(variable, geo_level, defaults),
            loss_weight=15.0,
        )
    if _is_aca_target(variable, domain_variable):
        return _policy(
            priority="P2",
            enforcement="warn",
            tolerance_pct=7.5,
            scale_floor=_scale_floor(variable, geo_level, defaults),
            loss_weight=6.0,
        )
    if _is_count_variable(variable):
        return _policy(
            priority="P2",
            enforcement="warn",
            tolerance_pct=_geo_tolerance(geo_level, 5.0, 5.0, 7.5),
            scale_floor=_scale_floor(variable, geo_level, defaults),
            loss_weight=6.0,
        )
    return _policy(
        priority="P3",
        enforcement="warn",
        tolerance_pct=_geo_tolerance(geo_level, 5.0, 5.0, 10.0),
        scale_floor=_scale_floor(variable, geo_level, defaults),
        loss_weight=3.0,
    )


def _policy(
    *,
    priority: str,
    enforcement: str,
    tolerance_pct: float,
    scale_floor: float,
    loss_weight: float,
) -> dict[str, Any]:
    return {
        "priority": priority,
        "enforcement": enforcement,
        "tolerance_pct": float(tolerance_pct),
        "scale_floor": float(scale_floor),
        "loss_weight": float(loss_weight),
    }


def _rule_matches(target: pd.Series, raw_match: Any) -> bool:
    if raw_match is None:
        raw_match = {}
    if not isinstance(raw_match, Mapping):
        raise ValueError("target policy rule match must be a mapping")
    for key in ("variable", "geo_level", "domain_variable"):
        if key in raw_match and not _matches_value(
            _target_string(target, key),
            raw_match[key],
        ):
            return False
    if "domain_variable_contains_any" in raw_match:
        domain_variable = _target_string(target, "domain_variable")
        tokens = _as_sequence(raw_match["domain_variable_contains_any"])
        if not any(str(token) in domain_variable for token in tokens):
            return False
    return True


def _rule_updates(rule: Mapping[str, Any]) -> dict[str, Any]:
    allowed = {
        "priority",
        "enforcement",
        "tolerance_pct",
        "scale_floor",
        "loss_weight",
    }
    updates = {key: rule[key] for key in allowed if key in rule}
    return updates


def _matches_value(value: str, expected: Any) -> bool:
    return value in {str(item) for item in _as_sequence(expected)}


def _as_sequence(value: Any) -> Sequence[Any]:
    if isinstance(value, list | tuple | set):
        return tuple(value)
    return (value,)


def _target_string(target: pd.Series, key: str) -> str:
    value = target.get(key, "")
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value)


def _policy_group_key(target: pd.Series, policy: Mapping[str, Any]) -> str:
    return "|".join(
        [
            str(policy["priority"]),
            str(policy["enforcement"]),
            _target_string(target, "geo_level"),
            _target_string(target, "variable"),
            _target_string(target, "domain_variable"),
        ]
    )


def _geo_tolerance(
    geo_level: str,
    national: float,
    state: float,
    district: float,
) -> float:
    if geo_level == "district":
        return district
    if geo_level == "state":
        return state
    return national


def _scale_floor(
    variable: str,
    geo_level: str,
    defaults: Mapping[str, Any],
) -> float:
    if variable == "net_worth":
        return float(defaults["net_worth_scale_floor"])
    if _is_count_variable(variable):
        floors = defaults["count_scale_floors"]
    else:
        floors = defaults["dollar_scale_floors"]
    if geo_level == "district":
        return float(floors["district"])
    if geo_level == "state":
        return float(floors["state"])
    return float(floors["national"])


def _is_count_variable(variable: str) -> bool:
    return variable in COUNT_VARIABLES or variable.endswith("_count")


def _is_aca_target(variable: str, domain_variable: str) -> bool:
    text = f"{variable},{domain_variable}".lower()
    return any(token in text for token in ACA_TOKENS)
