from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

try:
    from .calibration_profiles import (
        classify_calibration_quality,
        get_profile,
        validate_calibration_audit,
    )
except ImportError:  # pragma: no cover - script execution fallback
    from calibration_profiles import (
        classify_calibration_quality,
        get_profile,
        validate_calibration_audit,
    )


CONTRACT_VERSION = 1
MANIFEST_FILENAME = "calibration_manifest.json"


def metadata_path_for(h5_path: str | Path) -> Path:
    return Path(f"{Path(h5_path)}.metadata.json")


def normalize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    normalized = json.loads(json.dumps(metadata))
    normalized.setdefault("contract_version", CONTRACT_VERSION)

    profile_data = normalized.get("profile", {})
    audit = normalized.setdefault("calibration_audit", {})
    constraints = audit.get("constraints", {})

    if "max_constraint_pct_error" not in audit:
        audit["max_constraint_pct_error"] = float(
            max(
                (
                    abs(stats.get("pct_error", 0.0))
                    for stats in constraints.values()
                ),
                default=0.0,
            )
        )

    if audit.get("lp_fallback_used"):
        realized_error = float(audit.get("max_constraint_pct_error", 0.0))
        stored_error = audit.get("approximate_solution_error_pct")
        if stored_error is None or float(stored_error) < realized_error:
            audit["approximate_solution_error_pct"] = realized_error

    if "calibration_quality" not in audit and profile_data.get("name"):
        try:
            profile = get_profile(profile_data["name"])
        except ValueError:
            profile = None
        if profile is not None:
            canonical_profile = profile.to_dict()
            merged_profile = json.loads(json.dumps(canonical_profile))
            merged_profile.update(profile_data)
            normalized["profile"] = merged_profile
            audit["calibration_quality"] = classify_calibration_quality(
                audit,
                profile,
                year=normalized.get("year"),
            )

    if audit.get("lp_fallback_used"):
        quality = audit.get("calibration_quality")
        if quality == "exact":
            audit["approximation_method"] = "lp_minimax_exact"
            audit["approximate_solution_used"] = False
        elif quality == "approximate":
            audit["approximation_method"] = "lp_minimax"
            audit["approximate_solution_used"] = True

    if "validation_passed" not in audit and profile_data.get("name"):
        try:
            profile = get_profile(profile_data["name"])
        except ValueError:
            profile = None
        if profile is not None:
            issues = validate_calibration_audit(
                audit,
                profile,
                year=normalized.get("year"),
            )
            audit["validation_passed"] = not bool(issues)
            audit.setdefault("validation_issues", issues)

    return normalized


def write_year_metadata(
    h5_path: str | Path,
    *,
    year: int,
    base_dataset_path: str,
    profile: dict[str, Any],
    calibration_audit: dict[str, Any],
    target_source: dict[str, Any] | None = None,
) -> Path:
    metadata = {
        "contract_version": CONTRACT_VERSION,
        "year": year,
        "base_dataset_path": base_dataset_path,
        "profile": profile,
        "calibration_audit": calibration_audit,
    }
    if target_source is not None:
        metadata["target_source"] = target_source
    metadata = normalize_metadata(metadata)
    metadata_path = metadata_path_for(h5_path)
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata_path


def update_dataset_manifest(
    output_dir: str | Path,
    *,
    year: int,
    h5_path: str | Path,
    metadata_path: str | Path,
    base_dataset_path: str,
    profile: dict[str, Any],
    calibration_audit: dict[str, Any],
    target_source: dict[str, Any] | None = None,
) -> Path:
    output_dir = Path(output_dir)
    manifest_path = output_dir / MANIFEST_FILENAME
    profile = json.loads(json.dumps(profile))
    target_source = json.loads(json.dumps(target_source))

    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {
            "contract_version": CONTRACT_VERSION,
            "generated_at": None,
            "base_dataset_path": base_dataset_path,
            "profile": profile,
            "target_source": target_source,
            "years": [],
            "datasets": {},
        }

    if manifest["base_dataset_path"] != base_dataset_path:
        raise ValueError(
            "Output directory already contains a different base dataset path: "
            f"{manifest['base_dataset_path']} != {base_dataset_path}"
        )
    manifest_profile = json.loads(json.dumps(manifest["profile"]))
    if manifest_profile != profile:
        if manifest_profile.get("name") == profile.get(
            "name"
        ) and manifest_profile.get("calibration_method") == profile.get(
            "calibration_method"
        ):
            manifest["profile"] = profile
        else:
            raise ValueError(
                "Output directory already contains a different calibration profile: "
                f"{manifest['profile'].get('name')} != {profile.get('name')}"
            )
    if manifest.get("target_source") is None and target_source is not None:
        manifest["target_source"] = target_source
    elif manifest.get("target_source") != target_source:
        raise ValueError(
            "Output directory already contains a different target source: "
            f"{manifest.get('target_source')} != {target_source}"
        )

    datasets = manifest.setdefault("datasets", {})
    datasets[str(year)] = {
        "h5": Path(h5_path).name,
        "metadata": Path(metadata_path).name,
        "calibration_quality": calibration_audit.get("calibration_quality"),
        "method_used": calibration_audit.get("method_used"),
        "fell_back_to_ipf": calibration_audit.get("fell_back_to_ipf"),
        "age_max_pct_error": calibration_audit.get("age_max_pct_error"),
        "max_constraint_pct_error": calibration_audit.get(
            "max_constraint_pct_error"
        ),
        "negative_weight_pct": calibration_audit.get("negative_weight_pct"),
        "negative_weight_household_pct": calibration_audit.get(
            "negative_weight_household_pct"
        ),
        "validation_passed": calibration_audit.get("validation_passed"),
        "validation_issue_count": len(
            calibration_audit.get("validation_issues", [])
        ),
    }

    year_set = {int(value) for value in manifest.get("years", [])}
    year_set.add(year)
    manifest["years"] = sorted(year_set)
    manifest["year_range"] = {
        "start": min(year_set),
        "end": max(year_set),
    }
    manifest["generated_at"] = datetime.now(timezone.utc).isoformat()
    manifest["contains_invalid_artifacts"] = any(
        entry.get("validation_passed") is False for entry in datasets.values()
    )

    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def rebuild_dataset_manifest(output_dir: str | Path) -> Path:
    return rebuild_dataset_manifest_with_target_source(output_dir)


def rebuild_dataset_manifest_with_target_source(
    output_dir: str | Path,
    *,
    target_source: dict[str, Any] | None = None,
) -> Path:
    output_dir = Path(output_dir)
    metadata_files = sorted(output_dir.glob("*.h5.metadata.json"))
    if not metadata_files:
        raise FileNotFoundError(f"No metadata sidecars found in {output_dir}")

    manifest_path: Path | None = None
    for metadata_file in metadata_files:
        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        metadata = normalize_metadata(metadata)
        if target_source is not None:
            metadata["target_source"] = target_source
        metadata_file.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        year = int(metadata["year"])
        h5_path = output_dir / f"{year}.h5"
        manifest_path = update_dataset_manifest(
            output_dir,
            year=year,
            h5_path=h5_path,
            metadata_path=metadata_file,
            base_dataset_path=metadata["base_dataset_path"],
            profile=metadata["profile"],
            calibration_audit=metadata["calibration_audit"],
            target_source=metadata.get("target_source"),
        )

    assert manifest_path is not None
    return manifest_path
