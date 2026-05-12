from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from importlib import metadata as importlib_metadata
import json
from pathlib import Path
import subprocess
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
SUPPORT_AUGMENTATION_REPORT_FILENAME = "support_augmentation_report.json"


def metadata_path_for(h5_path: str | Path) -> Path:
    return Path(f"{Path(h5_path)}.metadata.json")


def _json_clone(value: Any) -> Any:
    return json.loads(json.dumps(value))


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _find_git_repo_root(path: Path) -> Path | None:
    current = path if path.is_dir() else path.parent
    for candidate in (current, *current.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def capture_policyengine_us_provenance() -> dict[str, Any]:
    import policyengine_us

    package_file = Path(policyengine_us.__file__).resolve()
    version = getattr(policyengine_us, "__version__", None)
    if version is None:
        try:
            version = importlib_metadata.version("policyengine-us")
        except importlib_metadata.PackageNotFoundError:
            version = None
    provenance: dict[str, Any] = {
        "package_file": str(package_file),
        "package_file_sha256": _sha256_file(package_file),
        "package_mtime_ns": package_file.stat().st_mtime_ns,
        "package_size": package_file.stat().st_size,
        "version": version,
    }
    repo_root = _find_git_repo_root(package_file)
    if repo_root is None:
        return provenance

    provenance["repo_root"] = str(repo_root)
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if head.returncode == 0:
        provenance["git_head"] = head.stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain=v1"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if status.returncode == 0:
        provenance["git_dirty"] = bool(status.stdout.strip())
    return provenance


def _resolve_base_dataset_path(base_dataset_path: str) -> Path | None:
    if base_dataset_path.startswith("hf://"):
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            return None
        rel = base_dataset_path.removeprefix("hf://")
        parts = rel.split("/")
        if len(parts) < 3:
            return None
        repo_id = "/".join(parts[:2])
        filename = "/".join(parts[2:])
        try:
            return Path(
                hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    local_files_only=True,
                )
            ).resolve()
        except Exception:
            return None

    candidate = Path(base_dataset_path).expanduser()
    if candidate.exists():
        return candidate.resolve()
    return None


def capture_base_dataset_snapshot(base_dataset_path: str) -> dict[str, Any]:
    snapshot: dict[str, Any] = {"requested_path": base_dataset_path}
    resolved = _resolve_base_dataset_path(base_dataset_path)
    if resolved is None or not resolved.exists():
        return snapshot

    snapshot["resolved_path"] = str(resolved)
    snapshot["resolved_file_sha256"] = _sha256_file(resolved)
    snapshot["resolved_size"] = resolved.stat().st_size
    snapshot["resolved_mtime_ns"] = resolved.stat().st_mtime_ns
    if "snapshots" in resolved.parts:
        snapshot_index = resolved.parts.index("snapshots")
        if snapshot_index + 1 < len(resolved.parts):
            snapshot["huggingface_snapshot"] = resolved.parts[snapshot_index + 1]
    return snapshot


def _normalize_tax_assumption_contract(
    value: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if value is None:
        return None
    normalized = _json_clone(value)
    # One-year long-run runs stamp the run year into end_year even when the
    # underlying tax-assumption contract is otherwise identical. Ignore that
    # run-local field when deciding whether multiple artifacts share a manifest.
    normalized.pop("end_year", None)
    return normalized


def _normalize_support_augmentation_contract(
    value: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if value is None:
        return None
    normalized = _json_clone(value)
    # Dynamic support augmentation records run-local reporting details that vary
    # by year even when the augmentation contract is otherwise identical.
    normalized.pop("target_year", None)
    normalized.pop("report_file", None)
    normalized.pop("report_summary", None)
    return normalized


def normalize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    normalized = json.loads(json.dumps(metadata))
    normalized.setdefault("contract_version", CONTRACT_VERSION)

    profile_data = normalized.get("profile", {})
    audit = normalized.setdefault("calibration_audit", {})
    constraints = audit.get("constraints", {})

    if "max_constraint_pct_error" not in audit:
        audit["max_constraint_pct_error"] = float(
            max(
                (abs(stats.get("pct_error", 0.0)) for stats in constraints.values()),
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
    tax_assumption: dict[str, Any] | None = None,
    support_augmentation: dict[str, Any] | None = None,
    policyengine_us: dict[str, Any] | None = None,
    base_dataset_snapshot: dict[str, Any] | None = None,
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
    if tax_assumption is not None:
        metadata["tax_assumption"] = tax_assumption
    if support_augmentation is not None:
        metadata["support_augmentation"] = support_augmentation
    if policyengine_us is not None:
        metadata["policyengine_us"] = policyengine_us
    if base_dataset_snapshot is not None:
        metadata["base_dataset_snapshot"] = base_dataset_snapshot
    metadata = normalize_metadata(metadata)
    metadata_path = metadata_path_for(h5_path)
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata_path


def write_support_augmentation_report(
    output_dir: str | Path,
    report: dict[str, Any],
    *,
    filename: str = SUPPORT_AUGMENTATION_REPORT_FILENAME,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / filename
    report_path.write_text(
        json.dumps(json.loads(json.dumps(report)), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report_path


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
    tax_assumption: dict[str, Any] | None = None,
    support_augmentation: dict[str, Any] | None = None,
    policyengine_us: dict[str, Any] | None = None,
    base_dataset_snapshot: dict[str, Any] | None = None,
) -> Path:
    output_dir = Path(output_dir)
    manifest_path = output_dir / MANIFEST_FILENAME
    profile = _json_clone(profile)
    target_source = _json_clone(target_source)
    tax_assumption = _json_clone(tax_assumption)
    support_augmentation = _json_clone(support_augmentation)
    policyengine_us = _json_clone(policyengine_us)
    base_dataset_snapshot = _json_clone(base_dataset_snapshot)

    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {
            "contract_version": CONTRACT_VERSION,
            "generated_at": None,
            "base_dataset_path": base_dataset_path,
            "profile": profile,
            "target_source": target_source,
            "tax_assumption": tax_assumption,
            "support_augmentation": support_augmentation,
            "policyengine_us": policyengine_us,
            "base_dataset_snapshot": base_dataset_snapshot,
            "years": [],
            "datasets": {},
        }

    if manifest["base_dataset_path"] != base_dataset_path:
        raise ValueError(
            "Output directory already contains a different base dataset path: "
            f"{manifest['base_dataset_path']} != {base_dataset_path}"
        )
    manifest_profile = _json_clone(manifest["profile"])
    if manifest_profile != profile:
        if manifest_profile.get("name") == profile.get("name") and manifest_profile.get(
            "calibration_method"
        ) == profile.get("calibration_method"):
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
    if manifest.get("tax_assumption") is None and tax_assumption is not None:
        manifest["tax_assumption"] = tax_assumption
    elif _normalize_tax_assumption_contract(
        manifest.get("tax_assumption")
    ) != _normalize_tax_assumption_contract(tax_assumption):
        raise ValueError(
            "Output directory already contains a different tax assumption: "
            f"{manifest.get('tax_assumption')} != {tax_assumption}"
        )
    elif tax_assumption is not None:
        manifest["tax_assumption"] = tax_assumption
    if (
        manifest.get("support_augmentation") is None
        and support_augmentation is not None
    ):
        manifest["support_augmentation"] = support_augmentation
    elif _normalize_support_augmentation_contract(
        manifest.get("support_augmentation")
    ) != _normalize_support_augmentation_contract(support_augmentation):
        raise ValueError(
            "Output directory already contains a different support augmentation: "
            f"{manifest.get('support_augmentation')} != {support_augmentation}"
        )
    elif support_augmentation is not None:
        manifest["support_augmentation"] = support_augmentation
    if manifest.get("policyengine_us") is None and policyengine_us is not None:
        manifest["policyengine_us"] = policyengine_us
    elif manifest.get("policyengine_us") != policyengine_us:
        raise ValueError(
            "Output directory already contains a different policyengine_us provenance: "
            f"{manifest.get('policyengine_us')} != {policyengine_us}"
        )
    if (
        manifest.get("base_dataset_snapshot") is None
        and base_dataset_snapshot is not None
    ):
        manifest["base_dataset_snapshot"] = base_dataset_snapshot
    elif manifest.get("base_dataset_snapshot") != base_dataset_snapshot:
        raise ValueError(
            "Output directory already contains a different base dataset snapshot: "
            f"{manifest.get('base_dataset_snapshot')} != {base_dataset_snapshot}"
        )

    datasets = manifest.setdefault("datasets", {})
    datasets[str(year)] = {
        "h5": Path(h5_path).name,
        "metadata": Path(metadata_path).name,
        "calibration_quality": calibration_audit.get("calibration_quality"),
        "method_used": calibration_audit.get("method_used"),
        "fell_back_to_ipf": calibration_audit.get("fell_back_to_ipf"),
        "age_max_pct_error": calibration_audit.get("age_max_pct_error"),
        "max_constraint_pct_error": calibration_audit.get("max_constraint_pct_error"),
        "negative_weight_pct": calibration_audit.get("negative_weight_pct"),
        "negative_weight_household_pct": calibration_audit.get(
            "negative_weight_household_pct"
        ),
        "validation_passed": calibration_audit.get("validation_passed"),
        "validation_issue_count": len(calibration_audit.get("validation_issues", [])),
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
            tax_assumption=metadata.get("tax_assumption"),
            support_augmentation=metadata.get("support_augmentation"),
            policyengine_us=metadata.get("policyengine_us"),
            base_dataset_snapshot=metadata.get("base_dataset_snapshot"),
        )

    assert manifest_path is not None
    return manifest_path
