"""Guardrails for structured pipeline documentation metadata."""

from __future__ import annotations

import ast
import re
import shlex
from pathlib import Path
from typing import Any, get_args

from policyengine_us_data.pipeline_schema import (
    EdgeType,
    NodeStatus,
    NodeType,
    Stability,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SUBSTAGE_ID_PATTERN = re.compile(r"^[1-5][a-z]_[a-z0-9_]+$")


try:
    from scripts import extract_pipeline_docs
except ModuleNotFoundError as exc:  # pragma: no cover - environment guard
    extract_pipeline_docs = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


def _load_bundle() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    assert extract_pipeline_docs is not None
    manifest = extract_pipeline_docs.load_pipeline_map(
        extract_pipeline_docs.DEFAULT_MAP
    )
    objects = extract_pipeline_docs.scan_decorated_objects()
    bundle = extract_pipeline_docs.merge_map(manifest, objects)
    return manifest, objects, bundle


def _decorated_objects_by_id() -> dict[str, list[str]]:
    assert extract_pipeline_docs is not None
    objects_by_id: dict[str, list[str]] = {}
    for source_root in extract_pipeline_docs.SOURCE_ROOTS:
        if not source_root.exists():
            continue
        for path in sorted(source_root.rglob("*.py")):
            try:
                tree = ast.parse(path.read_text(), filename=str(path))
            except SyntaxError:
                continue
            visitor = extract_pipeline_docs.DecoratedObjectVisitor(
                extract_pipeline_docs.module_path_for_file(path),
                str(path.relative_to(REPO_ROOT)),
            )
            visitor.visit(tree)
            for obj in visitor.objects:
                objects_by_id.setdefault(obj.id, []).append(obj.object_path)
    return objects_by_id


def _check_duplicate_decorator_ids() -> list[str]:
    violations = []
    for node_id, object_paths in _decorated_objects_by_id().items():
        if len(object_paths) > 1:
            locations = ", ".join(object_paths)
            violations.append(
                f"duplicate @pipeline_node id {node_id!r} appears on {locations}"
            )
    return violations


def _check_stage_metadata(manifest: dict[str, Any]) -> list[str]:
    violations = []
    canonical_stage_ids = {
        stage["id"]: set(stage.get("manifest_step_ids", []))
        for stage in manifest.get("canonical_stages", [])
    }

    if not canonical_stage_ids:
        violations.append("docs/pipeline_map.yaml must define canonical_stages")

    for stage in manifest.get("stages", []):
        stage_id = stage.get("id", "")
        canonical_stage_id = stage.get("canonical_stage_id", "")
        if not SUBSTAGE_ID_PATTERN.fullmatch(stage_id):
            violations.append(
                f"{stage_id}: substage id must match "
                "'<canonical-stage-number><letter>_<name>'"
            )
        if canonical_stage_id not in canonical_stage_ids:
            violations.append(
                f"{stage_id}: canonical_stage_id {canonical_stage_id!r} is unknown"
            )
        elif stage_id and stage_id[0] != canonical_stage_id[0]:
            violations.append(
                f"{stage_id}: substage prefix does not match "
                f"canonical_stage_id {canonical_stage_id!r}"
            )

        allowed_manifest_steps = canonical_stage_ids.get(canonical_stage_id, set())
        unknown_manifest_steps = sorted(
            set(stage.get("manifest_step_ids", [])) - allowed_manifest_steps
        )
        if unknown_manifest_steps:
            violations.append(
                f"{stage_id}: manifest_step_ids are not declared by "
                f"{canonical_stage_id}: {unknown_manifest_steps}"
            )

        if stage.get("status", "unknown") not in get_args(NodeStatus):
            violations.append(f"{stage_id}: unknown status {stage.get('status')!r}")
        if stage.get("stability", "unknown") not in get_args(Stability):
            violations.append(
                f"{stage_id}: unknown stability {stage.get('stability')!r}"
            )
        if not stage.get("edges"):
            violations.append(f"{stage_id}: substage must declare at least one edge")

    return violations


def _check_node_and_edge_metadata(bundle: dict[str, Any]) -> list[str]:
    violations = []
    node_types = set(get_args(NodeType))
    statuses = set(get_args(NodeStatus))
    stabilities = set(get_args(Stability))
    edge_types = set(get_args(EdgeType))

    for stage in bundle.get("stages", []):
        stage_id = stage["id"]
        for node in stage.get("nodes", []):
            node_id = node.get("id", "")
            if node.get("node_type", "process") not in node_types:
                violations.append(
                    f"{stage_id}/{node_id}: unknown node_type {node.get('node_type')!r}"
                )
            if node.get("status", "unknown") not in statuses:
                violations.append(
                    f"{stage_id}/{node_id}: unknown status {node.get('status')!r}"
                )
            if node.get("stability", "unknown") not in stabilities:
                violations.append(
                    f"{stage_id}/{node_id}: unknown stability {node.get('stability')!r}"
                )
            violations.extend(
                _check_validation_commands(
                    node.get("validation_commands", []),
                    f"{stage_id}/{node_id}",
                )
            )

        for edge in stage.get("edges", []):
            if edge.get("edge_type", "data_flow") not in edge_types:
                violations.append(
                    f"{stage_id}: edge {edge.get('source')!r} -> "
                    f"{edge.get('target')!r} has unknown edge_type "
                    f"{edge.get('edge_type')!r}"
                )

    return violations


def _check_validation_commands(commands: list[str], context: str) -> list[str]:
    violations = []
    for command in commands:
        try:
            parts = shlex.split(command)
        except ValueError as exc:
            violations.append(
                f"{context}: invalid validation command {command!r}: {exc}"
            )
            continue

        for part in parts:
            if (
                part.startswith(("tests/", "scripts/"))
                and not (REPO_ROOT / part).exists()
            ):
                violations.append(
                    f"{context}: validation command references missing path {part}"
                )

    return violations


def check() -> list[str]:
    if IMPORT_ERROR is not None:
        return [
            "pipeline-docs guard requires PyYAML; run through the dev "
            f"environment before validating ({IMPORT_ERROR})."
        ]

    try:
        manifest, _objects, bundle = _load_bundle()
    except SystemExit as exc:
        return [f"pipeline map validation failed: {exc}"]

    return [
        *_check_duplicate_decorator_ids(),
        *_check_stage_metadata(manifest),
        *_check_node_and_edge_metadata(bundle),
    ]


def main() -> int:
    violations = check()
    if not violations:
        print("pipeline-docs guard passed")
        return 0

    print("pipeline-docs guard failed:")
    for violation in violations:
        print(f"  - {violation}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
