"""Guardrails for stable pydoc-facing pipeline objects."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]


try:
    from scripts import extract_pipeline_docs
except ModuleNotFoundError as exc:  # pragma: no cover - environment guard
    extract_pipeline_docs = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


def _module_path(source_file: str) -> str:
    return ".".join(Path(source_file).with_suffix("").parts)


def _top_level_name(object_path: str, source_file: str) -> str:
    module_path = _module_path(source_file)
    remainder = object_path.removeprefix(f"{module_path}.")
    return remainder.split(".", 1)[0]


def _declared_all(source_file: str) -> set[str] | None:
    path = REPO_ROOT / source_file
    try:
        tree = ast.parse(path.read_text(), filename=str(path))
    except SyntaxError:
        return None

    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == "__all__"
            for target in node.targets
        ):
            continue
        try:
            value = ast.literal_eval(node.value)
        except (ValueError, TypeError):
            return None
        if isinstance(value, (list, tuple, set)):
            return {item for item in value if isinstance(item, str)}
    return None


def _is_strict_pydoc_target(obj: Any) -> bool:
    metadata = obj.metadata
    return (
        metadata.get("pydoc", True)
        and metadata.get("node_type") == "library"
        and metadata.get("stability") == "stable"
    )


def _check_object(obj: Any) -> list[str]:
    violations = []
    context = f"{obj.source_file}:{obj.line} {obj.object_path}"

    if not obj.docstring.strip():
        violations.append(f"{context}: stable pydoc target needs a docstring")

    if obj.kind == "function" and " -> " not in obj.signature:
        violations.append(f"{context}: stable pydoc target needs a return annotation")

    declared_all = _declared_all(obj.source_file)
    if declared_all is not None:
        top_level_name = _top_level_name(obj.object_path, obj.source_file)
        if top_level_name not in declared_all:
            violations.append(
                f"{context}: {top_level_name!r} is missing from module __all__"
            )

    return violations


def check() -> list[str]:
    if IMPORT_ERROR is not None:
        return [
            "pydoc-completeness guard requires PyYAML; run through the dev "
            f"environment before validating ({IMPORT_ERROR})."
        ]

    objects = extract_pipeline_docs.scan_decorated_objects()
    violations = []
    for obj in objects.values():
        if _is_strict_pydoc_target(obj):
            violations.extend(_check_object(obj))
    return violations


def main() -> int:
    violations = check()
    if not violations:
        print("pydoc-completeness guard passed")
        return 0

    print("pydoc-completeness guard failed:")
    for violation in violations:
        print(f"  - {violation}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
