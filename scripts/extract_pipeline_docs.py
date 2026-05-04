"""Extract structured pipeline docs and pydoc-style object summaries.

The extractor reads ``@pipeline_node`` decorators without importing project
modules, merges them with ``docs/pipeline_map.yaml``, validates edges, and emits
neutral artifacts for downstream docs/diagram tooling.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MAP = REPO_ROOT / "docs" / "pipeline_map.yaml"
DEFAULT_JSON = REPO_ROOT / "docs" / "generated" / "pipeline_map.json"
DEFAULT_API_JSON = REPO_ROOT / "docs" / "generated" / "pipeline_api.json"
DEFAULT_MARKDOWN = REPO_ROOT / "docs" / "engineering" / "pipeline-map.md"
SOURCE_ROOTS = (
    REPO_ROOT / "policyengine_us_data",
    REPO_ROOT / "modal_app",
)

VALID_DECORATORS = {"pipeline_node"}


@dataclass
class DocumentedObject:
    """Static pydoc-style summary extracted from a Python object."""

    id: str
    object_path: str
    source_file: str
    line: int
    kind: str
    signature: str
    docstring: str
    metadata: dict[str, Any]


def _literal_dict_from_call(call: ast.Call) -> dict[str, Any]:
    data: dict[str, Any] = {}
    for keyword in call.keywords:
        if keyword.arg is None:
            continue
        try:
            data[keyword.arg] = ast.literal_eval(keyword.value)
        except (ValueError, TypeError):
            data[keyword.arg] = ast.unparse(keyword.value)
    return data


def _decorator_name(decorator: ast.AST) -> str:
    if isinstance(decorator, ast.Call):
        decorator = decorator.func
    if isinstance(decorator, ast.Name):
        return decorator.id
    if isinstance(decorator, ast.Attribute):
        return decorator.attr
    return ""


def _extract_decorator_data(decorator: ast.AST) -> dict[str, Any] | None:
    if not isinstance(decorator, ast.Call):
        return None
    if _decorator_name(decorator) not in VALID_DECORATORS:
        return None

    data: dict[str, Any] = {}
    if decorator.args and isinstance(decorator.args[0], ast.Call):
        data.update(_literal_dict_from_call(decorator.args[0]))
    data.update(_literal_dict_from_call(decorator))

    if "id" not in data:
        return None
    return data


def _annotation_text(node: ast.AST | None) -> str:
    if node is None:
        return ""
    return ast.unparse(node)


def _signature(node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> str:
    if isinstance(node, ast.ClassDef):
        bases = [ast.unparse(base) for base in node.bases]
        return (
            f"class {node.name}({', '.join(bases)})" if bases else f"class {node.name}"
        )

    parts: list[str] = []
    args = node.args
    defaults = [None] * (len(args.args) - len(args.defaults)) + list(args.defaults)
    for arg, default in zip(args.args, defaults):
        text = arg.arg
        annotation = _annotation_text(arg.annotation)
        if annotation:
            text += f": {annotation}"
        if default is not None:
            text += f" = {ast.unparse(default)}"
        parts.append(text)
    if args.vararg:
        parts.append(f"*{args.vararg.arg}")
    elif args.kwonlyargs:
        parts.append("*")
    kw_defaults = args.kw_defaults
    for arg, default in zip(args.kwonlyargs, kw_defaults):
        text = arg.arg
        annotation = _annotation_text(arg.annotation)
        if annotation:
            text += f": {annotation}"
        if default is not None:
            text += f" = {ast.unparse(default)}"
        parts.append(text)
    if args.kwarg:
        parts.append(f"**{args.kwarg.arg}")
    returns = _annotation_text(node.returns)
    suffix = f" -> {returns}" if returns else ""
    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    return f"{prefix} {node.name}({', '.join(parts)}){suffix}"


class DecoratedObjectVisitor(ast.NodeVisitor):
    """Collect decorated functions/classes with qualified names."""

    def __init__(self, module_path: str, source_file: str):
        self.module_path = module_path
        self.source_file = source_file
        self.stack: list[str] = []
        self.objects: list[DocumentedObject] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._maybe_record(node, "class")
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._maybe_record(node, "function")
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._maybe_record(node, "function")
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    def _maybe_record(
        self,
        node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef,
        kind: str,
    ) -> None:
        metadata = None
        for decorator in node.decorator_list:
            metadata = _extract_decorator_data(decorator)
            if metadata is not None:
                break
        if metadata is None:
            return

        qualname = ".".join([*self.stack, node.name])
        object_path = f"{self.module_path}.{qualname}"
        metadata.setdefault("source_file", self.source_file)
        metadata.setdefault("api_refs", [])
        if object_path not in metadata["api_refs"]:
            metadata["api_refs"].append(object_path)

        self.objects.append(
            DocumentedObject(
                id=metadata["id"],
                object_path=object_path,
                source_file=self.source_file,
                line=node.lineno,
                kind=kind,
                signature=_signature(node),
                docstring=ast.get_docstring(node) or "",
                metadata=metadata,
            )
        )


def module_path_for_file(path: Path) -> str:
    relative = path.relative_to(REPO_ROOT).with_suffix("")
    return ".".join(relative.parts)


def scan_decorated_objects() -> dict[str, DocumentedObject]:
    objects: dict[str, DocumentedObject] = {}
    for source_root in SOURCE_ROOTS:
        if not source_root.exists():
            continue
        for path in sorted(source_root.rglob("*.py")):
            try:
                tree = ast.parse(path.read_text(), filename=str(path))
            except SyntaxError as exc:
                print(f"WARNING: skipping {path}: {exc}", file=sys.stderr)
                continue
            source_file = str(path.relative_to(REPO_ROOT))
            visitor = DecoratedObjectVisitor(module_path_for_file(path), source_file)
            visitor.visit(tree)
            for obj in visitor.objects:
                if obj.id in objects:
                    print(
                        f"WARNING: duplicate pipeline id {obj.id!r}: "
                        f"{objects[obj.id].object_path} and {obj.object_path}",
                        file=sys.stderr,
                    )
                objects[obj.id] = obj
    return objects


def load_pipeline_map(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"stages": []}
    with path.open() as file:
        return yaml.safe_load(file) or {"stages": []}


def merge_map(
    manifest: dict[str, Any], objects: dict[str, DocumentedObject]
) -> dict[str, Any]:
    object_nodes = {
        node_id: asdict(obj)["metadata"] for node_id, obj in objects.items()
    }
    manifest_nodes = {
        node["id"]: node
        for stage_def in manifest.get("stages", [])
        for node in stage_def.get("extra_nodes", [])
    }
    canonical_stages = manifest.get("canonical_stages", [])
    canonical_stage_by_id = {
        canonical_stage["id"]: canonical_stage
        for canonical_stage in canonical_stages
    }
    used_node_ids: set[str] = set()
    stages: list[dict[str, Any]] = []
    errors = 0

    for stage_def in manifest.get("stages", []):
        canonical_stage_id = stage_def.get("canonical_stage_id")
        canonical_stage = (
            canonical_stage_by_id.get(canonical_stage_id)
            if canonical_stage_id
            else None
        )
        if canonical_stage_id and canonical_stage is None:
            print(
                f"ERROR: stage {stage_def.get('id')} references unknown "
                f"canonical stage {canonical_stage_id!r}",
                file=sys.stderr,
            )
            errors += 1

        nodes: list[dict[str, Any]] = []
        node_ids: set[str] = set()
        for node in stage_def.get("extra_nodes", []):
            nodes.append(node)
            node_ids.add(node["id"])

        for edge in stage_def.get("edges", []):
            for endpoint in (edge["source"], edge["target"]):
                if endpoint in object_nodes and endpoint not in node_ids:
                    nodes.append(object_nodes[endpoint])
                    node_ids.add(endpoint)
                    used_node_ids.add(endpoint)
                elif endpoint in manifest_nodes and endpoint not in node_ids:
                    nodes.append(manifest_nodes[endpoint])
                    node_ids.add(endpoint)

        for edge in stage_def.get("edges", []):
            for role in ("source", "target"):
                if edge[role] not in node_ids:
                    print(
                        f"ERROR: stage {stage_def.get('id')} edge references "
                        f"unknown {role} node {edge[role]!r}",
                        file=sys.stderr,
                    )
                    errors += 1

        stages.append(
            {
                "id": stage_def["id"],
                "label": stage_def["label"],
                "title": stage_def["title"],
                "description": stage_def.get("description", ""),
                "canonical_stage_id": canonical_stage_id,
                "canonical_stage_title": (
                    canonical_stage.get("title") if canonical_stage else None
                ),
                "legacy_stage_id": stage_def.get("legacy_stage_id"),
                "manifest_step_ids": stage_def.get("manifest_step_ids", []),
                "status": stage_def.get("status", "unknown"),
                "stability": stage_def.get("stability", "unknown"),
                "groups": stage_def.get("groups", []),
                "nodes": nodes,
                "edges": stage_def.get("edges", []),
            }
        )

    api_nodes = [
        object_nodes[node_id]
        for node_id in sorted(set(object_nodes) - used_node_ids)
        if object_nodes[node_id].get("pydoc", True)
    ]

    if errors:
        raise SystemExit(f"{errors} pipeline map validation error(s)")

    return {
        "canonical_stages": canonical_stages,
        "stages": stages,
        "api_nodes": api_nodes,
        "metadata": {
            "canonical_stage_count": len(canonical_stages),
            "stage_count": len(stages),
            "substage_count": len(stages),
            "decorated_object_count": len(objects),
            "mapped_decorated_node_count": len(used_node_ids),
            "api_node_count": len(api_nodes),
        },
    }


def render_markdown(
    bundle: dict[str, Any], objects: dict[str, DocumentedObject]
) -> str:
    lines = [
        "# Pipeline Map",
        "",
        "Generated from `docs/pipeline_map.yaml` and `@pipeline_node` decorators.",
        "",
    ]
    canonical_stages = bundle.get("canonical_stages", [])
    if canonical_stages:
        lines.extend(
            [
                "## Canonical Stages",
                "",
                "| Stage | Title | Manifest steps |",
                "| --- | --- | --- |",
            ]
        )
        for canonical_stage in canonical_stages:
            manifest_steps = ", ".join(
                f"`{step_id}`"
                for step_id in canonical_stage.get("manifest_step_ids", [])
            )
            lines.append(
                f"| `{canonical_stage['id']}` "
                f"{canonical_stage.get('label', '')} | "
                f"{canonical_stage.get('title', '')} | "
                f"{manifest_steps} |"
            )
        lines.append("")

    stages_by_canonical_id: dict[str | None, list[dict[str, Any]]] = {}
    for stage in bundle["stages"]:
        stages_by_canonical_id.setdefault(stage.get("canonical_stage_id"), []).append(
            stage
        )

    def render_stage(stage: dict[str, Any], *, heading: str = "###") -> None:
        lines.extend(
            [
                f"{heading} {stage['title']}",
                "",
                stage.get("description", ""),
                "",
                f"- Substage ID: `{stage['id']}`",
                f"- Canonical stage: "
                f"`{stage.get('canonical_stage_id') or 'unknown'}`",
                f"- Legacy stage: `{stage.get('legacy_stage_id', 'none')}`",
                "- Manifest steps: "
                + (
                    ", ".join(
                        f"`{step_id}`"
                        for step_id in stage.get("manifest_step_ids", [])
                    )
                    or "`none`"
                ),
                f"- Status: `{stage.get('status', 'unknown')}`",
                f"- Stability: `{stage.get('stability', 'unknown')}`",
                "",
                "| Node | Type | Status | Stability | API refs |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        for node in stage["nodes"]:
            refs = ", ".join(f"`{ref}`" for ref in node.get("api_refs", []))
            lines.append(
                f"| `{node['id']}` {node.get('label', '')} | "
                f"`{node.get('node_type', 'process')}` | "
                f"`{node.get('status', 'unknown')}` | "
                f"`{node.get('stability', 'unknown')}` | {refs} |"
            )
        edge_heading = "#" * (len(heading) + 1)
        lines.extend(["", f"{edge_heading} Edges", ""])
        for edge in stage["edges"]:
            label = f" ({edge['label']})" if edge.get("label") else ""
            lines.append(
                f"- `{edge['source']}` -> `{edge['target']}` "
                f"`{edge.get('edge_type', 'data_flow')}`{label}"
            )
        lines.append("")

    if canonical_stages:
        for canonical_stage in canonical_stages:
            lines.extend(
                [
                    f"## {canonical_stage.get('label', canonical_stage['id'])}: "
                    f"{canonical_stage.get('title', '')}",
                    "",
                    canonical_stage.get("description", ""),
                    "",
                ]
            )
            for stage in stages_by_canonical_id.get(canonical_stage["id"], []):
                render_stage(stage)
        for stage in stages_by_canonical_id.get(None, []):
            render_stage(stage)
    else:
        for stage in bundle["stages"]:
            render_stage(stage, heading="##")

    if bundle["api_nodes"]:
        lines.extend(["## Pydoc API Surface", ""])
        for node in bundle["api_nodes"]:
            obj = objects.get(node["id"])
            if not obj:
                continue
            doc = (
                obj.docstring.splitlines()[0]
                if obj.docstring
                else node.get("description", "")
            )
            lines.extend(
                [
                    f"### `{obj.object_path}`",
                    "",
                    f"```python\n{obj.signature}\n```",
                    "",
                    doc,
                    "",
                ]
            )
    return "\n".join(lines).rstrip() + "\n"


def write_outputs(
    bundle: dict[str, Any],
    objects: dict[str, DocumentedObject],
    json_path: Path,
    api_json_path: Path,
    markdown_path: Path,
) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    api_json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)

    json_path.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n")
    api_payload = {
        obj_id: asdict(obj)
        for obj_id, obj in sorted(objects.items())
        if obj.metadata.get("pydoc", True)
    }
    api_json_path.write_text(json.dumps(api_payload, indent=2, sort_keys=True) + "\n")
    markdown_path.write_text(render_markdown(bundle, objects))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", default=str(DEFAULT_MAP))
    parser.add_argument("--json", default=str(DEFAULT_JSON))
    parser.add_argument("--api-json", default=str(DEFAULT_API_JSON))
    parser.add_argument("--markdown", default=str(DEFAULT_MARKDOWN))
    args = parser.parse_args(argv)

    objects = scan_decorated_objects()
    manifest = load_pipeline_map(Path(args.map))
    bundle = merge_map(manifest, objects)
    write_outputs(
        bundle,
        objects,
        Path(args.json),
        Path(args.api_json),
        Path(args.markdown),
    )
    print(
        f"Extracted {len(objects)} decorated objects into "
        f"{args.json}, {args.api_json}, and {args.markdown}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
