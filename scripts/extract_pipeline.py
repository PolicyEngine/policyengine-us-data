"""Extract pipeline metadata from decorated source code and YAML manifest.

Reads @pipeline_node decorators from Python source files via AST (no imports),
merges with pipeline_stages.yaml for stage groupings and edges, validates
the schema, and exports pipeline.json for React visualization.

Usage:
    python scripts/extract_pipeline.py
    python scripts/extract_pipeline.py --output docs/pipeline-diagrams/app/pipeline.json
"""

import ast
import json
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
STAGES_YAML = REPO_ROOT / "pipeline_stages.yaml"
DEFAULT_OUTPUT = REPO_ROOT / "docs" / "pipeline-diagrams" / "app" / "pipeline.json"

# Valid node and edge types (from pipeline_schema.py)
VALID_NODE_TYPES = {
    "input",
    "output",
    "process",
    "utility",
    "external",
    "us_specific",
    "uk_specific",
    "missing",
    "absent",
}
VALID_EDGE_TYPES = {
    "data_flow",
    "produces_artifact",
    "uses_utility",
    "external_source",
    "runs_on_infra",
    "informational",
}

# Decorated code nodes that are intentionally not rendered as graph nodes.
# The diagrams document these flows through finer-grained YAML nodes instead.
IGNORED_CODE_NODE_IDS = {
    "create_stratified": "Stage 3b shows this wrapper as a visual group around AGI/top-1%/sample steps",
    "run_calibration": "Stages 5-6 show this wrapper as visual groups around build/fit/output steps",
}


def extract_pipeline_nodes_from_file(filepath: Path) -> list[dict]:
    """Parse a Python file's AST and extract @pipeline_node decorator data.

    Returns a list of node dicts extracted from PipelineNode(...) calls
    in decorators, without importing the module.
    """
    try:
        source = filepath.read_text()
        tree = ast.parse(source, filename=str(filepath))
    except SyntaxError as e:
        print(f"  WARNING: Syntax error in {filepath}: {e}", file=sys.stderr)
        return []

    nodes = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        for decorator in node.decorator_list:
            if not isinstance(decorator, ast.Call):
                continue

            # Check if it's @pipeline_node(...)
            func = decorator.func
            name = ""
            if isinstance(func, ast.Name):
                name = func.id
            elif isinstance(func, ast.Attribute):
                name = func.attr

            if name != "pipeline_node":
                continue

            # Extract the PipelineNode(...) argument
            if not decorator.args:
                continue

            arg = decorator.args[0]
            if not isinstance(arg, ast.Call):
                continue

            # Parse keyword arguments from PipelineNode(...)
            node_data = {}
            for kw in arg.keywords:
                try:
                    node_data[kw.arg] = ast.literal_eval(kw.value)
                except (ValueError, TypeError):
                    node_data[kw.arg] = str(ast.dump(kw.value))

            if "id" in node_data:
                nodes.append(node_data)

    return nodes


def scan_source_files(source_dir: Path) -> dict[str, list[dict]]:
    """Scan all Python files under source_dir for @pipeline_node decorators.

    Returns a dict mapping relative file paths to lists of node dicts.
    """
    results = {}
    for py_file in sorted(source_dir.rglob("*.py")):
        rel_path = str(py_file.relative_to(REPO_ROOT))
        nodes = extract_pipeline_nodes_from_file(py_file)
        if nodes:
            results[rel_path] = nodes
    return results


def load_stages_yaml() -> dict:
    """Load pipeline_stages.yaml manifest."""
    with open(STAGES_YAML) as f:
        return yaml.safe_load(f)


def build_pipeline_json(output_path: Path = DEFAULT_OUTPUT):
    """Build the full pipeline.json from source code + YAML manifest."""
    print("Extracting pipeline metadata...")

    # Step 1: Scan source code for @pipeline_node decorators
    source_dir = REPO_ROOT / "policyengine_us_data"
    file_nodes = scan_source_files(source_dir)

    total_nodes = sum(len(v) for v in file_nodes.values())
    print(
        f"  Found {total_nodes} @pipeline_node decorators across {len(file_nodes)} files"
    )

    # Build a lookup: node_id → node_data
    all_code_nodes = {}
    for filepath, nodes in file_nodes.items():
        for node in nodes:
            node_id = node["id"]
            if node_id in all_code_nodes:
                print(
                    f"  WARNING: Duplicate node ID '{node_id}' in {filepath} "
                    f"(already defined in {all_code_nodes[node_id].get('source_file', '?')})",
                    file=sys.stderr,
                )
            all_code_nodes[node_id] = node

    # Step 2: Load YAML manifest
    manifest = load_stages_yaml()
    stages_data = manifest.get("stages", [])

    # Step 3: Merge code nodes with YAML stages
    stages_output = []
    used_code_node_ids = set()
    for stage_def in stages_data:
        stage = {
            "id": stage_def["id"],
            "label": stage_def["label"],
            "title": stage_def["title"],
            "description": stage_def["description"],
            "country": stage_def.get("country", "us"),
            "nodes": [],
            "edges": [],
            "groups": stage_def.get("groups", []),
        }

        # Add extra nodes from YAML (inputs, outputs, utilities)
        for extra in stage_def.get("extra_nodes", []):
            node_type = extra.get("node_type", "input")
            if node_type not in VALID_NODE_TYPES:
                print(
                    f"  WARNING: Invalid node_type '{node_type}' for node '{extra['id']}'",
                    file=sys.stderr,
                )
            stage["nodes"].append(extra)

        # Add code-defined nodes that belong to this stage
        # (matched by presence in the YAML edges as source or target)
        existing_ids = {n["id"] for n in stage["nodes"]}
        for edge in stage_def.get("edges", []):
            for node_id in (edge["source"], edge["target"]):
                if node_id not in all_code_nodes:
                    continue
                used_code_node_ids.add(node_id)
                if node_id not in existing_ids:
                    stage["nodes"].append(all_code_nodes[node_id])
                    existing_ids.add(node_id)

        # Add edges
        for edge in stage_def.get("edges", []):
            edge_type = edge.get("edge_type", "data_flow")
            if edge_type not in VALID_EDGE_TYPES:
                print(
                    f"  WARNING: Invalid edge_type '{edge_type}' for edge "
                    f"'{edge['source']}' → '{edge['target']}'",
                    file=sys.stderr,
                )
            stage["edges"].append(edge)

        stages_output.append(stage)

    # Step 4: Validate
    errors = 0
    unused_code_node_ids = set(all_code_nodes) - used_code_node_ids
    ignored_unused = unused_code_node_ids & set(IGNORED_CODE_NODE_IDS)
    unexpected_unused = unused_code_node_ids - set(IGNORED_CODE_NODE_IDS)

    for node_id in sorted(ignored_unused):
        print(
            f"  INFO: Decorated node '{node_id}' is intentionally omitted: "
            f"{IGNORED_CODE_NODE_IDS[node_id]}"
        )

    for node_id in sorted(unexpected_unused):
        print(
            f"  ERROR: Decorated node '{node_id}' is not referenced by "
            "pipeline_stages.yaml edges and is not in IGNORED_CODE_NODE_IDS",
            file=sys.stderr,
        )
        errors += 1

    for stage in stages_output:
        node_ids = {n["id"] for n in stage["nodes"]}
        for edge in stage["edges"]:
            if edge["source"] not in node_ids:
                print(
                    f"  ERROR: Edge source '{edge['source']}' not found in "
                    f"stage {stage['id']} nodes",
                    file=sys.stderr,
                )
                errors += 1

        for group in stage.get("groups", []):
            missing_group_nodes = [
                node_id
                for node_id in group.get("node_ids", [])
                if node_id not in node_ids
            ]
            if missing_group_nodes:
                print(
                    f"  ERROR: Group '{group['id']}' references missing nodes "
                    f"in stage {stage['id']}: {', '.join(missing_group_nodes)}",
                    file=sys.stderr,
                )
                errors += 1
            if edge["target"] not in node_ids:
                print(
                    f"  ERROR: Edge target '{edge['target']}' not found in "
                    f"stage {stage['id']} nodes",
                    file=sys.stderr,
                )
                errors += 1

    if errors:
        raise SystemExit(f"\n  {errors} validation error(s) found")

    # Step 5: Build output
    pipeline_json = {
        "stages": stages_output,
        "metadata": {
            "total_nodes": sum(len(s["nodes"]) for s in stages_output),
            "total_edges": sum(len(s["edges"]) for s in stages_output),
        },
    }

    # Step 6: Write
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(pipeline_json, f, indent=2)
        f.write("\n")

    print(f"\nWrote {output_path}")
    print(
        f"  {len(stages_output)} stages, "
        f"{pipeline_json['metadata']['total_nodes']} nodes, "
        f"{pipeline_json['metadata']['total_edges']} edges"
    )

    return pipeline_json


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        "-o",
        default=str(DEFAULT_OUTPUT),
        help="Output path for pipeline.json",
    )
    args = parser.parse_args()
    build_pipeline_json(Path(args.output))
