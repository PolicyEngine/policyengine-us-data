"""Source-level contracts for Modal pipeline orchestration."""

from __future__ import annotations

import ast
from pathlib import Path


PIPELINE_SOURCE = Path("modal_app/pipeline.py")


def _function_def(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"Could not find function {name}")


def test_promote_run_passes_version_to_national_promotion() -> None:
    tree = ast.parse(PIPELINE_SOURCE.read_text())
    promote_run = _function_def(tree, "promote_run")

    national_calls = [
        node
        for node in ast.walk(promote_run)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "remote"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "promote_national_publish"
    ]

    assert len(national_calls) == 1
    keyword_names = {keyword.arg for keyword in national_calls[0].keywords}
    assert {"branch", "version", "run_id"}.issubset(keyword_names)


def test_run_pipeline_stage_1_builds_without_publishing() -> None:
    tree = ast.parse(PIPELINE_SOURCE.read_text())
    run_pipeline = _function_def(tree, "run_pipeline")

    build_calls = [
        node
        for node in ast.walk(run_pipeline)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "remote"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "build_datasets"
    ]

    assert len(build_calls) == 1
    upload_keywords = [
        keyword for keyword in build_calls[0].keywords if keyword.arg == "upload"
    ]
    assert len(upload_keywords) == 1
    assert isinstance(upload_keywords[0].value, ast.Constant)
    assert upload_keywords[0].value.value is False
