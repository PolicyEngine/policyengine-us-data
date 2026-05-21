"""Focused source guards for build-only unified calibration package behavior."""

import ast
from pathlib import Path


def _call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        owner = _call_name(node.value)
        return f"{owner}.{node.attr}" if owner else node.attr
    return None


def test_build_only_package_output_returns_before_weight_fitting():
    source = Path("policyengine_us_data/calibration/unified_calibration.py").read_text(
        encoding="utf-8"
    )
    module = ast.parse(source)
    run_calibration = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "run_calibration"
    )
    build_only_block = next(
        node
        for node in run_calibration.body
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Name)
        and node.test.id == "build_only"
    )

    build_only_calls = {
        _call_name(call.func)
        for call in ast.walk(build_only_block)
        if isinstance(call, ast.Call)
    }
    fit_calls_after_build_only = [
        call
        for statement in run_calibration.body[
            run_calibration.body.index(build_only_block) + 1 :
        ]
        for call in ast.walk(statement)
        if isinstance(call, ast.Call) and _call_name(call.func) == "fit_l0_weights"
    ]

    assert "validator.raise_for_failure" in build_only_calls
    assert any(isinstance(node, ast.Return) for node in build_only_block.body)
    assert fit_calls_after_build_only, "source guard should cover the L0 fit path"
