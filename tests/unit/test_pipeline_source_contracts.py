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


def test_run_pipeline_stage_1_stages_datasets_without_promoting() -> None:
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
    keywords = {keyword.arg: keyword.value for keyword in build_calls[0].keywords}
    assert isinstance(keywords["upload"], ast.Constant)
    assert keywords["upload"].value is True
    assert isinstance(keywords["stage_only"], ast.Constant)
    assert keywords["stage_only"].value is True


def test_promote_run_fails_closed_for_required_promotion_steps() -> None:
    tree = ast.parse(PIPELINE_SOURCE.read_text())
    promote_run = _function_def(tree, "promote_run")
    source = ast.get_source_segment(PIPELINE_SOURCE.read_text(), promote_run)

    assert "_fail_step_manifest(promote_manifest, exc, pipeline_volume)" in source
    assert "WARNING: Base dataset promotion" not in source
    assert "WARNING: Regional promote" not in source
    assert "WARNING: National promote" not in source
    assert "WARNING: Version registration failed" not in source
    assert "Registering version in manifest" not in source


def test_promote_run_uses_canonical_dataset_promote_only_path() -> None:
    tree = ast.parse(PIPELINE_SOURCE.read_text())
    promote_run = _function_def(tree, "promote_run")
    source = ast.get_source_segment(PIPELINE_SOURCE.read_text(), promote_run)

    assert "policyengine_us_data.storage.upload_completed_datasets" in source
    assert "upload_datasets(" in source
    assert "promote_only=True" in source
    assert "promote_staging_to_production_hf" not in source
    assert "base_files = [" not in source
    assert "policyengine_us_data.utils.version_manifest" not in source


def test_run_pipeline_refreshes_diagnostics_even_when_h5_outputs_reused() -> None:
    tree = ast.parse(PIPELINE_SOURCE.read_text())
    run_pipeline = _function_def(tree, "run_pipeline")
    source = ast.get_source_segment(PIPELINE_SOURCE.read_text(), run_pipeline)

    assert "H5 outputs skipped - manifests valid; refreshing diagnostics" in source
    assert "Upload validation diagnostics even when H5 outputs are reused." in source


def test_promote_run_defers_component_staging_cleanup_until_all_promotions_succeed():
    tree = ast.parse(PIPELINE_SOURCE.read_text())
    promote_run = _function_def(tree, "promote_run")
    source = ast.get_source_segment(PIPELINE_SOURCE.read_text(), promote_run)

    assert source.count("cleanup_staging=False") == 3
    assert source.index("upload_datasets(") < source.index("promote_publish.remote(")
    assert source.index("promote_publish.remote(") < source.index(
        "promote_national_publish.remote("
    )
    assert source.index("promote_national_publish.remote(") < source.index(
        "_cleanup_promoted_staging_artifacts"
    )
