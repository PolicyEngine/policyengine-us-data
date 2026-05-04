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


def test_promote_run_uses_single_full_release_promotion() -> None:
    tree = ast.parse(PIPELINE_SOURCE.read_text())
    promote_run = _function_def(tree, "promote_run")
    source = ast.get_source_segment(PIPELINE_SOURCE.read_text(), promote_run)

    assert "_promote_full_release_from_staging(run_id, version)" in source
    assert "promote_publish.remote(" not in source
    assert "promote_national_publish.remote(" not in source
    assert "upload_datasets(" not in source


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


def test_promote_run_uses_unified_staged_release_path() -> None:
    tree = ast.parse(PIPELINE_SOURCE.read_text())
    promote_full = _function_def(tree, "_promote_full_release_from_staging")
    source = ast.get_source_segment(PIPELINE_SOURCE.read_text(), promote_full)

    assert "policyengine_us_data.utils.data_upload" in source
    assert "promote_full_release_from_staging" in source
    assert "files_with_paths=files_with_paths" in source
    assert 'extra_cleanup_paths=["_run_context.json"]' in source


def test_run_pipeline_refreshes_diagnostics_even_when_h5_outputs_reused() -> None:
    tree = ast.parse(PIPELINE_SOURCE.read_text())
    run_pipeline = _function_def(tree, "run_pipeline")
    source = ast.get_source_segment(PIPELINE_SOURCE.read_text(), run_pipeline)

    assert "H5 outputs skipped - manifests valid; refreshing diagnostics" in source
    assert "Upload validation diagnostics even when H5 outputs are reused." in source


def test_full_release_path_combines_base_regional_and_national_outputs():
    tree = ast.parse(PIPELINE_SOURCE.read_text())
    helper = _function_def(tree, "_full_release_staging_rel_paths")
    source = ast.get_source_segment(PIPELINE_SOURCE.read_text(), helper)

    assert "BASE_DATASET_STAGING_REL_PATHS" in source
    assert "_regional_h5_staging_rel_paths(run_id)" in source
    assert '"national/US.h5"' in source


def test_full_release_manifest_files_use_pipeline_and_staging_volumes():
    tree = ast.parse(PIPELINE_SOURCE.read_text())
    helper = _function_def(tree, "_full_release_manifest_files")
    source = ast.get_source_segment(PIPELINE_SOURCE.read_text(), helper)

    assert "_artifacts_dir(run_id)" in source
    assert "Path(STAGING_MOUNT) / run_id" in source
    assert "BASE_DATASET_STAGING_REL_PATHS" in source
