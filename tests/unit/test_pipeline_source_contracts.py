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


def _name(node: ast.AST) -> str | None:
    return node.id if isinstance(node, ast.Name) else None


def test_promote_run_uses_single_full_release_promotion() -> None:
    tree = ast.parse(PIPELINE_SOURCE.read_text())
    promote_run = _function_def(tree, "promote_run")
    source = ast.get_source_segment(PIPELINE_SOURCE.read_text(), promote_run)

    assert "promotion_context = RunContext.from_mapping(" in source
    assert "_apply_run_context_env(promotion_context)" in source
    assert "_promote_full_release_from_staging(" in source
    assert "promotion_context.to_dict()" in source
    assert "_promotion_result_from_stdout(promotion_stdout)" in source
    assert "_write_release_promotion_contract_for_run(" in source
    assert "release_promotion_refs" in source
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
    assert isinstance(keywords["version"], ast.Name)
    assert keywords["version"].id == "candidate_version"


def test_calibration_package_parameters_record_target_config_identity() -> None:
    source_text = PIPELINE_SOURCE.read_text()
    tree = ast.parse(source_text)
    helper = _function_def(tree, "_calibration_package_parameters")
    source = ast.get_source_segment(source_text, helper)

    assert "resolve_target_config_identity(" in source
    assert '"target_config": target_config_identity.path' in source
    assert '"target_config_sha256": target_config_identity.sha256' in source
    assert '"target_config_mode": target_config_identity.mode' in source


def test_stage_2_manifest_records_package_and_contract_outputs() -> None:
    source_text = PIPELINE_SOURCE.read_text()
    tree = ast.parse(source_text)
    run_pipeline = _function_def(tree, "run_pipeline")
    source = ast.get_source_segment(source_text, run_pipeline)

    assert "package_context = stage2_build_context_for_run(" in source
    assert "package_context.input_bundle.manifest_inputs" in source
    assert 'package_inputs["input_validation"]' in source
    assert "package_artifacts = package_context.output_bundle" in source
    assert "package_artifacts.manifest_outputs" in source


def test_promote_run_fails_closed_for_required_promotion_steps() -> None:
    tree = ast.parse(PIPELINE_SOURCE.read_text())
    promote_run = _function_def(tree, "promote_run")
    source = ast.get_source_segment(PIPELINE_SOURCE.read_text(), promote_run)
    fail_step_calls = [
        node
        for node in ast.walk(promote_run)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_fail_step_manifest"
    ]

    assert any(
        [_name(arg) for arg in call.args[:3]]
        == ["promote_manifest", "exc", "pipeline_volume"]
        for call in fail_step_calls
    )
    assert any(
        any(keyword.arg == "traceback_ref" for keyword in call.keywords)
        for call in fail_step_calls
    )
    assert "meta.error =" in source
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
    assert "run_context = json.loads" in source
    assert "run_context=run_context" in source
    assert "files_with_paths=files_with_paths" in source
    assert 'extra_cleanup_paths=["_run_context.json"]' in source


def test_promotion_stdout_parser_uses_stage5_result_parser() -> None:
    tree = ast.parse(PIPELINE_SOURCE.read_text())
    helper = _function_def(tree, "_promotion_result_from_stdout")
    source = ast.get_source_segment(PIPELINE_SOURCE.read_text(), helper)

    assert "parse_full_promotion_result_json" in source
    assert "FullPromotionResult.from_legacy_dict" not in source
    assert "json.loads" not in source


def test_promote_run_writes_release_promotion_contract_output() -> None:
    tree = ast.parse(PIPELINE_SOURCE.read_text())
    helper = _function_def(tree, "_write_release_promotion_contract_for_run")
    stage4_helper = _function_def(
        tree,
        "_stage4_output_contract_repo_path_if_available",
    )
    helper_source = ast.get_source_segment(PIPELINE_SOURCE.read_text(), helper)
    stage4_source = ast.get_source_segment(PIPELINE_SOURCE.read_text(), stage4_helper)

    assert "release_promotion_contract_path(run_dir)" in helper_source
    assert "build_legacy_release_candidate_bundle(" in helper_source
    assert "build_published_artifact_index(" in helper_source
    assert "write_published_artifact_index(" in helper_source
    assert "write_release_promotion_contract(" in helper_source
    assert 'role="contract"' in helper_source
    assert 'role="index"' in helper_source
    assert 'media_type="application/json"' in helper_source
    assert "validation_report_paths=_run_validation_report_repo_paths_if_available" in (
        helper_source
    )
    assert (
        "diagnostics_manifest_path=_run_diagnostics_manifest_repo_path_if_available"
        in helper_source
    )
    assert 'media_type="application/jsonl"' in helper_source
    assert (
        "source_output_contract_path=_stage4_output_contract_repo_path_if_available"
        in (helper_source)
    )
    assert "published_artifact_index=published_index_artifact" in helper_source
    assert 'diagnostics" / "contracts" / "output_build_contract.json"' in stage4_source
    assert "calibration/runs/{run_id}/" in stage4_source


def test_run_pipeline_refreshes_diagnostics_even_when_h5_outputs_reused() -> None:
    tree = ast.parse(PIPELINE_SOURCE.read_text())
    run_pipeline = _function_def(tree, "run_pipeline")
    source = ast.get_source_segment(PIPELINE_SOURCE.read_text(), run_pipeline)

    assert "H5 outputs skipped - manifests valid; refreshing diagnostics" in source
    assert "Upload validation diagnostics even when H5 outputs are reused." in source


def test_run_pipeline_uses_stage_3_fit_specs_for_reuse_and_paths() -> None:
    source_text = PIPELINE_SOURCE.read_text()
    tree = ast.parse(source_text)
    run_pipeline = _function_def(tree, "run_pipeline")
    archive_diagnostics = _function_def(tree, "archive_diagnostics")
    source = ast.get_source_segment(source_text, run_pipeline)
    archive_source = ast.get_source_segment(source_text, archive_diagnostics)

    assert "fitted_weights_spec_for_scope(FitScope.REGIONAL)" in source
    assert "fitted_weights_spec_for_scope(FitScope.NATIONAL)" in source
    assert "fit_artifacts_for_scope(FitScope.REGIONAL)" in source
    assert "fit_artifacts_for_scope(FitScope.NATIONAL)" in source
    assert "regional_fit_spec.manifest_parameters(" in source
    assert "national_fit_spec.manifest_parameters(" in source
    assert "regional_fit_spec.runtime_kwargs()" in source
    assert "national_fit_spec.runtime_kwargs()" in source
    assert "regional_output.artifact_paths(_artifacts_dir(run_id))" in source
    assert "national_output.artifact_paths(_artifacts_dir(run_id))" in source
    assert "diagnostic_result_filenames()" in archive_source


def test_run_pipeline_converts_fit_results_to_scoped_output_bundles() -> None:
    source_text = PIPELINE_SOURCE.read_text()
    tree = ast.parse(source_text)
    run_pipeline = _function_def(tree, "run_pipeline")
    archive_diagnostics = _function_def(tree, "archive_diagnostics")
    source = ast.get_source_segment(source_text, run_pipeline)
    archive_source = ast.get_source_segment(source_text, archive_diagnostics)

    assert "FittedWeightsInputBundle(" in source
    assert "FittedWeightsOutputBundle.from_result_bytes(" in source
    assert "regional_output.write_artifacts(batch, artifacts_rel)" in source
    assert "national_output.write_artifacts(batch, artifacts_rel)" in source
    assert "regional_output.diagnostic_result_bytes()" in source
    assert "national_output.diagnostic_result_bytes()" in source
    assert "diagnostics=regional_diagnostics" in source
    assert "diagnostics=national_diagnostics" in source
    assert 'role="diagnostic"' in archive_source


def test_local_area_consumes_centralized_stage_3_artifact_specs() -> None:
    source = Path("modal_app/local_area.py").read_text()

    assert "fit_artifacts_for_scope(FitScope.REGIONAL)" in source
    assert "fit_artifacts_for_scope(FitScope.NATIONAL)" in source
    assert "regional_fit_artifacts.weights.filename" in source
    assert "national_fit_artifacts.weights.filename" in source


def test_run_pipeline_tolerates_post_h5_pipeline_volume_open_files() -> None:
    source_text = PIPELINE_SOURCE.read_text()
    tree = ast.parse(source_text)
    run_pipeline = _function_def(tree, "run_pipeline")
    reload_helper = _function_def(tree, "_try_reload_pipeline_volume_after_h5_builds")
    run_pipeline_source = ast.get_source_segment(source_text, run_pipeline)
    helper_source = ast.get_source_segment(source_text, reload_helper)

    assert "_try_reload_pipeline_volume_after_h5_builds(pipeline_volume)" in (
        run_pipeline_source
    )
    assert "pipeline_volume.reload()" not in run_pipeline_source
    assert "open files preventing the operation" in helper_source
    assert "return False" in helper_source


def test_run_pipeline_passes_candidate_version_to_h5_publishers() -> None:
    source_text = PIPELINE_SOURCE.read_text()
    tree = ast.parse(source_text)
    run_pipeline = _function_def(tree, "run_pipeline")
    source = ast.get_source_segment(source_text, run_pipeline)

    assert source.count("candidate_version=candidate_version") >= 2


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
