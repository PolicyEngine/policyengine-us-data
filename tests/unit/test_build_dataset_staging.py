import json
import sqlite3
from pathlib import Path

import h5py
import pytest

from policyengine_us_data.build_datasets import (
    DatasetBuildContext,
    DatasetBuildOutputContractBuilder,
    DatasetInventoryWriter,
    PipelineArtifactStager,
    SourceDatasetSchemaSummaryWriter,
    TargetDatabaseSchemaSummaryWriter,
    stage_1_pipeline_artifact_specs,
    write_stage_1_diagnostics,
)


def _context(tmp_path: Path) -> DatasetBuildContext:
    return DatasetBuildContext(
        run_id="run-123",
        branch="main",
        code_sha="abc123",
        package_version="1.98.2",
        artifacts_dir=tmp_path / "artifacts",
        storage_dir=tmp_path / "storage",
        work_dir=tmp_path / "work",
    )


def _write_h5(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5_file:
        person = h5_file.create_group("person")
        person.create_dataset("age/2024", data=[1, 2, 3])
        person.create_dataset("is_disabled", data=[0, 1, 0])
        household = h5_file.create_group("household")
        household.create_dataset("weight/2024", data=[10.0, 20.0])


def _write_sqlite(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE targets (id INTEGER PRIMARY KEY, value REAL)")
        conn.execute("CREATE TABLE notes (id INTEGER PRIMARY KEY, label TEXT)")
        conn.execute("INSERT INTO targets (value) VALUES (1.5), (2.5)")
        conn.execute("INSERT INTO notes (label) VALUES ('a')")


def _write_text(path: Path, payload: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload)


def _write_required_storage_artifacts(
    context: DatasetBuildContext,
    *,
    include_enhanced_cps: bool = True,
    include_stage_5: bool = True,
    include_optional_weights: bool = False,
) -> None:
    for spec in stage_1_pipeline_artifact_specs():
        if spec.yearless_alias or spec.storage_path is None:
            continue
        if not include_enhanced_cps and spec.skip_when_enhanced_cps_skipped:
            continue
        if not include_stage_5 and spec.skip_when_stage_5_skipped:
            continue
        if not spec.required and not include_optional_weights:
            continue
        path = context.source_path(spec.storage_path)
        if path.suffix == ".h5":
            _write_h5(path)
        elif path.suffix == ".db":
            _write_sqlite(path)
        else:
            _write_text(path, spec.logical_name)


def test_stager_copies_only_declared_artifacts(tmp_path):
    context = _context(tmp_path)
    _write_required_storage_artifacts(context, include_optional_weights=True)
    extra = context.storage_dir / "untracked.h5"
    _write_h5(extra)

    staged = PipelineArtifactStager(context=context).stage_declared_artifacts()

    staged_names = {path.name for path in staged}
    assert "untracked.h5" not in staged_names
    assert "acs_2022.h5" in staged_names
    assert "policy_data.db" in staged_names
    assert "calibration_weights.npy" in staged_names


def test_stager_creates_yearless_source_imputed_alias(tmp_path):
    context = _context(tmp_path)
    _write_required_storage_artifacts(context)

    PipelineArtifactStager(context=context).stage_declared_artifacts()

    assert (
        context.artifacts_dir / "source_imputed_stratified_extended_cps_2024.h5"
    ).exists()
    alias = context.artifacts_dir / "source_imputed_stratified_extended_cps.h5"
    assert alias.exists()
    with h5py.File(alias) as h5_file:
        assert list(h5_file["person"]["age"]["2024"]) == [1, 2, 3]


def test_stager_fails_on_missing_required_artifact(tmp_path):
    context = _context(tmp_path)

    with pytest.raises(FileNotFoundError, match="acs_2022.h5"):
        PipelineArtifactStager(context=context).stage_declared_artifacts()


def test_stager_respects_skip_flags_for_optional_ecps_paths(tmp_path):
    context = _context(tmp_path)
    _write_required_storage_artifacts(
        context,
        include_enhanced_cps=False,
        include_stage_5=False,
    )

    staged = PipelineArtifactStager(context=context).stage_declared_artifacts(
        skip_enhanced_cps=True,
        skip_stage_5=True,
    )

    staged_names = {path.name for path in staged}
    assert "enhanced_cps_2024.h5" not in staged_names
    assert "small_enhanced_cps_2024.h5" not in staged_names
    assert "source_imputed_stratified_extended_cps_2024.h5" not in staged_names
    assert "source_imputed_stratified_extended_cps.h5" not in staged_names


def test_dataset_inventory_contains_each_staged_artifact_once(tmp_path):
    context = _context(tmp_path)
    _write_required_storage_artifacts(context, include_optional_weights=True)
    stager = PipelineArtifactStager(context=context)
    stager.stage_declared_artifacts()
    stager.write_checkpoint_stats({"expected_outputs": 3})

    diagnostic = DatasetInventoryWriter(context=context).write()

    inventory_path = context.artifacts_dir / "dataset_inventory.json"
    payload = json.loads(inventory_path.read_text())
    logical_names = [artifact["logical_name"] for artifact in payload["artifacts"]]
    assert len(logical_names) == len(set(logical_names))
    assert "policy_data_db" in logical_names
    assert "data_build_checkpoint_stats" in logical_names
    assert diagnostic.artifact.logical_name == "dataset_inventory"
    assert diagnostic.summary["artifact_count"] == len(logical_names)


def test_source_dataset_schema_summary_is_metadata_only(tmp_path):
    context = _context(tmp_path)
    context.artifacts_dir.mkdir(parents=True)
    _write_h5(context.artifacts_dir / "source_imputed_stratified_extended_cps.h5")

    diagnostic = SourceDatasetSchemaSummaryWriter(context=context).write()

    payload = json.loads(
        (context.artifacts_dir / "source_dataset_schema_summary.json").read_text()
    )
    assert payload["logical_name"] == "source_imputed_stratified_extended_cps"
    assert payload["entities"]["person"]["variables"] == ["age", "is_disabled"]
    assert payload["entities"]["household"]["row_counts"] == {
        "household/weight/2024": 2
    }
    assert diagnostic.summary == {"dataset_count": 3, "entity_count": 2}


def test_target_database_summary_reports_tables_and_row_counts(tmp_path):
    context = _context(tmp_path)
    context.artifacts_dir.mkdir(parents=True)
    _write_sqlite(context.artifacts_dir / "policy_data.db")

    diagnostic = TargetDatabaseSchemaSummaryWriter(context=context).write()

    payload = json.loads(
        (context.artifacts_dir / "target_database_schema_summary.json").read_text()
    )
    assert [table["name"] for table in payload["tables"]] == ["notes", "targets"]
    row_counts = {table["name"]: table["row_count"] for table in payload["tables"]}
    assert row_counts == {"notes": 1, "targets": 2}
    assert payload["known_target_tables"] == ["targets"]
    assert diagnostic.summary["table_count"] == 2
    assert diagnostic.summary["known_target_tables"] == ("targets",)


def test_contract_builder_records_stage_1_diagnostics(tmp_path):
    context = _context(tmp_path)
    _write_required_storage_artifacts(context)
    stager = PipelineArtifactStager(context=context)
    stager.stage_declared_artifacts()
    stager.write_checkpoint_stats({"expected_outputs": 3})
    diagnostics = write_stage_1_diagnostics(context=context)

    contract = DatasetBuildOutputContractBuilder(context=context).build(
        checkpoint_stats={"expected_outputs": 3},
        started_at="2026-05-08T12:00:00Z",
        completed_at="2026-05-08T12:01:00Z",
        duration_s=60.0,
        upload_requested=True,
        stage_only=True,
        skip_enhanced_cps=False,
        diagnostics=diagnostics,
    )

    assert {diagnostic.name for diagnostic in contract.diagnostics} == {
        "dataset_inventory",
        "source_dataset_schema_summary",
        "target_database_schema_summary",
    }
    assert contract.metadata["diagnostic_count"] == 3
