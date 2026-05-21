import json

import pandas as pd
import pytest
from scipy import sparse

from policyengine_us_data.calibration_package.matrix import (
    MatrixBuildResult,
    MatrixBuildService,
    MatrixBuildSpec,
)
from policyengine_us_data.stage_contracts.calibration_package_schema import (
    MatrixBuildSummary,
)
from policyengine_us_data.utils.manifest import compute_file_checksum


def _targets_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "variable": ["income_tax", "snap"],
            "geo_level": ["state", "state"],
            "geographic_id": ["01", "02"],
            "value": [100.0, 200.0],
        }
    )


def _matrix() -> sparse.csr_matrix:
    return sparse.csr_matrix(
        [
            [1.0, 0.0, 2.0, 0.0],
            [0.0, 3.0, 0.0, 0.0],
        ]
    )


def _precompute_spec() -> MatrixBuildSpec:
    return MatrixBuildSpec(
        matrix_builder="precompute",
        base_n_records=2,
        n_clones=2,
        county_level=False,
        workers=4,
    )


def _chunked_spec(tmp_path) -> MatrixBuildSpec:
    return MatrixBuildSpec(
        matrix_builder="chunked",
        base_n_records=2,
        n_clones=2,
        chunk_size=10,
        chunk_dir=str(tmp_path / "matrix_build"),
        keep_chunks=True,
        resume_chunks=True,
        rerandomize_takeup=False,
    )


def test_matrix_build_result_summary_round_trips(tmp_path):
    result = MatrixBuildResult.from_builder_output(
        spec=_precompute_spec(),
        targets_df=_targets_df(),
        X_sparse=_matrix(),
        target_names=["state_income_tax_01", "state_snap_02"],
    )

    summary = result.summary()
    summary_path = result.write_summary(tmp_path / "matrix_summary.json")
    restored = MatrixBuildSummary.from_dict(
        json.loads(summary_path.read_text(encoding="utf-8"))
    )

    assert restored == summary
    assert summary.to_dict()["matrix_shape"] == (2, 4)
    assert summary.matrix_nnz == 3
    assert summary.matrix_density == 3 / 8
    assert summary.n_targets == 2
    assert summary.n_columns == 4
    assert summary.target_name_count == 2
    assert summary.base_n_records == 2
    assert summary.n_clones == 2
    assert summary.matrix_builder == "precompute"
    assert summary.workers == 4
    assert summary.target_order_sha256.startswith("sha256:")


def test_matrix_build_result_chunked_metadata_matches_standard_shape(tmp_path):
    standard = MatrixBuildResult.from_builder_output(
        spec=_precompute_spec(),
        targets_df=_targets_df(),
        X_sparse=_matrix(),
        target_names=["state_income_tax_01", "state_snap_02"],
    ).summary()
    chunked = MatrixBuildResult.from_builder_output(
        spec=_chunked_spec(tmp_path),
        targets_df=_targets_df(),
        X_sparse=_matrix(),
        target_names=["state_income_tax_01", "state_snap_02"],
    ).summary()

    assert chunked.matrix_shape == standard.matrix_shape
    assert chunked.matrix_nnz == standard.matrix_nnz
    assert chunked.matrix_density == standard.matrix_density
    assert chunked.n_targets == standard.n_targets
    assert chunked.n_columns == standard.n_columns
    assert chunked.target_name_count == standard.target_name_count
    assert chunked.base_n_records == standard.base_n_records
    assert chunked.n_clones == standard.n_clones
    assert chunked.matrix_builder == "chunked"
    assert chunked.chunk_size == 10
    assert chunked.keep_chunks is True
    assert chunked.resume_chunks is True


def test_matrix_build_result_records_chunk_manifest_lineage(tmp_path):
    spec = _chunked_spec(tmp_path)
    manifest_path = tmp_path / "matrix_build" / "chunk_manifest.json"
    shard_path = tmp_path / "matrix_build" / "coo" / "chunk_000000.npz"
    shard_path.parent.mkdir(parents=True)
    manifest_path.write_text('{"signature":{"dataset":"abc"}}\n', encoding="utf-8")
    shard_path.write_bytes(b"coo shard")

    result = MatrixBuildResult.from_builder_output(
        spec=spec,
        targets_df=_targets_df(),
        X_sparse=_matrix(),
        target_names=["state_income_tax_01", "state_snap_02"],
        chunk_manifest_path=manifest_path,
        chunk_shard_paths=(shard_path,),
    )

    summary = result.summary()

    assert summary.chunk_manifest_path == str(manifest_path)
    assert summary.chunk_manifest_sha256 == (
        f"sha256:{compute_file_checksum(manifest_path)}"
    )
    assert summary.chunk_shard_count == 1
    assert summary.chunk_shard_paths == (str(shard_path),)


def test_matrix_build_result_enforces_target_order_invariants():
    with pytest.raises(ValueError, match="target_names count"):
        MatrixBuildResult.from_builder_output(
            spec=_precompute_spec(),
            targets_df=_targets_df(),
            X_sparse=_matrix(),
            target_names=["only_one_name"],
        )


class _FakeBuilder:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def build_matrix(self, **kwargs):
        self.calls.append(("standard", kwargs))
        return _targets_df(), _matrix(), ["state_income_tax_01", "state_snap_02"]

    def build_matrix_chunked(self, **kwargs):
        self.calls.append(("chunked", kwargs))
        chunk_dir = kwargs.get("chunk_dir")
        if chunk_dir:
            manifest_path = kwargs["chunk_dir"] + "/chunk_manifest.json"
            shard_path = kwargs["chunk_dir"] + "/coo/chunk_000000.npz"
            from pathlib import Path

            Path(shard_path).parent.mkdir(parents=True)
            Path(manifest_path).write_text("{}", encoding="utf-8")
            Path(shard_path).write_bytes(b"coo shard")
        return _targets_df(), _matrix(), ["state_income_tax_01", "state_snap_02"]


def test_matrix_build_service_normalizes_standard_and_chunked_outputs(tmp_path):
    builder = _FakeBuilder()
    service = MatrixBuildService(builder=builder)

    standard = service.build(
        spec=_precompute_spec(),
        geography=object(),
        sim=object(),
        target_filter={"target_selection": object()},
    )
    chunked = service.build(
        spec=_chunked_spec(tmp_path),
        geography=object(),
        sim=object(),
        target_filter={"target_selection": object()},
    )

    assert [call[0] for call in builder.calls] == ["standard", "chunked"]
    assert builder.calls[0][1]["workers"] == 4
    assert builder.calls[1][1]["chunk_size"] == 10
    assert standard.summary().matrix_builder == "precompute"
    assert chunked.summary().matrix_builder == "chunked"
    assert chunked.summary().chunk_shard_count == 1
