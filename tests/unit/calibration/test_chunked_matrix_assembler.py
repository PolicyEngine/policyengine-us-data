"""Unit tests for ``ChunkedMatrixAssembler`` pure helpers and class.

Tests here avoid constructing a real ``Microsimulation``. The kernel
(``run_single_chunk``) is exercised by the integration suite at
``tests/integration/test_chunked_matrix_builder.py``. This file
covers: chunk partitioning, streaming CSR assembly (including memory
profile), and resume-skip behaviour on pre-staged shards.
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import List
from unittest import mock

import numpy as np
import pytest
from scipy import sparse

from policyengine_us_data.calibration.chunked_matrix_assembler import (
    ChunkResult,
    ChunkedMatrixAssembler,
    SharedBuildState,
    partition_chunks,
    stream_csr_from_shards,
)


def _write_shard(
    shard_dir: Path,
    chunk_id: int,
    rows: np.ndarray,
    cols: np.ndarray,
    vals: np.ndarray,
    col_start: int,
    col_end: int,
) -> Path:
    path = shard_dir / f"chunk_{chunk_id:06d}.npz"
    np.savez_compressed(
        str(path),
        rows=np.asarray(rows, dtype=np.int32),
        cols=np.asarray(cols, dtype=np.int32),
        vals=np.asarray(vals, dtype=np.float32),
        col_start=np.array([col_start], dtype=np.int64),
        col_end=np.array([col_end], dtype=np.int64),
    )
    return path


def _make_shared_state(
    n_records: int = 10,
    n_clones: int = 2,
    n_targets: int = 3,
) -> SharedBuildState:
    """Minimal ``SharedBuildState`` for tests that don't run the kernel."""
    n_total = n_records * n_clones
    return SharedBuildState(
        source_dataset_path="/nonexistent/fixture.h5",
        time_period=2024,
        rerandomize_takeup=False,
        n_records=n_records,
        n_clones=n_clones,
        n_targets=n_targets,
        target_variables=["x"] * n_targets,
        target_reform_ids=[0] * n_targets,
        target_geo_info=[("national", "US")] * n_targets,
        non_geo_constraints_list=[[] for _ in range(n_targets)],
        unique_variables={"x"},
        unique_constraint_vars=set(),
        reform_variables=set(),
        target_names=[f"t{i}" for i in range(n_targets)],
        base_entity_maps=None,
        block_geoid=np.zeros(n_total, dtype="U15"),
        cd_geoid=np.zeros(n_total, dtype="U4"),
        county_fips=np.zeros(n_total, dtype="U5"),
        state_fips=np.zeros(n_total, dtype=np.int32),
    )


# -----------------------------------------------------------------------
# partition_chunks
# -----------------------------------------------------------------------


def test_partition_chunks_exact_multiple(tmp_path: Path) -> None:
    plans = partition_chunks(
        n_total=1000, chunk_size=250, coo_dir=tmp_path / "coo", h5_dir=tmp_path / "h5"
    )
    assert len(plans) == 4
    assert [p.col_start for p in plans] == [0, 250, 500, 750]
    assert [p.col_end for p in plans] == [250, 500, 750, 1000]
    assert plans[0].coo_path.name == "chunk_000000.npz"
    assert plans[3].coo_path.name == "chunk_000003.npz"


def test_partition_chunks_remainder(tmp_path: Path) -> None:
    plans = partition_chunks(
        n_total=1050, chunk_size=250, coo_dir=tmp_path / "coo", h5_dir=tmp_path / "h5"
    )
    assert len(plans) == 5
    assert plans[-1].col_start == 1000
    assert plans[-1].col_end == 1050


def test_partition_chunks_zero_total(tmp_path: Path) -> None:
    plans = partition_chunks(
        n_total=0, chunk_size=100, coo_dir=tmp_path / "coo", h5_dir=tmp_path / "h5"
    )
    assert plans == []


def test_partition_chunks_rejects_invalid_chunk_size(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="chunk_size"):
        partition_chunks(n_total=10, chunk_size=0, coo_dir=tmp_path, h5_dir=tmp_path)


# -----------------------------------------------------------------------
# stream_csr_from_shards
# -----------------------------------------------------------------------


def test_stream_csr_from_shards_matches_coo_reference(tmp_path: Path) -> None:
    shard_dir = tmp_path
    # Three chunks, 4 cols per chunk, 5 targets.
    # Hand-built entries that span multiple rows and include duplicates
    # across shards (different cols) to exercise the scatter pass.
    _write_shard(
        shard_dir,
        0,
        rows=[0, 2, 4],
        cols=[0, 2, 3],
        vals=[1.0, 2.0, 3.0],
        col_start=0,
        col_end=4,
    )
    _write_shard(
        shard_dir,
        1,
        rows=[0, 1, 2],
        cols=[4, 5, 7],
        vals=[4.0, 5.0, 6.0],
        col_start=4,
        col_end=8,
    )
    _write_shard(
        shard_dir,
        2,
        rows=[],  # empty shard
        cols=[],
        vals=[],
        col_start=8,
        col_end=12,
    )
    n_targets = 5
    n_total = 12

    X = stream_csr_from_shards(
        shard_dir, n_chunks=3, n_targets=n_targets, n_total=n_total
    )

    # Reference: build the same CSR via scipy's COO constructor.
    all_rows = np.array([0, 2, 4, 0, 1, 2], dtype=np.int32)
    all_cols = np.array([0, 2, 3, 4, 5, 7], dtype=np.int32)
    all_vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)
    reference = sparse.csr_matrix(
        (all_vals, (all_rows, all_cols)),
        shape=(n_targets, n_total),
        dtype=np.float32,
    )
    reference.sort_indices()

    assert X.shape == reference.shape
    assert X.nnz == reference.nnz
    assert np.array_equal(X.indptr, reference.indptr)
    assert np.array_equal(X.indices, reference.indices)
    assert np.allclose(X.data, reference.data)


def test_stream_csr_from_shards_all_empty(tmp_path: Path) -> None:
    shard_dir = tmp_path
    for chunk_id in range(2):
        _write_shard(
            shard_dir,
            chunk_id,
            rows=[],
            cols=[],
            vals=[],
            col_start=chunk_id * 4,
            col_end=(chunk_id + 1) * 4,
        )
    X = stream_csr_from_shards(shard_dir, n_chunks=2, n_targets=3, n_total=8)
    assert X.shape == (3, 8)
    assert X.nnz == 0


def test_stream_csr_memory_within_bound(tmp_path: Path) -> None:
    """Peak RSS during streaming assembly should not exceed the final
    CSR arrays plus one shard by a wide margin. Guards against a future
    regression to list-then-concat assembly.
    """
    psutil = pytest.importorskip("psutil")
    rng = np.random.default_rng(0)
    n_targets = 500
    n_total = 500_000
    nnz_per_shard = 50_000
    n_chunks = 8  # ~400k total nnz

    for chunk_id in range(n_chunks):
        col_start = chunk_id * (n_total // n_chunks)
        col_end = col_start + (n_total // n_chunks)
        rows = rng.integers(0, n_targets, size=nnz_per_shard, dtype=np.int32)
        cols = rng.integers(col_start, col_end, size=nnz_per_shard, dtype=np.int32)
        vals = rng.standard_normal(nnz_per_shard).astype(np.float32)
        _write_shard(tmp_path, chunk_id, rows, cols, vals, col_start, col_end)

    gc.collect()
    process = psutil.Process()
    baseline = process.memory_info().rss

    # Sample RSS during assembly by wrapping ``stream_csr_from_shards``
    # and taking a reading after the inner arrays are allocated. Simpler
    # than a sampling thread, sufficient for a coarse bound check.
    X = stream_csr_from_shards(
        tmp_path, n_chunks=n_chunks, n_targets=n_targets, n_total=n_total
    )
    peak = process.memory_info().rss

    final_csr_bytes = X.data.nbytes + X.indices.nbytes + X.indptr.nbytes
    # Bound: 4x final CSR + 32 MB slack. Tight enough to catch a full
    # list-then-concat regression (which would be >6x at this size), loose
    # enough that Python interpreter overhead + heap fragmentation don't
    # make this flaky on shared CI runners.
    bound = 4 * final_csr_bytes + 32 * 1024 * 1024
    delta = peak - baseline
    assert delta < bound, (
        f"Peak RSS grew by {delta:,} bytes; expected <{bound:,} "
        f"(final CSR is {final_csr_bytes:,} bytes)"
    )


# -----------------------------------------------------------------------
# ChunkedMatrixAssembler resume semantics
# -----------------------------------------------------------------------


def test_assembler_skips_existing_shards_when_resume(tmp_path: Path) -> None:
    """A pre-staged shard with matching col_start/col_end should skip
    the kernel and return a cached ``ChunkResult``.
    """
    state = _make_shared_state(n_records=10, n_clones=2, n_targets=3)
    assembler = ChunkedMatrixAssembler(
        shared_state=state,
        chunk_root=tmp_path,
        chunk_size=10,
        resume=True,
        keep_chunks=False,
    )
    # Two plans: cols 0-9 and 10-19.
    plan0 = assembler.plans[0]
    _write_shard(
        assembler.coo_dir,
        0,
        rows=np.array([1, 2], dtype=np.int32),
        cols=np.array([3, 7], dtype=np.int32),
        vals=np.array([1.0, 2.0], dtype=np.float32),
        col_start=plan0.col_start,
        col_end=plan0.col_end,
    )
    # Run only chunk 0; kernel would fail (sim=None, fixture path does
    # not exist), so hitting the cache path is proof of the skip.
    result = assembler.run_single_chunk(0)
    assert result.cached is True
    assert result.nnz == 2
    assert result.chunk_id == 0


def test_assembler_rejects_shard_with_mismatched_range(tmp_path: Path) -> None:
    state = _make_shared_state(n_records=10, n_clones=2, n_targets=3)
    assembler = ChunkedMatrixAssembler(
        shared_state=state,
        chunk_root=tmp_path,
        chunk_size=10,
        resume=True,
        keep_chunks=False,
    )
    # Write a shard whose metadata claims col_start=5 (not 0).
    _write_shard(
        assembler.coo_dir,
        0,
        rows=np.array([], dtype=np.int32),
        cols=np.array([], dtype=np.int32),
        vals=np.array([], dtype=np.float32),
        col_start=5,
        col_end=15,
    )
    with pytest.raises(ValueError, match="expected 0-9"):
        assembler.run_single_chunk(0)


def test_assembler_run_chunks_dispatches_each_id(tmp_path: Path) -> None:
    state = _make_shared_state(n_records=10, n_clones=3, n_targets=2)
    assembler = ChunkedMatrixAssembler(
        shared_state=state,
        chunk_root=tmp_path,
        chunk_size=10,
        resume=False,
        keep_chunks=False,
    )
    assert assembler.n_chunks == 3

    observed: List[int] = []

    def fake_run(chunk_id: int) -> ChunkResult:
        observed.append(chunk_id)
        return ChunkResult(chunk_id=chunk_id, nnz=0, cached=False)

    with mock.patch.object(assembler, "run_single_chunk", side_effect=fake_run):
        results = assembler.run_chunks([0, 2])

    assert observed == [0, 2]
    assert [r.chunk_id for r in results] == [0, 2]
