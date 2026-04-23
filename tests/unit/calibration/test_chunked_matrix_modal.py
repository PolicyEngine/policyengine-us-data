"""Unit tests for the Modal dispatch layer.

Covers the pure partition helper plus ``dispatch_chunks_modal`` under
a fake worker function (via injected ``worker_function`` and
``volume`` overrides). No real Modal calls.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List
from unittest import mock

import numpy as np
import pytest

from policyengine_us_data.calibration.chunked_matrix_assembler import (
    SharedBuildState,
)
from policyengine_us_data.calibration.chunked_matrix_modal import (
    dispatch_chunks_modal,
    partition_chunk_ids_contiguous,
)


# -----------------------------------------------------------------------
# partition_chunk_ids_contiguous
# -----------------------------------------------------------------------


def test_contiguous_batch_exact_division() -> None:
    batches = partition_chunk_ids_contiguous(n_chunks=12, num_workers=3)
    assert batches == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]


def test_contiguous_batch_remainder() -> None:
    batches = partition_chunk_ids_contiguous(n_chunks=10, num_workers=3)
    # ceil(10/3) = 4, so batches are [0..3], [4..7], [8..9] — 3 workers.
    assert batches == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]]
    assert sum(len(b) for b in batches) == 10


def test_contiguous_batch_more_workers_than_chunks() -> None:
    batches = partition_chunk_ids_contiguous(n_chunks=3, num_workers=10)
    # ceil(3/10) = 1, so at most 3 non-empty batches.
    assert batches == [[0], [1], [2]]


def test_contiguous_batch_zero_chunks() -> None:
    assert partition_chunk_ids_contiguous(n_chunks=0, num_workers=5) == []


def test_contiguous_batch_rejects_non_positive_workers() -> None:
    with pytest.raises(ValueError, match="num_workers"):
        partition_chunk_ids_contiguous(n_chunks=5, num_workers=0)


# -----------------------------------------------------------------------
# dispatch_chunks_modal with injected worker + volume fakes
# -----------------------------------------------------------------------


def _minimal_shared_state(
    n_records: int = 10, n_clones: int = 2, chunk_size: int = 10
) -> SharedBuildState:
    n_total = n_records * n_clones
    return SharedBuildState(
        source_dataset_path="/nonexistent/fixture.h5",
        time_period=2024,
        rerandomize_takeup=False,
        n_records=n_records,
        n_clones=n_clones,
        n_targets=2,
        chunk_size=chunk_size,
        target_variables=["x", "y"],
        target_reform_ids=[0, 0],
        target_geo_info=[("national", "US"), ("national", "US")],
        non_geo_constraints_list=[[], []],
        unique_variables={"x", "y"},
        unique_constraint_vars=set(),
        reform_variables=set(),
        target_names=["t0", "t1"],
        base_entity_maps=None,
        block_geoid=np.zeros(n_total, dtype="U15"),
        cd_geoid=np.zeros(n_total, dtype="U4"),
        county_fips=np.zeros(n_total, dtype="U5"),
        state_fips=np.zeros(n_total, dtype=np.int32),
    )


class _FakeHandle:
    def __init__(self, result: Dict, *, raise_on_get: Exception = None):
        self._result = result
        self._raise = raise_on_get
        self.object_id = "fc-fake"

    def get(self) -> Dict:
        if self._raise is not None:
            raise self._raise
        return self._result


class _FakeVolume:
    def __init__(self) -> None:
        self.commit_count = 0
        self.reload_count = 0

    def commit(self) -> None:
        self.commit_count += 1

    def reload(self) -> None:
        self.reload_count += 1


def _write_fake_shard(
    shard_dir: Path, chunk_id: int, col_start: int, col_end: int
) -> None:
    shard_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(shard_dir / f"chunk_{chunk_id:06d}.npz"),
        rows=np.array([0], dtype=np.int32),
        cols=np.array([col_start], dtype=np.int32),
        vals=np.array([1.0], dtype=np.float32),
        col_start=np.array([col_start], dtype=np.int64),
        col_end=np.array([col_end], dtype=np.int64),
    )


def test_dispatch_spawns_per_batch_and_assembles(tmp_path: Path) -> None:
    state = _minimal_shared_state(n_records=10, n_clones=4, chunk_size=10)
    # n_total=40, chunk_size=10 -> 4 chunks. num_workers=2 -> 2 batches.
    n_chunks = 4

    # Fake worker writes shards as a side effect of .spawn() so that
    # by the time assemble_final() runs, the shard files exist.
    spawn_calls: List[Dict] = []

    def fake_spawn(*, run_id: str, chunk_ids: List[int]) -> _FakeHandle:
        spawn_calls.append({"run_id": run_id, "chunk_ids": list(chunk_ids)})
        for chunk_id in chunk_ids:
            col_start = chunk_id * state.chunk_size
            col_end = col_start + state.chunk_size
            _write_fake_shard(tmp_path / "coo", chunk_id, col_start, col_end)
        return _FakeHandle(
            {
                "chunk_ids": list(chunk_ids),
                "nnz_per_chunk": [1] * len(chunk_ids),
                "errors": [],
            }
        )

    fake_worker = mock.MagicMock()
    fake_worker.spawn.side_effect = fake_spawn
    volume = _FakeVolume()

    X = dispatch_chunks_modal(
        shared_state=state,
        chunk_root=tmp_path,
        run_id="run-test",
        num_workers=2,
        worker_function=fake_worker,
        volume=volume,
    )

    # 2 batches spawned, each covering 2 contiguous chunk ids.
    assert [c["chunk_ids"] for c in spawn_calls] == [[0, 1], [2, 3]]
    # Every spawn carried the run_id.
    assert all(c["run_id"] == "run-test" for c in spawn_calls)
    # Final CSR covers all 4 chunks' nnz.
    assert X.shape == (state.n_targets, state.n_total)
    assert X.nnz == n_chunks
    # Volume was committed (pre-spawn) and reloaded (pre-assemble).
    assert volume.commit_count >= 1
    assert volume.reload_count >= 1


def test_dispatch_short_circuits_when_zero_chunks(tmp_path: Path) -> None:
    state = _minimal_shared_state(n_records=0, n_clones=0, chunk_size=10)
    fake_worker = mock.MagicMock()
    volume = _FakeVolume()

    X = dispatch_chunks_modal(
        shared_state=state,
        chunk_root=tmp_path,
        run_id="run-test",
        num_workers=4,
        worker_function=fake_worker,
        volume=volume,
    )

    assert X.shape == (state.n_targets, 0)
    assert X.nnz == 0
    fake_worker.spawn.assert_not_called()


def test_dispatch_aggregates_worker_errors(tmp_path: Path) -> None:
    state = _minimal_shared_state(n_records=10, n_clones=2, chunk_size=10)
    # n_total=20, chunk_size=10 -> 2 chunks, 2 workers.

    def fake_spawn(*, run_id: str, chunk_ids: List[int]) -> _FakeHandle:
        # First worker returns a per-chunk error; second crashes in .get().
        if chunk_ids == [0]:
            return _FakeHandle(
                {
                    "chunk_ids": chunk_ids,
                    "nnz_per_chunk": [],
                    "errors": [{"chunk_id": 0, "error": "boom"}],
                }
            )
        return _FakeHandle(None, raise_on_get=RuntimeError("worker oom"))

    fake_worker = mock.MagicMock()
    fake_worker.spawn.side_effect = fake_spawn
    volume = _FakeVolume()

    with pytest.raises(RuntimeError, match="Parallel chunked matrix build failed"):
        dispatch_chunks_modal(
            shared_state=state,
            chunk_root=tmp_path,
            run_id="run-test",
            num_workers=2,
            worker_function=fake_worker,
            volume=volume,
        )


def test_dispatch_writes_shared_state_pickle(tmp_path: Path) -> None:
    import pickle

    state = _minimal_shared_state(n_records=5, n_clones=1, chunk_size=10)
    # n_total=5, chunk_size=10 -> 1 chunk, 1 worker.

    def fake_spawn(*, run_id: str, chunk_ids: List[int]) -> _FakeHandle:
        for chunk_id in chunk_ids:
            _write_fake_shard(tmp_path / "coo", chunk_id, 0, 5)
        return _FakeHandle({"chunk_ids": chunk_ids, "nnz_per_chunk": [1], "errors": []})

    fake_worker = mock.MagicMock()
    fake_worker.spawn.side_effect = fake_spawn
    volume = _FakeVolume()

    dispatch_chunks_modal(
        shared_state=state,
        chunk_root=tmp_path,
        run_id="run-test",
        num_workers=1,
        worker_function=fake_worker,
        volume=volume,
    )

    pickle_path = tmp_path / "chunk_build_state.pkl"
    assert pickle_path.exists()
    with open(pickle_path, "rb") as f:
        roundtripped = pickle.load(f)
    assert roundtripped.n_total == state.n_total
    assert roundtripped.chunk_size == state.chunk_size
    assert roundtripped.target_names == state.target_names
