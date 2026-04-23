"""Coordinator-side Modal dispatch for chunked matrix building.

Writes shared ``SharedBuildState`` to a pipeline-volume path,
spawns ``build_matrix_chunk_worker`` per contiguous batch of
chunk ids, collects per-worker results, then streams the final CSR
from all shards on the volume.

Kept separate from ``unified_matrix_builder`` so the core matrix
builder doesn't import Modal; only the dispatch path does.
"""

from __future__ import annotations

import logging
import math
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from scipy import sparse

from policyengine_us_data.calibration.chunked_matrix_assembler import (
    ChunkedMatrixAssembler,
    SharedBuildState,
)

logger = logging.getLogger(__name__)

DEFAULT_NUM_MATRIX_WORKERS = 50
MODAL_APP_NAME = "policyengine-us-data-fit-weights"
WORKER_FUNCTION_NAME = "build_matrix_chunk_worker"


def partition_chunk_ids_contiguous(n_chunks: int, num_workers: int) -> List[List[int]]:
    """Split ``range(n_chunks)`` into ``num_workers`` contiguous batches.

    Contiguous (not round-robin) so that a partial run leaves complete
    prefixes on disk that future `--resume-chunks` invocations can
    skip cleanly. Returns at most ``num_workers`` non-empty batches.
    """
    if n_chunks <= 0:
        return []
    if num_workers <= 0:
        raise ValueError("num_workers must be positive")
    batch_size = math.ceil(n_chunks / num_workers)
    batches: List[List[int]] = []
    for start in range(0, n_chunks, batch_size):
        end = min(start + batch_size, n_chunks)
        batches.append(list(range(start, end)))
    return batches


def _lookup_worker_function():
    """Resolve the deployed Modal worker function.

    Using ``Function.from_name`` avoids importing the worker module
    here (Modal imports are heavy and would pull into every caller).
    It also means unit tests can monkeypatch this function without
    touching the worker module.
    """
    import modal

    return modal.Function.from_name(MODAL_APP_NAME, WORKER_FUNCTION_NAME)


def dispatch_chunks_modal(
    *,
    shared_state: SharedBuildState,
    chunk_root: Path,
    run_id: str,
    num_workers: int = DEFAULT_NUM_MATRIX_WORKERS,
    worker_function: Optional[Any] = None,
    volume: Optional[Any] = None,
) -> sparse.csr_matrix:
    """Fan chunk materialization across Modal workers, then assemble.

    Args:
        shared_state: Read-only per-build state; pickled once to the
            volume so workers can reconstruct the assembler without
            receiving the arrays through ``.spawn()`` args.
        chunk_root: Directory on the pipeline volume where shards land
            (``{chunk_root}/coo/chunk_XXXXXX.npz``) and where the
            shared state pickle lives
            (``{chunk_root}/chunk_build_state.pkl``).
        run_id: Forwarded to each worker so its volume paths align
            with the coordinator's.
        num_workers: Upper bound on workers; actual count equals
            ``min(num_workers, n_chunks)``.
        worker_function: Override for the Modal function (tests only).
        volume: Override for the pipeline volume (tests only). When
            omitted, resolves ``modal.Volume.from_name("pipeline-artifacts")``.

    Raises:
        RuntimeError: if any worker reports one or more chunk errors
            after all workers finish. Raised after aggregating so no
            errors are silently dropped.
    """
    chunk_root = Path(chunk_root)
    chunk_root.mkdir(parents=True, exist_ok=True)
    state_path = chunk_root / "chunk_build_state.pkl"
    with open(state_path, "wb") as f:
        pickle.dump(shared_state, f)

    if volume is None:
        import modal

        volume = modal.Volume.from_name("pipeline-artifacts", create_if_missing=True)
    # Make the shared-state pickle visible to workers.
    volume.commit()

    n_chunks = math.ceil(shared_state.n_total / shared_state.chunk_size)
    batches = partition_chunk_ids_contiguous(n_chunks, num_workers)

    if not batches:
        # Nothing to materialize; fall through to assembly (which will
        # return an empty CSR).
        volume.reload()
        assembler = ChunkedMatrixAssembler(
            shared_state=shared_state,
            chunk_root=chunk_root,
            chunk_size=shared_state.chunk_size,
            resume=True,
            keep_chunks=False,
        )
        return assembler.assemble_final()

    if worker_function is None:
        worker_function = _lookup_worker_function()

    logger.info(
        "Dispatching %d chunks across %d workers (batch sizes: %s)",
        n_chunks,
        len(batches),
        [len(b) for b in batches[:5]] + (["..."] if len(batches) > 5 else []),
    )
    t_dispatch = time.time()
    handles = []
    for batch_idx, chunk_ids in enumerate(batches):
        handle = worker_function.spawn(run_id=run_id, chunk_ids=chunk_ids)
        logger.info(
            "Worker %d/%d: %d chunks (%d-%d), fc=%s",
            batch_idx + 1,
            len(batches),
            len(chunk_ids),
            chunk_ids[0],
            chunk_ids[-1],
            getattr(handle, "object_id", "unknown"),
        )
        handles.append((batch_idx, handle))

    aggregated_errors: List[Dict] = []
    for batch_idx, handle in handles:
        try:
            result = handle.get()
        except Exception as exc:
            aggregated_errors.append(
                {
                    "batch": batch_idx,
                    "error": f"Worker crashed: {exc}",
                }
            )
            logger.error("Worker %d crashed: %s", batch_idx, exc)
            continue
        if result is None:
            aggregated_errors.append(
                {"batch": batch_idx, "error": "Worker returned None"}
            )
            continue
        errors = result.get("errors", [])
        if errors:
            for err in errors:
                err_copy = dict(err)
                err_copy["batch"] = batch_idx
                aggregated_errors.append(err_copy)
        logger.info(
            "Worker %d done: %d chunks completed, %d errors",
            batch_idx,
            len(result.get("chunk_ids", [])) - len(errors),
            len(errors),
        )

    logger.info(
        "All workers finished in %.1fs; %d errors total",
        time.time() - t_dispatch,
        len(aggregated_errors),
    )

    if aggregated_errors:
        preview = "; ".join(
            f"batch {e['batch']}: {e.get('error', 'unknown')[:120]}"
            for e in aggregated_errors[:3]
        )
        raise RuntimeError(
            f"Parallel chunked matrix build failed with "
            f"{len(aggregated_errors)} error(s). First: {preview}"
        )

    # All shards present on the volume; reload and assemble.
    volume.reload()
    assembler = ChunkedMatrixAssembler(
        shared_state=shared_state,
        chunk_root=chunk_root,
        chunk_size=shared_state.chunk_size,
        resume=True,
        keep_chunks=False,
    )
    return assembler.assemble_final()
