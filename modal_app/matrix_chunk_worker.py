"""Modal worker that materializes a batch of matrix chunks.

Registered on the same ``policyengine-us-data-fit-weights`` app as
``build_package_remote`` so the coordinator can spawn workers via
``modal.Function.from_name`` from inside the package build's
subprocess. Each worker reads the shared ``ChunkedMatrixAssembler``
state from ``pipeline_volume``, materializes its assigned chunks to
COO shard files on the volume, and commits. The coordinator reads the
shards back after all workers finish and streams them into the final
CSR matrix.
"""

from __future__ import annotations

import pickle
import sys
import traceback
from pathlib import Path
from typing import Dict, List

_baked = "/root/policyengine-us-data"
_local = str(Path(__file__).resolve().parent.parent)
for _p in (_baked, _local):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from modal_app.images import cpu_image  # noqa: E402
from modal_app.remote_calibration_runner import (  # noqa: E402
    PIPELINE_MOUNT,
    app,
    hf_secret,
    pipeline_vol,
)


def _chunk_root(run_id: str) -> str:
    return f"{PIPELINE_MOUNT}/artifacts/{run_id}/matrix_build"


@app.function(
    image=cpu_image,
    secrets=[hf_secret],
    volumes={PIPELINE_MOUNT: pipeline_vol},
    memory=16384,
    cpu=1.0,
    timeout=28800,
    max_containers=50,
    nonpreemptible=True,
)
def build_matrix_chunk_worker(run_id: str, chunk_ids: List[int]) -> Dict:
    """Materialize ``chunk_ids`` from the pickled ``SharedBuildState``.

    Args:
        run_id: Pipeline run identifier; selects the volume path for
            this worker's shared state and shard output directory.
        chunk_ids: Chunk indices this worker is responsible for.

    Returns:
        Dict with ``chunk_ids``, ``nnz_per_chunk``, and ``errors``
        lists suitable for the coordinator to aggregate.
    """
    from policyengine_us_data.calibration.chunked_matrix_assembler import (
        ChunkedMatrixAssembler,
    )

    pipeline_vol.reload()
    chunk_root = Path(_chunk_root(run_id))
    state_path = chunk_root / "chunk_build_state.pkl"
    if not state_path.exists():
        return {
            "chunk_ids": list(chunk_ids),
            "nnz_per_chunk": [],
            "errors": [
                {
                    "chunk_ids": list(chunk_ids),
                    "error": f"Missing shared state at {state_path}",
                }
            ],
        }

    with open(state_path, "rb") as f:
        shared_state = pickle.load(f)

    assembler = ChunkedMatrixAssembler(
        shared_state=shared_state,
        chunk_root=chunk_root,
        chunk_size=shared_state.chunk_size,
        resume=True,
        keep_chunks=False,
    )

    errors: List[Dict] = []
    nnz_per_chunk: List[int] = []
    for chunk_id in chunk_ids:
        try:
            result = assembler.run_single_chunk(chunk_id)
            nnz_per_chunk.append(result.nnz)
        except Exception as exc:
            errors.append(
                {
                    "chunk_id": chunk_id,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )

    pipeline_vol.commit()
    return {
        "chunk_ids": list(chunk_ids),
        "nnz_per_chunk": nnz_per_chunk,
        "errors": errors,
    }
