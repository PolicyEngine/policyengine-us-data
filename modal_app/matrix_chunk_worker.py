"""Modal worker that materializes a batch of matrix chunks.

The ``@app.function`` decorator here attaches to ``_calibration_app``
(declared as ``policyengine-us-data-fit-weights`` in
``remote_calibration_runner.py``), alongside ``build_package_remote``.
At deploy time, ``modal_app/pipeline.py`` merges that app into the
pipeline app via ``app.include(_calibration_app)``, so after
``modal deploy modal_app/pipeline.py`` the function is registered
under ``policyengine-us-data-pipeline`` in Modal's registry — that's
the name ``dispatch_chunks_modal`` uses in its
``modal.Function.from_name`` lookup.

Each worker reads the shared ``ChunkedMatrixAssembler`` state from
``pipeline_volume``, materializes its assigned chunks to COO shard
files on the volume, and commits. The coordinator reads the shards
back after all workers finish and streams them into the final CSR
matrix.
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
def build_matrix_chunk_worker(
    request: Dict | None = None,
    run_id: str | None = None,
    chunk_ids: List[int] | None = None,
    resume_chunks: bool = False,
) -> Dict:
    """Materialize a typed chunk request from pickled ``SharedBuildState``.

    Args:
        request: Typed worker request material. Legacy ``run_id`` and
            ``chunk_ids`` arguments are accepted for compatibility with
            older local tests and undeployed call sites.
        run_id: Legacy pipeline run identifier.
        chunk_ids: Legacy chunk indices this worker is responsible for.
        resume_chunks: Legacy resume flag.

    Returns:
        Structured worker result material suitable for the coordinator to
        aggregate.
    """
    from policyengine_us_data.calibration.chunked_matrix_assembler import (
        ChunkedMatrixAssembler,
    )
    from policyengine_us_data.calibration.signatures import signature_mismatches
    from policyengine_us_data.calibration_package.matrix import (
        CHUNK_EXECUTION_SCHEMA_VERSION,
        ChunkBuildRequest,
        ChunkExecutionResult,
        ChunkWorkerResult,
        write_chunk_result_manifest,
    )

    pipeline_vol.reload()
    if request is None:
        if run_id is None or chunk_ids is None:
            raise ValueError("request or legacy run_id/chunk_ids are required")
        chunk_root = Path(_chunk_root(run_id))
        request_obj = ChunkBuildRequest(
            schema_version=CHUNK_EXECUTION_SCHEMA_VERSION,
            run_id=run_id,
            chunk_ids=tuple(chunk_ids),
            chunk_root=str(chunk_root),
            state_path=str(chunk_root / "chunk_build_state.pkl"),
            resume_chunks=resume_chunks,
            lineage_signature={},
        )
    else:
        request_obj = ChunkBuildRequest.from_dict(request)
    chunk_root = Path(request_obj.chunk_root)
    state_path = Path(request_obj.state_path)
    if not state_path.exists():
        chunk_results = tuple(
            ChunkExecutionResult.failure(
                run_id=request_obj.run_id,
                chunk_id=chunk_id,
                error=f"Missing shared state at {state_path}",
            )
            for chunk_id in request_obj.chunk_ids
        )
        for result in chunk_results:
            write_chunk_result_manifest(chunk_root, result)
        pipeline_vol.commit()
        return ChunkWorkerResult(
            schema_version=CHUNK_EXECUTION_SCHEMA_VERSION,
            run_id=request_obj.run_id,
            chunk_ids=request_obj.chunk_ids,
            chunk_results=chunk_results,
        ).to_dict()

    with open(state_path, "rb") as f:
        shared_state = pickle.load(f)
    if request_obj.lineage_signature:
        state_lineage_signature = getattr(shared_state, "lineage_signature", {})
        fatal, _ = signature_mismatches(
            state_lineage_signature,
            request_obj.lineage_signature,
        )
        if fatal:
            error = "Chunk request lineage mismatch: " + "; ".join(fatal)
            chunk_results = tuple(
                ChunkExecutionResult.failure(
                    run_id=request_obj.run_id,
                    chunk_id=chunk_id,
                    error=error,
                )
                for chunk_id in request_obj.chunk_ids
            )
            for result in chunk_results:
                write_chunk_result_manifest(chunk_root, result)
            pipeline_vol.commit()
            return ChunkWorkerResult(
                schema_version=CHUNK_EXECUTION_SCHEMA_VERSION,
                run_id=request_obj.run_id,
                chunk_ids=request_obj.chunk_ids,
                chunk_results=chunk_results,
            ).to_dict()

    assembler = ChunkedMatrixAssembler(
        shared_state=shared_state,
        chunk_root=chunk_root,
        chunk_size=shared_state.chunk_size,
        resume=request_obj.resume_chunks,
        keep_chunks=False,
    )

    chunk_results: List[ChunkExecutionResult] = []
    for chunk_id in request_obj.chunk_ids:
        try:
            result = assembler.run_single_chunk(chunk_id)
            chunk_results.append(
                ChunkExecutionResult.from_chunk_result(
                    run_id=request_obj.run_id,
                    result=result,
                )
            )
        except Exception as exc:
            traceback_text = traceback.format_exc()
            assembler.record_chunk_error(
                chunk_id=chunk_id,
                error=str(exc),
                traceback=traceback_text,
            )
            chunk_results.append(
                ChunkExecutionResult.failure(
                    run_id=request_obj.run_id,
                    chunk_id=chunk_id,
                    error=str(exc),
                    traceback=traceback_text,
                )
            )

    pipeline_vol.commit()
    return ChunkWorkerResult(
        schema_version=CHUNK_EXECUTION_SCHEMA_VERSION,
        run_id=request_obj.run_id,
        chunk_ids=request_obj.chunk_ids,
        chunk_results=tuple(chunk_results),
    ).to_dict()
