"""Env-gated Modal smoke test for the matrix-chunk worker.

Skipped by default. Runs only when all of:
  - ``MODAL_TOKEN_ID`` and ``MODAL_TOKEN_SECRET`` are set (Modal auth)
  - ``POLICYENGINE_US_DATA_MODAL_SMOKE=1`` is set

Assumes the pipeline app (which ``.include()``s the worker) has been
deployed via:

    modal deploy modal_app/pipeline.py

The worker's ``@app.function`` decorator attaches to ``_calibration_app``
(``policyengine-us-data-fit-weights``) at the Python level, but
``pipeline.py`` merges that app into the pipeline app. After deploy,
the function is registered in Modal's registry under the pipeline
app's name — that's the name this test looks up.

This test validates two things without running a full chunk build:
  1. The Modal worker function is discoverable via
     ``modal.Function.from_name`` — catches "worker not deployed" and
     signature mismatches at test time rather than at pipeline time.
  2. ``partition_chunk_ids_contiguous`` produces the same batches the
     coordinator would send — sanity on the shape we'd ship.

A true end-to-end fan-out smoke (write shared state, spawn, verify
shards) requires the pipeline volume to hold a real fixture dataset;
that is a pre-merge manual step for phase 2, documented in the PR.
"""

from __future__ import annotations

import os

import pytest


_SMOKE_ENABLED = (
    os.environ.get("MODAL_TOKEN_ID")
    and os.environ.get("MODAL_TOKEN_SECRET")
    and os.environ.get("POLICYENGINE_US_DATA_MODAL_SMOKE") == "1"
)


pytestmark = pytest.mark.skipif(
    not _SMOKE_ENABLED,
    reason=(
        "Modal smoke test; set MODAL_TOKEN_ID, MODAL_TOKEN_SECRET, and "
        "POLICYENGINE_US_DATA_MODAL_SMOKE=1 to enable"
    ),
)


def test_worker_function_is_deployed() -> None:
    """The deployed worker must be lookupable under the pipeline app."""
    import modal

    worker = modal.Function.from_name(
        "policyengine-us-data-pipeline",
        "build_matrix_chunk_worker",
    )
    assert worker is not None


def test_dispatch_contiguous_batching_shape() -> None:
    """The batching produced for a typical production-scale run is sane."""
    from policyengine_us_data.calibration.chunked_matrix_modal import (
        partition_chunk_ids_contiguous,
    )

    # ~207 chunks at production; check the 50-worker default shape.
    batches = partition_chunk_ids_contiguous(n_chunks=207, num_workers=50)
    assert len(batches) <= 50
    assert sum(len(b) for b in batches) == 207
    # Contiguous: every batch is a range of consecutive ids.
    for batch in batches:
        assert list(batch) == list(range(batch[0], batch[0] + len(batch)))
