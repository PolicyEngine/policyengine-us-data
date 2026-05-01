"""Pure helpers for assigning weighted work items to worker chunks."""

from __future__ import annotations

import heapq
from collections.abc import Mapping, Sequence
from typing import Any

from policyengine_us_data.pipeline_metadata import pipeline_node

WorkItem = Mapping[str, Any]
WorkItems = Sequence[WorkItem]
WorkChunks = list[list[WorkItem]]

__all__ = [
    "WorkChunks",
    "WorkItem",
    "WorkItems",
    "partition_weighted_work_items",
]


def work_item_key(item: WorkItem) -> str:
    """Return the stable completion key used by the current H5 workers."""

    return f"{item['type']}:{item['id']}"


@pipeline_node(
    id="local_h5_partition",
    label="Partition Local H5 Work",
    node_type="library",
    description="Assign weighted area work items to worker batches using longest-processing-time scheduling.",
    source_file="policyengine_us_data/calibration/local_h5/partitioning.py",
    status="current",
    stability="stable",
    pathways=["local_h5"],
    validation_commands=[
        "uv run pytest tests/unit/calibration/test_local_h5_partitioning.py"
    ],
)
def partition_weighted_work_items(
    work_items: WorkItems,
    num_workers: int,
    completed: set[str] | None = None,
) -> WorkChunks:
    """Partition remaining H5 work across worker chunks.

    The function uses longest-processing-time scheduling: uncompleted work items
    are sorted by descending `weight`, then assigned to the currently lightest
    worker chunk. This keeps expensive state or district builds from clustering
    on one worker.

    Args:
        work_items: Candidate work items. Each item must contain `type`, `id`,
            and numeric `weight` keys.
        num_workers: Maximum number of worker chunks to produce.
        completed: Stable completion keys, formatted as `"{type}:{id}"`, that
            should be skipped.

    Returns:
        Non-empty worker chunks. Returns an empty list when `num_workers <= 0` or
        every item is already completed.
    """

    if num_workers <= 0:
        return []

    completed = completed or set()
    remaining = [item for item in work_items if work_item_key(item) not in completed]
    remaining.sort(key=lambda item: -item["weight"])

    n_workers = min(num_workers, len(remaining))
    if n_workers == 0:
        return []

    heap: list[tuple[int | float, int]] = [(0, idx) for idx in range(n_workers)]
    chunks: WorkChunks = [[] for _ in range(n_workers)]

    for item in remaining:
        load, idx = heapq.heappop(heap)
        chunks[idx].append(item)
        heapq.heappush(heap, (load + item["weight"], idx))

    return [chunk for chunk in chunks if chunk]
