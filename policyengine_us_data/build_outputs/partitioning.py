"""Pure helpers for assigning weighted local H5 requests to worker chunks."""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from collections.abc import Mapping, Sequence
from typing import Any

from policyengine_us_data.pipeline_metadata import pipeline_node

WorkItem = Mapping[str, Any]
WorkItems = Sequence[WorkItem]
WorkChunks = list[list[WorkItem]]
WeightedAreaRequestChunks = list[list["WeightedAreaRequest"]]

__all__ = [
    "WeightedAreaRequest",
    "WeightedAreaRequestChunks",
    "WorkChunks",
    "WorkItem",
    "WorkItems",
    "partition_weighted_area_requests",
    "partition_weighted_work_items",
]


@pipeline_node(
    id="local_h5_weighted_area_request",
    label="WeightedAreaRequest",
    node_type="library",
    description="Typed local H5 request with coordinator scheduling weight.",
    source_file="policyengine_us_data/build_outputs/partitioning.py",
    status="current",
    stability="moving",
    pathways=["local_h5"],
    validation_commands=["uv run pytest tests/unit/build_outputs/test_partitioning.py"],
)
@dataclass(frozen=True)
class WeightedAreaRequest:
    """Area build request plus scheduling weight for coordinator partitioning."""

    request: Any
    weight: int | float = 1

    def __post_init__(self) -> None:
        if isinstance(self.weight, bool) or not isinstance(self.weight, int | float):
            raise TypeError("weight must be numeric")
        if self.weight <= 0:
            raise ValueError("weight must be positive")

    @property
    def key(self) -> str:
        """Return the stable completion key for this request."""

        return f"{self.request.area_type}:{self.request.area_id}"

    def to_worker_payload(self) -> dict[str, Any]:
        """Serialize the request for `modal_app.worker_script --requests-json`."""

        return self.request.to_dict()


def work_item_key(item: WorkItem) -> str:
    """Return the stable completion key used by the current H5 workers."""

    return f"{item['type']}:{item['id']}"


@pipeline_node(
    id="local_h5_partition",
    label="Partition Local H5 Work",
    node_type="library",
    description="Assign weighted area work items to worker batches using longest-processing-time scheduling.",
    source_file="policyengine_us_data/build_outputs/partitioning.py",
    status="current",
    stability="stable",
    pathways=["local_h5"],
    validation_commands=["uv run pytest tests/unit/build_outputs/test_partitioning.py"],
)
def partition_weighted_area_requests(
    requests: Sequence[WeightedAreaRequest],
    num_workers: int,
    completed: set[str] | None = None,
) -> WeightedAreaRequestChunks:
    """Partition remaining typed H5 requests across worker chunks."""

    return _partition_weighted_items(
        items=tuple(requests),
        num_workers=num_workers,
        completed=completed,
        key=lambda item: item.key,
        weight=lambda item: item.weight,
    )


@pipeline_node(
    id="local_h5_legacy_work_item_partition",
    label="Partition Legacy Local H5 Work Items",
    node_type="library",
    description="Compatibility wrapper for assigning legacy local H5 work items to workers.",
    source_file="policyengine_us_data/build_outputs/partitioning.py",
    status="legacy",
    stability="moving",
    pathways=["local_h5"],
    validation_commands=["uv run pytest tests/unit/build_outputs/test_partitioning.py"],
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

    return _partition_weighted_items(
        items=tuple(work_items),
        num_workers=num_workers,
        completed=completed,
        key=work_item_key,
        weight=lambda item: item["weight"],
    )


def _partition_weighted_items(
    *,
    items: tuple[Any, ...],
    num_workers: int,
    completed: set[str] | None,
    key,
    weight,
):
    if num_workers <= 0:
        return []

    completed = completed or set()
    remaining = [item for item in items if key(item) not in completed]
    remaining.sort(key=lambda item: -weight(item))

    n_workers = min(num_workers, len(remaining))
    if n_workers == 0:
        return []

    heap: list[tuple[int | float, int]] = [(0, idx) for idx in range(n_workers)]
    chunks = [[] for _ in range(n_workers)]

    for item in remaining:
        load, idx = heapq.heappop(heap)
        chunks[idx].append(item)
        heapq.heappush(heap, (load + weight(item), idx))

    return [chunk for chunk in chunks if chunk]
