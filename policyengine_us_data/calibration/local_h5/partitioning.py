"""Pure helpers for assigning weighted work items to worker chunks."""

from __future__ import annotations

import heapq
from collections.abc import Mapping, Sequence
from typing import Any


def work_item_key(item: Mapping[str, Any]) -> str:
    """Return the stable completion key used by the current H5 workers."""

    return f"{item['type']}:{item['id']}"


def partition_weighted_work_items(
    work_items: Sequence[Mapping[str, Any]],
    num_workers: int,
    completed: set[str] | None = None,
) -> list[list[Mapping[str, Any]]]:
    """Partition work items across workers using longest-processing-time first."""

    if num_workers <= 0:
        return []

    completed = completed or set()
    remaining = [item for item in work_items if work_item_key(item) not in completed]
    remaining.sort(key=lambda item: -item["weight"])

    n_workers = min(num_workers, len(remaining))
    if n_workers == 0:
        return []

    heap: list[tuple[int | float, int]] = [(0, idx) for idx in range(n_workers)]
    chunks: list[list[Mapping[str, Any]]] = [[] for _ in range(n_workers)]

    for item in remaining:
        load, idx = heapq.heappop(heap)
        chunks[idx].append(item)
        heapq.heappush(heap, (load + item["weight"], idx))

    return [chunk for chunk in chunks if chunk]
