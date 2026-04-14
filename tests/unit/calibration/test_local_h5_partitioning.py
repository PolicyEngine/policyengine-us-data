from tests.unit.calibration.fixtures.test_local_h5_partitioning import (
    load_partitioning_exports,
)


partitioning = load_partitioning_exports()
flatten_chunks = partitioning["flatten_chunks"]
partition_weighted_work_items = partitioning["partition_weighted_work_items"]
work_item_key = partitioning["work_item_key"]


def test_work_item_key_uses_existing_completion_shape():
    item = {"type": "district", "id": "CA-12", "weight": 1}
    assert work_item_key(item) == "district:CA-12"


def test_partition_filters_completed_items():
    work_items = [
        {"type": "state", "id": "CA", "weight": 3},
        {"type": "district", "id": "CA-12", "weight": 1},
        {"type": "city", "id": "NYC", "weight": 2},
    ]

    chunks = partition_weighted_work_items(
        work_items,
        num_workers=2,
        completed={"district:CA-12"},
    )

    flattened = flatten_chunks(chunks)
    assert all(item["id"] != "CA-12" for item in flattened)
    assert {item["id"] for item in flattened} == {"CA", "NYC"}


def test_partition_returns_empty_for_zero_workers_or_zero_remaining():
    work_items = [{"type": "state", "id": "CA", "weight": 1}]

    assert partition_weighted_work_items(work_items, num_workers=0) == []
    assert (
        partition_weighted_work_items(
            work_items,
            num_workers=3,
            completed={"state:CA"},
        )
        == []
    )


def test_partition_uses_no_more_workers_than_remaining_items():
    work_items = [
        {"type": "state", "id": "CA", "weight": 5},
        {"type": "state", "id": "NY", "weight": 4},
    ]

    chunks = partition_weighted_work_items(work_items, num_workers=10)

    assert len(chunks) == 2
    assert all(len(chunk) == 1 for chunk in chunks)


def test_partition_is_weight_balancing_and_deterministic_for_equal_weights():
    work_items = [
        {"type": "district", "id": "A", "weight": 5},
        {"type": "district", "id": "B", "weight": 5},
        {"type": "district", "id": "C", "weight": 2},
        {"type": "district", "id": "D", "weight": 2},
    ]

    chunks = partition_weighted_work_items(work_items, num_workers=2)

    ids_by_chunk = [[item["id"] for item in chunk] for chunk in chunks]
    loads = [sum(item["weight"] for item in chunk) for chunk in chunks]

    assert ids_by_chunk == [["A", "C"], ["B", "D"]]
    assert loads == [7, 7]
