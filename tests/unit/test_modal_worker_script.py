import json
from types import SimpleNamespace

from tests.unit.fixtures.test_modal_worker_script import (
    FakeAreaBuildRequest,
    FakeAreaCatalog,
    FakeRequest,
    load_worker_script_module,
)


worker_script = load_worker_script_module()


def test_parse_args_accepts_requests_json():
    args = worker_script.parse_args(
        [
            "--requests-json",
            "[]",
            "--weights-path",
            "/tmp/weights.npy",
            "--dataset-path",
            "/tmp/source.h5",
            "--db-path",
            "/tmp/policy_data.db",
            "--output-dir",
            "/tmp/out",
        ]
    )

    assert args.requests_json == "[]"
    assert args.work_items is None


def test_load_request_inputs_from_args_uses_request_payloads_when_present():
    args = SimpleNamespace(
        requests_json=json.dumps([{"area_type": "national", "area_id": "US"}]),
        work_items=None,
    )

    mode, requests = worker_script._load_request_inputs_from_args(
        args=args,
        area_build_request_cls=FakeAreaBuildRequest,
    )

    assert mode == "requests"
    assert len(requests) == 1
    assert requests[0].payload["area_id"] == "US"


def test_load_request_inputs_from_args_keeps_legacy_work_items_raw():
    args = SimpleNamespace(
        requests_json=None,
        work_items=json.dumps([{"type": "national", "id": "US"}]),
    )

    mode, work_items = worker_script._load_request_inputs_from_args(
        args=args,
        area_build_request_cls=FakeAreaBuildRequest,
    )

    assert mode == "work_items"
    assert work_items == ({"type": "national", "id": "US"},)


def test_work_item_key_handles_missing_fields():
    assert worker_script._work_item_key({"type": "district"}) == "district:<missing-id>"
    assert worker_script._work_item_key(["not-a-dict"]) == "unknown:<invalid-work-item>"


def test_resolve_request_input_keeps_typed_requests_unchanged():
    request = FakeRequest(area_type="national", area_id="US")

    request_key, resolved = worker_script._resolve_request_input(
        request_input_mode="requests",
        request_input=request,
        area_catalog=FakeAreaCatalog(),
        geography=object(),
    )

    assert request_key == "national:US"
    assert resolved is request


def test_resolve_request_input_converts_one_legacy_work_item_at_a_time():
    catalog = FakeAreaCatalog()
    geography = object()
    work_item = {"type": "district", "id": "AK-01"}

    request_key, request = worker_script._resolve_request_input(
        request_input_mode="work_items",
        request_input=work_item,
        area_catalog=catalog,
        geography=geography,
    )

    assert request_key == "district:AK-01"
    assert request.area_type == "district"
    assert request.area_id == "AK-01"
    assert catalog.received_item == (work_item, geography)


def test_resolve_request_input_skips_legacy_work_item_without_request():
    catalog = FakeAreaCatalog()
    geography = object()
    work_item = {"type": "state", "id": "WY"}
    catalog.none_for = work_item

    request_key, request = worker_script._resolve_request_input(
        request_input_mode="work_items",
        request_input=work_item,
        area_catalog=catalog,
        geography=geography,
    )

    assert request_key == "state:WY"
    assert request is None
    assert catalog.received_item == (work_item, geography)
