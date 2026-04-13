import json
from types import SimpleNamespace

from tests.unit.fixtures.test_modal_worker_script import (
    FakeAreaBuildRequest,
    FakeAreaCatalog,
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


def test_load_requests_from_args_uses_request_payloads_when_present():
    args = SimpleNamespace(
        requests_json=json.dumps([{"area_type": "national", "area_id": "US"}]),
        work_items=None,
    )

    requests = worker_script._load_requests_from_args(
        args=args,
        area_build_request_cls=FakeAreaBuildRequest,
        area_catalog=FakeAreaCatalog(()),
        geography=object(),
    )

    assert len(requests) == 1
    assert requests[0].payload["area_id"] == "US"


def test_load_requests_from_args_keeps_legacy_work_items_compatibility():
    catalog = FakeAreaCatalog(requests=("typed-request",))
    geography = object()
    args = SimpleNamespace(
        requests_json=None,
        work_items=json.dumps([{"type": "national", "id": "US"}]),
    )

    requests = worker_script._load_requests_from_args(
        args=args,
        area_build_request_cls=FakeAreaBuildRequest,
        area_catalog=catalog,
        geography=geography,
    )

    assert requests == ("typed-request",)
    assert catalog.received == ([{"type": "national", "id": "US"}], geography)
