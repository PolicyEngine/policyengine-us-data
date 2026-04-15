"""Fixture helpers for ``test_modal_worker_script.py``."""

from __future__ import annotations

import importlib

__test__ = False


def load_worker_script_module():
    """Import the worker script module for direct helper testing."""

    return importlib.import_module("modal_app.worker_script")


class FakeAreaBuildRequest:
    """Minimal request type for worker parsing tests."""

    def __init__(self, payload):
        self.payload = payload

    @classmethod
    def from_dict(cls, data):
        return cls(payload=data)


class FakeAreaCatalog:
    """Catalog double for worker-script request resolution tests."""

    def __init__(self, requests=()):
        self.requests = tuple(requests)
        self.received = None
        self.received_item = None
        self.raise_for = None
        self.none_for = None

    def build_requests_from_work_items(self, work_items, *, geography):
        self.received = (work_items, geography)
        return self.requests

    def build_request_from_work_item(self, work_item, *, geography):
        self.received_item = (work_item, geography)
        if work_item == self.raise_for:
            raise ValueError("bad work item")
        if work_item == self.none_for:
            return None
        return FakeRequest(area_type=work_item["type"], area_id=work_item["id"])


class FakeRequest:
    """Minimal typed request used by worker resolution tests."""

    def __init__(self, *, area_type, area_id):
        self.area_type = area_type
        self.area_id = area_id
