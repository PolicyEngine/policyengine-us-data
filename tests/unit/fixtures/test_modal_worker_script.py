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
    """Catalog double that records the legacy compatibility inputs it sees."""

    def __init__(self, requests):
        self.requests = tuple(requests)
        self.received = None

    def build_requests_from_work_items(self, work_items, *, geography):
        self.received = (work_items, geography)
        return self.requests
