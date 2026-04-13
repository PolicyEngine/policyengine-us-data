"""Build-result contracts for local H5 publication.

This package owns typed per-area and per-worker result structures.
Future result-aggregation helpers should live here rather than inside
coordinator adapters.
"""

from .contracts import AreaBuildResult, BuildStatus, WorkerResult

__all__ = ["AreaBuildResult", "BuildStatus", "WorkerResult"]
