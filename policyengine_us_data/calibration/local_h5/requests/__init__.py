"""Request-oriented local H5 contracts.

This package owns request shapes and related helpers for the H5
migration. Future request-building utilities should live here next to
the request contracts rather than inside a giant shared contracts file.
"""

from .contracts import AreaBuildRequest, AreaFilter, AreaType, FilterOp

__all__ = ["AreaBuildRequest", "AreaFilter", "AreaType", "FilterOp"]
