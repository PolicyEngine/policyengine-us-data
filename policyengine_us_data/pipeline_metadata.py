"""No-op decorators for structured pipeline and library documentation.

These decorators attach metadata to Python objects without changing runtime
behavior. The documentation extractor reads them statically with ``ast`` so
heavy modules do not need to be imported just to build docs.
"""

from collections.abc import Callable
from typing import TypeVar, overload

from policyengine_us_data.pipeline_schema import PipelineNode


T = TypeVar("T")


@overload
def pipeline_node(node: PipelineNode) -> Callable[[T], T]: ...


@overload
def pipeline_node(**kwargs) -> Callable[[T], T]: ...


def pipeline_node(node: PipelineNode | None = None, **kwargs) -> Callable[[T], T]:
    """Attach structured pipeline metadata to a function or class.

    The decorator supports both explicit dataclass usage and keyword shorthand:

    .. code-block:: python

        @pipeline_node(PipelineNode(id="build_h5", label="Build H5"))
        def build_h5(...):
            ...

        @pipeline_node(id="build_h5", label="Build H5", status="current")
        def build_h5(...):
            ...
    """

    metadata = node if node is not None else PipelineNode(**kwargs)

    def wrapper(obj: T) -> T:
        setattr(obj, "_pipeline_node", metadata)
        return obj

    return wrapper
