"""Structured documentation schema for pipeline and library waypoints.

The objects in this module are intentionally independent of any diagramming
frontend. They describe pipeline participation, library object identity,
temporary migration state, and validation hooks. Renderers can project this
metadata into graphs, Markdown, API pages, or agent-readable context.
"""

from dataclasses import asdict, dataclass, field
from typing import Literal


NodeType = Literal[
    "entrypoint",
    "stage",
    "process",
    "library",
    "artifact",
    "utility",
    "external",
    "validation",
    "infrastructure",
    "legacy",
    "missing",
]

EdgeType = Literal[
    "data_flow",
    "produces_artifact",
    "uses_library",
    "uses_utility",
    "external_source",
    "runs_on_infra",
    "validates",
    "documents",
    "migration",
    "informational",
]

NodeStatus = Literal[
    "current",
    "transitional",
    "legacy",
    "planned",
    "unknown",
]

Stability = Literal[
    "stable",
    "moving",
    "experimental",
    "unknown",
]


@dataclass
class PipelineNode:
    """One documented waypoint in the data pipeline or library surface.

    Args:
        id: Stable node identifier, usually snake_case.
        label: Human-readable display name.
        node_type: Broad role of the node.
        description: Short summary for maps and tables.
        details: Longer implementation notes.
        source_file: Repository-relative source path when known.
        status: Whether this node is current, transitional, legacy, planned,
            or unknown.
        stability: How safe it is for agents to treat this as a stable
            contract.
        pathways: Logical pathways this node participates in.
        api_refs: Import paths for library objects documented elsewhere.
        artifacts_in: Named input artifacts consumed by this node.
        artifacts_out: Named output artifacts produced by this node.
        validation_commands: Focused checks that exercise the node or seam.
        migration_target: Optional target node or architecture destination.
        notes: Additional cautionary context.
        pydoc: Whether the extractor should include object docs for this node.
    """

    id: str
    label: str
    node_type: NodeType = "process"
    description: str = ""
    details: str = ""
    source_file: str = ""
    status: NodeStatus = "unknown"
    stability: Stability = "unknown"
    pathways: list[str] = field(default_factory=list)
    api_refs: list[str] = field(default_factory=list)
    artifacts_in: list[str] = field(default_factory=list)
    artifacts_out: list[str] = field(default_factory=list)
    validation_commands: list[str] = field(default_factory=list)
    migration_target: str = ""
    notes: str = ""
    pydoc: bool = True

    def to_dict(self) -> dict:
        """Return a JSON-serializable representation."""
        return asdict(self)


@dataclass
class PipelineEdge:
    """A directed relationship between two documented nodes."""

    source: str
    target: str
    edge_type: EdgeType = "data_flow"
    label: str = ""
    status: NodeStatus = "unknown"
    stability: Stability = "unknown"
    notes: str = ""

    def to_dict(self) -> dict:
        """Return a JSON-serializable representation."""
        return asdict(self)


@dataclass
class PipelineGroup:
    """A named collection of related nodes within a stage."""

    id: str
    label: str
    description: str = ""
    node_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Return a JSON-serializable representation."""
        return asdict(self)


@dataclass
class PipelineStage:
    """A stage or pathway section in the structured pipeline map."""

    id: str
    label: str
    title: str
    description: str
    status: NodeStatus = "unknown"
    stability: Stability = "unknown"
    nodes: list[PipelineNode] = field(default_factory=list)
    edges: list[PipelineEdge] = field(default_factory=list)
    groups: list[PipelineGroup] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Return a JSON-serializable representation."""
        return asdict(self)
