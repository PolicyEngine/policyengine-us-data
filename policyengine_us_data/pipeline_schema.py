"""Pipeline documentation schema.

Defines typed dataclasses for pipeline nodes, edges, and stages.
Used by @pipeline_node decorators to annotate source code and by
the Griffe extraction script to generate pipeline.json.

Node and edge types match the visual taxonomy in
docs/pipeline-diagrams/CLAUDE.md and components/legends.js.
"""

from dataclasses import dataclass, field
from typing import Literal

NodeType = Literal[
    "input",
    "output",
    "process",
    "utility",
    "external",
    "us_specific",
    "uk_specific",
    "missing",
    "absent",
]

EdgeType = Literal[
    "data_flow",
    "produces_artifact",
    "uses_utility",
    "external_source",
    "runs_on_infra",
    "informational",
]

# Node type → fill color, border color (matching legends.js)
NODE_COLORS = {
    "input": {"fill": "#dbeafe", "border": "#3b82f6"},
    "output": {"fill": "#dcfce7", "border": "#22c55e"},
    "process": {"fill": "#ffedd5", "border": "#f97316"},
    "utility": {"fill": "#f3e8ff", "border": "#a855f7"},
    "external": {"fill": "#fef9c3", "border": "#eab308"},
    "us_specific": {"fill": "#fce7f3", "border": "#ec4899"},
    "uk_specific": {"fill": "#ccfbf1", "border": "#14b8a6"},
    "missing": {"fill": "#fee2e2", "border": "#ef4444"},
    "absent": {"fill": "#f3f4f6", "border": "#d1d5db"},
}

# Edge type → color, style, width (matching legends.js)
EDGE_STYLES = {
    "data_flow": {"color": "#334155", "style": "solid", "width": 2},
    "produces_artifact": {"color": "#16a34a", "style": "solid", "width": 2},
    "uses_utility": {"color": "#7c3aed", "style": "dashed", "width": 1.5},
    "external_source": {"color": "#b45309", "style": "dotted", "width": 1.5},
    "runs_on_infra": {"color": "#dc2626", "style": "dashed", "width": 1.5},
    "informational": {"color": "#9ca3af", "style": "dotted", "width": 1},
}


@dataclass
class PipelineNode:
    """A node in a pipeline stage diagram.

    Args:
        id: Unique identifier (snake_case, e.g., "add_rent").
        label: Display name (e.g., "Rent Imputation (QRF)").
        node_type: Visual category from NodeType.
        description: One-line summary shown in tooltips.
        details: Multi-line implementation notes shown in detail panel.
        source_file: Python file path relative to repo root.
    """

    id: str
    label: str
    node_type: NodeType
    description: str = ""
    details: str = ""
    source_file: str = ""


@dataclass
class PipelineEdge:
    """A directed edge between two pipeline nodes.

    Args:
        source: Source node ID.
        target: Target node ID.
        edge_type: Visual category from EdgeType.
        label: Artifact name or data description on the edge.
    """

    source: str
    target: str
    edge_type: EdgeType = "data_flow"
    label: str = ""


@dataclass
class PipelineStage:
    """A stage in the pipeline (e.g., Stage 1: Base Dataset Construction).

    Args:
        id: Stage number (0-8).
        label: Short label (e.g., "Stage 1").
        title: Full title (e.g., "Stage 1: Base Dataset Construction").
        description: What this stage does.
        country: Which country pipeline ("us" or "uk").
        nodes: Nodes in this stage.
        edges: Edges within this stage.
    """

    id: int
    label: str
    title: str
    description: str
    country: str = "us"
    nodes: list[PipelineNode] = field(default_factory=list)
    edges: list[PipelineEdge] = field(default_factory=list)
