"use client";

import { useCallback, useEffect, useState, type MouseEvent } from "react";
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  ReactFlowProvider,
  useReactFlow,
  useNodesState,
  useEdgesState,
  type Node,
  type Edge,
  BackgroundVariant,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import ELK from "elkjs/lib/elk.bundled.js";
import { NODE_COLORS, EDGE_STYLES } from "../colors";
import PipelineNode from "./PipelineNode";
import ElkEdge from "./ElkEdge";
import NodeDetailPanel from "./NodeDetailPanel";

const elk = new ELK();
const nodeTypes = { pipeline: PipelineNode };
const edgeTypes = { elk: ElkEdge };
const NODE_WIDTH = 240;
const NODE_HEIGHT = 64;
const NODE_HEIGHT_COMPACT = 58;

const ELK_OPTIONS: Record<string, string> = {
  "elk.algorithm": "layered",
  "elk.direction": "DOWN",
  "elk.edgeRouting": "ORTHOGONAL",
  "elk.spacing.nodeNode": "60",
  "elk.layered.spacing.nodeNodeBetweenLayers": "80",
  "elk.layered.spacing.edgeNodeBetweenLayers": "30",
  "elk.spacing.edgeEdge": "20",
  "elk.spacing.edgeNode": "30",
  "elk.layered.nodePlacement.strategy": "NETWORK_SIMPLEX",
  "elk.layered.crossingMinimization.strategy": "LAYER_SWEEP",
};

interface StageData {
  id: number | string;
  label: string;
  title: string;
  description: string;
  nodes: PipelineJsonNode[];
  edges: PipelineJsonEdge[];
}

interface PipelineJsonNode {
  id: string;
  label?: string;
  node_type?: string;
  description?: string;
  details?: string;
  source_file?: string;
}

interface PipelineJsonEdge {
  source: string;
  target: string;
  edge_type?: string;
  label?: string;
}

interface Point {
  x: number;
  y: number;
}

interface ElkRoute {
  startPoint?: Point;
  endPoint?: Point;
  bendPoints?: Point[];
}

interface ElkLayoutEdge {
  id?: string;
  sections?: ElkRoute[];
}

type PipelineNodeData = Record<string, unknown> & {
  label: string;
  nodeType: string;
  description: string;
  details: string;
  source_file: string;
  width: number;
  height: number;
};

type PipelineEdgeData = Record<string, unknown> & {
  elkRoute: ElkRoute | null;
};

type PipelineFlowNode = Node<PipelineNodeData, "pipeline">;
type PipelineFlowEdge = Edge<PipelineEdgeData, "elk">;

/**
 * Match the fixed dimensions used by PipelineNode so ELK and ReactFlow
 * agree on the node boxes the routes attach to.
 */
function estimateNodeSize(node: PipelineJsonNode) {
  return {
    width: NODE_WIDTH,
    height: node.description ? NODE_HEIGHT : NODE_HEIGHT_COMPACT,
  };
}

/**
 * Assign sourceHandle/targetHandle based on relative node positions.
 */
function assignHandles(
  sourcePos: { x: number; y: number },
  targetPos: { x: number; y: number }
) {
  const dx = targetPos.x - sourcePos.x;
  const dy = targetPos.y - sourcePos.y;

  if (Math.abs(dx) >= Math.abs(dy)) {
    return dx >= 0
      ? { sourceHandle: "sr", targetHandle: "tl" }
      : { sourceHandle: "sl", targetHandle: "tr" };
  }
  return dy >= 0
    ? { sourceHandle: "sb", targetHandle: "tt" }
    : { sourceHandle: "st", targetHandle: "tb" };
}

async function runElkLayout(
  pipelineNodes: PipelineJsonNode[],
  pipelineEdges: PipelineJsonEdge[]
): Promise<{ nodes: PipelineFlowNode[]; edges: PipelineFlowEdge[] }> {
  const nodeSizeMap: Record<string, { width: number; height: number }> = {};
  const graph = {
    id: "root",
    layoutOptions: ELK_OPTIONS,
    children: pipelineNodes.map((n) => {
      const size = estimateNodeSize(n);
      nodeSizeMap[n.id] = size;
      return { id: n.id, width: size.width, height: size.height };
    }),
    edges: pipelineEdges.map((e, i: number) => ({
      id: `e-${i}`,
      sources: [e.source],
      targets: [e.target],
    })),
  };

  const result = await elk.layout(graph);

  // Build position map
  const positionMap: Record<string, { x: number; y: number }> = {};
  for (const child of result.children || []) {
    positionMap[child.id] = { x: child.x || 0, y: child.y || 0 };
  }

  // Build edge route map from ELK sections
  const routeMap: Record<string, ElkRoute> = {};
  for (const edge of (result.edges || []) as ElkLayoutEdge[]) {
    if (edge.id && edge.sections && edge.sections.length > 0) {
      routeMap[edge.id] = edge.sections[0];
    }
  }

  // Position nodes
  const nodes: PipelineFlowNode[] = pipelineNodes.map((n) => ({
    id: n.id,
    type: "pipeline",
    position: positionMap[n.id] || { x: 0, y: 0 },
    style: {
      width: nodeSizeMap[n.id]?.width || NODE_WIDTH,
      height: nodeSizeMap[n.id]?.height || NODE_HEIGHT,
    },
    data: {
      label: n.label || n.id,
      nodeType: n.node_type || "process",
      description: n.description || "",
      details: n.details || "",
      source_file: n.source_file || "",
      width: nodeSizeMap[n.id]?.width || NODE_WIDTH,
      height: nodeSizeMap[n.id]?.height || NODE_HEIGHT,
    },
  }));

  // Enrich edges with handles + ELK routes
  const edges: PipelineFlowEdge[] = pipelineEdges.map((e, i: number) => {
    const edgeId = `e-${i}`;
    const srcPos = positionMap[e.source] || { x: 0, y: 0 };
    const tgtPos = positionMap[e.target] || { x: 0, y: 0 };
    const handles = assignHandles(srcPos, tgtPos);
    const route = routeMap[edgeId];
    const edgeStyle = EDGE_STYLES[e.edge_type ?? "data_flow"] || EDGE_STYLES.data_flow;

    return {
      id: edgeId,
      source: e.source,
      target: e.target,
      ...handles,
      type: "elk",
      label: e.label || undefined,
      data: {
        elkRoute: route || null,
      },
      style: {
        stroke: edgeStyle.color,
        strokeWidth: edgeStyle.width,
        strokeDasharray:
          edgeStyle.style === "dashed" ? "6 3" : edgeStyle.style === "dotted" ? "2 2" : undefined,
      },
    };
  });

  return { nodes, edges };
}

function DiagramInner({ stage }: { stage: StageData }) {
  const [nodes, setNodes] = useNodesState<PipelineFlowNode>([]);
  const [edges, setEdges] = useEdgesState<PipelineFlowEdge>([]);
  const [selectedNode, setSelectedNode] = useState<PipelineNodeData | null>(null);
  const { fitView } = useReactFlow();

  useEffect(() => {
    if (!stage?.nodes?.length) return;

    let cancelled = false;
    runElkLayout(stage.nodes, stage.edges).then(({ nodes, edges }) => {
      if (cancelled) return;
      setNodes(nodes);
      setEdges(edges);
      requestAnimationFrame(() => {
        if (cancelled) return;
        fitView({ padding: 0.15 });
      });
    });

    return () => {
      cancelled = true;
    };
  }, [stage, setNodes, setEdges, fitView]);

  const onNodeClick = useCallback((_: MouseEvent, node: PipelineFlowNode) => {
    setSelectedNode(node.data);
  }, []);

  const onPaneClick = useCallback(() => {
    setSelectedNode(null);
  }, []);

  if (!stage?.nodes?.length) {
    return (
      <div className="flex items-center justify-center h-full" style={{ background: "var(--pe-bg-secondary)" }}>
        <div className="text-center">
          <div className="text-sm font-medium" style={{ color: "var(--pe-text-tertiary)" }}>
            No diagram data available
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="h-full w-full relative" style={{ background: "var(--pe-bg-secondary)" }}>
      {/* Stage info */}
      <div
        className="absolute top-4 left-4 z-10 px-4 py-3"
        style={{
          background: "rgba(255,255,255,0.92)",
          backdropFilter: "blur(8px)",
          borderRadius: "var(--pe-radius-lg)",
          border: "1px solid var(--pe-gray-200)",
          boxShadow: "0 1px 3px rgba(0,0,0,0.04)",
          fontFamily: "var(--pe-font-primary)",
        }}
      >
        <h2 className="font-semibold" style={{ fontSize: "13px", color: "var(--pe-text-primary)" }}>
          {stage.title}
        </h2>
        <p style={{ fontSize: "11px", color: "var(--pe-text-secondary)", marginTop: "2px" }}>
          {stage.description}
        </p>
        <div className="flex items-center gap-2 mt-1">
          <span style={{ fontSize: "10px", color: "var(--pe-text-tertiary)" }}>
            {stage.nodes.length} nodes
          </span>
          <span style={{ fontSize: "10px", color: "var(--pe-gray-200)" }}>·</span>
          <span style={{ fontSize: "10px", color: "var(--pe-text-tertiary)" }}>
            {stage.edges.length} edges
          </span>
        </div>
      </div>

      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodeClick={onNodeClick}
        onPaneClick={onPaneClick}
        nodeTypes={nodeTypes}
        edgeTypes={edgeTypes}
        nodesDraggable={false}
        nodesConnectable={false}
        elementsSelectable={false}
        fitView
        fitViewOptions={{ padding: 0.15 }}
        minZoom={0.1}
        maxZoom={3}
        proOptions={{ hideAttribution: true }}
      >
        <Background variant={BackgroundVariant.Dots} gap={20} size={1} color="#E2E8F0" />
        <Controls position="bottom-right" showInteractive={false} />
        <MiniMap
          position="bottom-left"
          nodeColor={(n) => NODE_COLORS[n.data?.nodeType as string]?.border || "#9CA3AF"}
          maskColor="rgba(0,0,0,0.04)"
          style={{ width: 140, height: 90 }}
        />
      </ReactFlow>

      <NodeDetailPanel data={selectedNode} onClose={() => setSelectedNode(null)} />
    </div>
  );
}

export default function PipelineDiagram({ stage }: { stage: StageData }) {
  return (
    <ReactFlowProvider>
      <DiagramInner stage={stage} />
    </ReactFlowProvider>
  );
}
