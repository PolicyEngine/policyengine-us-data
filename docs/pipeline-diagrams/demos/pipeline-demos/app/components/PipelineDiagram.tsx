"use client";

import { useCallback, useEffect, useRef, useState } from "react";
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
  id: number;
  label: string;
  title: string;
  description: string;
  nodes: any[];
  edges: any[];
}

/**
 * Estimate node size to match the CSS:
 * min-w-[180px] max-w-[280px], px-3 py-2, 12px label + 10px description
 */
function estimateNodeSize(node: any) {
  let height = 16; // py-2 top
  height += 18;    // label line
  if (node.description) height += 16; // description line
  height += 14;    // py-2 bottom
  return { width: 240, height: Math.max(58, height) };
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
  pipelineNodes: any[],
  pipelineEdges: any[]
): Promise<{ nodes: Node[]; edges: Edge[] }> {
  const graph = {
    id: "root",
    layoutOptions: ELK_OPTIONS,
    children: pipelineNodes.map((n: any) => {
      const size = estimateNodeSize(n);
      return { id: n.id, width: size.width, height: size.height };
    }),
    edges: pipelineEdges.map((e: any, i: number) => ({
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
  const routeMap: Record<string, any> = {};
  for (const edge of (result.edges || []) as any[]) {
    if (edge.sections && edge.sections.length > 0) {
      routeMap[edge.id] = edge.sections[0];
    }
  }

  // Position nodes
  const nodes: Node[] = pipelineNodes.map((n: any) => ({
    id: n.id,
    type: "pipeline",
    position: positionMap[n.id] || { x: 0, y: 0 },
    data: {
      label: n.label || n.id,
      nodeType: n.node_type || "process",
      description: n.description || "",
      details: n.details || "",
      source_file: n.source_file || "",
    },
  }));

  // Enrich edges with handles + ELK routes
  const edges: Edge[] = pipelineEdges.map((e: any, i: number) => {
    const edgeId = `e-${i}`;
    const srcPos = positionMap[e.source] || { x: 0, y: 0 };
    const tgtPos = positionMap[e.target] || { x: 0, y: 0 };
    const handles = assignHandles(srcPos, tgtPos);
    const route = routeMap[edgeId];
    const edgeStyle = EDGE_STYLES[e.edge_type] || EDGE_STYLES.data_flow;

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
  const [nodes, setNodes, onNodesChange] = useNodesState<Node>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([]);
  const [selectedNode, setSelectedNode] = useState<any>(null);
  const [layoutDone, setLayoutDone] = useState(false);
  const { fitView } = useReactFlow();
  const hasFit = useRef(false);

  useEffect(() => {
    if (!stage?.nodes?.length) return;
    hasFit.current = false;
    setLayoutDone(false);
    runElkLayout(stage.nodes, stage.edges).then(({ nodes, edges }) => {
      setNodes(nodes);
      setEdges(edges);
      setLayoutDone(true);
    });
  }, [stage]);

  useEffect(() => {
    if (layoutDone && !hasFit.current) {
      hasFit.current = true;
      requestAnimationFrame(() => {
        fitView({ padding: 0.15 });
      });
    }
  }, [layoutDone, fitView]);

  const onNodeClick = useCallback((_: any, node: Node) => {
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
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onNodeClick={onNodeClick}
        onPaneClick={onPaneClick}
        nodeTypes={nodeTypes}
        edgeTypes={edgeTypes}
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
