"use client";

import { useCallback, useEffect, useState, type MouseEvent } from "react";
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  ReactFlowProvider,
  ViewportPortal,
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
const ELK_PORT_SIZE = 0;
const GROUP_PADDING_X = 18;
const GROUP_PADDING_Y = 18;
const GROUP_PADDING_TOP = 24;
const EDGE_LABEL_HEIGHT = 16;
const EDGE_LABEL_MIN_WIDTH = 40;
const EDGE_LABEL_MAX_WIDTH = 180;
const EDGE_LABEL_CHAR_WIDTH = 6;

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
  groups?: PipelineJsonGroup[];
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

interface PipelineJsonGroup {
  id: string;
  label: string;
  description?: string;
  node_ids: string[];
}

interface Point {
  x: number;
  y: number;
}

type HandleId = "tt" | "st" | "tr" | "sr" | "tb" | "sb" | "tl" | "sl";
type PortSide = "NORTH" | "EAST" | "SOUTH" | "WEST";

interface ElkRoute {
  startPoint?: Point;
  endPoint?: Point;
  bendPoints?: Point[];
}

interface ElkLabel extends Point {
  width?: number;
  height?: number;
  text?: string;
}

interface ElkLayoutEdge {
  id?: string;
  sections?: ElkRoute[];
  labels?: ElkLabel[];
}

interface NodeSize {
  width: number;
  height: number;
}

interface PositionedNode extends NodeSize {
  x: number;
  y: number;
}

interface EdgeHandleAssignment {
  sourceHandle: HandleId;
  targetHandle: HandleId;
}

interface ElkPortSpec {
  id: string;
  side: PortSide;
}

interface EdgePortAssignment extends EdgeHandleAssignment {
  sourcePortId: string;
  targetPortId: string;
  sourceSide: PortSide;
  targetSide: PortSide;
}

interface GroupBox {
  id: string;
  label: string;
  description: string;
  x: number;
  y: number;
  width: number;
  height: number;
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
  elkLabel: ElkLabel | null;
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

function estimateEdgeLabelSize(label: string) {
  return {
    width: Math.min(
      EDGE_LABEL_MAX_WIDTH,
      Math.max(EDGE_LABEL_MIN_WIDTH, label.length * EDGE_LABEL_CHAR_WIDTH)
    ),
    height: EDGE_LABEL_HEIGHT,
  };
}

function elkPortId(nodeId: string, edgeId: string, role: "source" | "target") {
  return `${nodeId}::${edgeId}::${role}`;
}

function sideForHandle(handleId: HandleId): PortSide {
  switch (handleId) {
    case "tt":
    case "st":
      return "NORTH";
    case "tr":
    case "sr":
      return "EAST";
    case "tb":
    case "sb":
      return "SOUTH";
    case "tl":
    case "sl":
      return "WEST";
  }
}

function elkPortsForNode(ports: ElkPortSpec[]) {
  return ports.map((port) => ({
    id: port.id,
    width: ELK_PORT_SIZE,
    height: ELK_PORT_SIZE,
    layoutOptions: {
      "org.eclipse.elk.port.side": port.side,
    },
  }));
}

/**
 * Assign sourceHandle/targetHandle based on relative node positions.
 */
function assignHandles(
  sourceNode: PositionedNode,
  targetNode: PositionedNode
): EdgeHandleAssignment {
  const sourceCenter = {
    x: sourceNode.x + sourceNode.width / 2,
    y: sourceNode.y + sourceNode.height / 2,
  };
  const targetCenter = {
    x: targetNode.x + targetNode.width / 2,
    y: targetNode.y + targetNode.height / 2,
  };
  const dx = targetCenter.x - sourceCenter.x;
  const dy = targetCenter.y - sourceCenter.y;

  if (Math.abs(dx) >= Math.abs(dy)) {
    return dx >= 0
      ? { sourceHandle: "sr", targetHandle: "tl" }
      : { sourceHandle: "sl", targetHandle: "tr" };
  }
  return dy >= 0
    ? { sourceHandle: "sb", targetHandle: "tt" }
    : { sourceHandle: "st", targetHandle: "tb" };
}

function assignEdgePorts(
  edgeId: string,
  edge: PipelineJsonEdge,
  sourceNode: PositionedNode,
  targetNode: PositionedNode
): EdgePortAssignment {
  const handles = assignHandles(sourceNode, targetNode);

  return {
    ...handles,
    sourcePortId: elkPortId(edge.source, edgeId, "source"),
    targetPortId: elkPortId(edge.target, edgeId, "target"),
    sourceSide: sideForHandle(handles.sourceHandle),
    targetSide: sideForHandle(handles.targetHandle),
  };
}

function getPositionMap(
  children: Array<{ id: string; x?: number; y?: number }> | undefined,
  nodeSizeMap: Record<string, NodeSize>
) {
  const positionMap: Record<string, PositionedNode> = {};

  for (const child of children || []) {
    const size = nodeSizeMap[child.id] || { width: NODE_WIDTH, height: NODE_HEIGHT };
    positionMap[child.id] = {
      x: child.x || 0,
      y: child.y || 0,
      width: size.width,
      height: size.height,
    };
  }

  return positionMap;
}

function buildNodePortMap(
  pipelineEdges: PipelineJsonEdge[],
  edgePorts: Record<string, EdgePortAssignment>
) {
  const nodePortMap: Record<string, ElkPortSpec[]> = {};

  pipelineEdges.forEach((edge, i: number) => {
    const edgeId = `e-${i}`;
    const assignment = edgePorts[edgeId];
    if (!assignment) return;

    nodePortMap[edge.source] = [
      ...(nodePortMap[edge.source] || []),
      { id: assignment.sourcePortId, side: assignment.sourceSide },
    ];
    nodePortMap[edge.target] = [
      ...(nodePortMap[edge.target] || []),
      { id: assignment.targetPortId, side: assignment.targetSide },
    ];
  });

  return nodePortMap;
}

function buildElkGraph(
  pipelineNodes: PipelineJsonNode[],
  pipelineEdges: PipelineJsonEdge[],
  nodeSizeMap: Record<string, NodeSize>,
  edgePorts?: Record<string, EdgePortAssignment>
) {
  const usePorts = Boolean(edgePorts);
  const nodePortMap = edgePorts ? buildNodePortMap(pipelineEdges, edgePorts) : {};

  return {
    id: "root",
    layoutOptions: ELK_OPTIONS,
    children: pipelineNodes.map((n) => {
      const size = nodeSizeMap[n.id];

      return {
        id: n.id,
        width: size.width,
        height: size.height,
        ...(usePorts
          ? {
              layoutOptions: {
                "org.eclipse.elk.portConstraints": "FIXED_SIDE",
              },
              ports: elkPortsForNode(nodePortMap[n.id] || []),
            }
          : {}),
      };
    }),
    edges: pipelineEdges.map((e, i: number) => {
      const edgeId = `e-${i}`;
      const ports = edgePorts?.[edgeId];

      return {
        id: edgeId,
        sources: [ports ? ports.sourcePortId : e.source],
        targets: [ports ? ports.targetPortId : e.target],
        ...(e.label
          ? {
              labels: [
                {
                  text: e.label,
                  ...estimateEdgeLabelSize(e.label),
                },
              ],
            }
          : {}),
      };
    }),
  };
}

function buildGroupBoxes(
  groups: PipelineJsonGroup[] | undefined,
  positionMap: Record<string, PositionedNode>
): GroupBox[] {
  return (groups || [])
    .map((group) => {
      const groupNodes = group.node_ids
        .map((nodeId) => positionMap[nodeId])
        .filter(Boolean) as PositionedNode[];

      if (groupNodes.length === 0) {
        return null;
      }

      const minX = Math.min(...groupNodes.map((node) => node.x));
      const minY = Math.min(...groupNodes.map((node) => node.y));
      const maxX = Math.max(...groupNodes.map((node) => node.x + node.width));
      const maxY = Math.max(...groupNodes.map((node) => node.y + node.height));

      return {
        id: group.id,
        label: group.label,
        description: group.description || "",
        x: minX - GROUP_PADDING_X,
        y: minY - GROUP_PADDING_TOP,
        width: maxX - minX + GROUP_PADDING_X * 2,
        height: maxY - minY + GROUP_PADDING_TOP + GROUP_PADDING_Y,
      };
    })
    .filter((group): group is GroupBox => group !== null);
}

async function runElkLayout(
  pipelineNodes: PipelineJsonNode[],
  pipelineEdges: PipelineJsonEdge[],
  pipelineGroups: PipelineJsonGroup[] | undefined
): Promise<{ nodes: PipelineFlowNode[]; edges: PipelineFlowEdge[]; groups: GroupBox[] }> {
  const nodeSizeMap: Record<string, NodeSize> = {};
  for (const node of pipelineNodes) {
    nodeSizeMap[node.id] = estimateNodeSize(node);
  }

  const initialResult = await elk.layout(buildElkGraph(pipelineNodes, pipelineEdges, nodeSizeMap));
  const initialPositionMap = getPositionMap(initialResult.children, nodeSizeMap);
  const edgePorts: Record<string, EdgePortAssignment> = {};

  pipelineEdges.forEach((e, i: number) => {
    const edgeId = `e-${i}`;
    const sourceNode = initialPositionMap[e.source] || { x: 0, y: 0, ...nodeSizeMap[e.source] };
    const targetNode = initialPositionMap[e.target] || { x: 0, y: 0, ...nodeSizeMap[e.target] };
    edgePorts[edgeId] = assignEdgePorts(edgeId, e, sourceNode, targetNode);
  });

  const result = await elk.layout(buildElkGraph(pipelineNodes, pipelineEdges, nodeSizeMap, edgePorts));
  const positionMap = getPositionMap(result.children, nodeSizeMap);
  const groups = buildGroupBoxes(pipelineGroups, positionMap);

  // Build edge route map from ELK sections
  const routeMap: Record<string, ElkRoute> = {};
  const labelMap: Record<string, ElkLabel> = {};
  for (const edge of (result.edges || []) as ElkLayoutEdge[]) {
    if (edge.id && edge.sections && edge.sections.length > 0) {
      routeMap[edge.id] = edge.sections[0];
    }
    if (edge.id && edge.labels && edge.labels.length > 0) {
      labelMap[edge.id] = edge.labels[0];
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
    const ports = edgePorts[edgeId];
    const route = routeMap[edgeId];
    const edgeStyle = EDGE_STYLES[e.edge_type ?? "data_flow"] || EDGE_STYLES.data_flow;

    return {
      id: edgeId,
      source: e.source,
      target: e.target,
      sourceHandle: ports.sourceHandle,
      targetHandle: ports.targetHandle,
      type: "elk",
      label: e.label || undefined,
      data: {
        elkRoute: route || null,
        elkLabel: labelMap[edgeId] || null,
      },
      style: {
        stroke: edgeStyle.color,
        strokeWidth: edgeStyle.width,
        strokeDasharray:
          edgeStyle.style === "dashed" ? "6 3" : edgeStyle.style === "dotted" ? "2 2" : undefined,
      },
    };
  });

  return { nodes, edges, groups };
}

function DiagramInner({ stage }: { stage: StageData }) {
  const [nodes, setNodes] = useNodesState<PipelineFlowNode>([]);
  const [edges, setEdges] = useEdgesState<PipelineFlowEdge>([]);
  const [groups, setGroups] = useState<GroupBox[]>([]);
  const [selectedNode, setSelectedNode] = useState<PipelineNodeData | null>(null);
  const { fitView } = useReactFlow();

  useEffect(() => {
    if (!stage?.nodes?.length) return;

    let cancelled = false;
    runElkLayout(stage.nodes, stage.edges, stage.groups).then(({ nodes, edges, groups }) => {
      if (cancelled) return;
      setNodes(nodes);
      setEdges(edges);
      setGroups(groups);
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
        <ViewportPortal>
          {groups.map((group) => (
            <div
              key={group.id}
              className="pointer-events-none absolute rounded-lg"
              style={{
                left: group.x,
                top: group.y,
                width: group.width,
                height: group.height,
                border: "1.5px dashed #94A3B8",
                background: "rgba(148, 163, 184, 0.08)",
                boxShadow: "inset 0 0 0 1px rgba(255,255,255,0.5)",
              }}
            >
              <div
                className="absolute rounded-md px-2 py-1 max-w-[220px]"
                style={{
                  top: 0,
                  left: 12,
                  transform: "translateY(-50%)",
                  background: "rgba(255,255,255,0.96)",
                  border: "1px solid #CBD5E1",
                  color: "var(--pe-text-secondary)",
                  fontFamily: "var(--pe-font-primary)",
                  boxShadow: "0 1px 2px rgba(0,0,0,0.04)",
                }}
                title={group.description || group.label}
              >
                <div
                  className="font-medium truncate"
                  style={{ fontSize: "11px", color: "var(--pe-text-primary)" }}
                >
                  {group.label}
                </div>
                {group.description && (
                  <div
                    className="mt-0.5 overflow-hidden"
                    style={{
                      fontSize: "10px",
                      lineHeight: 1.2,
                      color: "var(--pe-text-secondary)",
                      display: "-webkit-box",
                      WebkitLineClamp: 2,
                      WebkitBoxOrient: "vertical",
                    }}
                  >
                    {group.description}
                  </div>
                )}
              </div>
            </div>
          ))}
        </ViewportPortal>
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
