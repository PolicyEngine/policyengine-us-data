"use client";

import { EdgeLabelRenderer, getSmoothStepPath, type EdgeProps } from "@xyflow/react";

type ElkLabelPosition = {
  x: number;
  y: number;
  width?: number;
  height?: number;
};

/**
 * Custom edge that renders using ELK's computed edge section when available,
 * falling back to smoothstep for edges without routes.
 */
export default function ElkEdge({
  id,
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  style = {},
  markerEnd,
  label,
  data,
}: EdgeProps) {
  let edgePath: string;
  let labelX: number;
  let labelY: number;
  let elkLabel: ElkLabelPosition | null = null;

  if (data?.elkRoute) {
    const { startPoint, endPoint, bendPoints = [] } = data.elkRoute as {
      startPoint?: { x: number; y: number };
      endPoint?: { x: number; y: number };
      bendPoints?: { x: number; y: number }[];
    };
    const points = startPoint && endPoint ? [startPoint, ...bendPoints, endPoint] : [];

    if (points.length > 0) {
      edgePath = points.map((p, i) => `${i === 0 ? "M" : "L"} ${p.x} ${p.y}`).join(" ");

      if (points.length === 2) {
        labelX = (points[0].x + points[1].x) / 2;
        labelY = (points[0].y + points[1].y) / 2;
      } else {
        const mid = Math.floor(points.length / 2);
        labelX = points[mid].x;
        labelY = points[mid].y;
      }
    } else {
      const [path, lx, ly] = getSmoothStepPath({
        sourceX,
        sourceY,
        targetX,
        targetY,
        sourcePosition,
        targetPosition,
      });
      edgePath = path;
      labelX = lx;
      labelY = ly;
    }
    elkLabel = (data.elkLabel as ElkLabelPosition | null) || null;
  } else {
    // Fallback to smoothstep
    const [path, lx, ly] = getSmoothStepPath({
      sourceX,
      sourceY,
      targetX,
      targetY,
      sourcePosition,
      targetPosition,
    });
    edgePath = path;
    labelX = lx;
    labelY = ly;
  }

  return (
    <>
      <path
        id={id}
        className="react-flow__edge-path"
        d={edgePath}
        style={style}
        markerEnd={markerEnd}
        fill="none"
      />
      {label && (
        <EdgeLabelRenderer>
          <div
            style={{
              position: "absolute",
              transform: elkLabel
                ? `translate(${elkLabel.x}px, ${elkLabel.y}px)`
                : `translate(-50%, -50%) translate(${labelX}px, ${labelY}px)`,
              pointerEvents: "all",
              zIndex: 10,
            }}
            className="nodrag nopan"
          >
            <span
              style={{
                display: "inline-block",
                fontSize: 10,
                fontWeight: 500,
                background: "rgba(255,255,255,0.85)",
                padding: "1px 5px",
                borderRadius: 3,
                color: (style as React.CSSProperties)?.stroke?.toString() || "#334155",
                whiteSpace: "nowrap",
                fontFamily: "var(--pe-font-primary)",
                minWidth: elkLabel?.width,
                minHeight: elkLabel?.height,
              }}
            >
              {label}
            </span>
          </div>
        </EdgeLabelRenderer>
      )}
    </>
  );
}
