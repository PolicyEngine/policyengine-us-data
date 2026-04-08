"use client";

import { EdgeLabelRenderer, getSmoothStepPath, type EdgeProps } from "@xyflow/react";

/**
 * Custom edge that renders using ELK's computed bend points when available,
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

  if (data?.elkRoute) {
    const { startPoint, endPoint, bendPoints = [] } = data.elkRoute as {
      startPoint: { x: number; y: number };
      endPoint: { x: number; y: number };
      bendPoints?: { x: number; y: number }[];
    };
    const points = [startPoint, ...bendPoints, endPoint];

    edgePath = points
      .map((p, i) => `${i === 0 ? "M" : "L"} ${p.x} ${p.y}`)
      .join(" ");

    // Label at midpoint of the path
    const mid = Math.floor(points.length / 2);
    labelX = points[mid].x;
    labelY = points[mid].y;
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
              transform: `translate(-50%, -50%) translate(${labelX}px, ${labelY}px)`,
              pointerEvents: "all",
            }}
            className="nodrag nopan"
          >
            <span
              style={{
                fontSize: 10,
                fontWeight: 500,
                background: "rgba(255,255,255,0.85)",
                padding: "1px 5px",
                borderRadius: 3,
                color: (style as React.CSSProperties)?.stroke?.toString() || "#334155",
                whiteSpace: "nowrap",
                fontFamily: "var(--pe-font-primary)",
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
