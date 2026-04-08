"use client";

import { Handle, Position } from "@xyflow/react";
import { NODE_COLORS } from "../colors";

const handleClass = "!w-2 !h-2 !border-none";

interface PipelineNodeData {
  label: string;
  nodeType: string;
  description?: string;
  width?: number;
  height?: number;
}

/**
 * 8 handles — source + target on each of 4 sides.
 * Handle ID convention:
 *   "sr" = source-right,  "tl" = target-left   (horizontal flow →)
 *   "sb" = source-bottom, "tt" = target-top     (vertical flow ↓)
 *   "st" = source-top,    "tb" = target-bottom  (upward flow ↑)
 *   "sl" = source-left,   "tr" = target-right   (leftward flow ←)
 */
function AllHandles({ color }: { color: string }) {
  const s = { background: color };
  return (
    <>
      <Handle type="target" position={Position.Top} id="tt" className={handleClass} style={s} />
      <Handle type="source" position={Position.Top} id="st" className={handleClass} style={s} />
      <Handle type="target" position={Position.Right} id="tr" className={handleClass} style={s} />
      <Handle type="source" position={Position.Right} id="sr" className={handleClass} style={s} />
      <Handle type="target" position={Position.Bottom} id="tb" className={handleClass} style={s} />
      <Handle type="source" position={Position.Bottom} id="sb" className={handleClass} style={s} />
      <Handle type="target" position={Position.Left} id="tl" className={handleClass} style={s} />
      <Handle type="source" position={Position.Left} id="sl" className={handleClass} style={s} />
    </>
  );
}

export default function PipelineNode({ data }: { data: PipelineNodeData }) {
  const colors = NODE_COLORS[data.nodeType] || NODE_COLORS.process;
  const isDashed = data.nodeType === "missing" || data.nodeType === "absent";

  return (
    <div
      className="rounded-lg px-3 py-2 cursor-pointer shadow-sm hover:shadow-md transition-shadow duration-150 overflow-hidden"
      style={{
        width: data.width || 240,
        height: data.height || 64,
        backgroundColor: colors.fill,
        border: `2px ${isDashed ? "dashed" : "solid"} ${colors.border}`,
        fontFamily: "var(--pe-font-primary)",
      }}
    >
      <div
        className="font-semibold leading-tight truncate"
        style={{ fontSize: "12px", color: "var(--pe-text-primary)" }}
      >
        {data.label}
      </div>
      {data.description && (
        <div
          className="mt-0.5 leading-snug overflow-hidden"
          style={{
            fontSize: "10px",
            color: "var(--pe-text-secondary)",
            display: "-webkit-box",
            WebkitLineClamp: 2,
            WebkitBoxOrient: "vertical",
          }}
        >
          {data.description}
        </div>
      )}
      <AllHandles color={colors.border} />
    </div>
  );
}
