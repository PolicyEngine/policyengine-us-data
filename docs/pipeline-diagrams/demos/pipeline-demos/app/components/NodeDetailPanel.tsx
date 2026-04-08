"use client";

import { NODE_COLORS } from "../colors";

const TYPE_LABELS: Record<string, string> = {
  input: "Input data",
  output: "Output artifact",
  process: "Processing step",
  utility: "Utility module",
  external: "External service",
  us_specific: "US-specific",
  uk_specific: "UK-specific",
  missing: "Planned",
  absent: "Not applicable",
};

interface NodeDetailProps {
  data: {
    label: string;
    nodeType: string;
    description?: string;
    details?: string;
    source_file?: string;
  } | null;
  onClose: () => void;
}

export default function NodeDetailPanel({ data, onClose }: NodeDetailProps) {
  if (!data) return null;

  const colors = NODE_COLORS[data.nodeType] || NODE_COLORS.process;
  const typeLabel = TYPE_LABELS[data.nodeType] || data.nodeType;

  return (
    <div
      className="absolute top-4 right-4 w-[304px] z-50 overflow-hidden"
      style={{
        borderRadius: "var(--pe-radius-lg)",
        background: "var(--pe-bg-primary)",
        border: "1px solid var(--pe-gray-200)",
        boxShadow: "0 4px 12px rgba(0,0,0,0.08)",
        fontFamily: "var(--pe-font-primary)",
      }}
    >
      {/* Colored top border */}
      <div className="h-[3px]" style={{ background: colors.border }} />

      <div className="p-4">
        <div className="flex justify-between items-start">
          <div className="flex-1 min-w-0 pr-3">
            <h3 className="font-semibold leading-tight" style={{ fontSize: "14px", color: "var(--pe-text-primary)" }}>
              {data.label}
            </h3>
            <span
              className="inline-block mt-1.5 px-2 py-0.5 text-[10px] font-medium uppercase tracking-wider"
              style={{
                borderRadius: "var(--pe-radius-sm)",
                backgroundColor: colors.fill,
                color: colors.border,
                border: `1px solid ${colors.border}30`,
              }}
            >
              {typeLabel}
            </span>
          </div>
          <button
            onClick={onClose}
            className="w-6 h-6 flex items-center justify-center rounded-md hover:bg-gray-100 transition-colors"
            style={{ color: "var(--pe-text-tertiary)" }}
          >
            <svg width="10" height="10" viewBox="0 0 10 10" fill="none">
              <path d="M1 1L9 9M9 1L1 9" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
            </svg>
          </button>
        </div>

        {data.description && (
          <p className="mt-3 leading-relaxed" style={{ fontSize: "12px", color: "var(--pe-text-secondary)" }}>
            {data.description}
          </p>
        )}

        {data.details && (
          <div
            className="mt-3 px-3 py-2 leading-relaxed"
            style={{
              fontSize: "11px",
              color: "var(--pe-gray-500)",
              background: "var(--pe-bg-tertiary)",
              borderRadius: "var(--pe-radius-md)",
              border: "1px solid var(--pe-gray-100)",
            }}
          >
            {data.details}
          </div>
        )}

        {data.source_file && (
          <div className="mt-3 flex items-center gap-1.5">
            <svg width="10" height="10" viewBox="0 0 10 10" fill="none" className="flex-shrink-0" style={{ color: "var(--pe-text-tertiary)" }}>
              <rect x="1.5" y="1" width="7" height="8" rx="1" stroke="currentColor" strokeWidth="1" />
              <line x1="3.5" y1="3.5" x2="6.5" y2="3.5" stroke="currentColor" strokeWidth="0.7" />
              <line x1="3.5" y1="5.5" x2="5.5" y2="5.5" stroke="currentColor" strokeWidth="0.7" />
            </svg>
            <code style={{ fontSize: "10px", color: "var(--pe-text-tertiary)", fontFamily: "var(--pe-font-mono)" }}>
              {data.source_file}
            </code>
          </div>
        )}
      </div>
    </div>
  );
}
