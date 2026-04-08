// Node type → visual properties (matching pipeline_schema.py NODE_COLORS)
export const NODE_COLORS: Record<string, { fill: string; border: string }> = {
  input: { fill: "#dbeafe", border: "#3b82f6" },
  output: { fill: "#dcfce7", border: "#22c55e" },
  process: { fill: "#ffedd5", border: "#f97316" },
  utility: { fill: "#f3e8ff", border: "#a855f7" },
  external: { fill: "#fef9c3", border: "#eab308" },
  us_specific: { fill: "#fce7f3", border: "#ec4899" },
  uk_specific: { fill: "#ccfbf1", border: "#14b8a6" },
  missing: { fill: "#fee2e2", border: "#ef4444" },
  absent: { fill: "#f3f4f6", border: "#d1d5db" },
};

// Edge type → visual properties (matching pipeline_schema.py EDGE_STYLES)
export const EDGE_STYLES: Record<
  string,
  { color: string; style: string; width: number }
> = {
  data_flow: { color: "#334155", style: "solid", width: 2 },
  produces_artifact: { color: "#16a34a", style: "solid", width: 2 },
  uses_utility: { color: "#7c3aed", style: "dashed", width: 1.5 },
  external_source: { color: "#b45309", style: "dotted", width: 1.5 },
  runs_on_infra: { color: "#dc2626", style: "dashed", width: 1.5 },
  informational: { color: "#9ca3af", style: "dotted", width: 1 },
};
