"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { NODE_COLORS, EDGE_STYLES } from "../colors";
import pipelineData from "../pipeline.json";

const NODE_LEGEND = [
  { type: "input", label: "Input" },
  { type: "output", label: "Output" },
  { type: "process", label: "Process" },
  { type: "utility", label: "Utility" },
  { type: "external", label: "External" },
  { type: "us_specific", label: "US-specific" },
  { type: "missing", label: "Missing" },
  { type: "absent", label: "Absent" },
];

const EDGE_LEGEND = [
  { type: "data_flow", label: "Data flow" },
  { type: "produces_artifact", label: "Produces artifact" },
  { type: "uses_utility", label: "Uses utility" },
  { type: "external_source", label: "External source" },
  { type: "runs_on_infra", label: "Runs on infra" },
  { type: "informational", label: "Informational" },
];

const META_STAGES = [
  { label: "Shared build", stageIds: [0, 1, 2] },
  { label: "ECPS pathway (deprecated)", stageIds: ["3a"] },
  { label: "Local area pathway", stageIds: ["3b", 4, 5, 6, 7, 8] },
];

export default function Sidebar() {
  const pathname = usePathname();
  const country = "us";
  const stages = pipelineData.stages;
  const stageMap = Object.fromEntries(stages.map((s) => [String(s.id), s]));

  return (
    <aside
      className="w-[272px] flex flex-col h-full overflow-y-auto"
      style={{
        background: "#FFFFFF",
        borderRight: "1px solid var(--pe-gray-200)",
        fontFamily: "var(--pe-font-primary)",
      }}
    >
      {/* Header */}
      <div className="px-5 pt-5 pb-4" style={{ borderBottom: "1px solid var(--pe-gray-200)" }}>
        <div className="flex items-center gap-3">
          <div
            className="w-8 h-8 rounded-md flex items-center justify-center"
            style={{ background: "var(--pe-primary-500)" }}
          >
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
              <path d="M8 2L14 5.5V10.5L8 14L2 10.5V5.5L8 2Z" stroke="white" strokeWidth="1.5" fill="none" />
              <circle cx="8" cy="8" r="2" fill="white" />
            </svg>
          </div>
          <div>
            <h1 className="text-[14px] font-semibold" style={{ color: "var(--pe-text-primary)" }}>
              Pipeline explorer
            </h1>
            <p className="text-[11px]" style={{ color: "var(--pe-text-secondary)" }}>
              US data pipeline
            </p>
          </div>
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 px-3 py-4">
        <Link
          href={`/${country}`}
          className="flex items-center gap-2.5 px-3 py-2 rounded-md text-[13px] font-medium transition-colors duration-150"
          style={{
            color: pathname === `/${country}` ? "var(--pe-primary-600)" : "var(--pe-text-secondary)",
            background: pathname === `/${country}` ? "var(--pe-primary-50)" : "transparent",
          }}
        >
          Overview
        </Link>

        {META_STAGES.map((meta, metaIdx) => (
          <div key={meta.label} className={metaIdx === 0 ? "mt-4" : "mt-1"}>
            <p
              className="text-[10px] font-semibold uppercase tracking-[0.1em] px-3 mb-1.5 mt-3"
              style={{ color: "var(--pe-text-tertiary)" }}
            >
              {meta.label}
            </p>

            <div className="space-y-0.5">
              {meta.stageIds.map((stageId) => {
                const stage = stageMap[String(stageId)];
                if (!stage) return null;
                const isActive = pathname === `/${country}/stage/${stage.id}`;
                return (
                  <Link
                    key={`${meta.label}-${stage.id}`}
                    href={`/${country}/stage/${stage.id}`}
                    className="flex items-start gap-2.5 px-3 py-2 rounded-md transition-colors duration-150"
                    style={{
                      color: isActive ? "var(--pe-primary-600)" : "var(--pe-text-secondary)",
                      background: isActive ? "var(--pe-primary-50)" : "transparent",
                    }}
                  >
                    <span
                      className="inline-flex items-center justify-center w-5 h-5 rounded text-[10px] font-bold flex-shrink-0 mt-px"
                      style={{
                        background: isActive ? "var(--pe-primary-500)" : "var(--pe-gray-100)",
                        color: isActive ? "#FFFFFF" : "var(--pe-text-tertiary)",
                      }}
                    >
                      {stage.id}
                    </span>
                    <div className="min-w-0">
                      <div className="text-[12px] font-medium leading-tight truncate">
                        {stage.description}
                      </div>
                      <div className="text-[10px] mt-0.5" style={{ color: "var(--pe-text-tertiary)" }}>
                        {stage.nodes.length} nodes · {stage.edges.length} edges
                      </div>
                    </div>
                  </Link>
                );
              })}
            </div>

            {/* Divider between meta-stages */}
            {metaIdx < META_STAGES.length - 1 && (
              <div className="mx-3 mt-3" style={{ borderBottom: "1px solid var(--pe-gray-200)" }} />
            )}
          </div>
        ))}
      </nav>

      {/* Legend */}
      <div className="px-4 py-4" style={{ borderTop: "1px solid var(--pe-gray-200)" }}>
        <p className="text-[10px] font-semibold uppercase tracking-[0.1em] mb-3"
          style={{ color: "var(--pe-text-tertiary)" }}>
          Legend
        </p>

        <div className="grid grid-cols-2 gap-x-3 gap-y-1.5">
          {NODE_LEGEND.map(({ type, label }) => {
            const colors = NODE_COLORS[type];
            const isDashed = type === "missing" || type === "absent";
            return (
              <div key={type} className="flex items-center gap-1.5">
                <div
                  className="w-2.5 h-2.5 rounded-sm flex-shrink-0"
                  style={{
                    backgroundColor: colors.fill,
                    border: `1.5px ${isDashed ? "dashed" : "solid"} ${colors.border}`,
                  }}
                />
                <span className="text-[10px]" style={{ color: "var(--pe-text-secondary)" }}>
                  {label}
                </span>
              </div>
            );
          })}
        </div>

        <div className="mt-3 pt-3 space-y-1.5" style={{ borderTop: "1px solid var(--pe-gray-100)" }}>
          {EDGE_LEGEND.map(({ type, label }) => {
            const style = EDGE_STYLES[type];
            return (
              <div key={type} className="flex items-center gap-2">
                <svg width="20" height="6" className="flex-shrink-0">
                  <line x1="0" y1="3" x2="20" y2="3"
                    stroke={style.color} strokeWidth={Math.min(style.width, 1.5)}
                    strokeDasharray={style.style === "dashed" ? "3 2" : style.style === "dotted" ? "1 2" : undefined}
                  />
                </svg>
                <span className="text-[10px]" style={{ color: "var(--pe-text-secondary)" }}>
                  {label}
                </span>
              </div>
            );
          })}
        </div>
      </div>
    </aside>
  );
}
