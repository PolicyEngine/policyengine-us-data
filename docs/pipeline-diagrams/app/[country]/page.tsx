"use client";

import PipelineDiagram from "../components/PipelineDiagram";
import pipelineData from "../pipeline.json";

const OVERVIEW_EDGES: Array<[number | string, number | string]> = [
  [0, 1],
  [1, 2],
  [2, "3a"],
  [2, "3b"],
  ["3b", 4],
  [4, 5],
  [5, 6],
  [6, 7],
  [7, 8],
];

const stageIds = new Set(pipelineData.stages.map((stage) => String(stage.id)));

// Overview: show the real high-level branch structure.
const overviewStage = {
  id: -1,
  label: "Overview",
  title: "Pipeline Overview — Cross-Stage Data Flow",
  description: "High-level view of all pipeline stages",
  nodes: pipelineData.stages.map((s) => ({
    id: `stage_${s.id}`,
    label: s.title,
    node_type: "process",
    description: s.description,
  })),
  edges: OVERVIEW_EDGES.filter(
    ([source, target]) =>
      stageIds.has(String(source)) && stageIds.has(String(target))
  ).map(([source, target]) => ({
    source: `stage_${source}`,
    target: `stage_${target}`,
    edge_type: "data_flow",
  })),
};

export default function CountryOverview() {
  return <PipelineDiagram stage={overviewStage} />;
}
