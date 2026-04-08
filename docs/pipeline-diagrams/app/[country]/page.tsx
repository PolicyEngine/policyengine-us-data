"use client";

import PipelineDiagram from "../components/PipelineDiagram";
import pipelineData from "../pipeline.json";

// Overview: show all stages as a high-level flow
const overviewStage = {
  id: -1,
  label: "Overview",
  title: "Pipeline Overview — Cross-Stage Data Flow",
  description: "High-level view of all pipeline stages",
  nodes: pipelineData.stages.map((s) => ({
    id: `stage_${s.id}`,
    label: `${s.label}: ${s.description}`,
    node_type: "process",
    description: s.description,
  })),
  edges: pipelineData.stages.slice(0, -1).map((s, i) => ({
    source: `stage_${s.id}`,
    target: `stage_${pipelineData.stages[i + 1].id}`,
    edge_type: "data_flow",
  })),
};

export default function CountryOverview() {
  return <PipelineDiagram stage={overviewStage} />;
}
