"use client";

import { use } from "react";
import PipelineDiagram from "../../../components/PipelineDiagram";
import pipelineData from "../../../pipeline.json";

export default function StagePage({
  params,
}: {
  params: Promise<{ country: string; stageId: string }>;
}) {
  const { stageId } = use(params);
  const stage = pipelineData.stages.find((s) => String(s.id) === stageId);

  if (!stage) {
    return (
      <div className="flex items-center justify-center h-full text-gray-500">
        Stage {stageId} not found.
      </div>
    );
  }

  return <PipelineDiagram stage={stage} />;
}
