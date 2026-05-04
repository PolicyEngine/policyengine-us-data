#!/usr/bin/env bash
set -euo pipefail

workflow_file="${PIPELINE_WORKFLOW_FILE:-pipeline.yaml}"
workflow_ref="${PIPELINE_WORKFLOW_REF:-main}"

if [[ -z "${US_DATA_RUN_ID:-}" ]]; then
  echo "US_DATA_RUN_ID is required" >&2
  exit 1
fi

if [[ -z "${SOURCE_SHA:-}" ]]; then
  echo "SOURCE_SHA is required" >&2
  exit 1
fi

gh workflow run "${workflow_file}" \
  --ref "${workflow_ref}" \
  -f run_id="${US_DATA_RUN_ID}" \
  -f source_sha="${SOURCE_SHA}"

if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
  {
    echo "## Pipeline Dispatched"
    echo
    echo "| Field | Value |"
    echo "|-------|-------|"
    echo "| Run ID | \`${US_DATA_RUN_ID}\` |"
    echo "| Source SHA | \`${SOURCE_SHA}\` |"
    echo "| Workflow | \`${workflow_file}\` |"
    echo "| Workflow ref | \`${workflow_ref}\` |"
  } >> "${GITHUB_STEP_SUMMARY}"
fi
