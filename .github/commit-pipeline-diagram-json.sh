#!/usr/bin/env bash

set -euo pipefail

PIPELINE_JSON="docs/pipeline-diagrams/app/pipeline.json"

append_summary() {
  if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
    {
      echo "## Pipeline Diagram Docs"
      echo
      echo "$1"
    } >> "$GITHUB_STEP_SUMMARY"
  fi
}

if git diff --quiet -- "$PIPELINE_JSON"; then
  append_summary "No generated pipeline JSON changes detected."
  exit 0
fi

git config user.name "github-actions[bot]"
git config user.email "github-actions[bot]@users.noreply.github.com"
git add "$PIPELINE_JSON"
git commit -m "Auto-update pipeline JSON"
git push origin HEAD:main

append_summary "Updated \`$PIPELINE_JSON\` on \`main\`. Connected Vercel deployment will pick up that commit."
