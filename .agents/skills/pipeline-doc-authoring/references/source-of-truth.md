# Source of Truth

Use these files in this order when updating pipeline docs.

## Core contract

- `policyengine_us_data/pipeline_metadata.py`
  - Defines the `@pipeline_node` decorator used in source files.
- `policyengine_us_data/pipeline_schema.py`
  - Defines the JSON-facing schema for stages, nodes, edges, and groups.

## Extraction and validation

- `scripts/extract_pipeline.py`
  - Scans decorated source files.
  - Merges code nodes with `pipeline_stages.yaml`.
  - Fails on unexpected unused decorators.
  - Contains the allowlist for intentionally omitted wrapper nodes.

Important rule: stage edges determine which decorated nodes are included in a stage. If a node ID is
never referenced by an edge, it will not appear in the generated graph.

## Stage structure

- `pipeline_stages.yaml`
  - Stage descriptions
  - Manual inputs/outputs/utilities
  - Data-flow and utility edges
  - Visual wrapper groups

## Generated artifact

- `docs/pipeline-diagrams/app/pipeline.json`
  - Generated output consumed by the docs app
  - Never edit by hand

## Docs app

- `docs/pipeline-diagrams/app/components/PipelineDiagram.tsx`
  - Renders flat nodes plus wrapper groups
  - Most metadata-only changes should not require edits here
- `docs/pipeline-diagrams/README.md`
  - Operator workflow for regeneration and checks

## Validation commands

- `python scripts/extract_pipeline.py`
- `python -m py_compile scripts/extract_pipeline.py <changed-python-files>`
- `cd docs/pipeline-diagrams && npx tsc --noEmit`
- `git diff --check`
