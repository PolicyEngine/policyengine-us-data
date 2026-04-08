# Pipeline Diagrams

Interactive ReactFlow documentation for the PolicyEngine US data pipeline.

## Data Flow

The rendered graph is generated from two sources:

- `@pipeline_node(PipelineNode(...))` decorators in `policyengine_us_data/**/*.py`
- stage groupings, extra nodes, and edges in `pipeline_stages.yaml`

Regenerate the app-consumed JSON from the repository root:

```bash
python scripts/extract_pipeline.py
```

The extractor writes `docs/pipeline-diagrams/app/pipeline.json`. Do not edit that
file by hand.

## Local App

Use Node.js 20.9 or newer. CI uses Node.js 24.

```bash
cd docs/pipeline-diagrams
npm ci
npm run dev
```

Open http://localhost:3000/us.

## Checks

Run the generator and TypeScript check before committing diagram metadata:

```bash
python scripts/extract_pipeline.py
cd docs/pipeline-diagrams
npx tsc --noEmit
```

`npm run lint` currently includes renderer cleanup work outside the generation
path, so the automated update workflow uses the TypeScript check only.
