---
name: pipeline-doc-authoring
description: Add or update this repo's automated pipeline documentation for the US data build. Use when Codex needs to document a new pipeline step, refresh stale diagram metadata after code changes, add `@pipeline_node` decorators, update `pipeline_stages.yaml` edges or groups, regenerate `docs/pipeline-diagrams/app/pipeline.json`, or verify that the ReactFlow/ELK docs stay aligned with the underlying pipeline code.
---

# Pipeline Doc Authoring

Use this skill to extend or repair the repository's automated pipeline documentation. The
documentation system is metadata-driven: code decorators describe process nodes, YAML describes
stage structure and edges, the extractor merges them into generated JSON, and the docs app renders
that JSON.

## Workflow

1. Map the real pipeline flow before editing docs. Read
   [references/source-of-truth.md](references/source-of-truth.md) first. Then inspect the actual
   driver method or orchestration path that runs the behavior you want to document. Derive order
   from the code, not from old YAML or old rendered JSON.

1. Choose the right documentation shape.

   - Use a normal decorated node for a pipeline-visible step that meaningfully transforms data or
     produces an artifact.
   - Use a YAML `extra_node` for fixed inputs, outputs, utilities, or external systems.
   - Use a YAML `group` for orchestration wrappers whose important content is already expanded into
     substeps.
   - Do not create nodes for trivial helpers that are only implementation detail.

1. Update code metadata. Add or refresh `@pipeline_node(PipelineNode(...))` on the implementing
   function in its real source file. Keep the `id` stable and unique. Write `description` and
   `details` from the current behavior, not from historical intent.

1. Update stage structure in `pipeline_stages.yaml`. Add or update stage descriptions,
   `extra_nodes`, `groups`, and `edges`. Edges control both graph order and node membership in a
   stage. A decorated node will not render unless at least one stage edge references its `id`.

1. Regenerate and validate. Run:

   - `python scripts/extract_pipeline.py`
   - `python -m py_compile scripts/extract_pipeline.py <changed-python-files>`
   - `cd docs/pipeline-diagrams && npx tsc --noEmit`
   - `git diff --check`

   Run `cd docs/pipeline-diagrams && npm run lint` when you touch the renderer. For pure metadata
   changes, TypeScript plus extractor validation is usually enough.

1. Review the generated diff semantically. Confirm that the new node or group appears in the
   intended stage, with the intended neighbors, and that the stage description still matches the
   real `generate()` or orchestration order.

## Strong Patterns

- Read [references/examples-and-pitfalls.md](references/examples-and-pitfalls.md) before adding a
  new node type or stage pattern.
- Reuse a single decorated node ID across multiple stages only when the underlying function is
  genuinely the same conceptual step in both places.
- Prefer wrapper groups over duplicate wrapper nodes when a function is just an orchestrator around
  already-documented substeps.

## Pitfalls

- Do not edit `docs/pipeline-diagrams/app/pipeline.json` by hand. It is generated.
- Do not add a decorator without wiring the node into `pipeline_stages.yaml`; the extractor treats
  unexpected unused decorators as errors.
- Do not fix stale docs by changing YAML alone when the code metadata is also wrong; update both.
- Do not add fake edges just to make a wrapper function visible. Use `groups` when the wrapper is a
  visual boundary, not a data-flow step.
- Do not assume a clean textual rebase means the docs are aligned. Re-read the rebased code path
  after merges to `main`.
