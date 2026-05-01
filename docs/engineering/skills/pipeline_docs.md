# Pipeline Documentation

Use the structured pipeline docs when editing pipeline orchestration, calibration
steps, local H5 publishing, or reusable library functions that participate in
those flows.

## Sources Of Truth

- `@pipeline_node` attaches object-level metadata. It is a no-op runtime
  decorator and is extracted statically.
- `docs/pipeline_map.yaml` defines the stage-level pathway, cross-stage
  artifacts, and edges.
- `scripts/extract_pipeline_docs.py` merges both sources and writes:
  - `docs/generated/pipeline_map.json`
  - `docs/generated/pipeline_api.json`
  - `docs/engineering/pipeline-map.md`

## Annotation Rules

Annotate semantic waypoints, not every private helper. A waypoint is worth a
decorator when it is a pipeline entrypoint, a bundled transitional process, a
library function whose behavior affects artifacts, a validation seam, or a
stable utility that downstream docs should expose.

Keep decorator metadata compact. Put durable API details in the function or
class docstring and type signature so the pydoc-style API artifact can consume
them. Use decorator fields for graph identity, artifacts, pathways, status,
stability, and focused validation commands.

For modules intended as standard pydoc/autodoc targets, declare `__all__` with
the supported public classes, functions, and type aliases. Keep private helpers
undocumented unless they are deliberately promoted into that public surface.

Use stable snake_case `id` values. If a function moves during refactors, keep the
ID unless the semantic waypoint changes. If a waypoint is being migrated, set
`status="transitional"` and use `migration_target` or `notes` instead of
renaming IDs prematurely.

## Update Workflow

After adding or changing annotations or `docs/pipeline_map.yaml`, regenerate the
artifacts:

```bash
uv run python scripts/extract_pipeline_docs.py
```

Then run the focused extractor tests:

```bash
uv run pytest tests/unit/test_pipeline_docs_extractor.py
```

If the local platform cannot install the full project environment, use
`uv run --no-sync --with pyyaml ...` for these docs-only commands.
