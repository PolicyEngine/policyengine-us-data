# Pipeline Documentation

Use the structured pipeline docs when editing pipeline orchestration, calibration
steps, local H5 publishing, or reusable library functions that participate in
those flows.

## Sources Of Truth

- `@pipeline_node` attaches object-level metadata. It is a no-op runtime
  decorator and is extracted statically.
- `docs/pipeline_map.yaml` defines the five canonical stages, the detailed
  `1a_`/`1b_` substage pathway, cross-stage artifacts, and edges.
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

Use stable snake_case `id` values. Pipeline map substages should use the
canonical-stage prefix plus a letter, for example `1a_raw_data_download` or
`4a_local_area_h5_regional`, and should declare `canonical_stage_id`,
`legacy_stage_id`, and any PR-855-style `manifest_step_ids`. Keep execution-ledger
substage boundaries explicit for regional/national fitting, regional/national
H5 builds, base-data staging, diagnostics upload, validation, HuggingFace
promotion, GCS promotion, and version-manifest finalization. If a function moves
during refactors, keep the ID unless the semantic waypoint changes. If a
waypoint is being migrated, set `status="transitional"` and use
`migration_target` or `notes` instead of renaming IDs prematurely.

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

Run quality guards before committing pipeline documentation changes:

```bash
uv run --no-sync --with pyyaml python scripts/run_quality_guards.py
```

The `pipeline-docs` guard checks stage IDs, canonical-stage wiring, edge and
node metadata, duplicate decorator IDs, and validation command paths. The
`pydoc-completeness` guard checks stable public library nodes for docstrings,
return annotations, and `__all__` membership when the source module declares an
export list.

If the local platform cannot install the full project environment, use
`uv run --no-sync --with pyyaml ...` for these docs-only commands.
