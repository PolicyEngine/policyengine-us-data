# Documentation Review

Use this skill when reviewing a pull request that changes pipeline behavior,
stage boundaries, generated artifacts, or public library functions.

## Review Goal

Documentation review is a harness check, not copyediting. Confirm that durable
documentation still describes the code paths an agent or maintainer would use to
understand, validate, or modify the pipeline.

Do not put PR-specific confidence, impact, or reviewer notes into Pydoc
docstrings or pipeline decorators. Keep durable facts in the source docs and keep
review judgment in the PR description or review comment.

## Trigger Files

Run documentation review when a PR changes any of these paths:

- `modal_app/`
- `policyengine_us_data/calibration/`
- `policyengine_us_data/datasets/`
- `policyengine_us_data/pipeline_metadata.py`
- `policyengine_us_data/pipeline_schema.py`
- `docs/pipeline_map.yaml`
- `docs/generated/`
- `docs/engineering/pipeline-map.md`
- `scripts/extract_pipeline_docs.py`
- `scripts/guards/pipeline_docs.py`
- `scripts/guards/pydoc_completeness.py`

Also run it when a PR changes public import surfaces, artifact names, validation
commands, staging/promotion behavior, or step-manifest semantics even if the
changed path is not listed above.

## Pipeline Documentation Checks

Check that changed pipeline behavior has a durable documentation surface:

- New or changed semantic waypoints have a `@pipeline_node` decorator or an
  explicit node in `docs/pipeline_map.yaml`.
- Stage and substage placement matches the five canonical stages and the
  `1a_`/`1b_` substage naming scheme.
- Edges describe real data, artifact, validation, or orchestration relationships.
- `status` and `stability` values are honest for transitional code.
- `validation_commands` are focused and point to existing tests or scripts.
- Generated docs build when decorator, Pydoc, or map source changes. PRs that
  change decorator metadata, Pydoc-facing source, or `docs/pipeline_map.yaml`
  should refresh the checked-in generated artifacts in the same change.
- Stale architecture names, folder names, and artifact names are not preserved in
  durable documentation sources or generated output.

## Pydoc Checks

For changed public library functions and classes:

- Stable library functions/classes have useful docstrings.
- Function signatures include meaningful parameter and return types when
  practical.
- Modules that declare `__all__` include promoted public functions/classes.
- Docstrings explain durable behavior, inputs, outputs, and failure modes. They
  should not include PR-specific confidence, impact, or implementation debate.

## Commands

Run the documentation and quality harness before handing off a documentation
review:

```bash
uv run --no-sync --with pyyaml python scripts/run_quality_guards.py
uv run --no-sync --with pyyaml --with pytest pytest tests/unit/test_pipeline_docs_extractor.py tests/unit/test_pipeline_doc_guards.py
```

If reviewing generated pipeline-doc build behavior, run the extractor against a
temporary output directory:

```bash
out_dir="$(mktemp -d)"
uv run --no-sync --with pyyaml python scripts/extract_pipeline_docs.py \
  --json "$out_dir/pipeline_map.json" \
  --api-json "$out_dir/pipeline_api.json" \
  --markdown "$out_dir/pipeline-map.md"
```

If reviewing documentation build behavior, also run:

```bash
uv run --no-sync --with "mystmd>=1.7.0" make documentation
```

## Review Output

Use a concise PR-facing note. Include:

- Touched stages or substages, or `none`.
- Documentation changes observed, or `not required` with a reason.
- Commands run, or commands not run with a reason.
- Impact: `low`, `medium`, or `high`.
- Confidence: `low`, `medium`, or `high`.
- Known gaps that should be handled in this PR or a follow-up.

Prefer file/line references for requested documentation changes. Do not approve
documentation coverage only because the structural guards pass; those guards
catch shape errors, while review should catch missing or misleading content.
