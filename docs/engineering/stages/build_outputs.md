# Build Outputs Stage AI Guide

This guide is for AI agents and maintainers modifying Stage 4
(`4_build_outputs`) code. Stage 4 turns calibrated and staged pipeline artifacts
into publishable outputs, including local-area H5 files, national H5 files,
diagnostics, and release-staging artifacts.

The active local H5 seams live under `policyengine_us_data/build_outputs/`.
Treat this package as the place for reusable Stage 4 library boundaries. Keep
Modal orchestration, worker entrypoints, and release promotion behavior outside
these library seams unless a stage plan explicitly says otherwise.

## Local H5 Build Path

The transitional runtime entrypoint is still
`policyengine_us_data.calibration.publish_local_area.build_h5()`. It should stay
as a facade while Stage 4 is being migrated. New implementation logic should
move behind narrower build-output library seams instead of growing this facade.

The current in-memory local H5 path is:

1. `AreaSelector` selects active clone-household rows from clone weights and
   geography filters.
2. `EntityReindexer` creates output household, person, and subentity IDs.
3. `VariableCloner` copies allowed source variables into a period-grouped
   payload.
4. `LocalAreaDatasetBuilder` applies payload postprocessors in declared order.
5. `H5Writer` writes the final `H5Payload` and verifies summary counts.

When adding behavior to this path, decide whether it is a selection, reindexing,
source-variable cloning, postprocessing, or writing concern. Do not place
country-specific payload mutation in `build_h5()` when it can be represented as
a postprocessor.

## Worker Chunk Execution

The Modal coordinator builds canonical typed area requests before spawning
workers. Regional publish reads the target congressional district universe from
the staged target database through `TargetUniverseReader`, reads only
coordinator-needed CD and county geography fields through
`CalibrationGeographyLoader.load_index()`, then asks `USAreaCatalog` to define
the regional release shape: every configured state, every target congressional
district, and the explicitly supported city outputs such as NYC. The
coordinator wraps those requests in `WeightedAreaRequest`, partitions them with
`partition_weighted_area_requests()`, and sends workers typed
`--requests-json` payloads. Completion is measured against the explicit request
keys, not just a raw file count, so stale or unrelated H5 files cannot satisfy a
missing expected area.

`LocalH5WorkerService` is the reusable Stage 4 boundary for executing one
prepared local-H5 worker chunk. It consumes a `WorkerSession`, typed
`AreaBuildRequest` objects, and a `WorkerExecutionConfig`, then returns a
structured `WorkerResult`.

`WorkerSession` owns worker-scoped setup that is safe to reuse across the
queued requests, such as weights, geography, validation context, and bootstrap
metadata. Source dataset snapshots are loaded per request through
`WorkerSession.load_source()` because the PolicyEngine microsimulation behind a
snapshot is mutable; reusing it across multiple H5 outputs can leak calculated
holder state into later outputs.

`modal_app.worker_script` should remain a thin CLI/JSON adapter around this
service. It may parse legacy `--work-items` and typed `--requests-json`, prepare
the worker session, and print the legacy coordinator JSON shape, but it should
not regain build-loop, write-loop, or validation-loop logic.

The legacy `--work-items` input path remains compatibility-only while older
tests and explicit override callers are retired. New coordinator work should
prefer typed `AreaBuildRequest` objects and typed worker payloads.

For now, `WorkerResult.to_legacy_dict()` preserves the existing coordinator
contract with `completed`, `failed`, `errors`, `validation_rows`, and
`validation_summary`. New code should prefer the structured `results` and
`issues` fields. Validation exceptions remain visible in legacy `errors` so the
current coordinator does not drop them before it migrates to structured results.
Removing the legacy shape and moving the coordinator off worker subprocess JSON
is a later migration step.

## Payload Postprocessors

Payload postprocessors are ordered, country- or product-specific transformations
that consume an `H5Payload` and return either another `H5Payload` or a structured
result object exposing a `.payload` attribute.

Use a postprocessor when the operation:

- Mutates or adds payload variables after generic source-variable cloning.
- Depends on country-specific business rules.
- Needs focused unit tests independent of the full H5 builder.
- Should run after some other payload construction step.

Do not use a postprocessor for:

- Selecting active clones. Use `AreaSelector`.
- Reindexing entities. Use `EntityReindexer`.
- Copying source variables unchanged. Use `VariableCloner`.
- Writing H5 files. Use `H5Writer`.
- Modal orchestration, volume setup, or publication promotion.

## Postprocessor Spec Contract

Every postprocessor should expose a stable `spec`:

```python
spec = PayloadPostProcessorSpec(
    key="stable_unique_key",
    requires=("upstream_key",),
)
```

The `key` is a durable identifier for the processing step. Prefer short,
stage-specific names such as `us_entity`, `us_geography`, or `us_takeup`.
Do not use display names, class names, or generated values as the key when the
processor is part of a stable runtime path.

The `requires` tuple lists postprocessor keys that must already have run. This
declares ordering explicitly. It is not a substitute for validating the concrete
payload fields the postprocessor consumes.

`LocalAreaDatasetBuilder` validates the configured postprocessor sequence before
building:

- Duplicate `spec.key` values are rejected.
- A postprocessor whose `requires` keys have not appeared earlier is rejected.
- Processors without an explicit `spec` receive a fallback key based on class
  name. This fallback is for tests or transitional code only; production
  postprocessors should define stable keys.

If a processor consumes fields written by an earlier processor, define both the
dependency and a payload validation. The dependency catches bad builder
configuration early; payload validation catches direct processor use and
malformed payloads.

## Current US Postprocessors

The production US postprocessor sequence is defined by
`default_us_postprocessors()`:

1. `USEntityPostProcessor`
   - Key: `us_entity`
   - Dependencies: none
   - Adds output entity IDs and `household_weight`.

2. `USGeographyPostProcessor`
   - Key: `us_geography`
   - Dependencies: none
   - Derives geography from selected block GEOIDs and writes geography
     variables such as `state_fips`, `county_fips`, `zip_code`, and
     `congressional_district_geoid`.

3. `USTakeupPostProcessor`
   - Key: `us_takeup`
   - Dependencies: `us_entity`, `us_geography`
   - Applies take-up draws and writes take-up variables.
   - Validates that required reindexed subentities exist.
   - Validates that `state_fips` exists in the payload.
   - Validates that `person_tax_unit_id` and `tax_unit_id` exist when reported
     ACA anchors are present.

Keep this ordering unless you also update specs, structural validations, and
unit tests.

## Adding A Postprocessor

When adding a postprocessor:

1. Define a result dataclass if callers need metadata beyond the payload.
2. Define a stable `PayloadPostProcessorSpec`.
3. Add direct payload precondition checks for every field the processor consumes.
4. Preserve the incoming payload's `time_period`, `entity_lengths`, and
   `variable_entities` unless intentionally changing them.
5. When adding variables with non-obvious entity lengths, update
   `variable_entities` so `H5Payload` can validate their shapes.
6. Add the postprocessor to the production factory only if it belongs in the
   runtime path.
7. Add unit tests for the processor in `tests/unit/build_outputs/`.
8. Add or update a builder-order test if the processor has dependencies.

Prefer dependency injection for expensive or external behavior. For example,
`USTakeupPostProcessor` accepts a `takeup_applier` so unit tests can verify the
contract without loading rates or running the full pipeline.

## Testing Expectations

Unit tests should cover each new postprocessor directly. At minimum, test:

- The variables it writes.
- The payload fields it consumes.
- Its declared `spec.key` and `spec.requires` ordering.
- Failure for missing required payload fields.
- Failure for wrong-length generated arrays when the output entity is known.

Builder tests should cover:

- Missing dependency rejection.
- Duplicate postprocessor key rejection.
- Result recording through `PayloadPostProcessorRun`.

Integration tests should only be added when the behavior crosses module or
runtime boundaries that unit tests cannot represent. Do not add a second
integration test that proves the same seam.

## Documentation Expectations

When Stage 4 behavior changes, update the durable documentation surface:

- Add or update `@pipeline_node` metadata for new stable library seams.
- Update `docs/pipeline_map.yaml` when the stage graph or durable artifacts
  change.
- Keep generated docs out of manual PR edits unless the repository workflow
  specifically requires them.

Do not put PR-specific rationale in docstrings. Put durable behavior in source
docs and put review or migration rationale in PR descriptions, issues, or stage
planning docs.
