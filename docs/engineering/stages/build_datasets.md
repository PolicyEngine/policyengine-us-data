# Stage 1: Build Datasets

Stage 1 builds the public dataset artifacts consumed by later pipeline stages.
Its public status boundary is organized around the `1a_` through `1f_`
substeps, while the transitional Modal runtime still executes several
command-backed units inside some of those public substeps.

## Rerun And Reuse Model

Checkpoint reuse has two gates:

- The semantic gate compares current `Stage1IdentityMaterial` with a persisted
  identity from the checkpoint-scoped Stage 1 reuse manifest.
- The physical gate verifies that every expected checkpoint output exists and is
  non-empty before a unit is restored.

The physical checkpoint layout remains `/checkpoints/{branch}/{commit_sha}`.
The Stage 1 reuse manifest is adapter state in that same scope. Missing,
malformed, or unreadable manifest content must fail closed to recompute; it must
not authorize reuse by itself.

Keep reuse explanations in the existing Stage 1 contract metadata under
`dataset_build_output.json -> metadata.stage_1_status.reuse_reasoning`.
That metadata should distinguish semantic identity results from physical
checkpoint availability, including missing prior identity, identity mismatch,
identity match, missing checkpoint output, empty checkpoint output, and restored
checkpoint output.

## Identity Granularity

`substep_id` is the public reporting group. It is not always the right durable
manifest lookup key, because transitional Stage 1 substeps can contain multiple
independently runnable command or script units. For example, raw-data download
and uprating both report through `1a_raw_data_download`, while the base dataset
substep can run several dataset builders.

When persisting or looking up reuse identities for a command-backed unit,
`identity_key` is the stable execution identity key within the checkpoint scope.
It includes the public `substep_id` plus enough stable execution material to
distinguish the command or script and its expected reusable outputs. Keep
`substep_id` on the record for public status grouping.

Do not key multiple manifest records only by `substep_id` unless the record
represents an intentionally aggregated identity for the whole public substep.
Otherwise, later units in the same substep can overwrite earlier units and make
future reruns recompute despite valid checkpoints.

## Conditional Running

Unit-level conditional running is the compatibility path while Stage 1 is still
command-backed:

1. Build current identity material for the runnable unit.
2. Compare it with the previous manifest identity for that unit's identity key.
3. Consult physical checkpoints only when the semantic decision is `reuse`.
4. Restore and skip only that unit when both gates pass.
5. Recompute the unit and update the manifest only after successful output
   restoration or successful checkpoint save.

Public substep status should be aggregated from its unit results. A public
substep is fully `reused` only when every required unit in that substep was
reused. If any unit recomputes successfully, report the substep as completed
with reuse reasoning that explains the mixed path.

Stage-level conditional running is the same idea one level higher. Stage 1 may
skip all builder execution only when every required unit for the requested run
flags has a matching semantic identity and valid physical checkpoint outputs.
Until the canonical Stage 1 coordinator owns whole-stage planning, do not infer
stage-level reuse from a single substep or unit record.

## Documentation Expectations

When changing Stage 1 identity material, checkpoint reuse decisions, artifact
outputs, substep aggregation, or contract metadata, keep the durable
documentation surface synchronized:

- Update this guide when the Stage 1 rerun or checkpoint model changes.
- Update `docs/pipeline_map.yaml` and regenerate generated pipeline docs when
  the stage graph, artifact names, or pipeline-node metadata change.
- Keep `dataset_build_output.json` metadata documentation aligned with the
  status and reuse reasoning actually emitted by the Modal adapter.
- Put PR-specific migration rationale in the PR description, not in durable
  docs or docstrings.

Tests for Stage 1 reuse changes should cover missing and malformed manifests,
semantic mismatch, physical checkpoint miss or empty output, same-public-substep
units with distinct identity keys, and contract metadata explaining both gates.
