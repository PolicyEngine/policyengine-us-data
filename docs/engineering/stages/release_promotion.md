# Release Promotion Stage AI Guide

This guide is for AI agents and maintainers modifying Stage 5
(`5_validate_and_promote_release`) code. Stage 5 validates a staged release
candidate, promotes the exact candidate to public Hugging Face and GCS
destinations, writes release/version/completion metadata, and cleans staging
only after completion is certified.

## Candidate Identity

Use `policyengine_us_data.release_promotion.ReleasePromotionContext` as the
typed Stage 5 identity boundary. The context must keep these values distinct:

- `run_id`: the canonical publication run correlation key.
- `candidate_version`: the candidate staging scope used in Hugging Face staging
  paths such as `staging/{candidate_version}-{run_id}/...`.
- `release_version`: the final stable public release version.
- `base_release_version` and `release_bump`: optional provenance for how the
  candidate scope was chosen.

Do not resolve a different run ID from the environment inside lower-level
release-promotion logic. Environment resolution belongs at orchestration edges;
Stage 5 library code should receive explicit context.

## Release Candidate Bundles

Use `ReleaseCandidateInputBundle` to describe the artifacts Stage 5 is allowed
to validate and promote. Each artifact should be represented by a
`ReleaseArtifactSpec` with a production-relative path, artifact family, source
stage, and optional checksum/size metadata.

The current compatibility path may build a bundle from the legacy staged path
set produced by Modal orchestration. Mark that reader as compatibility-only and
keep it retirable.

The Stage 4 contract/inventory reader API now exists for migration work:
`build_release_candidate_bundle_from_stage4_contract()` accepts an in-memory
Stage 4 contract plus inventory records, and
`read_stage4_release_candidate_bundle()` reads the same shape from files.
Production Stage 5 code should not depend on Stage 4 contracts until the
contract and inventory are canonical, complete, and populated with semantic
artifact identity plus checksum/size material.

Candidate bundles may record validation reports as path-only
`validation_report_paths` for compatibility. When Stage 4 or another upstream
producer can provide report checksums, prefer `validation_report_refs` with
canonical `DiagnosticRef` / `ArtifactRef` identity so rerun comparison can
distinguish an overwritten report at the same diagnostics path.

## Validation Reports

Stage 5 must use the shared validation schema for durable validation output:

- `policyengine_us_data.stage_contracts.ValidationReport`
- `policyengine_us_data.stage_contracts.ValidationFinding`
- `policyengine_us_data.stage_contracts.DiagnosticRef`

Do not create a Stage 5-specific durable validation report, check, finding, or
error schema for contracts, diagnostics, release candidates, status endpoints,
or step manifests. Release-specific details such as missing staged artifacts,
missing validation reports, finalized-release conflicts, version mismatches, or
destination conflicts should live in canonical finding metadata.

Use `ReleaseCandidateValidator` for the `5a_validate_outputs` library seam.
It wraps `policyengine_us_data.validation_core` and calls existing release
guards through injected dependencies, including staged-artifact presence,
release-manifest preflight, matching finalized-manifest checks, and release
completion marker checks. It should require all base dataset artifacts plus the
regional/national H5 artifact families. By default, canonical
`validation_report.json` files must parse as shared `ValidationReport` payloads
with release-acceptable statuses. Manual overrides may set the validator's
`validation_report_policy` to `presence_only`, but that override only relaxes
report parsing/status checks; report files must still exist. Keep dependencies
injectable so unit tests do not need Hugging Face, GCS, Modal, or production
credentials.

## Rerun Comparison Material

Before public writes, rerun and reuse decisions should compare semantic
candidate identity rather than only checking whether output files exist. The
comparison material should include:

- run ID, candidate version, release version, HF repository, and GCS bucket;
- Stage 4 output contract fingerprint when available;
- output inventory paths/checksums when available;
- validation report paths and `DiagnosticRef` checksum identities when
  available;
- expected production-relative artifact paths;
- the Stage 5 candidate bundle fingerprint.

When required artifacts only have paths and no checksum/size identity, treat
the bundle as path-only and do not use its fingerprint for promotion reuse
decisions.

Already-finalized releases are an idempotency case, not a shortcut around
candidate identity. A finalized release can be reused only when its completion
marker is valid and it matches the requested candidate.

## Side Effects

Candidate builders, schema adapters, and rerun comparison helpers should not
perform Hugging Face writes, GCS uploads, Modal calls, staging cleanup, or
release-manifest publication. Keep those operations behind explicit adapters or
services so tests can exercise candidate shape and validation logic without
credentials or network access.

Use `FullPromotionResult` and its substep result objects when exposing Stage 5
promotion outcomes to contracts, status APIs, or orchestration summaries. The
current compatibility wrapper, `promote_full_release_with_result()`, must keep
calling the existing transaction engine first and only wrap its dictionary
output afterward so the promotion order remains unchanged.

Result objects should carry semantic public-output identity, not only counts.
Keep Hugging Face repo/type, staging prefix, promoted/no-op paths, and commit ID
when available on `HuggingFacePromotionResult`; GCS bucket, release version,
object paths, skipped paths, and failures on `GcsPromotionResult`; release
manifest and TRACE TRO paths on `ReleaseManifestPromotionResult`; version
manifest path/version/current version on `VersionManifestPromotionResult`;
completion marker path/tag/validity on `CompletionMarkerPromotionResult`; and
cleanup `status` as `skipped`, `completed`, or `failed` on
`CleanupPromotionResult`. Later contract, index, diagnostics, and status
writers should read this typed material instead of scraping logs or
reconstructing public paths independently.
