# Pipeline Operations

Use this skill when diagnosing a deployed Modal pipeline run, especially when a
GitHub Actions pipeline launch fails or a user asks for the status of a run.

## Source Of Truth

Treat the pipeline status endpoint and run-scoped error records as the first
diagnostic source. Modal dashboard logs are useful supporting evidence, but they
are not the durable error record for this repo.

The status system reports:

- the run-level manifest;
- all stage and substage manifests present for that run;
- missing expected runtime manifest IDs;
- the latest durable error record, when one exists;
- a redacted, bounded traceback when one exists.

## Status Surfaces

The structured status payload is canonical. Run-scoped pipeline apps expose
status for their own mounted pipeline volume:

- `get_pipeline_status`: Python-callable structured JSON for agents, scripts,
  dashboards, and tests. Prefer this for diagnosis and automation.
- `pipeline_status_endpoint`: protected HTTP endpoint returning the same
  structured JSON for non-Python clients. Use Modal proxy auth headers.
- `list_pipeline_runs`: Python-callable structured JSON index of recent runs.
  This is volume-local and only lists runs visible to that app's mounted
  `US_DATA_PIPELINE_VOLUME_NAME`.
- `pipeline_runs_endpoint`: protected HTTP endpoint returning the same
  volume-local recent-run index for non-Python clients.
- `pipeline_status_snippet`: human-readable text used by
  `modal run modal_app/pipeline.py::main --action status`. This is for quick
  terminal inspection only and must not be treated as a schema.

Cross-run discovery lives in the stable `policyengine-us-data-pipeline-status`
app, not in a run-scoped pipeline app:

- `list_deployed_pipeline_runs`: Python-callable structured JSON index of
  deployed publication pipeline apps. It discovers runs from Modal app names
  matching `usdata-gha<github_run_id>-a<attempt>`, then calls each app's
  `get_pipeline_status`.
- `deployed_pipeline_runs_endpoint`: protected HTTP endpoint returning the same
  cross-app discovery payload. Use this for dashboards that need to discover all
  deployed publication runs.

The stable discovery app requires the `modal-token` Modal Secret in its
environment. That Secret must contain `MODAL_TOKEN_ID` and
`MODAL_TOKEN_SECRET`, and it should be attached only to functions that need
Modal control-plane access.

## Fetch Status

First identify the run context from the GitHub Actions summary, workflow logs, or
run-context output:

- `run_id`
- `candidate_version` for the HF staging namespace
- `base_release_version` and `release_bump` for promotion-time versioning
- `release_version` for final manifests, tags, and release completion, once
  promotion computes it
- Modal app name
- Modal environment

For agent or CLI diagnosis, call the deployed Modal function:

```bash
uv run python - <<'PY'
import json
import modal

app_name = "POLICYENGINE_US_DATA_MODAL_APP"
environment_name = "main"
run_id = "US_DATA_RUN_ID"

fn = modal.Function.from_name(
    app_name,
    "get_pipeline_status",
    environment_name=environment_name,
)
print(json.dumps(fn.remote(run_id), indent=2))
PY
```

The status payload includes a traceback when one is available. Tracebacks are
redacted and bounded by keeping the newest text if they are very long.

If the local environment cannot sync the full project environment, use the same
snippet with a Modal-only temporary environment by replacing `uv run python`
with `uv run --no-sync --with modal python`.

To discover deployed publication runs before choosing a run ID, call the stable
status app:

```bash
uv run --no-sync --with modal python - <<'PY'
import json
import modal

fn = modal.Function.from_name(
    "policyengine-us-data-pipeline-status",
    "list_deployed_pipeline_runs",
    environment_name="main",
)
print(json.dumps(fn.remote(limit=25), indent=2))
PY
```

If using the HTTP endpoint, authenticate with Modal proxy auth headers. Do not
publish or paste proxy auth values into PRs, issues, logs, or docs.

```bash
curl \
  -H "Modal-Key: $MODAL_PROXY_TOKEN_ID" \
  -H "Modal-Secret: $MODAL_PROXY_TOKEN_SECRET" \
  "https://<status-endpoint>.modal.run?run_id=<run_id>"
```

## Interpret Results

Use `status` and `message` for the short answer. Then inspect:

- `error.stage_id`: canonical top-level stage, such as `3_fit_weights`;
- `error.substage_id`: narrower substage, such as
  `3a_weight_fitting_regional`;
- `error.record_path`: immutable error record path in the pipeline volume;
- `error.latest_path`: latest error pointer for the run;
- `stage_manifests[].manifest.error`: manifest-local failure details;
- `missing_expected_manifest_ids`: expected runtime manifests that have not yet
  been written.

When reporting back, name the failing stage and substage, summarize the exception
type and message, and cite whether the traceback came from the status endpoint or
from Modal dashboard logs.

When diagnosing staging or promotion, keep candidate and final versions
separate. Staged files live under
`staging/{candidate_version}-{run_id}/...`; final release records live under
`releases/{release_version}/...`, and production artifact paths remain at the
repository root.

## Safety Rules

- Do not paste tracebacks into PRs, issues, or chat unless the user needs that
  detail.
- Redact secrets before sharing command output, even though the status endpoint
  already applies obvious redaction.
- Do not infer that a missing later-stage manifest is a failure if the run is
  still running.
- If the run was hard-killed before Python exception handling ran, the endpoint
  may show a running run with no durable error. In that case, report the last
  completed/running manifest and then use Modal dashboard logs as secondary
  evidence.

## Local Publication Preflight

When you already have a locally built or checkpointed
`enhanced_cps_2024.h5`, run the publication preflight before launching or
resuming the long local-area publication stages:

```bash
uv run python scripts/run_publication_preflight.py \
  --enhanced-cps /path/to/enhanced_cps_2024.h5 \
  --calibration-log /path/to/calibration_log.csv
```

This reuses the upload dataset contract, computes baseline SPM, checks
`employment_income` against the BEA NIPA wages target with a tight tolerance,
and runs final-epoch JCT diagnostics plus ACA/Medicaid state checks unless
explicitly skipped. Do not treat a completed local data build as publication
ready until this preflight or the equivalent Stage 1 publication validation has
passed.
