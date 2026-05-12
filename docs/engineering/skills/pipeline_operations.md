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

## Fetch Status

First identify the run context from the GitHub Actions summary, workflow logs, or
run-context output:

- `run_id`
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
