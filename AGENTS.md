# Codex Instructions

These instructions apply repository-wide.

## PR review workflow

When the task is a pull request review:

1. Read `.github/review/global.md`.
2. Always read:
   - `.github/review/segments/general.md`
   - `.github/review/segments/priority_and_confidence.md`
3. Inspect the changed files and selectively read these additional segments:
   - `.github/review/segments/staged_prs.md`
     Use when the PR touches staged-migration areas such as `modal_app/local_area.py`, `modal_app/worker_script.py`, `modal_app/pipeline.py`, `policyengine_us_data/calibration/local_h5/`, or `policyengine_us_data/calibration/validate_staging.py`.
   - `.github/review/segments/testing.md`
     Use when the PR changes production code or tests.
4. Prioritize bugs, regressions, contract drift, scope drift, and missing tests.
5. Present findings first.
6. For every finding, include:
   - severity
   - confidence
   - basis
   - why it matters
   - suggested fix
7. If there are no findings, say so explicitly and still mention blind spots.

## General engineering expectations

- Prefer direct evidence over speculation.
- Flag missing execution context when confidence is limited.
- Focus on behavior and operational risk before style.
