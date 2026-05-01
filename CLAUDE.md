# Claude Instructions

These instructions apply repository-wide.

## Canonical guidance

Repository-wide AI-facing engineering guidance lives in `AGENTS.md`.
Canonical skills live under `docs/engineering/skills/`.

Use those files as the source of truth. This file is a Claude adapter and should
stay thin; do not duplicate detailed testing, CI, formatting, or architecture
rules here.

## Required skill lookup

When adding, moving, or reviewing tests, read
`docs/engineering/skills/testing.md` before editing. Then run
`python scripts/run_quality_guards.py` before handing off test-layout changes.

## Safety boundaries

Do not fabricate data, validation metrics, academic results, or performance
claims. If a result has not been computed from code or cited from a published
source, mark it as pending instead.

Do not upload, promote, or publish data artifacts unless the user explicitly
asks for that operation.
