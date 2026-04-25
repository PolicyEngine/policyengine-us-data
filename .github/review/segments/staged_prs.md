# Staged PR Review Segment

This repository often uses staged migration PRs with narrow scope limits.

Check for:
- scope drift beyond the intended phase
- contract breaks across staged seams
- compatibility regressions in dual-path or legacy-adapter code
- accidental schema or artifact format changes
- conflicting implementations that should have one clear owner
- code landing in the wrong layer, such as orchestration absorbing domain logic

Call out whether each finding is:
- a true merge blocker
- a follow-up that can wait

If the PR looks intentionally transitional, say so, but still flag broken boundaries.
