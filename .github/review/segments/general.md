# General Review Segment

Check for:
- obvious bugs and behavioral regressions
- changed control flow that no longer matches caller expectations
- signature drift between callers and callees
- data path mistakes, especially path handling, identifiers, and selection logic
- missing or misleading validation
- missing unit coverage for newly introduced logic

Bias toward concise, actionable findings. Do not manufacture issues to fill space.
