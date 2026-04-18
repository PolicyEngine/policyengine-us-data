# Priority and Confidence Segment

Use this segment to classify both finding priority and confidence level consistently.

## Priority rubric

Classify each finding as `high`, `medium`, or `low`.

- `high`
  A likely merge blocker. The issue can cause incorrect behavior, runtime failure, broken contracts, artifact corruption, publication mistakes, or materially misleading output.
- `medium`
  Important, but not always a blocker. The issue can plausibly cause regressions, maintenance traps, incomplete migrations, or missing coverage around meaningful new behavior.
- `low`
  Real but limited impact. The issue is worth fixing, but it is unlikely to cause immediate user-facing failure or operational damage.

If a concern is merely stylistic or speculative, do not promote it into a finding.

## Confidence rubric

Classify each finding as `high`, `medium`, or `low`.

- `high`
  Directly supported by code in the diff, surrounding code, or executed tests.
- `medium`
  Strong inference from the code path, but not fully validated by execution or complete context.
- `low`
  Plausible concern, but evidence is incomplete or significant context is missing.

Also state the basis for the finding:

- `direct_code_evidence`
- `test_evidence`
- `inference`
- `missing_context`

When confidence is not `high`, briefly say what is missing.
