# PR Review Instructions

Primary goal: identify bugs, regressions, missing tests, contract drift, scope drift, and hidden operational risk.

Review rules:
- Findings first. Do not lead with summary or praise.
- Prioritize behavior, correctness, migration boundaries, and release risk over style.
- Ignore purely cosmetic issues unless they hide a behavioral problem.
- Distinguish direct evidence from inference.
- Be explicit about blind spots such as unrun tests, missing optional dependencies, or unclear runtime context.

Required structure:
- `Severity`: `high`, `medium`, or `low`
- `Confidence`: `high`, `medium`, or `low`
- `Basis`: `direct_code_evidence`, `test_evidence`, `inference`, or `missing_context`
- `Why it matters`
- `Suggested fix`

Use `.github/review/segments/priority_and_confidence.md` for the detailed severity and confidence rubric.

If there are no findings, say so explicitly and still list residual risks or blind spots.
