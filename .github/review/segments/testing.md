# Testing Review Segment

Check whether the PR adds focused tests for the new behavior it introduces.

Look for:
- direct unit coverage for newly added branch logic
- overreliance on broad integration tests when a narrow unit test would be clearer
- tests that are brittle because they depend on ambient environment state
- module-reload or monkeypatch patterns that can poison the rest of the suite
- new code paths with no test exercising them

If coverage is partial, say which production files or behaviors remain uncovered.
