"""Internal package for the incremental local H5 migration.

The initial migration surface is intentionally small:

- ``requests.py`` for typed area request values
- ``partitioning.py`` for pure worker chunking helpers

Later modules should be added only when they become active runtime
boundaries rather than speculative contract placeholders.
"""
