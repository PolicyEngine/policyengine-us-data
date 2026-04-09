Replace legacy SQLModel `session.query(...)` lookups in the SOI ETL loaders and their focused tests with `session.exec(select(...))` to remove deprecation warnings in CI.
