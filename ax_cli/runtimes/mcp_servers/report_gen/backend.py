"""Backend abstraction — pick SQLite or Postgres at runtime.

Selected by `AX_REPORT_GEN_DB_KIND`:
- unset / `sqlite` → in-repo synthetic SQLite (default, no setup required)
- `postgres` → connect to the DSN at `AX_REPORT_GEN_PG_DSN_READER`
  (a SELECT-only role; the seeder uses `AX_REPORT_GEN_PG_DSN_OWNER`)

Both backends expose the same surface:
- `dialect: str` for the sqlglot AST check (`sqlite` or `postgres`)
- `get_schema() -> dict` returns the database description
- `run_query(sql, row_limit, timeout_s) -> dict` runs a validated SELECT

The driver-specific safety primitives (SQLite's `?mode=ro`, Postgres's
read-only role + `SET TRANSACTION READ ONLY`) live inside each backend.
The AST safety layer (sqlglot) lives in tools.py and runs before either
backend touches the database.
"""

from __future__ import annotations

import os
from typing import Any, Protocol


class Backend(Protocol):
    """The minimum surface tools.py needs from a backend."""

    dialect: str
    """sqlglot dialect name — `sqlite` or `postgres`."""

    def get_schema(self) -> dict[str, Any]:
        """Return the database schema as a structured dict."""
        ...

    def run_query(self, sql: str, row_limit: int, timeout_s: float) -> dict[str, Any]:
        """Execute the validated SELECT and return columns + rows + row_count."""
        ...


def select_backend() -> Backend:
    """Pick the backend driven by `AX_REPORT_GEN_DB_KIND`. Defaults to sqlite."""
    kind = (os.environ.get("AX_REPORT_GEN_DB_KIND") or "sqlite").strip().lower()
    if kind == "sqlite":
        from .sqlite_backend import SqliteBackend
        return SqliteBackend()
    if kind == "postgres":
        from .postgres_backend import PostgresBackend
        return PostgresBackend()
    raise ValueError(
        f"AX_REPORT_GEN_DB_KIND must be 'sqlite' or 'postgres'; got {kind!r}"
    )
