"""report_gen MCP tool definitions: db_schema + db_query.

Backend-agnostic. The backend (SQLite or Postgres) is selected by
`AX_REPORT_GEN_DB_KIND` and constructed via `select_backend()`; this module
focuses on the SQL safety layer (sqlglot AST) and the MCP tool surface.

SQL safety is enforced in two layers, mirrored across backends:

1. AST parsing via `sqlglot.parse(sql, dialect=<backend.dialect>)`.
   Top-level statement must be a SELECT; the parse tree is walked for
   write-shaped subtree nodes (Delete / Update / Insert / Drop / Create /
   Alter) and suspicious function calls (`load_extension`).

2. Driver-level read-only enforcement. SQLite: `?mode=ro` connection.
   Postgres: SELECT-only role + `SET TRANSACTION READ ONLY` per query.

If `sqlglot` is missing, `db_query` returns `code: SQLGLOT_MISSING` rather
than silently falling back to layer-2-only. Fail-closed.
"""

from __future__ import annotations

import json
import os
from typing import Any

from ..stdio_server import ToolSpec
from .backend import select_backend

QUERY_TIMEOUT_S = float(os.environ.get("AX_REPORT_GEN_QUERY_TIMEOUT_S") or 5.0)
ROW_LIMIT_DEFAULT = 500


class ReadOnlyViolation(Exception):
    """Raised when AST inspection rejects a query."""


class SqlGlotMissing(Exception):
    """Raised when sqlglot isn't installed and AST safety can't run."""


def _check_sql_readonly(sql: str, dialect: str = "sqlite") -> None:
    """Parse the incoming SQL and reject anything that isn't a plain SELECT."""
    try:
        import sqlglot
        from sqlglot import exp
    except ImportError as e:
        raise SqlGlotMissing(
            "sqlglot is required for SQL safety; install via `pip install ax-cli[mcp]`"
        ) from e

    try:
        parsed = sqlglot.parse(sql, dialect=dialect)
    except Exception as e:
        raise ReadOnlyViolation(f"SQL parse error: {e}") from e

    if not parsed:
        raise ReadOnlyViolation("empty SQL")

    write_node_types = (
        exp.Delete, exp.Update, exp.Insert,
        exp.Drop, exp.Create, exp.Alter,
        exp.TruncateTable,
    )
    suspicious_functions = {"load_extension", "attach", "detach", "pragma"}

    for stmt in parsed:
        if stmt is None:
            continue
        if not isinstance(stmt, exp.Select):
            raise ReadOnlyViolation(
                f"Only SELECT statements allowed; got {type(stmt).__name__}"
            )
        for node in stmt.walk():
            if isinstance(node, write_node_types):
                raise ReadOnlyViolation(
                    f"Write operation rejected: {type(node).__name__}"
                )
            if isinstance(node, exp.Anonymous):
                name = (node.name or "").lower()
                if name in suspicious_functions:
                    raise ReadOnlyViolation(f"Suspicious function rejected: {name}")


def get_db_schema() -> dict[str, Any]:
    """Return the active backend's schema description."""
    return select_backend().get_schema()


def run_query(sql: str, row_limit: int = ROW_LIMIT_DEFAULT) -> dict[str, Any]:
    """Validate `sql` and execute it against the active backend."""
    sql = (sql or "").strip()
    if not sql:
        return {"error": "empty SQL", "code": "EMPTY_SQL"}

    backend = select_backend()
    try:
        _check_sql_readonly(sql, dialect=backend.dialect)
    except SqlGlotMissing as e:
        return {"error": str(e), "code": "SQLGLOT_MISSING"}
    except ReadOnlyViolation as e:
        return {"error": str(e), "code": "READONLY_VIOLATION"}

    return backend.run_query(sql, row_limit=row_limit, timeout_s=QUERY_TIMEOUT_S)


def _tool_result(payload: dict[str, Any]) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": json.dumps(payload, indent=2, default=str)}]}


def _handle_db_schema(arguments: dict[str, Any]) -> dict[str, Any]:
    return _tool_result(get_db_schema())


def _handle_db_query(arguments: dict[str, Any]) -> dict[str, Any]:
    sql = str(arguments.get("sql") or "")
    raw_limit = arguments.get("row_limit")
    if raw_limit is None:
        row_limit = ROW_LIMIT_DEFAULT
    else:
        try:
            row_limit = int(raw_limit)
        except (TypeError, ValueError):
            row_limit = ROW_LIMIT_DEFAULT
    row_limit = max(1, min(row_limit, 10_000))
    return _tool_result(run_query(sql, row_limit=row_limit))


DB_SCHEMA_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {},
    "description": "No arguments. Returns the synthetic database schema.",
}

DB_QUERY_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "sql": {
            "type": "string",
            "description": "A SELECT statement (SQLite or Postgres dialect depending on "
                          "backend). Write operations, DDL, and extension-loading function "
                          "calls are rejected by the AST safety check before execution.",
        },
        "row_limit": {
            "type": "number",
            "description": f"Maximum rows returned (default {ROW_LIMIT_DEFAULT}, max 10000). "
                          "If the query produces more rows, the response sets truncated=true.",
        },
    },
    "required": ["sql"],
}


def build_tools() -> list[ToolSpec]:
    return [
        ToolSpec(
            name="db_schema",
            description=(
                "Return the synthetic military-logistics database schema "
                "(tables, columns, foreign keys, row counts). Call this first "
                "before writing any SQL — it tells you what tables exist and "
                "how they relate. Data is synthetic / non-classified."
            ),
            input_schema=DB_SCHEMA_INPUT_SCHEMA,
            handler=_handle_db_schema,
        ),
        ToolSpec(
            name="db_query",
            description=(
                "Run a read-only SELECT query against the synthetic database. "
                "Returns columns + rows as JSON. Write operations, DDL, "
                "extension-loading calls, and queries exceeding the timeout "
                "are rejected. Use db_schema first to discover table shapes."
            ),
            input_schema=DB_QUERY_INPUT_SCHEMA,
            handler=_handle_db_query,
        ),
    ]
