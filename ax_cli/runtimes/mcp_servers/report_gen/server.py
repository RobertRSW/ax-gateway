"""report_gen MCP server entrypoint."""

from __future__ import annotations

import os

from ..stdio_server import ServerConfig, serve
from .synthetic_db import ensure_database
from .tools import build_tools

SERVER_NAME = "ax-report-gen"
SERVER_VERSION = "0.1.0"
INSTRUCTIONS = (
    "Read-only SQL queries against a synthetic military-logistics SQLite "
    "database. Two tools:\n"
    "- db_schema(): no args. Returns tables, columns, foreign keys, row counts.\n"
    "- db_query(sql, row_limit=500): runs a SELECT, returns rows as JSON.\n\n"
    "Data is synthetic and non-classified. Tables: theater, unit, "
    "ammo_stockpile, personnel_readiness, supply_route. Call db_schema "
    "first to learn structure, then craft SELECT queries. Writes/DDL are "
    "rejected; queries exceeding 5s are aborted."
)


def main() -> None:
    # Pre-seed the synthetic DB so the first db_schema call doesn't pay the
    # seed cost. Idempotent — no-op if file already exists.
    ensure_database()
    config = ServerConfig(
        name=SERVER_NAME,
        version=SERVER_VERSION,
        instructions=INSTRUCTIONS,
        tools=build_tools(),
        debug=os.environ.get("AX_MCP_DEBUG", "").lower() in {"1", "true", "yes", "on"},
    )
    serve(config)


if __name__ == "__main__":
    main()
