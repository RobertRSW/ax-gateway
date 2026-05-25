"""SQLite backend — connection-level read-only + sqlite3.set_progress_handler timeout.

The default Phase 1 demo lane backend. Zero setup: the DB lives at
`~/.ax/mcp/report_gen/synthetic.db` (override via `AX_REPORT_GEN_DB_PATH`)
and seeds itself on first use.

Read-only enforcement: `sqlite3.connect("file:<path>?mode=ro", uri=True)`.
SQLite refuses to mutate the file regardless of SQL.
"""

from __future__ import annotations

import sqlite3
import time
from typing import Any

from .synthetic_db import ensure_database, open_readonly


class SqliteBackend:
    dialect = "sqlite"

    def get_schema(self) -> dict[str, Any]:
        db_path = ensure_database()
        conn = open_readonly(db_path)
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )
            table_names = [r[0] for r in cursor.fetchall()]
            tables = []
            for table_name in table_names:
                cursor.execute(f"PRAGMA table_info({table_name})")
                cols = [
                    {
                        "name": row[1],
                        "type": row[2],
                        "not_null": bool(row[3]),
                        "default": row[4],
                        "primary_key": bool(row[5]),
                    }
                    for row in cursor.fetchall()
                ]
                cursor.execute(f"PRAGMA foreign_key_list({table_name})")
                foreign_keys = [
                    {
                        "column": row[3],
                        "references_table": row[2],
                        "references_column": row[4],
                    }
                    for row in cursor.fetchall()
                ]
                cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                row_count = cursor.fetchone()[0]
                tables.append({
                    "name": table_name,
                    "columns": cols,
                    "foreign_keys": foreign_keys,
                    "row_count": row_count,
                })
            return {
                "backend": "sqlite",
                "database": str(db_path),
                "synthetic": True,
                "tables": tables,
            }
        finally:
            conn.close()

    def run_query(self, sql: str, row_limit: int, timeout_s: float) -> dict[str, Any]:
        db_path = ensure_database()
        conn = open_readonly(db_path)
        conn.row_factory = sqlite3.Row
        _install_timeout(conn, timeout_s)
        try:
            cursor = conn.execute(sql)
            columns = [d[0] for d in cursor.description] if cursor.description else []
            rows = []
            truncated = False
            for i, row in enumerate(cursor):
                if i >= row_limit:
                    truncated = True
                    break
                rows.append({col: row[col] for col in columns})
            return {
                "columns": columns,
                "rows": rows,
                "row_count": len(rows),
                "truncated": truncated,
                "row_limit": row_limit,
            }
        except sqlite3.OperationalError as e:
            msg = str(e).lower()
            if "interrupt" in msg:
                return {"error": f"query exceeded {timeout_s}s timeout", "code": "TIMEOUT"}
            if "readonly" in msg or "read-only" in msg:
                return {"error": str(e), "code": "READONLY_VIOLATION"}
            return {"error": str(e), "code": "SQLITE_ERROR"}
        except sqlite3.Error as e:
            return {"error": str(e), "code": "SQLITE_ERROR"}
        finally:
            conn.close()


def _install_timeout(conn: sqlite3.Connection, timeout_s: float) -> None:
    """Abort the running statement once wall time elapsed."""
    deadline = time.monotonic() + max(timeout_s, 0.1)

    def _check() -> int:
        return 1 if time.monotonic() >= deadline else 0

    conn.set_progress_handler(_check, 1_000)
