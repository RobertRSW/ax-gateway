"""Postgres backend — connects as a SELECT-only role with read-only transactions.

Production-shaped real-DB backend. Activated by
`AX_REPORT_GEN_DB_KIND=postgres`. Connects via the DSN at
`AX_REPORT_GEN_PG_DSN_READER` for queries (a role with only USAGE on
schema + SELECT on tables — no write privileges granted at the Postgres
layer, regardless of what SQL we send).

Seeding uses `AX_REPORT_GEN_PG_DSN_OWNER` (owner role with full
schema/table privileges) and is performed by `seed_postgres.py` — out of
band of the MCP request path.

Defense in depth:
- Layer 1: sqlglot AST check (in tools.py, dialect="postgres")
- Layer 2a: `SET TRANSACTION READ ONLY` before each query
- Layer 2b: role-based grants — the reader role physically cannot write
  even if both 1 and 2a fail. Strongest backstop available.

Query timeout: `SET LOCAL statement_timeout = '<ms>'` inside the read-only
transaction. Postgres aborts past the deadline.
"""

from __future__ import annotations

import os
from typing import Any


class PostgresMissing(Exception):
    """Raised when psycopg isn't installed."""


def _require_psycopg():
    try:
        import psycopg  # noqa: F401
        return psycopg
    except ImportError as e:
        raise PostgresMissing(
            "psycopg is required for the postgres backend; "
            "install via `pip install ax-cli[mcp-postgres]`"
        ) from e


def _reader_dsn() -> str:
    dsn = os.environ.get("AX_REPORT_GEN_PG_DSN_READER")
    if not dsn:
        raise RuntimeError(
            "AX_REPORT_GEN_PG_DSN_READER is required when AX_REPORT_GEN_DB_KIND=postgres"
        )
    return dsn


class PostgresBackend:
    dialect = "postgres"

    def get_schema(self) -> dict[str, Any]:
        psycopg = _require_psycopg()
        dsn = _reader_dsn()
        with psycopg.connect(dsn, autocommit=False) as conn:
            # Read-only transaction is also belt-and-suspenders for schema
            # introspection — the catalog reads are SELECTs anyway.
            with conn.cursor() as cur:
                cur.execute("SET TRANSACTION READ ONLY")
                cur.execute("SELECT current_database(), current_schema()")
                db_name, schema_name = cur.fetchone()

                cur.execute(
                    """
                    SELECT table_name
                    FROM information_schema.tables
                    WHERE table_schema = %s AND table_type = 'BASE TABLE'
                    ORDER BY table_name
                    """,
                    (schema_name,),
                )
                table_names = [r[0] for r in cur.fetchall()]

                tables = []
                for table_name in table_names:
                    cur.execute(
                        """
                        SELECT column_name, data_type, is_nullable, column_default
                        FROM information_schema.columns
                        WHERE table_schema = %s AND table_name = %s
                        ORDER BY ordinal_position
                        """,
                        (schema_name, table_name),
                    )
                    cols_raw = cur.fetchall()

                    cur.execute(
                        """
                        SELECT a.attname
                        FROM pg_index i
                        JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey)
                        WHERE i.indrelid = %s::regclass AND i.indisprimary
                        """,
                        (f"{schema_name}.{table_name}",),
                    )
                    pk_cols = {r[0] for r in cur.fetchall()}

                    cols = [
                        {
                            "name": col_name,
                            "type": data_type,
                            "not_null": (is_nullable == "NO"),
                            "default": col_default,
                            "primary_key": col_name in pk_cols,
                        }
                        for col_name, data_type, is_nullable, col_default in cols_raw
                    ]

                    # Use pg_catalog (privilege-independent) instead of
                    # information_schema, which hides FK metadata from roles
                    # that lack REFERENCES on the source table.
                    cur.execute(
                        """
                        SELECT
                          att.attname AS column_name,
                          ref_cls.relname AS references_table,
                          ref_att.attname AS references_column
                        FROM pg_constraint con
                        JOIN pg_class cls ON cls.oid = con.conrelid
                        JOIN pg_namespace nsp ON nsp.oid = cls.relnamespace
                        JOIN pg_class ref_cls ON ref_cls.oid = con.confrelid
                        JOIN unnest(con.conkey) WITH ORDINALITY AS k(attnum, ord)
                          ON true
                        JOIN pg_attribute att
                          ON att.attrelid = con.conrelid AND att.attnum = k.attnum
                        JOIN unnest(con.confkey) WITH ORDINALITY AS rk(attnum, ord)
                          ON rk.ord = k.ord
                        JOIN pg_attribute ref_att
                          ON ref_att.attrelid = con.confrelid AND ref_att.attnum = rk.attnum
                        WHERE con.contype = 'f'
                          AND nsp.nspname = %s
                          AND cls.relname = %s
                        ORDER BY con.conname, k.ord
                        """,
                        (schema_name, table_name),
                    )
                    foreign_keys = [
                        {
                            "column": col,
                            "references_table": ref_table,
                            "references_column": ref_col,
                        }
                        for col, ref_table, ref_col in cur.fetchall()
                    ]

                    cur.execute(
                        f'SELECT COUNT(*) FROM "{schema_name}"."{table_name}"'
                    )
                    row_count = cur.fetchone()[0]

                    tables.append({
                        "name": table_name,
                        "columns": cols,
                        "foreign_keys": foreign_keys,
                        "row_count": row_count,
                    })

                return {
                    "backend": "postgres",
                    "database": db_name,
                    "schema": schema_name,
                    "synthetic": True,
                    "tables": tables,
                }

    def run_query(self, sql: str, row_limit: int, timeout_s: float) -> dict[str, Any]:
        psycopg = _require_psycopg()
        dsn = _reader_dsn()
        timeout_ms = max(int(timeout_s * 1000), 100)
        try:
            with psycopg.connect(dsn, autocommit=False) as conn:
                with conn.cursor() as cur:
                    cur.execute("SET TRANSACTION READ ONLY")
                    cur.execute(f"SET LOCAL statement_timeout = '{timeout_ms}ms'")
                    cur.execute(sql)
                    columns = [d.name for d in (cur.description or [])]
                    rows = []
                    truncated = False
                    for i, row in enumerate(cur):
                        if i >= row_limit:
                            truncated = True
                            break
                        rows.append({col: row[idx] for idx, col in enumerate(columns)})
                    return {
                        "columns": columns,
                        "rows": rows,
                        "row_count": len(rows),
                        "truncated": truncated,
                        "row_limit": row_limit,
                    }
        except psycopg.errors.QueryCanceled:
            return {"error": f"query exceeded {timeout_s}s timeout", "code": "TIMEOUT"}
        except psycopg.errors.InsufficientPrivilege as e:
            # Layer 2b kicked in — reader role refused a write. Surface as
            # READONLY_VIOLATION so callers get a uniform error code regardless
            # of which layer rejected the write.
            return {"error": str(e).strip(), "code": "READONLY_VIOLATION"}
        except psycopg.errors.ReadOnlySqlTransaction as e:
            # Layer 2a kicked in — read-only transaction refused a write.
            return {"error": str(e).strip(), "code": "READONLY_VIOLATION"}
        except psycopg.Error as e:
            return {"error": str(e).strip(), "code": "POSTGRES_ERROR"}
