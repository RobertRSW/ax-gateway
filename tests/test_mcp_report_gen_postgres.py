"""Postgres-backend tests for report_gen MCP.

Skipped by default — they need a real Postgres instance. To run:

    export AX_REPORT_GEN_PG_TEST_DSN_OWNER='postgresql://owner:pw@host/db'
    export AX_REPORT_GEN_PG_TEST_DSN_READER='postgresql://reader:pw@host/db'
    pytest tests/test_mcp_report_gen_postgres.py

On the AX-Gateway VM both DSNs are persisted at /root/.ax/mcp/postgres.env;
source it before running:

    set -a; source /root/.ax/mcp/postgres.env; set +a
    AX_REPORT_GEN_PG_TEST_DSN_OWNER="$AX_REPORT_GEN_PG_DSN_OWNER" \\
    AX_REPORT_GEN_PG_TEST_DSN_READER="$AX_REPORT_GEN_PG_DSN_READER" \\
        pytest tests/test_mcp_report_gen_postgres.py

The tests verify the real Postgres path end-to-end:
- Seeder produces the expected row counts
- Reader role can SELECT but cannot DELETE / INSERT / CREATE
- AST check + connection-level read-only both fire
- Schema introspection returns FKs (regression test for the
  information_schema-vs-pg_catalog bug found during live testing)
- Query timeout aborts long-running statements
"""

from __future__ import annotations

import json
import os

import pytest

OWNER_DSN = os.environ.get("AX_REPORT_GEN_PG_TEST_DSN_OWNER")
READER_DSN = os.environ.get("AX_REPORT_GEN_PG_TEST_DSN_READER")

pytestmark = pytest.mark.skipif(
    not (OWNER_DSN and READER_DSN),
    reason="Postgres DSNs not set (AX_REPORT_GEN_PG_TEST_DSN_OWNER/READER)",
)

# Both psycopg and sqlglot are needed for the Postgres backend.
psycopg = pytest.importorskip("psycopg")
sqlglot = pytest.importorskip("sqlglot")


@pytest.fixture(scope="module", autouse=True)
def configure_pg_env():
    """Wire the test DSNs into the env vars the backend reads."""
    prior_kind = os.environ.get("AX_REPORT_GEN_DB_KIND")
    prior_owner = os.environ.get("AX_REPORT_GEN_PG_DSN_OWNER")
    prior_reader = os.environ.get("AX_REPORT_GEN_PG_DSN_READER")

    os.environ["AX_REPORT_GEN_DB_KIND"] = "postgres"
    os.environ["AX_REPORT_GEN_PG_DSN_OWNER"] = OWNER_DSN
    os.environ["AX_REPORT_GEN_PG_DSN_READER"] = READER_DSN

    # Seed before the suite so query tests have data.
    from ax_cli.runtimes.mcp_servers.report_gen.postgres_seed import seed_postgres
    seed_postgres(OWNER_DSN)

    yield

    if prior_kind is None:
        os.environ.pop("AX_REPORT_GEN_DB_KIND", None)
    else:
        os.environ["AX_REPORT_GEN_DB_KIND"] = prior_kind
    if prior_owner is None:
        os.environ.pop("AX_REPORT_GEN_PG_DSN_OWNER", None)
    else:
        os.environ["AX_REPORT_GEN_PG_DSN_OWNER"] = prior_owner
    if prior_reader is None:
        os.environ.pop("AX_REPORT_GEN_PG_DSN_READER", None)
    else:
        os.environ["AX_REPORT_GEN_PG_DSN_READER"] = prior_reader


def test_seed_produces_expected_row_counts():
    from ax_cli.runtimes.mcp_servers.report_gen.postgres_seed import seed_postgres
    counts = seed_postgres(OWNER_DSN)
    assert counts == {
        "theater": 5,
        "unit": 15,
        "ammo_stockpile": 18,
        "personnel_readiness": 15,
        "supply_route": 10,
    }


def test_backend_dialect_is_postgres():
    from ax_cli.runtimes.mcp_servers.report_gen.backend import select_backend
    backend = select_backend()
    assert backend.dialect == "postgres"
    assert backend.__class__.__name__ == "PostgresBackend"


def test_schema_includes_backend_field():
    from ax_cli.runtimes.mcp_servers.report_gen.tools import get_db_schema
    schema = get_db_schema()
    assert schema["backend"] == "postgres"
    assert schema["synthetic"] is True
    assert schema["schema"] == "public"


def test_schema_returns_all_five_tables():
    from ax_cli.runtimes.mcp_servers.report_gen.tools import get_db_schema
    schema = get_db_schema()
    table_names = {t["name"] for t in schema["tables"]}
    assert table_names == {
        "theater", "unit", "ammo_stockpile", "personnel_readiness", "supply_route"
    }


def test_schema_returns_foreign_keys_via_pg_catalog():
    """Regression for the information_schema bug — reader role couldn't see
    FKs because constraint_column_usage filters by privilege. Fixed by
    switching to pg_catalog views."""
    from ax_cli.runtimes.mcp_servers.report_gen.tools import get_db_schema
    schema = get_db_schema()
    by_name = {t["name"]: t for t in schema["tables"]}
    ammo_fks = by_name["ammo_stockpile"]["foreign_keys"]
    assert len(ammo_fks) == 1
    assert ammo_fks[0]["column"] == "theater_id"
    assert ammo_fks[0]["references_table"] == "theater"
    assert ammo_fks[0]["references_column"] == "id"


def test_query_centcom_ammo_returns_expected_rows():
    from ax_cli.runtimes.mcp_servers.report_gen.tools import run_query
    result = run_query(
        "SELECT t.name AS theater, a.ammo_type, a.quantity, a.units "
        "FROM ammo_stockpile a JOIN theater t ON a.theater_id = t.id "
        "WHERE t.name = 'CENTCOM' ORDER BY a.quantity DESC"
    )
    assert result["row_count"] == 5
    assert result["rows"][0]["ammo_type"] == "5.56mm"
    assert result["rows"][0]["quantity"] == 120000


@pytest.mark.parametrize("sql", [
    "DELETE FROM theater",
    "UPDATE theater SET name = 'X'",
    "INSERT INTO theater (id, name) VALUES (99, 'TEST')",
    "DROP TABLE theater",
])
def test_writes_rejected_by_ast_layer(sql):
    from ax_cli.runtimes.mcp_servers.report_gen.tools import run_query
    result = run_query(sql)
    assert result["code"] == "READONLY_VIOLATION"


def test_cte_smuggled_delete_rejected_with_postgres_dialect():
    from ax_cli.runtimes.mcp_servers.report_gen.tools import run_query
    result = run_query(
        "WITH x AS (DELETE FROM theater WHERE id = 1 RETURNING *) SELECT * FROM x"
    )
    assert result["code"] == "READONLY_VIOLATION"


def test_reader_role_cannot_write_even_if_ast_bypassed():
    """Layer 2b: even if we bypass the AST check entirely, the reader role's
    Postgres grants refuse the write. Strongest backstop."""
    with psycopg.connect(READER_DSN, autocommit=False) as conn:
        with conn.cursor() as cur:
            # SET TRANSACTION READ ONLY is layer 2a; this test verifies that
            # even without it, the role grants are sufficient.
            with pytest.raises(
                (psycopg.errors.InsufficientPrivilege,
                 psycopg.errors.ReadOnlySqlTransaction)
            ):
                cur.execute("DELETE FROM theater")


def test_query_timeout_aborts_long_running_select():
    """Use pg_sleep to force a wait longer than the 5s default."""
    from ax_cli.runtimes.mcp_servers.report_gen.tools import run_query
    # Override timeout to 1s for the test so it doesn't actually take 5s.
    import ax_cli.runtimes.mcp_servers.report_gen.tools as tools_mod
    prior_timeout = tools_mod.QUERY_TIMEOUT_S
    tools_mod.QUERY_TIMEOUT_S = 1.0
    try:
        result = run_query("SELECT pg_sleep(3)")
        assert result["code"] == "TIMEOUT"
    finally:
        tools_mod.QUERY_TIMEOUT_S = prior_timeout


def test_row_limit_truncates_postgres_results():
    from ax_cli.runtimes.mcp_servers.report_gen.tools import run_query
    result = run_query("SELECT * FROM ammo_stockpile ORDER BY id", row_limit=3)
    assert result["row_count"] == 3
    assert result["truncated"] is True


def test_handler_wraps_postgres_query_results_in_mcp_block():
    from ax_cli.runtimes.mcp_servers.report_gen.tools import _handle_db_query
    result = _handle_db_query({"sql": "SELECT COUNT(*) AS n FROM theater"})
    payload = json.loads(result["content"][0]["text"])
    assert payload["row_count"] == 1
    assert payload["rows"][0]["n"] == 5
