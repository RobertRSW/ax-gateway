"""Tests for the report_gen MCP server: schema, query, SQL safety."""

from __future__ import annotations

import json

import pytest

# sqlglot is gated behind the [mcp] optional-extra; tests need it.
sqlglot = pytest.importorskip("sqlglot")

from ax_cli.runtimes.mcp_servers.report_gen.synthetic_db import (  # noqa: E402
    seed_database,
)
from ax_cli.runtimes.mcp_servers.report_gen.tools import (  # noqa: E402
    ReadOnlyViolation,
    _check_sql_readonly,
    _handle_db_query,
    _handle_db_schema,
    build_tools,
    get_db_schema,
    run_query,
)


@pytest.fixture(scope="module", autouse=True)
def isolated_db(tmp_path_factory, monkeypatch_session=None):
    """Run all tests against a fresh DB in tmp so we don't touch the user's ~/.ax."""
    # Module-scoped monkeypatch can't use the function-scoped fixture; do it
    # manually with os.environ + cleanup.
    import os
    tmp = tmp_path_factory.mktemp("report_gen")
    db_path = tmp / "synthetic.db"
    prior = os.environ.get("AX_REPORT_GEN_DB_PATH")
    os.environ["AX_REPORT_GEN_DB_PATH"] = str(db_path)
    seed_database(db_path)
    yield db_path
    if prior is None:
        os.environ.pop("AX_REPORT_GEN_DB_PATH", None)
    else:
        os.environ["AX_REPORT_GEN_DB_PATH"] = prior


def test_build_tools_returns_db_schema_and_db_query():
    tools = build_tools()
    names = [t.name for t in tools]
    assert names == ["db_schema", "db_query"]


def test_get_db_schema_returns_all_five_tables():
    schema = get_db_schema()
    assert schema["synthetic"] is True
    table_names = {t["name"] for t in schema["tables"]}
    assert table_names == {
        "theater", "unit", "ammo_stockpile", "personnel_readiness", "supply_route"
    }


def test_get_db_schema_includes_foreign_keys():
    schema = get_db_schema()
    by_name = {t["name"]: t for t in schema["tables"]}
    fk = by_name["ammo_stockpile"]["foreign_keys"]
    assert any(
        f["column"] == "theater_id" and f["references_table"] == "theater"
        for f in fk
    )


def test_get_db_schema_reports_row_counts():
    schema = get_db_schema()
    by_name = {t["name"]: t for t in schema["tables"]}
    assert by_name["theater"]["row_count"] == 5
    assert by_name["unit"]["row_count"] == 15
    assert by_name["ammo_stockpile"]["row_count"] == 18


def test_run_query_simple_select():
    result = run_query("SELECT name, region FROM theater ORDER BY name")
    assert result["row_count"] == 5
    assert result["columns"] == ["name", "region"]
    centcom = next(r for r in result["rows"] if r["name"] == "CENTCOM")
    assert centcom["region"] == "Middle East"


def test_run_query_join_centcom_ammo():
    sql = (
        "SELECT t.name AS theater, a.ammo_type, a.quantity, a.units "
        "FROM ammo_stockpile a JOIN theater t ON a.theater_id = t.id "
        "WHERE t.name = 'CENTCOM' ORDER BY a.quantity DESC"
    )
    result = run_query(sql)
    assert result["row_count"] == 5
    assert result["rows"][0]["ammo_type"] == "5.56mm"
    assert result["rows"][0]["quantity"] == 120000


def test_run_query_empty_sql():
    result = run_query("")
    assert result["code"] == "EMPTY_SQL"


def test_run_query_row_limit_truncates():
    result = run_query("SELECT * FROM ammo_stockpile", row_limit=3)
    assert result["row_count"] == 3
    assert result["truncated"] is True
    assert result["row_limit"] == 3


def test_run_query_row_limit_not_triggered_when_below():
    result = run_query("SELECT * FROM theater", row_limit=10)
    assert result["truncated"] is False


# --- SQL safety: AST layer (sqlglot) ---


@pytest.mark.parametrize("sql", [
    "DELETE FROM theater WHERE id = 1",
    "UPDATE theater SET name = 'X' WHERE id = 1",
    "INSERT INTO theater (id, name) VALUES (99, 'NEWCOM')",
    "DROP TABLE theater",
    "CREATE TABLE foo (id INTEGER)",
    "ALTER TABLE theater ADD COLUMN evil TEXT",
])
def test_check_sql_readonly_rejects_writes(sql):
    with pytest.raises(ReadOnlyViolation):
        _check_sql_readonly(sql)


def test_check_sql_readonly_rejects_cte_smuggled_delete():
    """The classic attack the cheap startswith-SELECT check misses."""
    sql = "WITH x AS (DELETE FROM theater WHERE id = 1 RETURNING *) SELECT * FROM x"
    with pytest.raises(ReadOnlyViolation):
        _check_sql_readonly(sql)


def test_check_sql_readonly_rejects_load_extension_function():
    # load_extension parses as a normal function call; the Anonymous-node walk
    # catches it with a "Suspicious function" error.
    with pytest.raises(ReadOnlyViolation, match="Suspicious function"):
        _check_sql_readonly("SELECT load_extension('evil.so')")


@pytest.mark.parametrize("fn", ["attach", "detach", "pragma"])
def test_check_sql_readonly_rejects_reserved_extension_keywords(fn):
    # ATTACH/DETACH/PRAGMA are SQLite reserved keywords, so sqlglot can't
    # parse them as function calls inside a SELECT — they get rejected at
    # parse time with "SQL parse error", which still fails closed.
    with pytest.raises(ReadOnlyViolation):
        _check_sql_readonly(f"SELECT {fn}('arg')")


def test_check_sql_readonly_accepts_plain_select():
    # Should not raise
    _check_sql_readonly("SELECT * FROM theater")
    _check_sql_readonly("SELECT t.name FROM theater t JOIN unit u ON u.theater_id = t.id")


def test_check_sql_readonly_rejects_unparseable_sql():
    with pytest.raises(ReadOnlyViolation):
        _check_sql_readonly("not even sql")


# --- SQL safety: full run_query path returns structured errors ---


def test_run_query_returns_readonly_violation_code():
    result = run_query("DELETE FROM theater")
    assert result["code"] == "READONLY_VIOLATION"
    assert "Delete" in result["error"] or "SELECT" in result["error"]


def test_run_query_rejects_cte_smuggled_delete_via_error_code():
    result = run_query(
        "WITH x AS (DELETE FROM theater WHERE id = 1 RETURNING *) SELECT * FROM x"
    )
    assert result["code"] == "READONLY_VIOLATION"


# --- Connection-level read-only is the backstop layer ---


def test_connection_level_readonly_blocks_writes_even_if_ast_bypassed():
    """If somehow the AST check missed a write, SQLite still refuses."""
    import sqlite3

    from ax_cli.runtimes.mcp_servers.report_gen.synthetic_db import default_db_path, open_readonly

    conn = open_readonly(default_db_path())
    try:
        with pytest.raises(sqlite3.OperationalError, match="readonly|read-only"):
            conn.execute("DELETE FROM theater")
    finally:
        conn.close()


# --- Tool handlers wrap results in MCP content blocks ---


def test_db_schema_handler_returns_mcp_content_block():
    result = _handle_db_schema({})
    assert "content" in result
    payload = json.loads(result["content"][0]["text"])
    assert "tables" in payload
    assert len(payload["tables"]) == 5


def test_db_query_handler_returns_mcp_content_block():
    result = _handle_db_query(
        {"sql": "SELECT name FROM theater WHERE name = 'CENTCOM'"}
    )
    payload = json.loads(result["content"][0]["text"])
    assert payload["row_count"] == 1
    assert payload["rows"][0]["name"] == "CENTCOM"


def test_db_query_handler_clamps_huge_row_limit():
    # row_limit > 10000 should clamp; not an error
    result = _handle_db_query({"sql": "SELECT * FROM theater", "row_limit": 999_999})
    payload = json.loads(result["content"][0]["text"])
    assert payload["row_limit"] == 10_000


def test_db_query_handler_floors_zero_row_limit_to_one():
    result = _handle_db_query({"sql": "SELECT * FROM theater", "row_limit": 0})
    payload = json.loads(result["content"][0]["text"])
    assert payload["row_limit"] == 1


def test_db_query_invalid_row_limit_falls_back_to_default():
    result = _handle_db_query({"sql": "SELECT * FROM theater", "row_limit": "not a number"})
    payload = json.loads(result["content"][0]["text"])
    assert payload["row_limit"] == 500  # ROW_LIMIT_DEFAULT
