# report_gen MCP server

Read-only SQL queries against a synthetic military-logistics SQLite database.
Built for the theater-readiness briefing scenario: agent answers
"What's CENTCOM's ammo status?" by composing `db_schema` → `db_query` →
(`svg_viz.status_card` from the sibling server).

## Tools

### `db_schema()`

No arguments. Returns the full schema — tables, columns, foreign keys,
row counts. Call this first so you know what tables exist before writing
queries.

```json
{
  "database": "/home/me/.ax/mcp/report_gen/synthetic.db",
  "synthetic": true,
  "tables": [
    {
      "name": "theater",
      "columns": [
        {"name": "id", "type": "INTEGER", "not_null": false, "primary_key": true},
        {"name": "name", "type": "TEXT", "not_null": true, "primary_key": false}
      ],
      "foreign_keys": [],
      "row_count": 5
    }
  ]
}
```

### `db_query(sql, row_limit=500)`

Runs a SQLite SELECT statement. Returns columns + rows as JSON.

```json
{
  "columns": ["theater", "ammo_type", "quantity", "units"],
  "rows": [
    {"theater": "CENTCOM", "ammo_type": "5.56mm", "quantity": 120000, "units": "rounds"}
  ],
  "row_count": 1,
  "truncated": false,
  "row_limit": 500
}
```

On rejection, returns a structured error code:

| Code | When |
|---|---|
| `EMPTY_SQL` | Empty or whitespace-only `sql`. |
| `READONLY_VIOLATION` | AST check rejected the query (write op, DDL, CTE-smuggled write, `load_extension`, unparseable, reserved extension keyword). |
| `SQLGLOT_MISSING` | `sqlglot` not installed — install `ax-cli[mcp]` extra. |
| `TIMEOUT` | Query ran past `AX_REPORT_GEN_QUERY_TIMEOUT_S` (default 5s). |
| `SQLITE_ERROR` | Other SQLite operational error (bad column name, etc.). |

## Synthetic schema (military-logistics narrative)

Five tables seeded with plausible-but-fabricated values:

- **`theater`** — 5 rows: CENTCOM, INDOPACOM, EUCOM, AFRICOM, NORTHCOM
- **`unit`** — 15 rows: divisions, fleets, MEUs, fighter wings, etc.
- **`ammo_stockpile`** — 18 rows: small arms, artillery, missiles by theater
- **`personnel_readiness`** — 15 rows: C-1 / C-2 / C-3 levels per unit
- **`supply_route`** — 10 rows: air / sea / land between theaters, with status

The data shape is what matters — narratively suggestive enough to support
"agent reports on CENTCOM status" demos, transparently mock enough that
nobody mistakes it for real military data.

## Demo prompts

Once registered with Claude Code (see [../README.md](../README.md)):

> *"What's the ammo status for CENTCOM? Show it as a status card."*
> *"Which theaters have the most Javelins? Make me a bar chart."*
> *"Are any supply routes contested right now?"*
> *"Show me readiness levels across all Army units as a status card."*

The agent calls `db_schema` to learn the tables, writes the appropriate
`db_query`, and (if `svg_viz` is also registered) hands the rows to
`status_card` or `chart` for inline rendering.

## Re-seeding the database

The DB is seeded once at first run. To regenerate from scratch (e.g., after
schema-evolution work), delete the file:

```bash
rm ~/.ax/mcp/report_gen/synthetic.db
# Next call to db_schema or db_query re-seeds.
```

Or point at a fresh path:

```bash
AX_REPORT_GEN_DB_PATH=/tmp/mcp-test.db python -m ax_cli.runtimes.mcp_servers.report_gen
```

## Security posture (defense in depth)

See [../README.md](../README.md) for the two-layer model (sqlglot AST +
connection-level read-only). Phase 1 demo lane covers single-user dev only;
multi-tenant permission gating is a Phase 2 deferred item.
