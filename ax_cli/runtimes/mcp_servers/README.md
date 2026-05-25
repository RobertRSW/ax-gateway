# MCP servers shipped with ax-cli

Hand-rolled JSON-RPC stdio MCP servers (protocol version `2025-11-25`)
matching the `ax_cli/commands/channel.py` pattern. Harness-agnostic — any
MCP client (Claude Code, LangGraph via an MCP-to-tool adapter, Hermes via
its tool registry connector) can drive them.

| Server | Tools | Required deps | Purpose |
|---|---|---|---|
| `report_gen` | `db_schema`, `db_query` | `[mcp]` extra (`sqlglot`) | Read-only SQL queries against a synthetic military-logistics SQLite database. SQL safety enforced by AST parsing + connection-level read-only. |
| `svg_viz` | `chart`, `status_card` | none (stdlib only) | SVG generation for inline chat rendering: bar / line / donut charts and status briefing cards. |

See the per-server READMEs (`report_gen/README.md`, `svg_viz/README.md`)
for full rationale, deferrals, and security model.

## Install

```bash
# SQLite backend (default, zero database setup):
pip install ax-cli[mcp]

# Postgres backend (real-DB, role-based read-only):
pip install ax-cli[mcp-postgres]
```

The `[mcp]` extra pulls in `sqlglot` for `report_gen`'s SQL safety layer.
The `[mcp-postgres]` extra additionally installs `psycopg[binary]` for the
Postgres backend. `svg_viz` has no extra deps — it works with a base
`pip install ax-cli`.

## Choosing a backend

`report_gen` ships two backends, selected by `AX_REPORT_GEN_DB_KIND`:

| Backend | When | Setup | Read-only enforcement |
|---|---|---|---|
| **sqlite** (default) | Demo / dev / CI. Zero external deps beyond `[mcp]`. | None — DB seeds itself at `~/.ax/mcp/report_gen/synthetic.db` on first use. | Connection-level `?mode=ro` URI flag. |
| **postgres** | Real-DB validation, customer demos with production-shaped infrastructure. | Postgres server + `ax_report_gen` database + owner/reader roles. See [Postgres setup](#postgres-setup) below. | Three layers: (1) sqlglot AST check with `dialect="postgres"`, (2) `SET TRANSACTION READ ONLY` per query, (3) reader role with no write grants (strongest backstop — physically cannot write). |

Switching is a single env-var change at MCP launch time:

```bash
# SQLite (default):
python -m ax_cli.runtimes.mcp_servers.report_gen

# Postgres:
AX_REPORT_GEN_DB_KIND=postgres \
AX_REPORT_GEN_PG_DSN_READER='postgresql://reader:pw@host/ax_report_gen' \
    python -m ax_cli.runtimes.mcp_servers.report_gen
```

## Postgres setup

On a Debian/Ubuntu host:

```bash
apt-get install -y postgresql postgresql-contrib

# Create roles + database (run as postgres superuser):
sudo -u postgres psql <<SQL
CREATE ROLE ax_report_gen_owner WITH LOGIN PASSWORD '<owner-pw>' NOSUPERUSER NOCREATEDB NOCREATEROLE;
CREATE ROLE ax_report_gen_reader WITH LOGIN PASSWORD '<reader-pw>' NOSUPERUSER NOCREATEDB NOCREATEROLE;
CREATE DATABASE ax_report_gen OWNER ax_report_gen_owner;
\c ax_report_gen
REVOKE ALL ON SCHEMA public FROM PUBLIC;
GRANT USAGE ON SCHEMA public TO ax_report_gen_reader;
ALTER DEFAULT PRIVILEGES FOR ROLE ax_report_gen_owner IN SCHEMA public
    GRANT SELECT ON TABLES TO ax_report_gen_reader;
SQL

# Seed the schema (owner DSN — has CREATE TABLE privilege):
AX_REPORT_GEN_PG_DSN_OWNER='postgresql://ax_report_gen_owner:<owner-pw>@127.0.0.1/ax_report_gen' \
    python -m ax_cli.runtimes.mcp_servers.report_gen.postgres_seed
```

After that, the MCP server runs as the reader role (no write privileges)
and the seeder is never invoked again unless you explicitly re-seed.

## Register with Claude Code

Drop a `.mcp.json` into the workspace where you'll run Claude Code:

```json
{
  "mcpServers": {
    "report_gen": {
      "type": "stdio",
      "command": "python",
      "args": ["-m", "ax_cli.runtimes.mcp_servers.report_gen"]
    },
    "svg_viz": {
      "type": "stdio",
      "command": "python",
      "args": ["-m", "ax_cli.runtimes.mcp_servers.svg_viz"]
    }
  }
}
```

Then launch Claude Code in that workspace. The two MCPs auto-register and
their tools appear in Claude's tool list as `report_gen__db_schema`,
`report_gen__db_query`, `svg_viz__chart`, `svg_viz__status_card`.

## Smoke test from a shell

Each server reads JSON-RPC from stdin and writes JSON-RPC to stdout.
You can drive either one directly:

```bash
printf '%s\n' \
  '{"jsonrpc":"2.0","id":1,"method":"initialize"}' \
  '{"jsonrpc":"2.0","id":2,"method":"tools/list"}' \
  '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"db_schema","arguments":{}}}' \
  | python -m ax_cli.runtimes.mcp_servers.report_gen
```

## Environment variables

| Variable | Server | Default | Purpose |
|---|---|---|---|
| `AX_REPORT_GEN_DB_KIND` | `report_gen` | `sqlite` | Backend selector. `sqlite` (default) or `postgres`. |
| `AX_REPORT_GEN_DB_PATH` | `report_gen` | `~/.ax/mcp/report_gen/synthetic.db` | SQLite file path. Created + seeded at first run. SQLite backend only. |
| `AX_REPORT_GEN_PG_DSN_READER` | `report_gen` | (required when kind=postgres) | Postgres DSN for the SELECT-only reader role. The MCP server connects with this. |
| `AX_REPORT_GEN_PG_DSN_OWNER` | `postgres_seed` | (required for seeding) | Postgres DSN for the schema-owner role. Used only by `postgres_seed`; NOT read by the MCP server itself. |
| `AX_REPORT_GEN_QUERY_TIMEOUT_S` | `report_gen` | `5.0` | Per-query wall-clock timeout in seconds. Both backends honor it. |
| `AX_MCP_DEBUG` | both | (unset) | Set to `1`/`true`/`yes`/`on` to log per-request dispatch to stderr. |

## Security model (report_gen)

Defense in depth, scaling with the backend's underlying mechanisms:

**Layer 1 — AST parsing** (both backends). `sqlglot.parse(sql, dialect=<sqlite|postgres>)`.
Top-level statement must be a SELECT; the parse tree is walked for
write-shaped subtree nodes (Delete / Update / Insert / Drop / Create /
Alter / TruncateTable) and suspicious function calls (`load_extension`).
`ATTACH`/`DETACH`/`PRAGMA` are SQLite reserved keywords so they get
rejected at parse time — also fail-closed.

**Layer 2 — Driver-level read-only** (backend-specific):

- **SQLite:** `sqlite3.connect("file:<path>?mode=ro", uri=True)`. SQLite
  refuses to mutate the file regardless of what SQL it receives.
- **Postgres:** `SET TRANSACTION READ ONLY` set on every query transaction.
  Postgres aborts any write inside a read-only transaction with
  `ReadOnlySqlTransaction`.

**Layer 3 — Role grants** (Postgres only — strongest backstop). The
`ax_report_gen_reader` role has only `USAGE` on `public` schema and
`SELECT` on tables (via `ALTER DEFAULT PRIVILEGES`). It has no `INSERT` /
`UPDATE` / `DELETE` / `CREATE` grants — the database physically refuses
the operation regardless of whether layers 1 or 2 fired.

If `sqlglot` is missing, `db_query` returns `code: SQLGLOT_MISSING` with an
actionable error message rather than silently falling back to layer-2-only.

## Why hand-rolled stdio, not the `mcp` PyPI package

`ax_cli/commands/channel.py` already implements the MCP protocol over stdio
JSON-RPC, hand-coded, no external deps. These servers follow the same
pattern (`stdio_server.py` is the shared ~100-line loop) to keep the
dependency footprint minimal and the code easy to read/modify.

If the in-repo MCP count grows past ~5 servers, revisit whether the `mcp`
PyPI package is worth the dependency add.

## Inline rendering in aX chat

The MCPs are pure tools — they produce SVG strings and JSON rows. They
don't know about aX. To render an SVG inline in the chat stream, the
calling agent uploads the SVG as a context resource via `axctl context add`
and posts a message with `metadata.ui.widget` referencing the context key.
See `docs/mcp-app-signal-adapter.md` for the full pattern.
