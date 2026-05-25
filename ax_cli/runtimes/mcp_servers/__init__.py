"""ax-cli MCP servers — harness-agnostic tool servers.

These are JSON-RPC stdio MCP servers (protocol version 2025-11-25), matching
the hand-rolled pattern from `ax_cli/commands/channel.py`. Any MCP client
(Claude Code, LangGraph via an MCP-to-tool adapter, Hermes via its tool
registry connector) can invoke them.

Servers shipped here:

- `report_gen` — read-only SQL queries against a synthetic SQLite database.
  Two tools: `db_schema` and `db_query`. SQL safety enforced by sqlglot AST
  parsing + connection-level read-only.

- `svg_viz` — SVG generation. Two tools: `chart` (bar / line / donut) and
  `status_card` (briefing-style report card with colored status pills).

Each server has a `__main__` entrypoint so it runs as
`python -m ax_cli.runtimes.mcp_servers.<name>`.
"""
