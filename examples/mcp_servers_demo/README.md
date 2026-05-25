# MCP servers — end-to-end demo

Demo lane: prove the `report_gen` + `svg_viz` MCP composition
end-to-end on a theater-readiness briefing scenario:

> *"What's the CENTCOM ammo status?"* → status briefing card rendered inline.

Two paths to drive the demo:

## Path 1: scripted (no LLM, fastest validation)

```bash
# If you ran `pip install -e .` from the repo root, the package is on
# the path and this works as-is:
python examples/mcp_servers_demo/demo_centcom_ammo.py

# Otherwise (raw checkout, no editable install), run from the repo root
# with PYTHONPATH=.:
PYTHONPATH=. python examples/mcp_servers_demo/demo_centcom_ammo.py
```

This calls the tool handlers directly (skipping the LLM) and writes
`centcom_status.svg` next to the script. Open it in a browser to inspect.
Useful for verifying the data shape + SVG output before bringing a real
agent into the loop.

The script also prints the underlying ammo + readiness JSON rows to stdout
so you can sanity-check the data the agent would see.

## Path 2: through Claude Code (real LLM driving both MCPs)

1. `pip install ax-cli[mcp]` — installs `sqlglot` for SQL safety.
2. From this directory, launch Claude Code:
   ```bash
   claude --strict-mcp-config --mcp-config .mcp.json
   ```
3. In the Claude Code prompt:
   > *"Use db_schema to learn the database. Then query CENTCOM's ammo
   > stockpile and personnel readiness. Use status_card to render the
   > combined report. Save the SVG to centcom_via_claude.svg."*
4. Claude calls `db_schema`, writes appropriate SELECT queries, hands the
   rows to `status_card`, and writes the SVG out.

The `.mcp.json` in this directory wires both servers up via
`python -m ax_cli.runtimes.mcp_servers.<name>`. Tools surface in Claude's
tool list as `report_gen__db_schema`, `report_gen__db_query`,
`svg_viz__chart`, `svg_viz__status_card`.

## Path 3: inline in aX chat (the actual demo target)

After Path 1 + 2 verify the MCPs work in isolation, the SVG gets surfaced
in aX chat via the apps signal adapter (`docs/mcp-app-signal-adapter.md`):
upload the SVG via `axctl context add`, post a message with
`metadata.ui.widget` referencing the returned context key.

That's the path the Phase 1 demo runs through to the customer.

## What this proves

- The shared `stdio_server.py` loop dispatches `initialize` / `tools/list`
  / `tools/call` correctly.
- `report_gen` schema introspection works against the seeded SQLite DB.
- `report_gen` SQL safety rejects writes (verified by tests; the demo
  script only runs safe SELECTs).
- `svg_viz` produces valid SVG that opens in a browser.
- The composition is real — same tool handlers an LLM-driven agent uses.

## What this does NOT prove

- That a specific LLM provider chooses sensible queries (varies by model).
- That the SVG renders correctly in the aX chat signal-card panel (needs
  the apps signal adapter path-test from Path 3).
- Multi-tenant / production-grade hardening (Phase 2; see design doc §10).
