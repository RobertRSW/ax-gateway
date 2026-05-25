#!/usr/bin/env python3
"""End-to-end demo: CENTCOM ammo + readiness → status card SVG.

Calls the report_gen + svg_viz tool handlers directly (skipping the LLM
loop) to validate the MCP composition end-to-end. Writes the SVG to
centcom_status.svg next to this script.

The real demo runs the same composition through an LLM-driven agent (see
this directory's README). This script proves the data + render path
without needing the LLM in the loop.
"""

from __future__ import annotations

import json
from pathlib import Path

from ax_cli.runtimes.mcp_servers.report_gen.tools import run_query
from ax_cli.runtimes.mcp_servers.svg_viz.renderers import render_status_card


def _readiness_to_status(readiness_level: str) -> str:
    """Map C-1/C-2/C-3/C-4 to status pill colors."""
    mapping = {"C-1": "ok", "C-2": "warning", "C-3": "alert", "C-4": "alert"}
    return mapping.get(readiness_level, "warning")


def _stockpile_to_status(quantity: int, ammo_type: str) -> str:
    """Heuristic: very low values get an alert pill so the demo has variety.

    Real production would use per-type thresholds set by ops. Here we just
    want the demo card to show all three pill colors.
    """
    if ammo_type.lower().endswith(("missiles", "missile")) or ammo_type in {
        "Javelin", "Stinger", "Hellfire", "SM-6", "Harpoon"
    }:
        if quantity < 300:
            return "alert"
        if quantity < 500:
            return "warning"
        return "ok"
    # Conventional rounds / shells
    if quantity < 10_000:
        return "warning"
    return "ok"


def main() -> None:
    print("[1] Querying ammo stockpile for CENTCOM...")
    ammo = run_query(
        "SELECT a.ammo_type, a.quantity, a.units "
        "FROM ammo_stockpile a JOIN theater t ON a.theater_id = t.id "
        "WHERE t.name = 'CENTCOM' ORDER BY a.quantity DESC"
    )
    print(json.dumps(ammo, indent=2, default=str))

    print("\n[2] Querying personnel readiness for CENTCOM units...")
    readiness = run_query(
        "SELECT u.name AS unit, u.branch, r.readiness_level, r.notes "
        "FROM personnel_readiness r "
        "JOIN unit u ON r.unit_id = u.id "
        "JOIN theater t ON u.theater_id = t.id "
        "WHERE t.name = 'CENTCOM' ORDER BY r.readiness_level, u.name"
    )
    print(json.dumps(readiness, indent=2, default=str))

    print("\n[3] Composing status card...")
    sections = [
        {
            "heading": "Ammo Stockpile",
            "rows": [
                {
                    "label": row["ammo_type"],
                    "value": f"{row['quantity']:,} {row['units']}",
                    "status": _stockpile_to_status(row["quantity"], row["ammo_type"]),
                }
                for row in ammo["rows"]
            ],
        },
        {
            "heading": "Personnel Readiness",
            "rows": [
                {
                    "label": row["unit"],
                    "value": row["readiness_level"],
                    "status": _readiness_to_status(row["readiness_level"]),
                }
                for row in readiness["rows"]
            ],
        },
    ]
    svg = render_status_card(
        title="CENTCOM Status Report",
        sections=sections,
        options={
            "subtitle": "Composed from report_gen + svg_viz MCPs",
            "footer": "Synthetic data — non-classified. Demo only.",
            "width": 900,
        },
    )

    out_path = Path(__file__).parent / "centcom_status.svg"
    out_path.write_text(svg, encoding="utf-8")
    print(f"\n[4] Wrote {out_path} ({len(svg):,} bytes)")
    print("    Open it in a browser to inspect.")


if __name__ == "__main__":
    main()
