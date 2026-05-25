"""svg_viz MCP tool definitions and dispatchers.

Two tools:

- `chart(type, data, options)` — bar / line / donut. Returns `{"svg": "..."}`.
- `status_card(title, sections, options)` — status briefing layout with
  per-row status pills (ok / warning / alert). Returns `{"svg": "..."}`.

The tool handlers return MCP tool-call payloads (`content: [{type: text, ...}]`).
The SVG itself is wrapped in a single JSON object so callers can attach it
verbatim to the apps signal adapter (context upload + ui.widget) per the
design doc.
"""

from __future__ import annotations

import json
from typing import Any

from ..stdio_server import ToolSpec
from .renderers import render_chart, render_status_card

CHART_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "type": {
            "type": "string",
            "description": "Chart type: 'bar', 'line', or 'donut'.",
            "enum": ["bar", "line", "donut"],
        },
        "data": {
            "type": "array",
            "description": "List of data points.",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string", "description": "X-axis or legend label."},
                    "value": {"type": "number", "description": "Numeric value."},
                },
                "required": ["label", "value"],
            },
        },
        "options": {
            "type": "object",
            "description": "Optional rendering tweaks.",
            "properties": {
                "title": {"type": "string", "description": "Chart title rendered at the top."},
                "color_scheme": {
                    "type": "string",
                    "description": "Color palette: 'default' (blue-leaning), 'alert' (red-leaning), 'ok' (green-leaning).",
                    "enum": ["default", "alert", "ok"],
                },
                "width": {"type": "number", "description": "SVG width in pixels (default 600)."},
                "height": {"type": "number", "description": "SVG height in pixels (default 400)."},
            },
        },
    },
    "required": ["type", "data"],
}

STATUS_CARD_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "title": {"type": "string", "description": "Card title shown in the header bar."},
        "sections": {
            "type": "array",
            "description": "Status sections, each with a heading and a list of rows.",
            "items": {
                "type": "object",
                "properties": {
                    "heading": {"type": "string", "description": "Section heading."},
                    "rows": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "label": {"type": "string", "description": "Row label (left side)."},
                                "value": {"type": "string", "description": "Row value (right side)."},
                                "status": {
                                    "type": "string",
                                    "description": "Status pill color: 'ok' (green), 'warning' (amber), 'alert' (red). Any other value renders gray.",
                                    "enum": ["ok", "warning", "alert"],
                                },
                            },
                            "required": ["label", "value"],
                        },
                    },
                },
                "required": ["heading", "rows"],
            },
        },
        "options": {
            "type": "object",
            "description": "Optional rendering tweaks.",
            "properties": {
                "subtitle": {"type": "string", "description": "Secondary heading shown below title."},
                "footer": {"type": "string", "description": "Footer text (good place for disclaimers or generation stamps)."},
                "width": {"type": "number", "description": "SVG width in pixels (default 800). Height auto-fits sections."},
            },
        },
    },
    "required": ["title", "sections"],
}


def _wrap_svg(svg: str) -> dict[str, Any]:
    """Return an MCP tool-call result wrapping the SVG as JSON text."""
    payload = json.dumps({"svg": svg})
    return {"content": [{"type": "text", "text": payload}]}


def _handle_chart(arguments: dict[str, Any]) -> dict[str, Any]:
    chart_type = str(arguments.get("type") or "").strip()
    data = arguments.get("data") or []
    options = arguments.get("options") or {}
    if not chart_type:
        raise ValueError("chart.type is required")
    if not isinstance(data, list):
        raise ValueError("chart.data must be a list of {label, value} objects")
    svg = render_chart(chart_type, data, options)
    return _wrap_svg(svg)


def _handle_status_card(arguments: dict[str, Any]) -> dict[str, Any]:
    title = str(arguments.get("title") or "").strip()
    sections = arguments.get("sections") or []
    options = arguments.get("options") or {}
    if not title:
        raise ValueError("status_card.title is required")
    if not isinstance(sections, list):
        raise ValueError("status_card.sections must be a list")
    svg = render_status_card(title, sections, options)
    return _wrap_svg(svg)


def build_tools() -> list[ToolSpec]:
    return [
        ToolSpec(
            name="chart",
            description=(
                "Render a bar / line / donut chart as SVG. Returns a JSON object "
                "with key 'svg' containing the full SVG document string. Useful "
                "for visualizing query results inline in the chat."
            ),
            input_schema=CHART_INPUT_SCHEMA,
            handler=_handle_chart,
        ),
        ToolSpec(
            name="status_card",
            description=(
                "Render a status briefing card with title, optional subtitle, "
                "and one or more sections of labeled rows with status pills "
                "(ok / warning / alert). Designed for command-status reports "
                "(e.g., 'CENTCOM Ammo Status'). Returns a JSON object with key "
                "'svg' containing the full SVG document string."
            ),
            input_schema=STATUS_CARD_INPUT_SCHEMA,
            handler=_handle_status_card,
        ),
    ]
