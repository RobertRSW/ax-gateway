"""Tests for the svg_viz MCP server: stdio dispatch + SVG shape."""

from __future__ import annotations

import json

import pytest

from ax_cli.runtimes.mcp_servers.svg_viz.renderers import render_chart, render_status_card
from ax_cli.runtimes.mcp_servers.svg_viz.tools import (
    _handle_chart,
    _handle_status_card,
    build_tools,
)


def test_build_tools_returns_chart_and_status_card():
    tools = build_tools()
    names = [t.name for t in tools]
    assert names == ["chart", "status_card"]
    for tool in tools:
        assert tool.input_schema["type"] == "object"
        assert "required" in tool.input_schema


@pytest.mark.parametrize("chart_type", ["bar", "line", "donut"])
def test_render_chart_supported_types(chart_type):
    data = [
        {"label": "CENTCOM", "value": 12500},
        {"label": "INDOPACOM", "value": 8200},
        {"label": "EUCOM", "value": 4100},
    ]
    svg = render_chart(chart_type, data, {"title": "Ammo Stockpile"})
    assert svg.startswith("<svg")
    assert svg.endswith("</svg>")
    assert "Ammo Stockpile" in svg
    assert "viewBox=" in svg


def test_render_chart_rejects_unknown_type():
    with pytest.raises(ValueError, match="Unsupported chart type"):
        render_chart("scatter", [{"label": "a", "value": 1}])


def test_render_chart_handles_empty_data():
    svg = render_chart("bar", [], {"title": "Empty"})
    assert "(no data)" in svg


def test_render_chart_donut_empty_when_total_zero():
    svg = render_chart("donut", [{"label": "a", "value": 0}, {"label": "b", "value": 0}])
    assert "(no data)" in svg


def test_render_status_card_basic():
    svg = render_status_card(
        "CENTCOM Status",
        [
            {
                "heading": "Ammo Stockpile",
                "rows": [
                    {"label": "5.56mm", "value": "120,000 rounds", "status": "ok"},
                    {"label": "Javelin", "value": "350 missiles", "status": "alert"},
                ],
            }
        ],
        {"subtitle": "Generated 2026-05-24", "footer": "Synthetic data"},
    )
    assert svg.startswith("<svg")
    assert "CENTCOM Status" in svg
    assert "5.56mm" in svg
    assert "120,000 rounds" in svg
    assert "Generated 2026-05-24" in svg
    assert "Synthetic data" in svg
    # Status pill colors should be present
    assert "#10b981" in svg  # ok
    assert "#ef4444" in svg  # alert


def test_status_card_pill_color_fallback_for_unknown_status():
    svg = render_status_card(
        "Title",
        [{"heading": "S", "rows": [{"label": "row", "value": "v", "status": "mystery"}]}],
    )
    assert "#6b7280" in svg  # PILL_FALLBACK gray


def test_render_status_card_escapes_html_in_values():
    svg = render_status_card(
        "<script>alert(1)</script>",
        [{"heading": "X", "rows": [{"label": "<b>l</b>", "value": "v&v", "status": "ok"}]}],
    )
    assert "<script>" not in svg
    assert "&lt;script&gt;" in svg
    assert "&amp;" in svg


def test_chart_handler_wraps_svg_in_json_payload():
    result = _handle_chart(
        {"type": "bar", "data": [{"label": "a", "value": 1}], "options": {"title": "T"}}
    )
    assert "content" in result
    text = result["content"][0]["text"]
    payload = json.loads(text)
    assert "svg" in payload
    assert payload["svg"].startswith("<svg")


def test_status_card_handler_wraps_svg_in_json_payload():
    result = _handle_status_card(
        {
            "title": "T",
            "sections": [{"heading": "H", "rows": [{"label": "l", "value": "v"}]}],
        }
    )
    text = result["content"][0]["text"]
    payload = json.loads(text)
    assert "svg" in payload


def test_chart_handler_requires_type():
    with pytest.raises(ValueError, match="chart.type is required"):
        _handle_chart({"data": []})


def test_chart_handler_rejects_non_list_data():
    with pytest.raises(ValueError, match="must be a list"):
        _handle_chart({"type": "bar", "data": "not a list"})


def test_status_card_handler_requires_title():
    with pytest.raises(ValueError, match="status_card.title is required"):
        _handle_status_card({"title": "", "sections": []})


def test_status_card_handler_rejects_non_list_sections():
    with pytest.raises(ValueError, match="must be a list"):
        _handle_status_card({"title": "T", "sections": "nope"})


def test_color_scheme_alert_uses_red_palette():
    svg = render_chart("bar", [{"label": "a", "value": 1}], {"color_scheme": "alert"})
    assert "#ef4444" in svg


def test_color_scheme_ok_uses_green_palette():
    svg = render_chart("bar", [{"label": "a", "value": 1}], {"color_scheme": "ok"})
    assert "#10b981" in svg
