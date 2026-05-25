"""Pure-Python SVG renderers — no external deps.

Two renderers:
- `render_chart(chart_type, data, options)` for bar / line / donut
- `render_status_card(title, sections, options)` for a title +
  sectioned-rows + colored-status-pill briefing layout

SVG is generated as straight strings (not via svgwrite or similar) so the
demo lane keeps a zero-dep footprint. If chart variety grows past ~5 types
this should be revisited per the design doc trade-off table.

Status pill colors map the design doc convention:
- ok      → green
- warning → amber
- alert   → red
- (other) → gray
"""

from __future__ import annotations

import html
import math
from typing import Any

# Color palette — slightly muted defense/ops aesthetic, not consumer-app bright
COLORS = {
    "default": ["#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#06b6d4"],
    "alert": ["#ef4444", "#dc2626", "#b91c1c", "#991b1b", "#7f1d1d", "#fca5a5"],
    "ok": ["#10b981", "#059669", "#047857", "#065f46", "#064e3b", "#6ee7b7"],
}
STATUS_PILL = {
    "ok": "#10b981",
    "warning": "#f59e0b",
    "alert": "#ef4444",
}
PILL_FALLBACK = "#6b7280"
BG = "#0f172a"
FG = "#e2e8f0"
MUTED = "#64748b"
GRID = "#1e293b"


def _esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _palette(scheme: str | None, count: int) -> list[str]:
    base = COLORS.get((scheme or "default").lower(), COLORS["default"])
    if count <= len(base):
        return base[:count]
    # Cycle if more values than palette entries
    return [base[i % len(base)] for i in range(count)]


def render_chart(chart_type: str, data: list[dict[str, Any]], options: dict[str, Any] | None = None) -> str:
    """Render a bar/line/donut chart as a complete SVG document string."""
    options = options or {}
    width = int(options.get("width") or 600)
    height = int(options.get("height") or 400)
    title = str(options.get("title") or "")
    scheme = options.get("color_scheme")

    chart_type = (chart_type or "").lower()
    if chart_type == "bar":
        body = _render_bar(data, width, height, scheme)
    elif chart_type == "line":
        body = _render_line(data, width, height, scheme)
    elif chart_type == "donut":
        body = _render_donut(data, width, height, scheme)
    else:
        raise ValueError(f"Unsupported chart type: {chart_type!r} (supported: bar, line, donut)")

    title_block = ""
    if title:
        title_block = f'<text x="{width // 2}" y="28" text-anchor="middle" fill="{FG}" '
        title_block += f'font-family="sans-serif" font-size="16" font-weight="600">{_esc(title)}</text>'

    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}">'
        f'<rect width="{width}" height="{height}" fill="{BG}"/>'
        f"{title_block}{body}</svg>"
    )


def _render_bar(data: list[dict[str, Any]], width: int, height: int, scheme: str | None) -> str:
    if not data:
        return _empty_chart_text(width, height)
    padding_l, padding_r, padding_t, padding_b = 60, 30, 50, 60
    plot_w = width - padding_l - padding_r
    plot_h = height - padding_t - padding_b

    values = [float(d.get("value", 0) or 0) for d in data]
    max_v = max(values + [1.0])
    palette = _palette(scheme, len(data))

    n = len(data)
    bar_gap = max(4, plot_w // (n * 4))
    bar_w = max(8, (plot_w - bar_gap * (n + 1)) // n)

    parts = [_axes(padding_l, padding_t, plot_w, plot_h, max_v)]
    for i, d in enumerate(data):
        v = float(d.get("value", 0) or 0)
        x = padding_l + bar_gap + i * (bar_w + bar_gap)
        h = int(plot_h * (v / max_v)) if max_v > 0 else 0
        y = padding_t + plot_h - h
        label = _esc(d.get("label", ""))
        parts.append(f'<rect x="{x}" y="{y}" width="{bar_w}" height="{h}" fill="{palette[i]}" rx="2"/>')
        parts.append(
            f'<text x="{x + bar_w // 2}" y="{y - 6}" text-anchor="middle" '
            f'fill="{FG}" font-family="sans-serif" font-size="11">{_fmt_value(v)}</text>'
        )
        parts.append(
            f'<text x="{x + bar_w // 2}" y="{padding_t + plot_h + 16}" text-anchor="middle" '
            f'fill="{MUTED}" font-family="sans-serif" font-size="11">{label}</text>'
        )
    return "".join(parts)


def _render_line(data: list[dict[str, Any]], width: int, height: int, scheme: str | None) -> str:
    if not data:
        return _empty_chart_text(width, height)
    padding_l, padding_r, padding_t, padding_b = 60, 30, 50, 60
    plot_w = width - padding_l - padding_r
    plot_h = height - padding_t - padding_b

    values = [float(d.get("value", 0) or 0) for d in data]
    max_v = max(values + [1.0])
    palette = _palette(scheme, 1)
    n = len(data)
    step = plot_w / max(n - 1, 1) if n > 1 else 0

    points = []
    for i, v in enumerate(values):
        x = padding_l + (i * step if n > 1 else plot_w / 2)
        y = padding_t + plot_h - (plot_h * (v / max_v)) if max_v > 0 else padding_t + plot_h
        points.append((x, y))

    parts = [_axes(padding_l, padding_t, plot_w, plot_h, max_v)]
    path = "M " + " L ".join(f"{x:.1f} {y:.1f}" for x, y in points)
    parts.append(f'<path d="{path}" stroke="{palette[0]}" stroke-width="2" fill="none"/>')
    for i, (x, y) in enumerate(points):
        v = values[i]
        label = _esc(data[i].get("label", ""))
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" fill="{palette[0]}"/>')
        parts.append(
            f'<text x="{x:.1f}" y="{padding_t + plot_h + 16}" text-anchor="middle" '
            f'fill="{MUTED}" font-family="sans-serif" font-size="11">{label}</text>'
        )
        parts.append(
            f'<text x="{x:.1f}" y="{y - 8:.1f}" text-anchor="middle" '
            f'fill="{FG}" font-family="sans-serif" font-size="10">{_fmt_value(v)}</text>'
        )
    return "".join(parts)


def _render_donut(data: list[dict[str, Any]], width: int, height: int, scheme: str | None) -> str:
    if not data:
        return _empty_chart_text(width, height)
    cx, cy = width // 2, height // 2 + 10
    r_outer = min(width, height) // 3
    r_inner = r_outer // 2
    palette = _palette(scheme, len(data))

    total = sum(float(d.get("value", 0) or 0) for d in data)
    if total <= 0:
        return _empty_chart_text(width, height)

    parts = []
    start_angle = -math.pi / 2  # Start at top
    for i, d in enumerate(data):
        v = float(d.get("value", 0) or 0)
        if v <= 0:
            continue
        sweep = (v / total) * 2 * math.pi
        end_angle = start_angle + sweep
        parts.append(_donut_slice(cx, cy, r_outer, r_inner, start_angle, end_angle, palette[i]))
        start_angle = end_angle

    # Legend along the right edge
    legend_x = cx + r_outer + 30
    legend_y = cy - (len(data) * 10)
    for i, d in enumerate(data):
        y = legend_y + i * 22
        if y > height - 20:
            break
        v = float(d.get("value", 0) or 0)
        pct = (v / total * 100) if total > 0 else 0
        label = _esc(d.get("label", ""))
        parts.append(f'<rect x="{legend_x}" y="{y}" width="12" height="12" fill="{palette[i]}" rx="2"/>')
        parts.append(
            f'<text x="{legend_x + 18}" y="{y + 10}" fill="{FG}" '
            f'font-family="sans-serif" font-size="11">{label} ({pct:.1f}%)</text>'
        )
    return "".join(parts)


def _donut_slice(
    cx: int, cy: int, r_outer: int, r_inner: int, start: float, end: float, color: str
) -> str:
    x1 = cx + r_outer * math.cos(start)
    y1 = cy + r_outer * math.sin(start)
    x2 = cx + r_outer * math.cos(end)
    y2 = cy + r_outer * math.sin(end)
    x3 = cx + r_inner * math.cos(end)
    y3 = cy + r_inner * math.sin(end)
    x4 = cx + r_inner * math.cos(start)
    y4 = cy + r_inner * math.sin(start)
    large_arc = 1 if (end - start) > math.pi else 0
    path = (
        f"M {x1:.1f} {y1:.1f} "
        f"A {r_outer} {r_outer} 0 {large_arc} 1 {x2:.1f} {y2:.1f} "
        f"L {x3:.1f} {y3:.1f} "
        f"A {r_inner} {r_inner} 0 {large_arc} 0 {x4:.1f} {y4:.1f} Z"
    )
    return f'<path d="{path}" fill="{color}"/>'


def _axes(x0: int, y0: int, w: int, h: int, max_v: float) -> str:
    # Light horizontal gridlines at 25/50/75/100% with value labels
    parts = []
    for frac in (0, 0.25, 0.5, 0.75, 1.0):
        y = y0 + h - h * frac
        parts.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x0 + w}" y2="{y:.1f}" stroke="{GRID}" stroke-width="1"/>')
        label_v = max_v * frac
        parts.append(
            f'<text x="{x0 - 8}" y="{y + 4:.1f}" text-anchor="end" '
            f'fill="{MUTED}" font-family="sans-serif" font-size="10">{_fmt_value(label_v)}</text>'
        )
    return "".join(parts)


def _fmt_value(v: float) -> str:
    if v >= 1_000_000:
        return f"{v / 1_000_000:.1f}M"
    if v >= 10_000:
        return f"{v / 1_000:.1f}K"
    if v == int(v):
        return f"{int(v)}"
    return f"{v:.1f}"


def _empty_chart_text(width: int, height: int) -> str:
    return (
        f'<text x="{width // 2}" y="{height // 2}" text-anchor="middle" '
        f'fill="{MUTED}" font-family="sans-serif" font-size="14">(no data)</text>'
    )


def render_status_card(
    title: str, sections: list[dict[str, Any]], options: dict[str, Any] | None = None
) -> str:
    """Render a status briefing card as SVG.

    Layout: title bar at top, optional subtitle, then one box per section with
    a heading and rows of (label / value / colored status pill).
    """
    options = options or {}
    width = int(options.get("width") or 800)
    subtitle = str(options.get("subtitle") or "")
    footer = str(options.get("footer") or "")

    header_h = 60 if not subtitle else 78
    section_padding = 14
    row_h = 26
    section_header_h = 30
    section_gap = 12

    section_heights = [
        section_header_h + section_padding * 2 + row_h * len(s.get("rows") or [])
        for s in sections
    ]
    body_h = sum(section_heights) + section_gap * max(len(sections) - 1, 0)
    footer_h = 28 if footer else 0
    height = header_h + 16 + body_h + footer_h + 16

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}">',
        f'<rect width="{width}" height="{height}" fill="{BG}"/>',
        f'<rect x="0" y="0" width="{width}" height="{header_h}" fill="#1e293b"/>',
        f'<text x="24" y="38" fill="{FG}" font-family="sans-serif" '
        f'font-size="20" font-weight="700">{_esc(title)}</text>',
    ]
    if subtitle:
        parts.append(
            f'<text x="24" y="62" fill="{MUTED}" font-family="sans-serif" '
            f'font-size="12">{_esc(subtitle)}</text>'
        )

    y = header_h + 16
    for i, section in enumerate(sections):
        heading = section.get("heading", "")
        rows = section.get("rows") or []
        section_h = section_header_h + section_padding * 2 + row_h * len(rows)
        parts.append(
            f'<rect x="16" y="{y}" width="{width - 32}" height="{section_h}" '
            f'fill="#1e293b" rx="6"/>'
        )
        parts.append(
            f'<text x="28" y="{y + 22}" fill="{FG}" font-family="sans-serif" '
            f'font-size="13" font-weight="600">{_esc(heading)}</text>'
        )
        row_y = y + section_header_h + section_padding
        for row in rows:
            label = _esc(row.get("label", ""))
            value = _esc(row.get("value", ""))
            status = str(row.get("status", "")).lower()
            pill_color = STATUS_PILL.get(status, PILL_FALLBACK)
            parts.append(
                f'<circle cx="32" cy="{row_y + 8}" r="5" fill="{pill_color}"/>'
            )
            parts.append(
                f'<text x="46" y="{row_y + 12}" fill="{FG}" font-family="sans-serif" '
                f'font-size="12">{label}</text>'
            )
            parts.append(
                f'<text x="{width - 28}" y="{row_y + 12}" fill="{FG}" font-family="sans-serif" '
                f'font-size="12" font-weight="600" text-anchor="end">{value}</text>'
            )
            row_y += row_h
        y += section_h + section_gap

    if footer:
        parts.append(
            f'<text x="24" y="{height - 14}" fill="{MUTED}" font-family="sans-serif" '
            f'font-size="10">{_esc(footer)}</text>'
        )

    parts.append("</svg>")
    return "".join(parts)
