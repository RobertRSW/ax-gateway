# svg_viz MCP server

Pure-Python SVG generation. Two tools: `chart` (bar / line / donut) and
`status_card` (briefing-card layout with ok / warning / alert pills).
Zero runtime deps beyond Python stdlib.

## Tools

### `chart(type, data, options)`

```json
{
  "type": "bar",
  "data": [
    {"label": "CENTCOM", "value": 12500},
    {"label": "INDOPACOM", "value": 8200},
    {"label": "EUCOM", "value": 4100}
  ],
  "options": {
    "title": "Ammo Stockpile by Theater (rounds)",
    "color_scheme": "default",
    "width": 600,
    "height": 400
  }
}
```

Returns:

```json
{"svg": "<svg xmlns=\"http://www.w3.org/2000/svg\" ...>...</svg>"}
```

Supported `type` values: `bar`, `line`, `donut`. Supported `color_scheme`
values: `default` (blue-leaning), `alert` (red-leaning), `ok` (green-leaning).

### `status_card(title, sections, options)`

```json
{
  "title": "CENTCOM Status — 2026-05-24",
  "sections": [
    {
      "heading": "Ammo Stockpile",
      "rows": [
        {"label": "5.56mm", "value": "120,000 rounds", "status": "ok"},
        {"label": "155mm artillery", "value": "8,400 shells", "status": "warning"},
        {"label": "Javelin", "value": "350 missiles", "status": "alert"}
      ]
    }
  ],
  "options": {
    "subtitle": "Generated 2026-05-24 14:30 UTC",
    "footer": "Synthetic data — non-classified",
    "width": 800
  }
}
```

Returns the same `{"svg": "..."}` shape. Each row gets a colored status
pill: green for `ok`, amber for `warning`, red for `alert`, gray for
anything else.

## Why no svgwrite / matplotlib

SVG is generated as straight strings — keeps the dep footprint at zero,
gives complete control of the demo aesthetic. If chart variety grows past
~5 types this should be revisited per design doc §6 trade-off table.

## Inline rendering in aX chat

The SVG content goes through the apps signal adapter (`docs/mcp-app-signal-adapter.md`):
upload the SVG via `axctl context add`, post a message with
`metadata.ui.widget` referencing the returned context key, the frontend
renders the message as a folded signal card.
