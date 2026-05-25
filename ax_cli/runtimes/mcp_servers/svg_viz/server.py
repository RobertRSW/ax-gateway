"""svg_viz MCP server entrypoint."""

from __future__ import annotations

import os

from ..stdio_server import ServerConfig, serve
from .tools import build_tools

SERVER_NAME = "ax-svg-viz"
SERVER_VERSION = "0.1.0"
INSTRUCTIONS = (
    "SVG visualization tools. Two tools available:\n"
    "- chart(type, data, options): bar / line / donut charts.\n"
    "- status_card(title, sections, options): status briefing card with "
    "ok/warning/alert pills.\n"
    "Both tools return a JSON object with key 'svg' containing the full "
    "SVG document string. To render in aX chat, upload the SVG via "
    "axctl context add and post a message with a metadata.ui.widget "
    "signal referencing the context key."
)


def main() -> None:
    config = ServerConfig(
        name=SERVER_NAME,
        version=SERVER_VERSION,
        instructions=INSTRUCTIONS,
        tools=build_tools(),
        debug=os.environ.get("AX_MCP_DEBUG", "").lower() in {"1", "true", "yes", "on"},
    )
    serve(config)


if __name__ == "__main__":
    main()
