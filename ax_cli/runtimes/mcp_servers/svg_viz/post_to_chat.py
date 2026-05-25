"""svg_viz post_to_chat tool — uploads an SVG via the AX context API and
posts an ``apps signal`` message so paxai's chat renders it as a folded
signal card (Context Explorer panel), not as a code block.

The render-as-card path is documented in
``docs/mcp-app-signal-adapter.md``:

1. ``client.set_context(space_id, key, value=svg_string)`` — stores the SVG
   as a context resource keyed for retrieval.
2. ``client.send_message(space_id, body, metadata={...ui.widget...})`` —
   posts a message whose ``metadata.ui.widget.resource_uri`` points at
   ``ui://context/explorer`` and whose ``metadata.ui.widget.initial_data``
   carries the context key. paxai's frontend renders the message as a
   folded signal card; clicking it opens the Context Explorer panel which
   resolves the key and renders the SVG.

Identity: this tool uses ``ax_cli.config.get_client()`` which reads the
agent's credentials from the env Gateway passes to the bridge subprocess
(``AX_TOKEN_FILE`` / ``AX_BASE_URL`` / ``AX_AGENT_ID`` / ``AX_SPACE_ID``).
No new credential handling here.

Failure mode: returns a structured error dict instead of raising. The
calling LLM gets a readable error and can recover; the rest of the
agent loop continues. We deliberately do NOT fall back to "post the
SVG inline" — that's the failure case this tool exists to prevent.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from typing import Any

CONTEXT_KEY_PREFIX = "svg_viz"
"""Prefix on auto-generated context keys so the Context Explorer can group them."""

DEFAULT_TTL_S = 60 * 60 * 24 * 7
"""Auto-expire SVG context entries after 7 days (override via the ttl arg)."""

APP_KEY = "context"
RESOURCE_URI = "ui://context/explorer"
"""Per docs/mcp-app-signal-adapter.md — SVG signals open the Context Explorer panel."""


def _make_context_key(title: str) -> str:
    """Build a stable-shaped context key for a posted SVG.

    Pattern: ``svg_viz:<timestamp>:<random>:<title-slug>``. The random
    component keeps simultaneous posts from colliding; the title slug
    is a human-readable hint for anyone browsing context.
    """
    ts = int(time.time())
    short_id = uuid.uuid4().hex[:8]
    slug = "".join(c if c.isalnum() else "-" for c in title.lower())[:48]
    slug = slug.strip("-") or "svg"
    return f"{CONTEXT_KEY_PREFIX}:{ts}:{short_id}:{slug}"


def _build_signal_metadata(
    *,
    context_key: str,
    title: str,
    summary: str,
    space_id: str,
    agent_name: str | None,
    severity: str = "info",
) -> dict[str, Any]:
    """Construct the ``metadata.ui.widget`` + ``metadata.ui.cards`` payload.

    Mirrors the shape ``axctl apps signal context`` emits (see
    ``ax_cli/commands/apps.py::_build_signal_metadata``) so the
    transcript-signal rendering path treats SVG posts identically to
    CLI-authored signals.
    """
    tool_call_id = f"svg_viz_post:{uuid.uuid4().hex[:12]}"
    return {
        "top_level_ingress": False,
        "signal_only": True,
        "app_signal": {
            "source": "svg_viz_post_to_chat",
            "app": APP_KEY,
            "action": "get",
            "signal_only": True,
            "severity": severity,
        },
        "ui": {
            "widget": {
                "resource_uri": RESOURCE_URI,
                "title": title,
                "initial_data": {
                    "context_key": context_key,
                    "space_id": space_id,
                    "kind": "svg",
                },
            },
            "cards": [
                {
                    "type": "context_artifact",
                    "title": title,
                    "summary": summary,
                    "payload": {
                        "context_key": context_key,
                        "kind": "svg",
                        "source": "svg_viz_post_to_chat",
                        "author_agent": agent_name,
                        "tool_call_id": tool_call_id,
                    },
                }
            ],
        },
        "tool_call_id": tool_call_id,
    }


def post_svg_to_chat(
    svg: str,
    title: str,
    summary: str = "",
    *,
    space_id: str | None = None,
    ttl_seconds: int = DEFAULT_TTL_S,
) -> dict[str, Any]:
    """Upload `svg` to the context API and post a signal-card message.

    Returns a dict with ``message_id``, ``context_key``, ``space_id``,
    and ``title``. On failure returns ``{"error": "...", "code": "..."}``.

    The caller (typically the LLM through MCP) supplies the SVG string
    produced by ``chart`` or ``status_card``, a title, and an optional
    summary. The agent's AX identity comes from the env Gateway already
    set on the bridge subprocess; we don't override it.
    """
    if not isinstance(svg, str) or not svg.strip().startswith("<svg"):
        return {
            "error": "svg argument must be a string starting with '<svg'",
            "code": "INVALID_SVG",
        }
    if not isinstance(title, str) or not title.strip():
        return {"error": "title is required", "code": "MISSING_TITLE"}

    try:
        from ax_cli.config import get_client, resolve_space_id
    except Exception as exc:  # noqa: BLE001 - import is the failure
        return {
            "error": f"ax_cli not available in subprocess env: {exc}",
            "code": "AX_CLI_UNAVAILABLE",
        }

    try:
        client = get_client()
    except Exception as exc:  # noqa: BLE001 - credential resolution
        return {
            "error": f"could not build AX client: {exc}",
            "code": "NO_CREDENTIALS",
        }

    try:
        resolved_space = resolve_space_id(client, explicit=space_id)
    except Exception as exc:  # noqa: BLE001 - space resolution
        return {"error": f"could not resolve space: {exc}", "code": "NO_SPACE"}
    if not resolved_space:
        return {
            "error": "space_id is required (set AX_SPACE_ID or pass space_id)",
            "code": "NO_SPACE",
        }

    context_key = _make_context_key(title)
    try:
        client.set_context(resolved_space, context_key, svg, ttl=ttl_seconds)
    except Exception as exc:  # noqa: BLE001 - HTTP errors etc
        return {
            "error": f"context upload failed: {exc}",
            "code": "CONTEXT_UPLOAD_FAILED",
            "context_key": context_key,
        }

    agent_name = os.environ.get("AX_AGENT_NAME") or os.environ.get("AX_GATEWAY_AGENT_NAME")
    metadata = _build_signal_metadata(
        context_key=context_key,
        title=title,
        summary=summary or title,
        space_id=resolved_space,
        agent_name=agent_name,
    )

    message_body = summary.strip() or f"📊 {title}"

    try:
        result = client.send_message(
            resolved_space,
            message_body,
            metadata=metadata,
            message_type="system",
        )
    except Exception as exc:  # noqa: BLE001 - HTTP errors etc
        return {
            "error": f"signal message post failed: {exc}",
            "code": "MESSAGE_POST_FAILED",
            "context_key": context_key,
        }

    message = result.get("message", result) if isinstance(result, dict) else {}
    return {
        "ok": True,
        "message_id": message.get("id") or message.get("message_id"),
        "context_key": context_key,
        "space_id": resolved_space,
        "title": title,
        "resource_uri": RESOURCE_URI,
    }


# ── MCP tool wrapper ──────────────────────────────────────────────────────


POST_TO_CHAT_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "svg": {
            "type": "string",
            "description": (
                "Full SVG document string (must start with '<svg'). Typically "
                "the value returned by the chart or status_card tool's 'svg' field."
            ),
        },
        "title": {
            "type": "string",
            "description": (
                "Human-readable title shown on the folded signal card and on "
                "the opened panel (e.g., 'CENTCOM Ammo Status')."
            ),
        },
        "summary": {
            "type": "string",
            "description": (
                "One-sentence description of what the SVG shows. Used as the "
                "card subtitle AND as the chat message body. Defaults to the title."
            ),
        },
        "space_id": {
            "type": "string",
            "description": (
                "Optional: target aX space. Defaults to AX_SPACE_ID from env (the space the agent is bound to)."
            ),
        },
        "ttl_seconds": {
            "type": "number",
            "description": (
                "Optional: how long the SVG context entry lives. Default 604800 "
                "(7 days). Use a smaller value (e.g., 3600 for 1 hour) for "
                "ephemeral demos."
            ),
        },
    },
    "required": ["svg", "title"],
}


def _handle_post_to_chat(arguments: dict[str, Any]) -> dict[str, Any]:
    result = post_svg_to_chat(
        svg=arguments.get("svg") or "",
        title=arguments.get("title") or "",
        summary=arguments.get("summary") or "",
        space_id=arguments.get("space_id"),
        ttl_seconds=int(arguments.get("ttl_seconds") or DEFAULT_TTL_S),
    )
    return {"content": [{"type": "text", "text": json.dumps(result)}]}
