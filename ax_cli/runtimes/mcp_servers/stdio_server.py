"""Shared JSON-RPC stdio MCP server loop.

Hand-rolled to match the pattern in `ax_cli/commands/channel.py` — no `mcp`
PyPI dependency. The protocol version (`2025-11-25`) is identical so any
MCP client that talks to `ax channel` also talks to these servers.

Servers built on top of this base define `ToolSpec` entries and a single
synchronous `dispatch(name, arguments)` handler. The loop covers
`initialize` / `tools/list` / `tools/call` / `ping` / empty list endpoints
and surfaces dispatch exceptions as JSON-RPC errors.

Synchronous on purpose: report_gen and svg_viz are pure request/response —
they don't push notifications back like channel.py does. A blocking stdin
read loop keeps the code under ~100 lines.
"""

from __future__ import annotations

import json
import sys
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable

PROTOCOL_VERSION = "2025-11-25"


@dataclass
class ToolSpec:
    """One MCP tool: name, human description, JSON Schema, handler."""

    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[[dict[str, Any]], dict[str, Any]]


@dataclass
class ServerConfig:
    """Per-server identity + tool table for the shared loop."""

    name: str
    version: str
    instructions: str
    tools: list[ToolSpec] = field(default_factory=list)
    debug: bool = False


def _log(config: ServerConfig, message: str) -> None:
    if not config.debug:
        return
    print(f"[{config.name}] {message}", file=sys.stderr, flush=True)


def _write(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def _send_response(request_id: Any, result: dict[str, Any]) -> None:
    _write({"jsonrpc": "2.0", "id": request_id, "result": result})


def _send_error(request_id: Any, code: int, message: str) -> None:
    _write({"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}})


def _tool_text_response(text: str, is_error: bool = False) -> dict[str, Any]:
    payload: dict[str, Any] = {"content": [{"type": "text", "text": text}]}
    if is_error:
        payload["isError"] = True
    return payload


def _handle_tool_call(config: ServerConfig, request_id: Any, params: dict[str, Any]) -> None:
    # `params` and `params["arguments"]` must be objects. A malformed client
    # could send a string/list/null; guard before calling .get() so a bad
    # request returns a JSON-RPC error instead of crashing the server loop.
    if not isinstance(params, dict):
        _send_error(request_id, -32602, "Invalid params: expected an object")
        return
    name = params.get("name")
    arguments = params.get("arguments") or {}
    if not isinstance(arguments, dict):
        _send_error(request_id, -32602, "Invalid params: 'arguments' must be an object")
        return
    spec = next((t for t in config.tools if t.name == name), None)
    if spec is None:
        _send_error(request_id, -32601, f"Unknown tool: {name}")
        return
    try:
        result = spec.handler(arguments)
    except Exception as exc:  # pragma: no cover - handler-specific
        _log(config, f"tool {name} raised: {exc}\n{traceback.format_exc()}")
        _send_response(request_id, _tool_text_response(f"{name} failed: {exc}", is_error=True))
        return
    _send_response(request_id, result)


def _handle_request(config: ServerConfig, request: dict[str, Any]) -> None:
    request_id = request.get("id")
    method = request.get("method")
    params = request.get("params")
    # Normalize params to a dict — JSON-RPC allows omitting it, but a malformed
    # client could send a non-object. Downstream handlers assume a dict.
    if not isinstance(params, dict):
        params = {}

    if method == "initialize":
        _send_response(
            request_id,
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": config.name, "version": config.version},
                "instructions": config.instructions,
            },
        )
    elif method == "tools/list":
        _send_response(
            request_id,
            {
                "tools": [
                    {"name": t.name, "description": t.description, "inputSchema": t.input_schema} for t in config.tools
                ]
            },
        )
    elif method == "tools/call":
        _handle_tool_call(config, request_id, params)
    elif method == "resources/list":
        _send_response(request_id, {"resources": []})
    elif method == "resources/templates/list":
        _send_response(request_id, {"resourceTemplates": []})
    elif method == "prompts/list":
        _send_response(request_id, {"prompts": []})
    elif method == "ping":
        _send_response(request_id, {})
    else:
        _send_error(request_id, -32601, f"Method not found: {method}")


def serve(config: ServerConfig) -> None:
    """Block on stdin, dispatch JSON-RPC requests until EOF."""
    _log(config, f"starting ({len(config.tools)} tools)")
    for line in sys.stdin:
        raw = line.strip()
        if not raw:
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            _log(config, f"bad JSON: {exc}")
            continue
        if not isinstance(payload, dict):
            continue
        method = payload.get("method", "")
        # Notifications (no id) — channel.py sets self.initialized on
        # notifications/initialized; these servers don't need that state.
        if "id" not in payload:
            _log(config, f"notification: {method}")
            continue
        # A single malformed request must never crash the whole server loop.
        # Per-request handlers already convert known-bad input into JSON-RPC
        # errors; this is the backstop for anything they miss.
        try:
            _handle_request(config, payload)
        except Exception as exc:  # noqa: BLE001 - daemon resilience
            _log(config, f"request handling failed: {exc}\n{traceback.format_exc()}")
            _send_error(payload.get("id"), -32603, f"Internal error: {exc}")
    _log(config, "stdin closed; shutting down")
