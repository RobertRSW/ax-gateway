"""Wrap MCP tools as LangChain BaseTools for use in LangGraph ToolNode.

Bridges the protocol gap: MCP describes tools via JSON Schema, LangChain
expects Pydantic models. This adapter spawns each configured MCP server
(via `McpStdioClient`), calls `tools/list` once, and produces a list of
`BaseTool` instances the LangGraph `ToolNode` can dispatch to.

Usage from langgraph_bridge.py:

    from ax_cli.runtimes.mcp_servers.langchain_adapter import load_mcps_from_env
    extra_tools, mcp_clients = load_mcps_from_env()
    tools = _default_tools() + extra_tools
    # ... build ToolNode(tools, wrap_tool_call=_make_security_wrap(workdir))
    # ... atexit.register(lambda: [c.close() for c in mcp_clients])

Configuration comes from a single env var `AX_BRIDGE_MCP_SERVERS`, a JSON
object mapping server label → spawn config:

    {
      "report_gen": {
        "command": ["python", "-m", "ax_cli.runtimes.mcp_servers.report_gen"],
        "env": {"AX_REPORT_GEN_DB_KIND": "postgres"}
      },
      "svg_viz": {
        "command": ["python", "-m", "ax_cli.runtimes.mcp_servers.svg_viz"]
      }
    }

The env value can be inline JSON or a `@/path/to/config.json` reference
so the deploy story can keep DSN secrets in a mode-600 file rather than
the daemon's env.

Tool naming: when multiple MCPs are loaded, tool names are prefixed with
the server label to avoid collisions (e.g. `report_gen__db_query`).
Single-server setups skip the prefix for cleaner LLM prompts.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

from .mcp_client import McpStdioClient, McpToolError, McpToolSpec


def load_mcps_from_env(
    *,
    config_env: str = "AX_BRIDGE_MCP_SERVERS",
    debug: bool = False,
) -> tuple[list[Any], list[McpStdioClient]]:
    """Read the MCP config from env, spawn servers, return (tools, clients).

    Returns `([], [])` when the env var is unset or empty so the bridge
    behaves identically to the pre-MCP state when no MCPs are configured.

    Caller is responsible for `client.close()` on the returned clients at
    bridge shutdown (e.g. via `atexit.register`).
    """
    raw = (os.environ.get(config_env) or "").strip()
    if not raw:
        return [], []

    try:
        config = _load_config(raw)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        _stderr(f"[mcp-adapter] failed to parse {config_env}: {exc}")
        return [], []

    if not isinstance(config, dict) or not config:
        return [], []

    tools: list[Any] = []
    clients: list[McpStdioClient] = []
    prefix_names = len(config) > 1

    for label, server_cfg in config.items():
        if not isinstance(server_cfg, dict):
            _stderr(f"[mcp-adapter] {label}: config must be an object; got {type(server_cfg).__name__}")
            continue
        command = server_cfg.get("command")
        if not isinstance(command, list) or not command:
            _stderr(f"[mcp-adapter] {label}: missing or invalid 'command' (must be a non-empty list)")
            continue
        env_override = server_cfg.get("env") or None
        cwd = server_cfg.get("cwd") or None
        timeout_s = float(server_cfg.get("timeout_s") or 30.0)

        client = McpStdioClient(
            command=command,
            env=env_override,
            cwd=cwd,
            timeout_s=timeout_s,
            debug=debug,
        )
        try:
            client.start()
            client.initialize()
            specs = client.list_tools()
        except Exception as exc:  # noqa: BLE001 — bridge degrades gracefully
            _stderr(f"[mcp-adapter] {label}: failed to initialize: {exc}")
            client.close()
            continue

        if not specs:
            _stderr(f"[mcp-adapter] {label}: server exposes no tools")
            client.close()
            continue

        clients.append(client)
        for spec in specs:
            tool_name = f"{label}__{spec.name}" if prefix_names else spec.name
            tools.append(_make_langchain_tool(client, spec, tool_name=tool_name))
        _stderr(f"[mcp-adapter] {label}: registered {len(specs)} tool(s) ({', '.join(s.name for s in specs)})")

    return tools, clients


def _load_config(raw: str) -> Any:
    """Inline JSON OR `@/path/to/file.json`."""
    if raw.startswith("@"):
        path = raw[1:].strip()
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    return json.loads(raw)


def _make_langchain_tool(client: McpStdioClient, spec: McpToolSpec, *, tool_name: str) -> Any:
    """Construct a LangChain StructuredTool that routes calls to `client`."""
    from langchain_core.tools import StructuredTool

    args_schema = _json_schema_to_pydantic(spec.input_schema, tool_name)

    def _invoke(**kwargs: Any) -> str:
        try:
            return client.call_tool(spec.name, kwargs)
        except McpToolError as exc:
            return f"tool {spec.name} returned error: {exc}"
        except Exception as exc:  # noqa: BLE001
            return f"tool {spec.name} dispatch failed: {exc}"

    return StructuredTool.from_function(
        func=_invoke,
        name=tool_name,
        description=spec.description or f"MCP tool {spec.name}",
        args_schema=args_schema,
    )


def _json_schema_to_pydantic(schema: dict[str, Any], tool_name: str) -> Any:
    """Build a Pydantic v2 model from an MCP tool's JSON Schema.

    Best-effort: we handle the schema shapes our own MCPs emit (object with
    `properties` + `required`, primitive types, arrays of objects). Anything
    unrecognized falls back to a permissive `Any` field so the LangGraph
    layer doesn't reject otherwise-valid tool calls.
    """
    from pydantic import BaseModel, Field, create_model

    properties = schema.get("properties") or {}
    required = set(schema.get("required") or [])

    fields: dict[str, tuple[Any, Any]] = {}
    for prop_name, prop_schema in properties.items():
        if not isinstance(prop_schema, dict):
            field_type: Any = Any
        else:
            field_type = _json_type_to_python(prop_schema)
        description = (prop_schema or {}).get("description", "") if isinstance(prop_schema, dict) else ""
        if prop_name in required:
            fields[prop_name] = (field_type, Field(..., description=description))
        else:
            fields[prop_name] = (field_type, Field(default=None, description=description))

    if not fields:
        # LangChain requires *some* args_schema; return a no-arg empty model.
        return create_model(f"{tool_name}__Args", __base__=BaseModel)

    return create_model(f"{tool_name}__Args", __base__=BaseModel, **fields)


def _json_type_to_python(schema: dict[str, Any]) -> Any:
    """Map a JSON-Schema fragment to a Python type hint for Pydantic."""
    t = schema.get("type")
    if t == "string":
        return str
    if t == "number":
        return float
    if t == "integer":
        return int
    if t == "boolean":
        return bool
    if t == "array":
        items = schema.get("items")
        if isinstance(items, dict):
            inner = _json_type_to_python(items)
            return list[inner]  # type: ignore[valid-type]
        return list
    if t == "object":
        return dict
    return Any


def _stderr(message: str) -> None:
    print(message, file=sys.stderr, flush=True)
