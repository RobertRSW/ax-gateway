"""Tests for the shared MCP stdio server loop (JSON-RPC dispatch)."""

from __future__ import annotations

import io
import json
import sys
from typing import Any

import pytest

from ax_cli.runtimes.mcp_servers.stdio_server import (
    PROTOCOL_VERSION,
    ServerConfig,
    ToolSpec,
    serve,
)


def _make_config(tools: list[ToolSpec] | None = None) -> ServerConfig:
    return ServerConfig(
        name="test-server",
        version="0.1.0",
        instructions="Test instructions",
        tools=tools or [],
    )


def _drive(config: ServerConfig, requests: list[dict[str, Any]], capsys) -> list[dict[str, Any]]:
    """Pipe `requests` through `serve()` via stdin redirection; parse stdout responses."""
    stdin_text = "\n".join(json.dumps(r) for r in requests) + "\n"
    prior_stdin = sys.stdin
    sys.stdin = io.StringIO(stdin_text)
    try:
        serve(config)
    finally:
        sys.stdin = prior_stdin
    captured = capsys.readouterr()
    return [json.loads(line) for line in captured.out.strip().splitlines() if line.strip()]


def test_initialize_returns_protocol_and_server_info(capsys):
    responses = _drive(_make_config(), [{"jsonrpc": "2.0", "id": 1, "method": "initialize"}], capsys)
    assert len(responses) == 1
    result = responses[0]["result"]
    assert result["protocolVersion"] == PROTOCOL_VERSION
    assert result["serverInfo"]["name"] == "test-server"
    assert result["serverInfo"]["version"] == "0.1.0"
    assert result["capabilities"] == {"tools": {}}
    assert result["instructions"] == "Test instructions"


def test_tools_list_returns_registered_tools(capsys):
    config = _make_config(tools=[
        ToolSpec(
            name="echo",
            description="Echo back",
            input_schema={"type": "object", "properties": {"text": {"type": "string"}}},
            handler=lambda args: {"content": [{"type": "text", "text": args.get("text", "")}]},
        ),
    ])
    responses = _drive(config, [{"jsonrpc": "2.0", "id": 1, "method": "tools/list"}], capsys)
    tools = responses[0]["result"]["tools"]
    assert len(tools) == 1
    assert tools[0]["name"] == "echo"
    assert tools[0]["description"] == "Echo back"


def test_tool_call_dispatches_to_handler(capsys):
    config = _make_config(tools=[
        ToolSpec(
            name="echo",
            description="",
            input_schema={"type": "object"},
            handler=lambda args: {"content": [{"type": "text", "text": f"got: {args.get('text')}"}]},
        ),
    ])
    responses = _drive(
        config,
        [{"jsonrpc": "2.0", "id": 1, "method": "tools/call",
          "params": {"name": "echo", "arguments": {"text": "hi"}}}],
        capsys,
    )
    assert responses[0]["result"]["content"][0]["text"] == "got: hi"


def test_unknown_tool_returns_method_not_found_error(capsys):
    responses = _drive(
        _make_config(),
        [{"jsonrpc": "2.0", "id": 1, "method": "tools/call",
          "params": {"name": "no_such_tool", "arguments": {}}}],
        capsys,
    )
    assert "error" in responses[0]
    assert responses[0]["error"]["code"] == -32601
    assert "Unknown tool" in responses[0]["error"]["message"]


def test_handler_exception_returned_as_iserror_text(capsys):
    def _boom(args: dict[str, Any]) -> dict[str, Any]:
        raise ValueError("kaboom")

    config = _make_config(tools=[
        ToolSpec(name="bad", description="", input_schema={"type": "object"}, handler=_boom),
    ])
    responses = _drive(
        config,
        [{"jsonrpc": "2.0", "id": 1, "method": "tools/call",
          "params": {"name": "bad", "arguments": {}}}],
        capsys,
    )
    result = responses[0]["result"]
    assert result["isError"] is True
    assert "kaboom" in result["content"][0]["text"]


def test_unknown_method_returns_error(capsys):
    responses = _drive(
        _make_config(),
        [{"jsonrpc": "2.0", "id": 1, "method": "nonsense/method"}],
        capsys,
    )
    assert responses[0]["error"]["code"] == -32601


def test_notifications_are_ignored(capsys):
    # No 'id' field = notification per JSON-RPC 2.0; loop should skip it
    responses = _drive(
        _make_config(),
        [
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            {"jsonrpc": "2.0", "id": 1, "method": "initialize"},
        ],
        capsys,
    )
    # Only the initialize gets a response
    assert len(responses) == 1
    assert responses[0]["id"] == 1


def test_ping_returns_empty_result(capsys):
    responses = _drive(_make_config(), [{"jsonrpc": "2.0", "id": 1, "method": "ping"}], capsys)
    assert responses[0]["result"] == {}


@pytest.mark.parametrize("method,result_key", [
    ("resources/list", "resources"),
    ("resources/templates/list", "resourceTemplates"),
    ("prompts/list", "prompts"),
])
def test_empty_list_endpoints(capsys, method, result_key):
    responses = _drive(_make_config(), [{"jsonrpc": "2.0", "id": 1, "method": method}], capsys)
    assert responses[0]["result"] == {result_key: []}


def test_bad_json_lines_are_skipped(capsys):
    """The loop should keep going past malformed JSON instead of crashing."""
    stdin_text = "not json\n" + json.dumps({"jsonrpc": "2.0", "id": 1, "method": "ping"}) + "\n"
    prior_stdin = sys.stdin
    sys.stdin = io.StringIO(stdin_text)
    try:
        serve(_make_config())
    finally:
        sys.stdin = prior_stdin
    captured = capsys.readouterr()
    responses = [json.loads(line) for line in captured.out.strip().splitlines() if line.strip()]
    assert len(responses) == 1
    assert responses[0]["id"] == 1


# ── Malformed-params robustness (eugeneluzgin PR #100 review) ───────────────


def test_tool_call_with_string_params_returns_error_not_crash(capsys):
    """A non-object `params` must not crash the server (params.get would raise)."""
    config = _make_config(tools=[
        ToolSpec(name="echo", description="", input_schema={"type": "object"},
                 handler=lambda a: {"content": [{"type": "text", "text": "ok"}]}),
    ])
    responses = _drive(
        config,
        [{"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": "not-an-object"}],
        capsys,
    )
    # Normalized to {} upstream, so it falls through to "Unknown tool: None"
    assert "error" in responses[0]
    assert responses[0]["error"]["code"] in (-32601, -32602)


def test_tool_call_with_list_params_returns_error_not_crash(capsys):
    config = _make_config(tools=[
        ToolSpec(name="echo", description="", input_schema={"type": "object"},
                 handler=lambda a: {"content": [{"type": "text", "text": "ok"}]}),
    ])
    responses = _drive(
        config,
        [{"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": [1, 2, 3]}],
        capsys,
    )
    assert "error" in responses[0]


def test_tool_call_with_non_dict_arguments_returns_error(capsys):
    """`arguments` must be an object; a string/list there should error cleanly."""
    config = _make_config(tools=[
        ToolSpec(name="echo", description="", input_schema={"type": "object"},
                 handler=lambda a: {"content": [{"type": "text", "text": "ok"}]}),
    ])
    responses = _drive(
        config,
        [{"jsonrpc": "2.0", "id": 1, "method": "tools/call",
          "params": {"name": "echo", "arguments": "not-an-object"}}],
        capsys,
    )
    assert responses[0]["error"]["code"] == -32602
    assert "arguments" in responses[0]["error"]["message"]


def test_malformed_request_does_not_kill_the_loop(capsys):
    """A request that triggers an unexpected error must not stop later requests."""
    config = _make_config(tools=[
        ToolSpec(name="echo", description="", input_schema={"type": "object"},
                 handler=lambda a: {"content": [{"type": "text", "text": "ok"}]}),
    ])
    responses = _drive(
        config,
        [
            {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": "garbage"},
            {"jsonrpc": "2.0", "id": 2, "method": "ping"},  # must still be served
        ],
        capsys,
    )
    # Two responses: the error for #1 AND the successful ping for #2
    by_id = {r.get("id"): r for r in responses}
    assert 1 in by_id
    assert 2 in by_id
    assert by_id[2]["result"] == {}
