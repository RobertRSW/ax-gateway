"""Tests for the svg_viz post_to_chat tool.

post_to_chat lazy-imports `ax_cli.config.get_client` + `resolve_space_id`
from inside the function body, so we test the seam by injecting a fake
`ax_cli.config` module into ``sys.modules`` before the call. This avoids
pulling in the real `ax_cli.config` (which depends on typer + httpx and
isn't necessary for these unit tests).

The full live path gets exercised on the VM separately — see the
corresponding work log.
"""

from __future__ import annotations

import json
import sys
import types
from unittest.mock import MagicMock

import pytest

from ax_cli.runtimes.mcp_servers.svg_viz.post_to_chat import (
    CONTEXT_KEY_PREFIX,
    RESOURCE_URI,
    _build_signal_metadata,
    _handle_post_to_chat,
    _make_context_key,
    post_svg_to_chat,
)
from ax_cli.runtimes.mcp_servers.svg_viz.tools import build_tools

# ── pure unit tests (no client, no env) ────────────────────────────────────


def test_build_tools_includes_post_to_chat():
    tools = build_tools()
    names = [t.name for t in tools]
    assert names == ["chart", "status_card", "post_to_chat"]


def test_make_context_key_includes_prefix_and_slug():
    key = _make_context_key("CENTCOM Status Report")
    assert key.startswith(f"{CONTEXT_KEY_PREFIX}:")
    assert key.endswith(":centcom-status-report")
    parts = key.split(":")
    assert len(parts) == 4  # svg_viz : ts : random : slug
    assert parts[1].isdigit()  # timestamp
    assert len(parts[2]) == 8  # short random id


def test_make_context_key_handles_punctuation_safely():
    key = _make_context_key("/etc/passwd & other ../surprises!")
    parts = key.split(":")
    slug = parts[3]
    assert "/" not in slug
    assert ".." not in slug
    assert "&" not in slug


def test_make_context_key_truncates_long_titles():
    key = _make_context_key("a" * 200)
    slug = key.split(":")[3]
    assert len(slug) <= 48


def test_make_context_key_handles_empty_title_safely():
    key = _make_context_key("!!!")
    slug = key.split(":")[3]
    assert slug  # not empty


def test_build_signal_metadata_shape_matches_apps_signal_path():
    """Mirrors the metadata shape from axctl apps signal context — see
    docs/mcp-app-signal-adapter.md §Identity Checks for the contract."""
    metadata = _build_signal_metadata(
        context_key="svg_viz:1:abc:test",
        title="Test Title",
        summary="Test summary",
        space_id="space-1",
        agent_name="test-agent",
    )

    # Top-level signal flags
    assert metadata["top_level_ingress"] is False
    assert metadata["signal_only"] is True

    # app_signal block
    assert metadata["app_signal"]["source"] == "svg_viz_post_to_chat"
    assert metadata["app_signal"]["app"] == "context"
    assert metadata["app_signal"]["signal_only"] is True

    # ui.widget (the actual render-as-card trigger)
    widget = metadata["ui"]["widget"]
    assert widget["resource_uri"] == RESOURCE_URI
    assert widget["title"] == "Test Title"
    assert widget["initial_data"]["context_key"] == "svg_viz:1:abc:test"
    assert widget["initial_data"]["space_id"] == "space-1"
    assert widget["initial_data"]["kind"] == "svg"

    # ui.cards (folded card payload)
    cards = metadata["ui"]["cards"]
    assert len(cards) == 1
    assert cards[0]["type"] == "context_artifact"
    assert cards[0]["title"] == "Test Title"
    assert cards[0]["payload"]["context_key"] == "svg_viz:1:abc:test"
    assert cards[0]["payload"]["author_agent"] == "test-agent"
    assert cards[0]["payload"]["source"] == "svg_viz_post_to_chat"

    # tool_call_id linkage
    assert metadata["tool_call_id"].startswith("svg_viz_post:")
    assert metadata["tool_call_id"] == cards[0]["payload"]["tool_call_id"]


# ── input validation ───────────────────────────────────────────────────────


def test_post_svg_rejects_non_svg_string():
    result = post_svg_to_chat(svg="not svg content", title="title")
    assert result["code"] == "INVALID_SVG"


def test_post_svg_rejects_empty_title():
    result = post_svg_to_chat(svg="<svg></svg>", title="")
    assert result["code"] == "MISSING_TITLE"


def test_post_svg_rejects_whitespace_only_title():
    result = post_svg_to_chat(svg="<svg></svg>", title="   ")
    assert result["code"] == "MISSING_TITLE"


def test_post_svg_rejects_non_string_svg():
    result = post_svg_to_chat(svg=12345, title="title")  # type: ignore[arg-type]
    assert result["code"] == "INVALID_SVG"


# ── happy-path with a fake ax_cli.config injected into sys.modules ─────────


def _install_fake_ax_cli_config(
    monkeypatch: pytest.MonkeyPatch,
    *,
    get_client_impl=None,
    resolve_space_id_impl=None,
):
    """Replace `ax_cli.config` in sys.modules with a fake exposing the two
    callables post_to_chat depends on. Reverts on test teardown via monkeypatch."""
    fake_module = types.ModuleType("ax_cli.config")
    fake_module.get_client = get_client_impl or (lambda: MagicMock())
    fake_module.resolve_space_id = resolve_space_id_impl or (
        lambda client, explicit=None: explicit or "space-default"
    )
    monkeypatch.setitem(sys.modules, "ax_cli.config", fake_module)


@pytest.fixture
def mocked_client():
    """A mock AxClient that records context-set and message-send calls."""
    client = MagicMock()
    client.set_context.return_value = {"ok": True}
    client.send_message.return_value = {"message": {"id": "msg-test-123"}}
    return client


def test_post_svg_happy_path_calls_set_context_then_send_message(monkeypatch, mocked_client):
    """Verify the two-step flow: upload context, then post signal message."""
    _install_fake_ax_cli_config(
        monkeypatch,
        get_client_impl=lambda: mocked_client,
    )

    result = post_svg_to_chat(
        svg="<svg xmlns='http://www.w3.org/2000/svg'></svg>",
        title="Status Report",
        summary="A test summary",
    )

    assert result["ok"] is True
    assert result["message_id"] == "msg-test-123"
    assert result["space_id"] == "space-default"
    assert result["title"] == "Status Report"
    assert result["resource_uri"] == RESOURCE_URI

    # set_context was called with the SVG content
    set_ctx_call = mocked_client.set_context.call_args
    assert set_ctx_call.args[0] == "space-default"
    assert set_ctx_call.args[1].startswith(f"{CONTEXT_KEY_PREFIX}:")
    assert set_ctx_call.args[2].startswith("<svg")

    # send_message was called with the signal metadata
    send_call = mocked_client.send_message.call_args
    metadata = send_call.kwargs["metadata"]
    assert metadata["ui"]["widget"]["resource_uri"] == RESOURCE_URI
    assert send_call.kwargs["message_type"] == "system"


def test_post_svg_propagates_context_upload_failure(monkeypatch):
    """If set_context raises, return CONTEXT_UPLOAD_FAILED and DON'T call send_message."""
    failing_client = MagicMock()
    failing_client.set_context.side_effect = Exception("network down")
    failing_client.send_message = MagicMock()

    _install_fake_ax_cli_config(monkeypatch, get_client_impl=lambda: failing_client)

    result = post_svg_to_chat(svg="<svg></svg>", title="t")
    assert result["code"] == "CONTEXT_UPLOAD_FAILED"
    assert "network down" in result["error"]
    failing_client.send_message.assert_not_called()


def test_post_svg_propagates_send_message_failure(monkeypatch):
    """set_context works but send_message fails -> MESSAGE_POST_FAILED.

    Returns the context_key so callers can clean up if they want to.
    """
    half_failing = MagicMock()
    half_failing.set_context.return_value = {"ok": True}
    half_failing.send_message.side_effect = Exception("server 500")

    _install_fake_ax_cli_config(monkeypatch, get_client_impl=lambda: half_failing)

    result = post_svg_to_chat(svg="<svg></svg>", title="t")
    assert result["code"] == "MESSAGE_POST_FAILED"
    assert "server 500" in result["error"]
    assert "context_key" in result


def test_post_svg_returns_no_credentials_when_get_client_raises(monkeypatch):
    def _boom():
        raise RuntimeError("no token file")

    _install_fake_ax_cli_config(monkeypatch, get_client_impl=_boom)

    result = post_svg_to_chat(svg="<svg></svg>", title="t")
    assert result["code"] == "NO_CREDENTIALS"


def test_post_svg_returns_no_space_when_resolution_returns_none(monkeypatch):
    _install_fake_ax_cli_config(
        monkeypatch,
        get_client_impl=lambda: MagicMock(),
        resolve_space_id_impl=lambda client, explicit=None: None,
    )

    result = post_svg_to_chat(svg="<svg></svg>", title="t")
    assert result["code"] == "NO_SPACE"


# ── MCP tool handler wrapping ──────────────────────────────────────────────


def test_handler_wraps_result_as_mcp_text_content(monkeypatch, mocked_client):
    _install_fake_ax_cli_config(monkeypatch, get_client_impl=lambda: mocked_client)

    result = _handle_post_to_chat(
        {"svg": "<svg></svg>", "title": "x", "summary": "y"}
    )
    assert "content" in result
    payload = json.loads(result["content"][0]["text"])
    assert payload["ok"] is True
    assert payload["message_id"] == "msg-test-123"


def test_handler_passes_through_validation_errors():
    """Handler should NOT raise on bad input — returns structured error in MCP content."""
    result = _handle_post_to_chat({"svg": "not svg", "title": "x"})
    payload = json.loads(result["content"][0]["text"])
    assert payload["code"] == "INVALID_SVG"
