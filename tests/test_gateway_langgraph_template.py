"""Regression: LangGraph template registers correctly and the bridge
emits the Gateway lifecycle contract.

A Gateway-managed LangGraph runtime should be declarable from the
dashboard / CLI like any other template, and the bridge subprocess
should emit AX_GATEWAY_EVENT lines that map to the three signals
operators rely on (online via heartbeat, accept-work via intake_model,
response-path via return_paths).

This file locks the initial contract:
  - the template appears in agent_template_catalog with the right shape
  - the template appears in the default agent_template_list ordering
  - the bridge file exists at the path the template advertises
  - the bridge emits a "processing" event and a "completed" event around
    a stub prompt round trip
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

from ax_cli.gateway_runtime_types import (
    agent_template_catalog,
    agent_template_definition,
    agent_template_list,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
BRIDGE_PATH = REPO_ROOT / "examples" / "gateway_langgraph" / "langgraph_bridge.py"


def _install_fake_langgraph(monkeypatch) -> None:
    """Stub `langgraph.graph` in sys.modules so the bridge's
    `from langgraph.graph import END, START, StateGraph` resolves
    without requiring the optional langgraph package to be installed.

    Implements the minimal surface the bridge uses: `StateGraph(dict)`,
    `.add_node(name, fn)`, `.add_edge(source, target)`, `.compile()`
    returning an app whose `.invoke(state)` walks START -> ... -> END,
    executing each node and merging dict-returns into state. Works for
    the bridge's one-node graph (START -> node -> END) without modeling
    branches or conditional edges.

    Same lesson as the merged groq tests: stub the optional dep at
    sys.modules so CI runs without it.
    """
    import types as _types

    _START = "__LANGGRAPH_STUB_START__"
    _END = "__LANGGRAPH_STUB_END__"

    class _StateGraph:
        def __init__(self, _state_cls) -> None:
            self._nodes: dict[str, object] = {}
            self._edges: list[tuple[str, str]] = []

        def add_node(self, name: str, fn) -> None:
            self._nodes[name] = fn

        def add_edge(self, source: str, target: str) -> None:
            self._edges.append((source, target))

        def compile(self):
            nodes = dict(self._nodes)
            edges = dict(self._edges)  # linear graphs only

            class _App:
                def invoke(self, state):
                    state = dict(state)
                    current = edges.get(_START)
                    while current and current != _END:
                        result = nodes[current](state)
                        if isinstance(result, dict):
                            state.update(result)
                        current = edges.get(current)
                    return state

            return _App()

    fake_graph_module = _types.ModuleType("langgraph.graph")
    fake_graph_module.START = _START
    fake_graph_module.END = _END
    fake_graph_module.StateGraph = _StateGraph

    fake_pkg = _types.ModuleType("langgraph")
    fake_pkg.graph = fake_graph_module

    monkeypatch.setitem(sys.modules, "langgraph", fake_pkg)
    monkeypatch.setitem(sys.modules, "langgraph.graph", fake_graph_module)


def test_langgraph_template_is_registered() -> None:
    catalog = agent_template_catalog()
    assert "langgraph" in catalog, (
        "langgraph template missing from agent_template_catalog. "
        "Should sit alongside ollama / hermes / claude_code_channel."
    )

    template = agent_template_definition("langgraph")
    assert template["id"] == "langgraph"
    assert template["runtime_type"] == "exec", (
        "Reuses the exec runtime adapter (same precedent as ollama). "
        "A dedicated 'langgraph' runtime_type is a follow-up."
    )
    assert template["intake_model"] == "launch_on_send"
    assert template["return_paths"] == ["inline_reply"]
    assert template["availability"] == "ready"
    assert template["launchable"] is True


def test_langgraph_template_default_exec_command_points_at_bridge() -> None:
    template = agent_template_definition("langgraph")
    defaults = template.get("defaults") or {}
    exec_command = str(defaults.get("exec_command") or "")
    assert "examples/gateway_langgraph/langgraph_bridge.py" in exec_command, (
        f"langgraph template's default exec_command should run the stub "
        f"bridge at examples/gateway_langgraph/langgraph_bridge.py. Got: {exec_command!r}"
    )


def test_langgraph_template_listed_in_default_ordering() -> None:
    listed_ids = [item["id"] for item in agent_template_list()]
    assert "langgraph" in listed_ids, (
        "langgraph template should appear in the default (non-advanced) "
        "template list so the dashboard's Add Agent modal can render it."
    )


def test_langgraph_bridge_file_exists() -> None:
    assert BRIDGE_PATH.exists(), (
        f"langgraph bridge file missing at {BRIDGE_PATH}. The default "
        "exec_command in the template registration points at it."
    )


def test_langgraph_bridge_emits_lifecycle_events(monkeypatch, capsys) -> None:
    """Run the bridge's main() inline on the STUB path and confirm it
    emits processing and completed AX_GATEWAY_EVENT lines around the
    round trip. GROQ_API_KEY is explicitly unset so the bridge picks
    the stub-ack node rather than the LLM node."""
    _install_fake_langgraph(monkeypatch)
    sys.path.insert(0, str(BRIDGE_PATH.parent))
    try:
        import langgraph_bridge as bridge
    finally:
        sys.path.pop(0)

    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr(sys, "argv", ["langgraph_bridge.py", "test prompt"])
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))
    monkeypatch.setenv("AX_GATEWAY_AGENT_NAME", "langgraph-test")

    rc = bridge.main()
    captured = capsys.readouterr()

    assert rc == 0, f"bridge main() returned {rc}; expected 0"

    event_lines = [line for line in captured.out.splitlines() if line.startswith(bridge.EVENT_PREFIX)]
    statuses = []
    completed_detail = None
    for line in event_lines:
        payload = json.loads(line[len(bridge.EVENT_PREFIX) :])
        if payload.get("kind") == "status":
            statuses.append(payload.get("status"))
            if payload.get("status") == "completed":
                completed_detail = payload.get("detail") or {}

    assert "processing" in statuses, f"bridge did not emit a processing status event. statuses={statuses}"
    assert "completed" in statuses, f"bridge did not emit a completed status event. statuses={statuses}"
    assert completed_detail is not None, "completed event missing detail block"
    assert completed_detail.get("used_llm") is False, (
        f"stub path should report used_llm=False in the completed event detail. got: {completed_detail!r}"
    )

    reply_lines = [line for line in captured.out.splitlines() if line and not line.startswith(bridge.EVENT_PREFIX)]
    assert reply_lines, "bridge did not print a reply line on stdout"
    assert "test prompt" in reply_lines[-1], (
        f"bridge reply should echo the prompt in the stub. last line: {reply_lines[-1]!r}"
    )


def _make_stream_chunk(content):
    """Build a single streaming-mode chunk shaped like the Groq SDK yields."""
    import types as _types

    delta = _types.SimpleNamespace(content=content)
    choice = _types.SimpleNamespace(delta=delta)
    return _types.SimpleNamespace(choices=[choice])


def test_langgraph_bridge_calls_groq_llm_when_configured(monkeypatch, capsys) -> None:
    """When GROQ_API_KEY is set AND the groq SDK is importable, the
    bridge's LangGraph node should stream a Groq chat completion and
    return the joined model response.

    The groq SDK is stubbed via sys.modules so this test runs offline
    and does not consume API credits. Same pattern as the merged Groq
    runtime tests (_install_fake_groq).
    """
    import types as _types
    from unittest.mock import MagicMock

    _install_fake_langgraph(monkeypatch)
    sys.path.insert(0, str(BRIDGE_PATH.parent))
    try:
        import langgraph_bridge as bridge
    finally:
        sys.path.pop(0)

    # Build a stub groq module with a mocked client. The bridge does
    # `from groq import Groq`, instantiates Groq(), and calls
    # client.chat.completions.create(stream=True, ...), then iterates
    # the returned stream consuming `.choices[0].delta.content` per
    # chunk. The fake returns an iterator of three streaming chunks
    # that join into a recognizable answer.
    fake_stream = iter(
        [
            _make_stream_chunk("The speed of light "),
            _make_stream_chunk("is approximately "),
            _make_stream_chunk("299,792 km/s."),
        ]
    )
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = fake_stream

    fake_groq = _types.ModuleType("groq")
    fake_groq.Groq = MagicMock(return_value=fake_client)
    monkeypatch.setitem(sys.modules, "groq", fake_groq)

    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    monkeypatch.setenv("AX_BRIDGE_LLM_MODEL", "test-model-x")
    monkeypatch.setattr(sys, "argv", ["langgraph_bridge.py", "what is the speed of light in km/s"])
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))
    monkeypatch.setenv("AX_GATEWAY_AGENT_NAME", "langgraph-test")

    rc = bridge.main()
    captured = capsys.readouterr()

    assert rc == 0, f"bridge main() returned {rc}; expected 0"

    # Groq was called exactly once with the configured model, stream=True,
    # and the prompt as the user message.
    assert fake_client.chat.completions.create.call_count == 1
    call_kwargs = fake_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["model"] == "test-model-x", (
        f"bridge should forward AX_BRIDGE_LLM_MODEL to Groq. got model={call_kwargs.get('model')!r}"
    )
    assert call_kwargs.get("stream") is True, (
        "bridge must request streaming so the activity feed stays live during the call"
    )
    messages = call_kwargs["messages"]
    assert messages[0]["role"] == "system"
    assert "langgraph-test" in messages[0]["content"], (
        "system prompt should name the routed agent so the model knows who it is replying as"
    )
    assert messages[-1]["role"] == "user"
    assert messages[-1]["content"] == "what is the speed of light in km/s"

    # Completion event reports used_llm=True (and back-compat stub=False).
    event_lines = [line for line in captured.out.splitlines() if line.startswith(bridge.EVENT_PREFIX)]
    completed_detail = None
    for line in event_lines:
        payload = json.loads(line[len(bridge.EVENT_PREFIX) :])
        if payload.get("kind") == "status" and payload.get("status") == "completed":
            completed_detail = payload.get("detail") or {}
    assert completed_detail is not None, "completed event missing"
    assert completed_detail.get("used_llm") is True, (
        f"LLM path should report used_llm=True in the completed event detail. got: {completed_detail!r}"
    )
    assert completed_detail.get("stub") is False, (
        "stub flag is kept for back-compat with the pre-LLM-validation schema; "
        f"LLM path should report stub=False. got: {completed_detail!r}"
    )

    # The joined streamed response (not a synthetic ack) lands on stdout.
    reply_lines = [line for line in captured.out.splitlines() if line and not line.startswith(bridge.EVENT_PREFIX)]
    assert reply_lines, "bridge did not print a reply line on stdout"
    assert "299,792" in reply_lines[-1], (
        f"bridge reply should be the joined streamed response, not a stub ack. last line: {reply_lines[-1]!r}"
    )


def test_langgraph_bridge_streams_activity_events_during_llm_call(monkeypatch, capsys) -> None:
    """Streaming path should emit a `processing` status when the first
    token arrives and at least one throttled `activity` event with a
    rolling preview. This locks in the chatty-observability contract
    Andrew's review called out: a synchronous Groq call would leave
    the activity feed silent for the duration of the call.

    Time is faked so the heartbeat fires deterministically without
    sleeping.
    """
    import types as _types
    from unittest.mock import MagicMock

    _install_fake_langgraph(monkeypatch)
    sys.path.insert(0, str(BRIDGE_PATH.parent))
    try:
        import langgraph_bridge as bridge
    finally:
        sys.path.pop(0)

    # Drive time.monotonic deterministically so each yielded chunk
    # advances the clock past the ACTIVITY_HEARTBEAT_SECONDS threshold.
    # Each call to monotonic() returns the next value in the list.
    fake_now = iter([0.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0])

    def _next_monotonic() -> float:
        try:
            return next(fake_now)
        except StopIteration:
            return 99.0

    monkeypatch.setattr(bridge.time, "monotonic", _next_monotonic)

    fake_stream = iter(
        [
            _make_stream_chunk("Light "),
            _make_stream_chunk("travels at "),
            _make_stream_chunk("about "),
            _make_stream_chunk("299,792 km/s."),
        ]
    )
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = fake_stream

    fake_groq = _types.ModuleType("groq")
    fake_groq.Groq = MagicMock(return_value=fake_client)
    monkeypatch.setitem(sys.modules, "groq", fake_groq)

    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    monkeypatch.setattr(sys, "argv", ["langgraph_bridge.py", "tell me about light"])
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))
    monkeypatch.setenv("AX_GATEWAY_AGENT_NAME", "langgraph-test")

    rc = bridge.main()
    captured = capsys.readouterr()
    assert rc == 0, f"bridge main() returned {rc}; expected 0"

    event_lines = [line for line in captured.out.splitlines() if line.startswith(bridge.EVENT_PREFIX)]
    processing_messages: list[str] = []
    streaming_activities: list[str] = []
    for line in event_lines:
        payload = json.loads(line[len(bridge.EVENT_PREFIX) :])
        if payload.get("kind") == "status" and payload.get("status") == "processing":
            processing_messages.append(str(payload.get("message") or ""))
        if payload.get("kind") == "activity":
            activity = str(payload.get("activity") or "")
            if "test-model" in activity or "Streaming response" in activity or "Light" in activity:
                streaming_activities.append(activity)

    assert any("Groq is responding" in m for m in processing_messages), (
        "bridge should emit a `Groq is responding` processing status on first streamed token. "
        f"got processing messages: {processing_messages!r}"
    )
    assert streaming_activities, (
        "bridge should emit at least one throttled activity event with rolling preview "
        f"during streaming. all events: {event_lines!r}"
    )


def test_langgraph_bridge_honors_ax_bridge_system_prompt(monkeypatch, capsys) -> None:
    """AX_BRIDGE_SYSTEM_PROMPT overrides the default system-prompt tail
    ("Reply concisely.") that follows the agent-name framing.
    """
    import types as _types
    from unittest.mock import MagicMock

    _install_fake_langgraph(monkeypatch)
    sys.path.insert(0, str(BRIDGE_PATH.parent))
    try:
        import langgraph_bridge as bridge
    finally:
        sys.path.pop(0)

    fake_stream = iter([_make_stream_chunk("ok.")])
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = fake_stream

    fake_groq = _types.ModuleType("groq")
    fake_groq.Groq = MagicMock(return_value=fake_client)
    monkeypatch.setitem(sys.modules, "groq", fake_groq)

    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    monkeypatch.setenv("AX_BRIDGE_SYSTEM_PROMPT", "Answer in formal English only.")
    monkeypatch.setattr(sys, "argv", ["langgraph_bridge.py", "hello"])
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))
    monkeypatch.setenv("AX_GATEWAY_AGENT_NAME", "langgraph-test")

    rc = bridge.main()
    capsys.readouterr()
    assert rc == 0

    call_kwargs = fake_client.chat.completions.create.call_args.kwargs
    system_content = call_kwargs["messages"][0]["content"]
    assert "Answer in formal English only." in system_content, (
        f"AX_BRIDGE_SYSTEM_PROMPT should be threaded into the system message. got: {system_content!r}"
    )
    assert "Reply concisely." not in system_content, (
        "default tail should be replaced, not appended, when AX_BRIDGE_SYSTEM_PROMPT is set"
    )


def test_langgraph_bridge_falls_back_to_stub_when_groq_sdk_missing(monkeypatch, capsys) -> None:
    """If GROQ_API_KEY is set but the groq SDK is NOT importable, the
    bridge should fall back to the stub-ack node cleanly and emit an
    activity event explaining the fallback. Backward-compat path so
    operators with stale environments still get a working bridge."""
    _install_fake_langgraph(monkeypatch)
    sys.path.insert(0, str(BRIDGE_PATH.parent))
    try:
        import langgraph_bridge as bridge
    finally:
        sys.path.pop(0)

    # Force `from groq import Groq` to raise ImportError by setting
    # the entry in sys.modules to None.
    monkeypatch.setitem(sys.modules, "groq", None)
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    monkeypatch.setattr(sys, "argv", ["langgraph_bridge.py", "test prompt"])
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))
    monkeypatch.setenv("AX_GATEWAY_AGENT_NAME", "langgraph-test")

    rc = bridge.main()
    captured = capsys.readouterr()

    assert rc == 0
    event_lines = [line for line in captured.out.splitlines() if line.startswith(bridge.EVENT_PREFIX)]
    activities = []
    completed_detail = None
    for line in event_lines:
        payload = json.loads(line[len(bridge.EVENT_PREFIX) :])
        if payload.get("kind") == "activity":
            activities.append(payload.get("activity"))
        if payload.get("kind") == "status" and payload.get("status") == "completed":
            completed_detail = payload.get("detail") or {}

    assert any("groq SDK not installed" in a for a in activities), (
        f"fallback activity event should mention the missing groq SDK. got: {activities!r}"
    )
    assert completed_detail is not None
    assert completed_detail.get("used_llm") is False, (
        "fallback path should report used_llm=False in the completed event detail"
    )

    reply_lines = [line for line in captured.out.splitlines() if line and not line.startswith(bridge.EVENT_PREFIX)]
    assert reply_lines
    assert "test prompt" in reply_lines[-1], "fallback reply should echo the prompt via the stub ack node"
