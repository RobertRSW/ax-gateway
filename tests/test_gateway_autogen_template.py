"""Regression: AutoGen template registers correctly and the bridge
emits the Gateway lifecycle contract.

A Gateway-managed AutoGen runtime should be declarable from the
dashboard / CLI like any other template, and the bridge subprocess
should emit AX_GATEWAY_EVENT lines that map to the three signals
operators rely on (online via heartbeat, accept-work via intake_model,
response-path via return_paths).

Mirrors test_gateway_langgraph_template.py — same shape, adapted for
the AutoGen bridge across all three execution tiers.

This file locks the contract for:
  - the template appears in agent_template_catalog with the right shape
  - the template appears in the default agent_template_list ordering
  - the bridge file exists at the path the template advertises
  - the bridge emits a "processing" event and a "completed" event
    around a stub-path prompt round trip
  - the bridge wires an AutoGen AssistantAgent to a Groq-backed
    OpenAIChatCompletionClient and runs on_messages() when
    GROQ_API_KEY is set, returning the model's reply
  - AX_BRIDGE_SYSTEM_PROMPT threads into the Agent system_message
  - the bridge falls back to the stub agent path when autogen-ext is
    missing
  - the bridge falls back to the string template when
    autogen-agentchat itself is missing
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
BRIDGE_PATH = REPO_ROOT / "examples" / "gateway_autogen" / "autogen_bridge.py"


def _install_fake_autogen(
    monkeypatch,
    on_messages_reply="stub autogen reply",
    stream_chunks=None,
):
    """Stub the autogen-agentchat / autogen-ext / autogen-core packages
    in sys.modules so the bridge's lazy imports resolve without
    requiring the real packages.

    Implements the minimal surface the bridge uses. AssistantAgent
    constructor captures kwargs and exposes both an async `on_messages`
    (returns a Response for back-compat) and an async-generator
    `on_messages_stream` (yields per-token ModelClientStreamingChunkEvent
    items if `stream_chunks` is provided, then a final Response with
    `chat_message.content` matching `on_messages_reply`).
    OpenAIChatCompletionClient captures kwargs.

    Same lesson as the merged groq / mistral tests: stub the optional
    dep at sys.modules so CI runs without it.
    """
    import types as _types

    captured = {"agents": [], "clients": [], "calls": []}

    class _TextMessage:
        def __init__(self, content, source):
            self.content = content
            self.source = source

    class _CancellationToken:
        pass

    class _ChatMessage:
        def __init__(self, content):
            self.content = content

    class _Response:
        def __init__(self, content):
            self.chat_message = _ChatMessage(content)

    class _StreamingChunkEvent:
        """ModelClientStreamingChunkEvent stand-in. Carries partial token
        text as `content`, no `chat_message` attribute (the bridge's
        duck-typed dispatch keys off that to distinguish chunks from
        the final Response)."""

        def __init__(self, content):
            self.content = content

    class _OpenAIChatCompletionClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            captured["clients"].append(self)

        async def close(self):
            return None

    class _AssistantAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.name = kwargs.get("name", "")
            self.system_message = kwargs.get("system_message", "")
            captured["agents"].append(self)

        async def on_messages(self, messages, cancellation_token=None):
            captured["calls"].append(
                {
                    "method": "on_messages",
                    "messages": [(m.content, m.source) for m in messages],
                }
            )
            return _Response(on_messages_reply)

        async def on_messages_stream(self, messages, cancellation_token=None):
            captured["calls"].append(
                {
                    "method": "on_messages_stream",
                    "messages": [(m.content, m.source) for m in messages],
                }
            )
            for chunk_content in stream_chunks or []:
                yield _StreamingChunkEvent(chunk_content)
            yield _Response(on_messages_reply)

    fake_ac = _types.ModuleType("autogen_agentchat")
    fake_agents = _types.ModuleType("autogen_agentchat.agents")
    fake_agents.AssistantAgent = _AssistantAgent
    fake_messages = _types.ModuleType("autogen_agentchat.messages")
    fake_messages.TextMessage = _TextMessage

    fake_core = _types.ModuleType("autogen_core")
    fake_core.CancellationToken = _CancellationToken

    fake_ext = _types.ModuleType("autogen_ext")
    fake_ext_models = _types.ModuleType("autogen_ext.models")
    fake_ext_models_openai = _types.ModuleType("autogen_ext.models.openai")
    fake_ext_models_openai.OpenAIChatCompletionClient = _OpenAIChatCompletionClient

    monkeypatch.setitem(sys.modules, "autogen_agentchat", fake_ac)
    monkeypatch.setitem(sys.modules, "autogen_agentchat.agents", fake_agents)
    monkeypatch.setitem(sys.modules, "autogen_agentchat.messages", fake_messages)
    monkeypatch.setitem(sys.modules, "autogen_core", fake_core)
    monkeypatch.setitem(sys.modules, "autogen_ext", fake_ext)
    monkeypatch.setitem(sys.modules, "autogen_ext.models", fake_ext_models)
    monkeypatch.setitem(sys.modules, "autogen_ext.models.openai", fake_ext_models_openai)

    return captured


def test_autogen_template_is_registered() -> None:
    catalog = agent_template_catalog()
    assert "autogen" in catalog, (
        "autogen template missing from agent_template_catalog. Should sit alongside langgraph / strands / ollama."
    )

    template = agent_template_definition("autogen")
    assert template["id"] == "autogen"
    assert template["runtime_type"] == "exec", (
        "Reuses the exec runtime adapter (same precedent as langgraph). "
        "A dedicated 'autogen' runtime_type is a follow-up."
    )
    assert template["intake_model"] == "launch_on_send"
    assert template["return_paths"] == ["inline_reply"]
    assert template["availability"] == "ready"
    assert template["launchable"] is True


def test_autogen_template_default_exec_command_points_at_bridge() -> None:
    template = agent_template_definition("autogen")
    defaults = template.get("defaults") or {}
    exec_command = str(defaults.get("exec_command") or "")
    assert "examples/gateway_autogen/autogen_bridge.py" in exec_command, (
        f"autogen template's default exec_command should run the bridge "
        f"at examples/gateway_autogen/autogen_bridge.py. Got: {exec_command!r}"
    )


def test_autogen_template_listed_in_default_ordering() -> None:
    listed_ids = [item["id"] for item in agent_template_list()]
    assert "autogen" in listed_ids, (
        "autogen template should appear in the default (non-advanced) "
        "template list so the dashboard's Add Agent modal can render it."
    )


def test_autogen_bridge_file_exists() -> None:
    assert BRIDGE_PATH.exists(), (
        f"autogen bridge file missing at {BRIDGE_PATH}. The default "
        "exec_command in the template registration points at it."
    )


def test_autogen_bridge_emits_lifecycle_events(monkeypatch, capsys) -> None:
    """Run the bridge's main() inline on the STUB agent path and confirm
    it emits processing and completed AX_GATEWAY_EVENT lines around the
    round trip. GROQ_API_KEY is explicitly unset so the bridge picks
    the stub-agent path rather than the LLM path."""
    _install_fake_autogen(monkeypatch)
    sys.path.insert(0, str(BRIDGE_PATH.parent))
    try:
        import autogen_bridge as bridge
    finally:
        sys.path.pop(0)

    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr(sys, "argv", ["autogen_bridge.py", "test prompt"])
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))
    monkeypatch.setenv("AX_GATEWAY_AGENT_NAME", "autogen-test")

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


def test_autogen_bridge_calls_groq_llm_when_configured(monkeypatch, capsys) -> None:
    """When GROQ_API_KEY is set AND autogen-ext is importable, the
    bridge should build an OpenAIChatCompletionClient pointed at Groq's
    OpenAI-compatible endpoint, wire it into an AssistantAgent with
    `model_client_stream=True`, and drive a single turn via
    `on_messages_stream()` with the prompt as a TextMessage.
    """
    captured = _install_fake_autogen(
        monkeypatch,
        on_messages_reply="The speed of light is approximately 299,792 km/s.",
    )

    sys.path.insert(0, str(BRIDGE_PATH.parent))
    try:
        import autogen_bridge as bridge
    finally:
        sys.path.pop(0)

    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    monkeypatch.setenv("AX_BRIDGE_LLM_MODEL", "test-model-x")
    monkeypatch.setattr(sys, "argv", ["autogen_bridge.py", "what is the speed of light in km/s"])
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))
    monkeypatch.setenv("AX_GATEWAY_AGENT_NAME", "autogen-test")

    rc = bridge.main()
    out = capsys.readouterr()

    assert rc == 0, f"bridge main() returned {rc}; expected 0"

    # Exactly one model client, one agent, one on_messages call.
    assert len(captured["clients"]) == 1
    assert len(captured["agents"]) == 1
    assert len(captured["calls"]) == 1

    # Model client was wired with the configured model + Groq endpoint + API key.
    client_kwargs = captured["clients"][0].kwargs
    assert client_kwargs["model"] == "test-model-x", (
        f"bridge should forward AX_BRIDGE_LLM_MODEL to the model client. got model={client_kwargs.get('model')!r}"
    )
    assert client_kwargs["base_url"] == "https://api.groq.com/openai/v1", (
        "bridge should point the OpenAI-compat client at Groq's endpoint"
    )
    assert client_kwargs["api_key"] == "gsk_test", "bridge should pass GROQ_API_KEY through to the model client"

    # Agent was wired with the model client, model_client_stream=True for
    # per-token streaming events, and a system_message that names the agent.
    agent_kwargs = captured["agents"][0].kwargs
    assert agent_kwargs.get("model_client") is captured["clients"][0], (
        "Agent should be wired to the constructed Groq-backed model client"
    )
    assert agent_kwargs.get("model_client_stream") is True, (
        "Agent must be constructed with model_client_stream=True so the "
        "underlying OpenAIChatCompletionClient emits per-token chunks during "
        "on_messages_stream(). Without that flag the bridge's activity feed "
        "stays silent during the call."
    )
    assert "autogen-test" in agent_kwargs.get("system_message", ""), (
        "Agent system_message should name the routed agent so the model knows who it is replying as"
    )
    # AutoGen names must be valid Python identifiers, so the hyphen in
    # `autogen-test` should be substituted with an underscore.
    assert agent_kwargs.get("name") == "autogen_test", (
        f"bridge should sanitize agent name for AutoGen (hyphens to underscores). got: {agent_kwargs.get('name')!r}"
    )

    # The bridge calls the streaming variant, not the non-streaming one,
    # so the operator sees per-token activity events during the call.
    call = captured["calls"][0]
    assert call["method"] == "on_messages_stream", (
        f"bridge should call on_messages_stream() to surface token-level "
        f"activity events, got method={call.get('method')!r}"
    )
    assert call["messages"] == [("what is the speed of light in km/s", "user")]

    # Completion event reports used_llm=True (and back-compat stub=False).
    event_lines = [line for line in out.out.splitlines() if line.startswith(bridge.EVENT_PREFIX)]
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

    # The on_messages reply (not a synthetic ack) lands on stdout.
    reply_lines = [line for line in out.out.splitlines() if line and not line.startswith(bridge.EVENT_PREFIX)]
    assert reply_lines, "bridge did not print a reply line on stdout"
    assert "299,792" in reply_lines[-1], (
        f"bridge reply should be the model's response, not a stub ack. last line: {reply_lines[-1]!r}"
    )


def test_autogen_bridge_streams_activity_events_during_llm_call(monkeypatch, capsys) -> None:
    """The streaming path should emit a `processing` status when the
    first token arrives and at least one throttled `activity` event
    with a rolling preview. This locks in the chatty-observability
    contract that PR #38's review pass enforced on the langgraph
    bridge. Without per-token signals the activity feed would go
    silent for the duration of the LLM call.

    Time is faked so the heartbeat fires deterministically without
    sleeping.
    """
    captured = _install_fake_autogen(
        monkeypatch,
        on_messages_reply="Final answer: 299,792 km/s.",
        stream_chunks=[
            "Light ",
            "travels at ",
            "about ",
            "299,792 km/s.",
        ],
    )

    sys.path.insert(0, str(BRIDGE_PATH.parent))
    try:
        import autogen_bridge as bridge
    finally:
        sys.path.pop(0)

    # Drive time.monotonic deterministically so each yielded chunk
    # advances the clock past the ACTIVITY_HEARTBEAT_SECONDS threshold.
    # Each call to monotonic() returns the next value in the list.
    fake_now = iter([0.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0])

    def _next_monotonic() -> float:
        try:
            return next(fake_now)
        except StopIteration:
            return 99.0

    monkeypatch.setattr(bridge.time, "monotonic", _next_monotonic)

    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    monkeypatch.setenv("AX_BRIDGE_LLM_MODEL", "test-model-x")
    monkeypatch.setattr(sys, "argv", ["autogen_bridge.py", "tell me about light"])
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))
    monkeypatch.setenv("AX_GATEWAY_AGENT_NAME", "autogen-test")

    rc = bridge.main()
    out = capsys.readouterr()
    assert rc == 0, f"bridge main() returned {rc}; expected 0"

    event_lines = [line for line in out.out.splitlines() if line.startswith(bridge.EVENT_PREFIX)]
    processing_messages: list[str] = []
    streaming_activities: list[str] = []
    for line in event_lines:
        payload = json.loads(line[len(bridge.EVENT_PREFIX) :])
        if payload.get("kind") == "status" and payload.get("status") == "processing":
            processing_messages.append(str(payload.get("message") or ""))
        if payload.get("kind") == "activity":
            activity = str(payload.get("activity") or "")
            if "test-model-x" in activity or "Streaming response" in activity:
                streaming_activities.append(activity)

    assert any("Groq is responding" in m for m in processing_messages), (
        "bridge should emit a `Groq is responding` processing status on the first streamed token. "
        f"got processing messages: {processing_messages!r}"
    )
    assert streaming_activities, (
        "bridge should emit at least one throttled activity event with rolling preview "
        f"during on_messages_stream(). all events: {event_lines!r}"
    )

    # The final Response content lands on stdout, not a chunk concatenation.
    reply_lines = [line for line in out.out.splitlines() if line and not line.startswith(bridge.EVENT_PREFIX)]
    assert reply_lines, "bridge did not print a reply line"
    assert "299,792" in reply_lines[-1], (
        f"bridge reply should be the final Response content, not the concatenated chunks. "
        f"last line: {reply_lines[-1]!r}"
    )

    # Sanity check that the streaming method (not on_messages) was the one called.
    assert len(captured["calls"]) == 1
    assert captured["calls"][0]["method"] == "on_messages_stream"


def test_autogen_bridge_falls_back_to_chunks_when_no_final_response(monkeypatch, capsys) -> None:
    """If the stream completes without yielding a final Response
    (some AutoGen builds may close the stream via an internal
    terminator instead), the bridge should fall back to the
    concatenated chunk text so a reply still lands on stdout.

    Stubs the agent to yield only chunk events, no Response.
    """
    import types as _types

    sys.path.insert(0, str(BRIDGE_PATH.parent))
    try:
        import autogen_bridge as bridge
    finally:
        sys.path.pop(0)

    class _StreamingChunkEvent:
        def __init__(self, content):
            self.content = content

    class _AssistantAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.name = kwargs.get("name", "")

        async def on_messages_stream(self, messages, cancellation_token=None):
            for chunk_content in ["Partial ", "reply ", "only."]:
                yield _StreamingChunkEvent(chunk_content)

    class _OpenAIChatCompletionClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        async def close(self):
            return None

    class _TextMessage:
        def __init__(self, content, source):
            self.content = content
            self.source = source

    class _CancellationToken:
        pass

    fake_ac = _types.ModuleType("autogen_agentchat")
    fake_agents = _types.ModuleType("autogen_agentchat.agents")
    fake_agents.AssistantAgent = _AssistantAgent
    fake_messages = _types.ModuleType("autogen_agentchat.messages")
    fake_messages.TextMessage = _TextMessage
    fake_core = _types.ModuleType("autogen_core")
    fake_core.CancellationToken = _CancellationToken
    fake_ext = _types.ModuleType("autogen_ext")
    fake_ext_models = _types.ModuleType("autogen_ext.models")
    fake_ext_models_openai = _types.ModuleType("autogen_ext.models.openai")
    fake_ext_models_openai.OpenAIChatCompletionClient = _OpenAIChatCompletionClient
    monkeypatch.setitem(sys.modules, "autogen_agentchat", fake_ac)
    monkeypatch.setitem(sys.modules, "autogen_agentchat.agents", fake_agents)
    monkeypatch.setitem(sys.modules, "autogen_agentchat.messages", fake_messages)
    monkeypatch.setitem(sys.modules, "autogen_core", fake_core)
    monkeypatch.setitem(sys.modules, "autogen_ext", fake_ext)
    monkeypatch.setitem(sys.modules, "autogen_ext.models", fake_ext_models)
    monkeypatch.setitem(sys.modules, "autogen_ext.models.openai", fake_ext_models_openai)

    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    monkeypatch.setattr(sys, "argv", ["autogen_bridge.py", "hello"])
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))
    monkeypatch.setenv("AX_GATEWAY_AGENT_NAME", "autogen-test")

    rc = bridge.main()
    out = capsys.readouterr()
    assert rc == 0

    reply_lines = [line for line in out.out.splitlines() if line and not line.startswith(bridge.EVENT_PREFIX)]
    assert reply_lines, "bridge should fall back to chunk concatenation when no final Response"
    assert "Partial reply only." in reply_lines[-1], (
        f"bridge reply should be the chunk concatenation, got: {reply_lines[-1]!r}"
    )


def test_autogen_bridge_honors_ax_bridge_system_prompt(monkeypatch, capsys) -> None:
    """AX_BRIDGE_SYSTEM_PROMPT overrides the default trailing instruction
    ("Reply concisely.") in the agent's system_message.
    """
    captured = _install_fake_autogen(monkeypatch, on_messages_reply="ok.")

    sys.path.insert(0, str(BRIDGE_PATH.parent))
    try:
        import autogen_bridge as bridge
    finally:
        sys.path.pop(0)

    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    monkeypatch.setenv("AX_BRIDGE_SYSTEM_PROMPT", "Answer in formal English only.")
    monkeypatch.setattr(sys, "argv", ["autogen_bridge.py", "hello"])
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))
    monkeypatch.setenv("AX_GATEWAY_AGENT_NAME", "autogen-test")

    rc = bridge.main()
    capsys.readouterr()
    assert rc == 0

    agent_kwargs = captured["agents"][0].kwargs
    system_message = agent_kwargs.get("system_message", "")
    assert "Answer in formal English only." in system_message, (
        f"AX_BRIDGE_SYSTEM_PROMPT should be threaded into the Agent system_message. got: {system_message!r}"
    )
    assert "Reply concisely." not in system_message, (
        "default tail should be replaced, not appended, when AX_BRIDGE_SYSTEM_PROMPT is set"
    )


def test_autogen_bridge_falls_back_to_stub_when_autogen_ext_missing(monkeypatch, capsys) -> None:
    """If GROQ_API_KEY is set, autogen-agentchat is importable, but
    autogen-ext (OpenAIChatCompletionClient) is NOT importable, the
    bridge should fall back to the stub agent path cleanly and emit an
    activity event explaining the fallback.
    """
    _install_fake_autogen(monkeypatch)

    sys.path.insert(0, str(BRIDGE_PATH.parent))
    try:
        import autogen_bridge as bridge
    finally:
        sys.path.pop(0)

    # Force `from autogen_ext.models.openai import OpenAIChatCompletionClient`
    # to raise ImportError by setting the entry in sys.modules to None.
    monkeypatch.setitem(sys.modules, "autogen_ext.models.openai", None)
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    monkeypatch.setattr(sys, "argv", ["autogen_bridge.py", "test prompt"])
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))
    monkeypatch.setenv("AX_GATEWAY_AGENT_NAME", "autogen-test")

    rc = bridge.main()
    out = capsys.readouterr()

    assert rc == 0
    event_lines = [line for line in out.out.splitlines() if line.startswith(bridge.EVENT_PREFIX)]
    activities = []
    completed_detail = None
    for line in event_lines:
        payload = json.loads(line[len(bridge.EVENT_PREFIX) :])
        if payload.get("kind") == "activity":
            activities.append(payload.get("activity"))
        if payload.get("kind") == "status" and payload.get("status") == "completed":
            completed_detail = payload.get("detail") or {}

    assert any("autogen-ext not installed" in a for a in activities), (
        f"fallback activity event should mention the missing autogen-ext package. got: {activities!r}"
    )
    assert completed_detail is not None
    assert completed_detail.get("used_llm") is False, (
        "fallback path should report used_llm=False in the completed event detail"
    )

    reply_lines = [line for line in out.out.splitlines() if line and not line.startswith(bridge.EVENT_PREFIX)]
    assert reply_lines
    assert "test prompt" in reply_lines[-1], "fallback reply should echo the prompt via the stub agent path"


def test_autogen_bridge_falls_back_to_string_template_when_autogen_not_installed(monkeypatch, capsys) -> None:
    """If autogen-agentchat itself is NOT importable, the bridge should
    fall back to the plain string-template path. Lifecycle events still
    fire so an operator's activity feed looks consistent."""
    sys.path.insert(0, str(BRIDGE_PATH.parent))
    try:
        import autogen_bridge as bridge
    finally:
        sys.path.pop(0)

    # Force `from autogen_agentchat.agents import AssistantAgent` to
    # raise ImportError by setting the entry in sys.modules to None.
    monkeypatch.setitem(sys.modules, "autogen_agentchat.agents", None)
    monkeypatch.setattr(sys, "argv", ["autogen_bridge.py", "test prompt"])
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))
    monkeypatch.setenv("AX_GATEWAY_AGENT_NAME", "autogen-test")

    rc = bridge.main()
    out = capsys.readouterr()

    assert rc == 0
    event_lines = [line for line in out.out.splitlines() if line.startswith(bridge.EVENT_PREFIX)]
    activities = []
    completed_detail = None
    for line in event_lines:
        payload = json.loads(line[len(bridge.EVENT_PREFIX) :])
        if payload.get("kind") == "activity":
            activities.append(payload.get("activity"))
        if payload.get("kind") == "status" and payload.get("status") == "completed":
            completed_detail = payload.get("detail") or {}

    assert any("autogen-agentchat not installed" in a for a in activities), (
        f"fallback activity event should mention the missing autogen-agentchat package. got: {activities!r}"
    )
    assert completed_detail is not None
    assert completed_detail.get("used_llm") is False
    assert completed_detail.get("stub") is True

    reply_lines = [line for line in out.out.splitlines() if line and not line.startswith(bridge.EVENT_PREFIX)]
    assert reply_lines
    assert "test prompt" in reply_lines[-1], "string-template fallback reply should echo the prompt"
