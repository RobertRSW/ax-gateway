#!/usr/bin/env python3
"""Gateway-managed bridge for an AutoGen agent.

This bridge is designed for `ax gateway agents add ... --template autogen`.
It runs once per inbound mention: read the prompt, route it through an
AutoGen `AssistantAgent`, and print the reply on stdout.

Three execution tiers, picked at runtime by what is installed and
configured. Mirrors the langgraph bridge contract so an operator's
activity feed looks consistent regardless of which template a Gateway
agent runs.

  1. Real LLM path. If `autogen-agentchat` AND `autogen-ext` are
     importable AND GROQ_API_KEY is set, the bridge builds an
     AssistantAgent wired to a Groq-backed OpenAIChatCompletionClient
     pointed at https://api.groq.com/openai/v1 (Groq is
     OpenAI-compatible at that endpoint, so no Groq-specific AutoGen
     extension is needed). The bridge then drives a single turn via
     `agent.on_messages_stream()` with `model_client_stream=True` so
     token-level chunks surface as throttled ~1s activity events with
     a rolling preview, keeping the operator's activity feed live
     during the call. Env overrides: AX_BRIDGE_LLM_MODEL (default
     llama-3.3-70b-versatile), AX_BRIDGE_SYSTEM_PROMPT (default
     "Reply concisely.", appended to the agent's system message).

  2. Stub agent path. If `autogen-agentchat` is importable but Groq is
     not configured, the bridge emits an activity event explaining the
     fallback and returns a stub ack without constructing the agent
     (AutoGen's AssistantAgent requires a working model_client, so
     "construct but don't call" is not a clean pattern the way it is
     for CrewAI). The same lifecycle events still fire.

  3. String fallback path. If `autogen-agentchat` itself is not
     installed, the bridge returns a plain string template. Same
     lifecycle events still fire.

The three-tier shape lets the bridge round-trip a reply through the
Gateway end to end in CI / dev without LLM creds, and switch to real
LLM execution in production environments where GROQ_API_KEY is
provisioned. Multi-agent teams (RoundRobinGroupChat, SelectorGroupChat)
and tool calls are deliberate follow-ups.

Target AutoGen line: modern `autogen-agentchat` >=0.4 (async-first).
Not the legacy `pyautogen` 0.2 line, which Microsoft is phasing out.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from typing import Any, NamedTuple

EVENT_PREFIX = "AX_GATEWAY_EVENT "


def emit_event(payload: dict[str, Any]) -> None:
    print(f"{EVENT_PREFIX}{json.dumps(payload, sort_keys=True)}", flush=True)


def _read_prompt() -> str:
    if len(sys.argv) > 1 and sys.argv[-1] != "-":
        return sys.argv[-1]
    env_prompt = os.environ.get("AX_MENTION_CONTENT", "").strip()
    if env_prompt:
        return env_prompt
    return sys.stdin.read().strip()


def _agent_name() -> str:
    return (
        os.environ.get("AX_GATEWAY_AGENT_NAME", "").strip()
        or os.environ.get("AX_AGENT_NAME", "").strip()
        or "autogen-bot"
    )


DEFAULT_LLM_MODEL = "llama-3.3-70b-versatile"
GROQ_OPENAI_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_SYSTEM_PROMPT_TAIL = "Reply concisely."
ACTIVITY_HEARTBEAT_SECONDS = 1.0
PREVIEW_MAX_CHARS = 180


class RunResult(NamedTuple):
    reply: str
    used_llm: bool


def _system_prompt_tail() -> str:
    return os.environ.get("AX_BRIDGE_SYSTEM_PROMPT", "").strip() or DEFAULT_SYSTEM_PROMPT_TAIL


def _autogen_safe_agent_name(name: str) -> str:
    """AutoGen agent names must be valid Python identifiers.

    Gateway agent names typically include hyphens (e.g. `autogen-bot`).
    Substitute hyphens with underscores so the name is acceptable to
    AutoGen's AssistantAgent constructor, which validates against
    Python identifier rules.
    """
    safe = name.replace("-", "_")
    if not safe.isidentifier():
        return "ax_autogen_bot"
    return safe


def _build_model_client(model: str):
    """Build an OpenAIChatCompletionClient pointed at Groq's OpenAI-compatible endpoint.

    Raises ImportError if `autogen-ext` is not installed (caller treats
    as a signal to fall back to the stub path). We do not require the
    `openai` SDK to be importable here, because `autogen-ext` already
    declares it as a transitive dependency.

    The model_info dict tells AutoGen what capabilities to advertise
    for this model. function_calling is enabled because Groq's Llama
    models support tool use through the OpenAI-compatible API.
    """
    from autogen_ext.models.openai import OpenAIChatCompletionClient

    return OpenAIChatCompletionClient(
        model=model,
        base_url=GROQ_OPENAI_BASE_URL,
        api_key=os.environ["GROQ_API_KEY"],
        model_info={
            "vision": False,
            "function_calling": True,
            "json_output": False,
            "family": "unknown",
            "structured_output": False,
        },
    )


async def _run_agent_stream(agent, prompt: str, model: str) -> str:
    """Send a single user message to the AutoGen agent and consume the
    streaming events, surfacing throttled activity events to the
    Gateway activity feed during the call.

    Uses `agent.on_messages_stream()` (async iterator) so operators
    see the bridge thinking instead of staring at silence during the
    multi-second LLM call. Matches the LangGraph bridge's pattern from
    PR #38 review pass. The agent's `model_client_stream=True` flag is
    what makes the underlying OpenAIChatCompletionClient emit
    `ModelClientStreamingChunkEvent` items during the call. Without
    that flag, the stream still works but only yields the final
    Response at the end (no per-token signal).

    Event types in the stream, identified by duck-typing.

    - Final `Response` carries a `chat_message` attribute. We extract
      the chat_message content as the canonical reply text.
    - `ModelClientStreamingChunkEvent` carries a `content` string with
      the partial token text. We accumulate these and emit throttled
      `activity` events with a rolling preview (~1s heartbeat).

    Other inner event types (tool-call events, etc.) are ignored at
    this layer.
    """
    from autogen_agentchat.messages import TextMessage
    from autogen_core import CancellationToken

    chunks: list[str] = []
    final_reply = ""
    first_chunk_seen = False
    last_activity_at = 0.0

    stream = agent.on_messages_stream(
        [TextMessage(content=prompt, source="user")],
        cancellation_token=CancellationToken(),
    )
    async for event in stream:
        chat_message = getattr(event, "chat_message", None)
        if chat_message is not None:
            inner_content = getattr(chat_message, "content", None)
            final_reply = str(inner_content or "")
            continue

        content = getattr(event, "content", None)
        if not isinstance(content, str) or not content:
            continue

        chunks.append(content)
        now = time.monotonic()
        if not first_chunk_seen:
            first_chunk_seen = True
            emit_event(
                {
                    "kind": "status",
                    "status": "processing",
                    "message": f"Groq is responding ({model})",
                }
            )
        if now - last_activity_at >= ACTIVITY_HEARTBEAT_SECONDS:
            preview = "".join(chunks).strip().replace("\n", " ")
            if len(preview) > PREVIEW_MAX_CHARS:
                preview = "..." + preview[-(PREVIEW_MAX_CHARS - 3) :]
            emit_event(
                {
                    "kind": "activity",
                    "activity": (f"{model}: {preview}" if preview else f"Streaming response from {model}..."),
                }
            )
            last_activity_at = now

    # Prefer the final Response content if it landed, else fall back to
    # the accumulated chunks (some AutoGen builds may not emit a final
    # Response if the stream completes via an internal terminator).
    return final_reply or "".join(chunks).strip()


def _run_autogen(prompt: str) -> RunResult:
    """Run an AutoGen AssistantAgent (real LLM or stub) if autogen-agentchat
    is available, else a plain string template.

    Returns a RunResult naming the reply and whether the real LLM path
    was taken so main() can report it accurately in the completion
    event without a side-channel. See the module docstring for the
    three-tier behavior.
    """
    try:
        from autogen_agentchat.agents import AssistantAgent
    except ImportError:
        emit_event(
            {
                "kind": "activity",
                "activity": "autogen-agentchat not installed; using stub reply (install autogen-agentchat for real agent execution)",
            }
        )
        return RunResult(
            reply=f"AutoGen stub ack from @{_agent_name()}: {prompt}",
            used_llm=False,
        )

    groq_key = os.environ.get("GROQ_API_KEY", "").strip()
    model = os.environ.get("AX_BRIDGE_LLM_MODEL", DEFAULT_LLM_MODEL).strip() or DEFAULT_LLM_MODEL
    model_client = None
    if groq_key:
        try:
            model_client = _build_model_client(model)
        except ImportError:
            emit_event(
                {
                    "kind": "activity",
                    "activity": "GROQ_API_KEY set but autogen-ext not installed; falling back to stub agent",
                }
            )

    agent_name = _agent_name()
    safe_name = _autogen_safe_agent_name(agent_name)
    system_message = f"You are @{agent_name}, an assistant routed through the aX Gateway. {_system_prompt_tail()}"

    if model_client is not None:
        emit_event(
            {
                "kind": "activity",
                "activity": f"building AutoGen AssistantAgent with Groq model client (model={model})",
            }
        )
        emit_event(
            {
                "kind": "status",
                "status": "processing",
                "message": f"Calling Groq ({model}) via AutoGen",
            }
        )
        agent = AssistantAgent(
            name=safe_name,
            model_client=model_client,
            model_client_stream=True,
            system_message=system_message,
        )
        try:
            reply = asyncio.run(_run_agent_stream(agent, prompt, model))
        finally:
            # AutoGen model clients hold an underlying httpx client that
            # should be closed to avoid "unclosed connection" warnings
            # on subprocess exit. Best-effort, the bridge is one-shot so
            # leaks would not accumulate across runs.
            try:
                asyncio.run(model_client.close())
            except Exception:
                pass
        return RunResult(reply=reply.strip(), used_llm=True)

    emit_event(
        {
            "kind": "activity",
            "activity": "AutoGen installed but no Groq model client wired; using stub reply (set GROQ_API_KEY for real agent execution)",
        }
    )
    return RunResult(
        reply=f"AutoGen ack from @{agent_name}: {prompt}",
        used_llm=False,
    )


def main() -> int:
    prompt = _read_prompt()
    if not prompt:
        print("(no mention content received)", file=sys.stderr)
        return 1

    started = time.monotonic()
    emit_event(
        {
            "kind": "status",
            "status": "processing",
            "message": "Routing prompt through AutoGen bridge",
        }
    )

    try:
        result = _run_autogen(prompt)
    except Exception as exc:
        emit_event({"kind": "status", "status": "error", "error_message": str(exc)})
        print(f"AutoGen bridge failed: {exc}", file=sys.stderr)
        return 1

    duration_ms = int((time.monotonic() - started) * 1000)
    emit_event(
        {
            "kind": "status",
            "status": "completed",
            "message": f"AutoGen bridge completed in {duration_ms}ms",
            "detail": {
                "duration_ms": duration_ms,
                # `stub` is kept for back-compat with the pre-LLM-validation
                # bridge schema; downstream consumers may still key off it.
                # New consumers should prefer `used_llm`.
                "stub": not result.used_llm,
                "used_llm": result.used_llm,
            },
        }
    )
    print(result.reply or f"AutoGen bridge for @{_agent_name()} finished without text.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
