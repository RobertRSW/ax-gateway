# NEW: not yet vendored from ax-agents. Pending upstream PR before the next
# vendor sync. See ax_cli/runtimes/hermes/README.md for vendoring guidance.
"""LeapfrogAI SDK runtime — wraps Defense Unicorns' OpenAI-compatible endpoint.

Phase 3: multi-turn agent loop with tool calls. The runtime streams a
chat.completions call against a self-hosted LeapfrogAI deployment via the
official `openai` Python SDK pointed at an operator-supplied base_url,
accumulates assistant text and tool_calls across chunks, executes
requested tools through the shared `tools` module, and loops until the
model emits a final text-only reply (or MAX_TURNS is hit).

LeapfrogAI exposes the OpenAI Chat Completions API specification, so the
wire format is identical to openai_sdk's chat.completions path — no
schema adapter or history shape conversion is needed (unlike Gemini).

Deferred to Phase 4: streaming on_text_delta during the turn (currently
buffered for parity with sibling runtimes), gRPC backend path (the
OpenAI SDK uses LeapfrogAI's REST gateway), session continuity beyond
per-call history.

Auth:
  LEAPFROG_API_KEY  — bearer token issued by the LeapfrogAI deployment
  LEAPFROG_BASE_URL — endpoint URL of the deployment (no public default;
                      each deployment is private)

Both are required. Unlike openai_sdk (which has a hardcoded ChatGPT
backend URL) or gemini_sdk (which talks to a public Google endpoint),
LeapfrogAI deployments are per-customer, so the operator MUST supply
the endpoint URL alongside the API key.

Models: operator-configured per deployment. Default is
`llama-3.3-70b-instruct` (the most commonly deployed Llama model on
Defense Unicorns reference deployments). Operators override via the
`model` field in the managed-agent registry entry.
"""

from __future__ import annotations

import json
import logging
import os
import time

from . import BaseRuntime, RuntimeResult, StreamCallback, register

log = logging.getLogger("runtime.leapfrog_sdk")

DEFAULT_MODEL = "llama-3.3-70b-instruct"
MAX_TURNS = 25
TOOL_OUTPUT_CAP = 10_000  # bytes of tool output fed back to the model per call

LEAPFROG_API_KEY_ENV = "LEAPFROG_API_KEY"
LEAPFROG_BASE_URL_ENV = "LEAPFROG_BASE_URL"


def _resolve_auth() -> tuple[tuple[str, str] | None, str]:
    """Read LEAPFROG_API_KEY + LEAPFROG_BASE_URL from the environment.

    Returns ((api_key, base_url), "") on success or (None, error_message)
    when either var is missing. The error message names the missing var
    so the operator can act on it.
    """
    api_key = os.environ.get(LEAPFROG_API_KEY_ENV, "").strip()
    base_url = os.environ.get(LEAPFROG_BASE_URL_ENV, "").strip()
    if not api_key:
        return None, f"{LEAPFROG_API_KEY_ENV} not set"
    if not base_url:
        return (
            None,
            f"{LEAPFROG_BASE_URL_ENV} not set "
            "(LeapfrogAI deployments are private; operator must provide the endpoint URL)",
        )
    return (api_key, base_url), ""


def _to_chat_completion_tool(rd_tool: dict) -> dict:
    """Convert a Responses-API tool definition (flat `name`) into the
    chat-completions `{"type": "function", "function": {...}}` shape that
    LeapfrogAI's OpenAI-compat endpoint accepts.

    Unlike Gemini, no schema field stripping is needed: LeapfrogAI passes
    JSON-Schema through to the underlying model unchanged, so `default`,
    `examples`, `$ref`, and friends are all fine.
    """
    return {
        "type": "function",
        "function": {
            "name": rd_tool["name"],
            "description": rd_tool.get("description", ""),
            "parameters": rd_tool.get("parameters", {}),
        },
    }


def _tool_display(name: str, args: dict) -> str:
    """Human-readable one-liner for tool activity log."""
    if name in ("read_file", "write_file", "edit_file"):
        p = args.get("path", "")
        verb = {"read_file": "Read", "write_file": "Write", "edit_file": "Edit"}[name]
        tail = p.rsplit("/", 1)[-1] if "/" in p else p
        return f"{verb} {tail}"
    if name == "bash":
        cmd = str(args.get("command", ""))[:60]
        return f"Run: {cmd}"
    if name == "grep":
        return f"Search: {args.get('pattern', '')}"
    if name == "glob_files":
        return f"Find: {args.get('pattern', '')}"
    return name


@register("leapfrog_sdk")
class LeapfrogSDKRuntime(BaseRuntime):
    """Runs agent turns via the OpenAI Python SDK pointed at a LeapfrogAI deployment.

    Phase 3: multi-turn loop with tool calling. Buffers text deltas
    locally per turn (only emits via StreamCallback.on_text_complete once
    the turn is confirmed text-only — prevents pre-tool chatter from
    leaking as visible chat content and suppressing the sentinel's
    tool-progress UI). Accumulates tool_call fragments across the
    streaming chunks, executes tools through the shared `tools` module,
    and loops until the model produces a final text-only reply or
    MAX_TURNS is reached.

    Mirrors the deadline-checked + clamped-tool-timeout pattern from
    gemini_sdk.py so a single tool cannot block the sentinel past its
    --timeout budget.
    """

    def execute(
        self,
        message: str,
        *,
        workdir: str,
        model: str | None = None,
        system_prompt: str | None = None,
        session_id: str | None = None,
        stream_cb: StreamCallback | None = None,
        timeout: int = 300,
        extra_args: dict | None = None,
    ) -> RuntimeResult:
        auth, err = _resolve_auth()
        if auth is None:
            log.error(f"leapfrog_sdk: auth resolution failed: {err}")
            return RuntimeResult(
                text=f"Agent could not authenticate with LeapfrogAI ({err}).",
                exit_reason="crashed",
                elapsed_seconds=0,
            )
        api_key, base_url = auth

        try:
            from openai import (
                OpenAI,
                APIStatusError,
                APITimeoutError,
                AuthenticationError,
                InternalServerError,
                PermissionDeniedError,
                RateLimitError,
            )
        except ImportError as e:
            # pyproject.toml does not declare `openai` as a hard dep — the
            # sibling openai_sdk.py is also lazy-imported. Packaged axctl
            # installs may not have it. Surface a clean RuntimeResult so the
            # sentinel can render an actionable message instead of crashing
            # on a bare ModuleNotFoundError.
            log.error(f"leapfrog_sdk: openai Python SDK is not installed ({e})")
            return RuntimeResult(
                text=(
                    "Agent could not start because the `openai` Python package "
                    "is not installed in this runtime environment. "
                    "Install it with `pip install openai` and retry."
                ),
                exit_reason="crashed",
                elapsed_seconds=0,
            )

        # Absolute import matches the sibling runtimes. The Hermes sentinel
        # prepends ax_cli/runtimes/hermes to sys.path and loads this module
        # as `runtimes.leapfrog_sdk`, so a relative `from ..tools` would
        # escape past the top-level package and raise ImportError at runtime.
        # tests/test_leapfrog_sdk_runtime.py inserts the same hermes
        # directory into sys.path so the absolute form resolves there too.
        from tools import TOOL_DEFINITIONS, execute_tool

        cb = stream_cb or StreamCallback()
        model = model or DEFAULT_MODEL
        instructions = system_prompt or "You are a helpful coding assistant."

        tools = [_to_chat_completion_tool(t) for t in TOOL_DEFINITIONS]

        start_time = time.time()
        deadline = start_time + timeout

        # Build the messages list. LeapfrogAI uses the OpenAI chat-completions
        # message shape natively, so history rows pass straight through.
        history: list[dict] = list((extra_args or {}).get("history", []))
        history.append({"role": "user", "content": message})

        final_text = ""
        tool_count = 0
        files_written: list[str] = []

        client = OpenAI(api_key=api_key, base_url=base_url)

        for turn in range(MAX_TURNS):
            now = time.time()
            remaining = deadline - now
            if remaining <= 0:
                log.warning(
                    f"leapfrog_sdk: timeout exceeded at turn {turn + 1} "
                    f"(budget={timeout}s, elapsed {int(now - start_time)}s)"
                )
                return RuntimeResult(
                    text=(final_text or "Agent timed out before producing a final answer."),
                    history=history,
                    session_id=None,
                    tool_count=tool_count,
                    files_written=files_written,
                    exit_reason="timeout",
                    elapsed_seconds=int(now - start_time),
                )

            log.info(f"leapfrog_sdk: turn {turn + 1}, {len(history)} messages")

            # Chat-completions expects a `system` message at the front, not a
            # separate `system_instruction` kwarg (that's a Gemini-ism). Prepend
            # one each turn so the model's behavior is consistent across turns.
            messages = [{"role": "system", "content": instructions}, *history]

            try:
                stream = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    tools=tools,
                    stream=True,
                    timeout=remaining,
                )
            except RateLimitError as e:
                # 429. Throttle, surface the status code so the operator can
                # tell rate-limit from auth-fail at a glance.
                log.warning(f"leapfrog_sdk: rate limited (HTTP {e.status_code}): {e.message}")
                return RuntimeResult(
                    text=(
                        final_text
                        or f"LeapfrogAI deployment rate-limited (HTTP {e.status_code}). Retry after a short delay."
                    ),
                    history=history,
                    tool_count=tool_count,
                    files_written=files_written,
                    exit_reason="rate_limited",
                    elapsed_seconds=int(time.time() - start_time),
                )
            except APITimeoutError as e:
                # Connection or read timeout from the openai SDK (httpx-backed).
                # Distinct from a sentinel-budget timeout, but maps to the same
                # exit_reason since the user-visible cause is identical.
                log.error(f"leapfrog_sdk: API timeout: {e}")
                return RuntimeResult(
                    text=(
                        final_text
                        or "Agent timed out while waiting for the model."
                    ),
                    history=history,
                    tool_count=tool_count,
                    files_written=files_written,
                    exit_reason="timeout",
                    elapsed_seconds=int(time.time() - start_time),
                )
            except (AuthenticationError, PermissionDeniedError) as e:
                # 401 / 403. Operator-actionable — the user must see this in
                # the chat reply so they can rotate LEAPFROG_API_KEY or fix
                # deployment ACLs. Never silently swallow auth failures.
                log.error(f"leapfrog_sdk: auth failed (HTTP {e.status_code}): {e.message}")
                return RuntimeResult(
                    text=f"LeapfrogAI authentication failed (HTTP {e.status_code}). Check LEAPFROG_API_KEY and deployment ACLs.",
                    history=history,
                    tool_count=tool_count,
                    files_written=files_written,
                    exit_reason="auth_error",
                    elapsed_seconds=int(time.time() - start_time),
                )
            except InternalServerError as e:
                # 5xx from the LeapfrogAI deployment. Retry is plausible;
                # signal that to the operator.
                log.error(f"leapfrog_sdk: server error (HTTP {e.status_code}): {e.message}")
                return RuntimeResult(
                    text=(
                        final_text
                        or f"LeapfrogAI deployment returned HTTP {e.status_code}. Retry may succeed."
                    ),
                    history=history,
                    tool_count=tool_count,
                    files_written=files_written,
                    exit_reason="server_error",
                    elapsed_seconds=int(time.time() - start_time),
                )
            except APIStatusError as e:
                # Any other 4xx not matched above (e.g. 400 BadRequest,
                # 422 UnprocessableEntity, 404 NotFound). Surface the status
                # and the message so the operator knows what to fix.
                log.error(f"leapfrog_sdk: API error (HTTP {e.status_code}): {e.message}")
                return RuntimeResult(
                    text=(
                        final_text
                        or f"LeapfrogAI API error (HTTP {e.status_code}): {e.message}"
                    ),
                    history=history,
                    tool_count=tool_count,
                    files_written=files_written,
                    exit_reason="api_error",
                    elapsed_seconds=int(time.time() - start_time),
                )
            except Exception as e:
                # Catch-all for anything outside the openai SDK's typed
                # exception hierarchy (network adapter bugs, connection
                # refused before an APIConnectionError, etc.). Logged with
                # full repr so the underlying type is visible in ops triage.
                log.error(f"leapfrog_sdk: unexpected error opening stream: {e!r}")
                return RuntimeResult(
                    text=final_text or "Agent encountered an unexpected error and could not complete the task.",
                    history=history,
                    tool_count=tool_count,
                    files_written=files_written,
                    exit_reason="crashed",
                    elapsed_seconds=int(time.time() - start_time),
                )

            # Accumulate text and tool_call fragments across the stream.
            #
            # Why we buffer text instead of streaming via on_text_delta: if the
            # model says "Let me check that..." and then makes function calls,
            # that pre-tool chatter would leak as visible chat content and
            # suppress the sentinel's tool-progress UI. We only emit via
            # cb.on_text_complete once the turn is confirmed text-only.
            # Mirrors the buffering pattern in openai_sdk.py and gemini_sdk.py.
            #
            # OpenAI tool_call streaming: each chunk's delta.tool_calls is a
            # sparse list keyed by `index` — fragments arrive across many
            # chunks. We accumulate by index into a dict, then materialize at
            # turn end.
            turn_text = ""
            tool_call_fragments: dict[int, dict] = {}

            try:
                for chunk in stream:
                    choices = getattr(chunk, "choices", None) or []
                    for choice in choices:
                        delta = getattr(choice, "delta", None)
                        if delta is None:
                            continue

                        # Text content fragment
                        content = getattr(delta, "content", None)
                        if content:
                            turn_text += content

                        # Tool call fragments (sparse, indexed)
                        deltas = getattr(delta, "tool_calls", None) or []
                        for tc_delta in deltas:
                            idx = getattr(tc_delta, "index", 0)
                            slot = tool_call_fragments.setdefault(
                                idx,
                                {"id": "", "name": "", "arguments": ""},
                            )
                            tc_id = getattr(tc_delta, "id", None)
                            if tc_id:
                                slot["id"] = tc_id
                            fn = getattr(tc_delta, "function", None)
                            if fn is not None:
                                fn_name = getattr(fn, "name", None)
                                if fn_name:
                                    slot["name"] = fn_name
                                fn_args = getattr(fn, "arguments", None)
                                if fn_args:
                                    slot["arguments"] += fn_args
            except Exception as e:
                log.error(f"leapfrog_sdk: stream error after {len(turn_text)} chars: {e}")
                partial = turn_text.strip()
                if partial:
                    history.append({"role": "assistant", "content": partial})
                return RuntimeResult(
                    text=partial or "Agent encountered a stream error mid-response.",
                    history=history,
                    tool_count=tool_count,
                    files_written=files_written,
                    exit_reason="crashed",
                    elapsed_seconds=int(time.time() - start_time),
                )

            # Materialize tool calls in arrival order so call_id/name/args
            # alignment matches what the model emitted. Skip any slot that
            # never got a name (defensive — shouldn't happen on a healthy stream).
            tool_calls = [
                {
                    "id": slot["id"] or f"call_{turn}_{idx}",
                    "name": slot["name"],
                    "arguments": slot["arguments"],
                }
                for idx, slot in sorted(tool_call_fragments.items())
                if slot["name"]
            ]

            if tool_calls:
                # Append the assistant turn carrying the tool calls in
                # chat-completions shape (already the native shape — no
                # back-conversion needed, unlike Gemini).
                assistant_tool_calls = [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": tc["arguments"] or "{}",
                        },
                    }
                    for tc in tool_calls
                ]
                history.append(
                    {
                        "role": "assistant",
                        "content": turn_text or None,
                        "tool_calls": assistant_tool_calls,
                    }
                )

                for tc in tool_calls:
                    # Re-check the deadline before each tool. A long-running
                    # tool can otherwise block the listener well past the
                    # operator's --timeout.
                    now_tool = time.time()
                    remaining_for_tool = deadline - now_tool
                    if remaining_for_tool <= 0:
                        log.warning(
                            f"leapfrog_sdk: timeout exceeded before tool "
                            f"{tc['name']} (elapsed {int(now_tool - start_time)}s)"
                        )
                        return RuntimeResult(
                            text=(final_text or "Agent timed out before completing tool calls."),
                            history=history,
                            session_id=None,
                            tool_count=tool_count,
                            files_written=files_written,
                            exit_reason="timeout",
                            elapsed_seconds=int(now_tool - start_time),
                        )

                    tool_count += 1
                    name = tc["name"]
                    try:
                        args = json.loads(tc["arguments"]) if tc["arguments"] else {}
                    except json.JSONDecodeError:
                        args = {}

                    # Clamp any model-supplied "timeout" arg to the remaining
                    # wall-clock budget. Tools like `bash` honor args["timeout"]
                    # directly, so without this a model could request a 600s
                    # bash inside a 30s sentinel budget. Tools without a
                    # "timeout" arg are unaffected.
                    if "timeout" in args:
                        try:
                            args["timeout"] = min(
                                int(args["timeout"]),
                                max(1, int(remaining_for_tool)),
                            )
                        except (TypeError, ValueError):
                            args["timeout"] = max(1, int(remaining_for_tool))

                    summary = _tool_display(name, args)
                    log.info(f"leapfrog_sdk: tool {name}({json.dumps(args, default=str)[:80]})")
                    cb.on_tool_start(name, summary)
                    result = execute_tool(name, args, workdir)

                    if name == "write_file" and not result.is_error:
                        files_written.append(args.get("path", ""))

                    short = result.output[:200] if result.output else ""
                    cb.on_tool_end(name, short)

                    # Cap tool output at TOOL_OUTPUT_CAP bytes to bound context
                    # growth, and surface a truncation marker when we hit the
                    # cap so the model can tell content was clipped (otherwise
                    # it may reason as if it has the full output, e.g. assume a
                    # large file was fully read). Mirrors groq_sdk.py 411-427.
                    full_output = result.output or ""
                    if len(full_output) > TOOL_OUTPUT_CAP:
                        tool_content = full_output[:TOOL_OUTPUT_CAP] + "\n[output truncated]"
                    else:
                        tool_content = full_output
                    history.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "content": tool_content,
                        }
                    )

                cb.on_status("thinking")
                continue  # Next turn: model sees tool results.

            # No tool calls — text-only response. Treat as final.
            visible = turn_text.strip()
            if visible:
                final_text = visible
                cb.on_text_complete(final_text)
                history.append({"role": "assistant", "content": visible})
            break
        else:
            # The for-loop completed without break, meaning every turn produced
            # tool calls and the model never finalized. Surface this as
            # iteration_limit so the sentinel renders a bounded-loop notice
            # rather than a misleading "Completed with no text output".
            elapsed = int(time.time() - start_time)
            log.warning(
                f"leapfrog_sdk: hit MAX_TURNS={MAX_TURNS} without final answer "
                f"(elapsed {elapsed}s, {tool_count} tools)"
            )
            return RuntimeResult(
                text=(final_text or "Agent hit the maximum turn limit without producing a final answer."),
                history=history,
                session_id=None,
                tool_count=tool_count,
                files_written=files_written,
                exit_reason="iteration_limit",
                elapsed_seconds=elapsed,
            )

        elapsed = int(time.time() - start_time)
        log.info(f"leapfrog_sdk: done in {elapsed}s, {tool_count} tools, {len(final_text)} chars")
        return RuntimeResult(
            text=final_text,
            history=history,
            session_id=None,
            tool_count=tool_count,
            files_written=files_written,
            exit_reason="done",
            elapsed_seconds=elapsed,
        )
