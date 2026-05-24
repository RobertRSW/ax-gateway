# NEW: not yet vendored from ax-agents. Pending upstream PR before the next
# vendor sync. See ax_cli/runtimes/hermes/README.md for vendoring guidance.
"""Gemini SDK runtime - wraps Google's generativeai API.

Phase 3: multi-turn agent loop with tool calls. The runtime streams a
generate_content call, accumulates text and any tool-call (function-call)
parts, executes requested tools via the shared `tools` module, and
loops until the model emits a final text-only reply (or max_turns is hit).

Tool definitions in this codebase are stored in OpenAI Responses-API
shape (flat `name` field). Gemini expects a nested
`tools=[{"function_declarations": [...]}]` shape, so we adapt on the
way out.

Deferred to Phase 4: Vertex AI / GovCloud auth, rate-limit backoff,
session continuity beyond per-call history.

Auth: GOOGLE_API_KEY environment variable.
Models: https://ai.google.dev/gemini-api/docs/models
        (default: gemini-2.5-flash)
"""

from __future__ import annotations

import json
import logging
import os
import re
import time

from . import BaseRuntime, RuntimeResult, StreamCallback, register

log = logging.getLogger("runtime.gemini_sdk")

DEFAULT_MODEL = "gemini-2.5-flash"
MAX_TURNS = 25
TOOL_OUTPUT_CAP = 10_000  # bytes of tool output fed back to the model per call


# Gemini's Schema proto rejects unknown fields. The shared TOOL_DEFINITIONS use
# OpenAI Responses-shape parameters, which include OpenAPI/JSON-Schema niceties
# like `default`, `examples`, `$schema`, `$ref`, and `title`. Gemini's Schema
# only knows about: type, format, description, nullable, enum, items, properties,
# required, anyOf, propertyOrdering. Anything else crashes the SDK with
# "Unknown field for Schema: <key>" at GenerativeModel(...) instantiation time.
#
# We allowlist what Gemini accepts and recurse into items/properties so nested
# schemas get the same treatment.
_GEMINI_ALLOWED_SCHEMA_KEYS = frozenset(
    {
        "type",
        "format",
        "description",
        "nullable",
        "enum",
        "items",
        "properties",
        "required",
        "anyOf",
        "propertyOrdering",
    }
)


def _sanitize_schema_for_gemini(schema: dict) -> dict:
    """Recursively strip JSON-Schema fields Gemini's Schema proto doesn't accept.

    Examples of fields removed: `default` (most common, used in our tool defs),
    `examples`, `$ref`, `$schema`, `title`. Everything else passes through.
    The recursion handles nested `items` (arrays) and `properties` (objects).
    """
    if not isinstance(schema, dict):
        return schema
    cleaned: dict = {}
    for key, value in schema.items():
        if key not in _GEMINI_ALLOWED_SCHEMA_KEYS:
            continue
        if key == "properties" and isinstance(value, dict):
            cleaned[key] = {k: _sanitize_schema_for_gemini(v) for k, v in value.items()}
        elif key == "items" and isinstance(value, dict):
            cleaned[key] = _sanitize_schema_for_gemini(value)
        elif key == "anyOf" and isinstance(value, list):
            cleaned[key] = [_sanitize_schema_for_gemini(item) for item in value]
        else:
            cleaned[key] = value
    return cleaned


def _to_gemini_function_declaration(rd_tool: dict) -> dict:
    """Convert a Responses-API tool definition to Gemini function_declaration shape.

    Strips JSON-Schema fields that Gemini's Schema proto doesn't accept
    (`default`, `examples`, etc. — see `_sanitize_schema_for_gemini`).
    """
    return {
        "name": rd_tool["name"],
        "description": rd_tool.get("description", ""),
        "parameters": _sanitize_schema_for_gemini(rd_tool.get("parameters", {})),
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


def _history_to_gemini_contents(history: list[dict]) -> list[dict]:
    """Convert the runtime's chat-completions-shape history to Gemini Contents.

    Input rows look like (chat.completions shape, what other runtimes write):
      {"role": "user", "content": "..."}
      {"role": "assistant", "content": "..." | None, "tool_calls": [...]}
      {"role": "tool", "tool_call_id": "...", "content": "..."}

    Gemini Contents shape:
      {"role": "user" | "model" | "function", "parts": [{"text": "..."} | {"function_call": ...} | {"function_response": ...}]}

    The `tool_call_id` doesn't survive the conversion - Gemini pairs
    function_call + function_response by `name` and turn order, not by id.
    """
    contents: list[dict] = []
    for row in history:
        role = row.get("role")
        if role == "user":
            text = row.get("content") or ""
            contents.append({"role": "user", "parts": [{"text": text}]})
        elif role == "assistant":
            parts: list[dict] = []
            text = row.get("content")
            if text:
                parts.append({"text": text})
            for tc in row.get("tool_calls") or []:
                fn = tc.get("function") or {}
                raw_args = fn.get("arguments") or ""
                try:
                    args = json.loads(raw_args) if raw_args else {}
                except json.JSONDecodeError:
                    args = {}
                parts.append({"function_call": {"name": fn.get("name", ""), "args": args}})
            if parts:
                contents.append({"role": "model", "parts": parts})
        elif role == "tool":
            # Gemini pairs response to call by name; we lost call_id but kept content.
            # The caller appends this immediately after the matching assistant turn,
            # so name-resolution works as long as turns alternate correctly.
            contents.append(
                {
                    "role": "function",
                    "parts": [
                        {
                            "function_response": {
                                "name": row.get("_tool_name", ""),
                                "response": {"content": row.get("content") or ""},
                            }
                        }
                    ],
                }
            )
    return contents


@register("gemini_sdk")
class GeminiSDKRuntime(BaseRuntime):
    """Runs agent turns via the Google generativeai SDK.

    Phase 3: multi-turn loop with tool calling. Buffers text deltas
    locally per turn (only emits via StreamCallback.on_text_complete
    once the turn is confirmed text-only - prevents pre-tool chatter
    from leaking as visible chat content and suppressing the sentinel's
    tool-progress UI). Accumulates function_call parts across the
    streaming chunks, executes tools through the shared `tools` module,
    and loops until the model produces a final text-only reply or
    MAX_TURNS is reached.
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
        api_key = os.environ.get("GOOGLE_API_KEY", "").strip()
        if not api_key:
            log.error("gemini_sdk: GOOGLE_API_KEY not set in environment")
            return RuntimeResult(
                text="Agent could not authenticate with Gemini (GOOGLE_API_KEY not set).",
                exit_reason="crashed",
                elapsed_seconds=0,
            )

        try:
            import google.generativeai as genai
        except ImportError as e:
            # pyproject.toml does not declare `google-generativeai` as a hard
            # dependency, so packaged axctl installs will not have it. Surface
            # a clean RuntimeResult so the sentinel can render an actionable
            # message instead of crashing on a bare ModuleNotFoundError.
            log.error(f"gemini_sdk: google-generativeai Python SDK is not installed ({e})")
            return RuntimeResult(
                text=(
                    "Agent could not start because the `google-generativeai` "
                    "Python package is not installed in this runtime environment. "
                    "Install it with `pip install google-generativeai` and retry."
                ),
                exit_reason="crashed",
                elapsed_seconds=0,
            )

        # Absolute import matches openai_sdk.py and the other sibling runtimes.
        # The Hermes sentinel prepends ax_cli/runtimes/hermes to sys.path and
        # loads this module as `runtimes.gemini_sdk`, so a relative
        # `from ..tools` would escape past the top-level package and raise
        # ImportError at runtime. Tests in tests/test_gemini_sdk_runtime.py
        # insert the same hermes directory into sys.path so the absolute form
        # resolves there too.
        from tools import TOOL_DEFINITIONS, execute_tool

        cb = stream_cb or StreamCallback()
        model = model or DEFAULT_MODEL
        instructions = system_prompt or "You are a helpful coding assistant."

        function_declarations = [_to_gemini_function_declaration(t) for t in TOOL_DEFINITIONS]
        gemini_tools = [{"function_declarations": function_declarations}]

        start_time = time.time()
        deadline = start_time + timeout

        # Keep history in the chat-completions shape (same as other runtimes)
        # so RuntimeResult.history stays portable across providers. Convert to
        # Gemini Contents at API-call time.
        history: list[dict] = list((extra_args or {}).get("history", []))
        history.append({"role": "user", "content": message})

        final_text = ""
        tool_count = 0
        files_written: list[str] = []

        genai.configure(api_key=api_key)
        gemini_model = genai.GenerativeModel(
            model_name=model,
            system_instruction=instructions,
            tools=gemini_tools,
        )

        for turn in range(MAX_TURNS):
            now = time.time()
            remaining = deadline - now
            if remaining <= 0:
                log.warning(
                    f"gemini_sdk: timeout exceeded at turn {turn + 1} "
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

            log.info(f"gemini_sdk: turn {turn + 1}, {len(history)} messages")
            contents = _history_to_gemini_contents(history)

            try:
                stream = gemini_model.generate_content(
                    contents,
                    stream=True,
                    request_options={"timeout": remaining},
                )
            except Exception as e:
                error_str = str(e)
                log.error(f"gemini_sdk: API error opening stream: {error_str}")
                lower_err = error_str.lower()
                # Why string-matching instead of catching google.api_core.exceptions
                # subclasses directly: genai.GenerativeModel.generate_content does
                # not consistently surface the typed PermissionDenied /
                # ResourceExhausted / DeadlineExceeded classes through the
                # streaming code path — they often arrive wrapped or stringified.
                # A regex classifier on the message string is the defensive
                # choice. groq_sdk.py and leapfrog_sdk.py both use the typed
                # path because their underlying SDKs surface exceptions cleanly.
                is_timeout = (
                    "timeout" in lower_err
                    or "deadline" in lower_err
                    or "timed out" in lower_err
                )
                # Word-boundary matching so substrings like "rate" don't
                # accidentally match unrelated words (e.g. "generateContent"
                # in a 404 model-not-found error, which would misclassify a
                # crash as a rate limit).
                is_rate_limit = (
                    "429" in error_str
                    or "resourceexhausted" in lower_err
                    or re.search(r"\brate\b", lower_err) is not None
                    or re.search(r"\bquota\b", lower_err) is not None
                )
                # Auth failures (401 / 403) are operator-actionable — the user
                # must see them in the chat reply so they can rotate or fix
                # GOOGLE_API_KEY. Mirrors groq_sdk.py's auth_error exit_reason.
                is_auth_error = (
                    "401" in error_str
                    or "403" in error_str
                    or "permission denied" in lower_err
                    or "permissiondenied" in lower_err
                    or "unauthenticated" in lower_err
                    or re.search(r"\bauthentication\b", lower_err) is not None
                    or re.search(r"\bapi[_ ]key\b", lower_err) is not None
                )
                if is_timeout:
                    return RuntimeResult(
                        text=(final_text or "Agent timed out while waiting for the model."),
                        history=history,
                        tool_count=tool_count,
                        files_written=files_written,
                        exit_reason="timeout",
                        elapsed_seconds=int(time.time() - start_time),
                    )
                if is_rate_limit:
                    return RuntimeResult(
                        text="",
                        history=history,
                        tool_count=tool_count,
                        files_written=files_written,
                        exit_reason="rate_limited",
                        elapsed_seconds=int(time.time() - start_time),
                    )
                if is_auth_error:
                    return RuntimeResult(
                        text="Gemini authentication failed. Check GOOGLE_API_KEY.",
                        history=history,
                        tool_count=tool_count,
                        files_written=files_written,
                        exit_reason="auth_error",
                        elapsed_seconds=int(time.time() - start_time),
                    )
                return RuntimeResult(
                    text=final_text or "Agent encountered an API error and could not complete the task.",
                    history=history,
                    tool_count=tool_count,
                    files_written=files_written,
                    exit_reason="crashed",
                    elapsed_seconds=int(time.time() - start_time),
                )

            # Accumulate text and function_call parts across the stream.
            #
            # Why we buffer text instead of streaming it: if the model says
            # "Let me check that..." and then makes function calls, that
            # pre-tool chatter would leak as visible chat content and
            # suppress the sentinel's tool-progress UI. We only emit via
            # cb.on_text_complete once the turn is confirmed text-only.
            # Mirrors the buffering pattern in openai_sdk.py and groq_sdk.py.
            turn_text = ""
            function_calls: list[dict] = []

            try:
                for chunk in stream:
                    candidates = getattr(chunk, "candidates", None) or []
                    for cand in candidates:
                        content = getattr(cand, "content", None)
                        if content is None:
                            continue
                        for part in getattr(content, "parts", None) or []:
                            # Text part
                            text = getattr(part, "text", None)
                            if text:
                                turn_text += text
                            # Function-call part
                            fc = getattr(part, "function_call", None)
                            if fc is not None and getattr(fc, "name", None):
                                # fc.args is a proto MapComposite; coerce to dict.
                                try:
                                    fc_args = dict(fc.args) if fc.args else {}
                                except Exception:
                                    fc_args = {}
                                function_calls.append(
                                    {
                                        "name": fc.name,
                                        "args": fc_args,
                                    }
                                )
            except Exception as e:
                log.error(f"gemini_sdk: stream error after {len(turn_text)} chars: {e}")
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

            # If the model requested tools, execute them and continue the loop.
            if function_calls:
                # Append the assistant turn carrying the function calls.
                # Use chat-completions shape (with tool_calls) so history
                # stays portable; _history_to_gemini_contents handles the
                # back-conversion to Gemini's parts format on the next turn.
                assistant_tool_calls = []
                for i, fc in enumerate(function_calls):
                    assistant_tool_calls.append(
                        {
                            "id": f"call_{turn}_{i}",
                            "type": "function",
                            "function": {
                                "name": fc["name"],
                                "arguments": json.dumps(fc["args"]),
                            },
                        }
                    )
                history.append(
                    {
                        "role": "assistant",
                        "content": turn_text or None,
                        "tool_calls": assistant_tool_calls,
                    }
                )

                for fc in function_calls:
                    # Re-check the deadline before each tool. A long-running
                    # tool can otherwise block the listener well past the
                    # operator's --timeout.
                    now_tool = time.time()
                    remaining_for_tool = deadline - now_tool
                    if remaining_for_tool <= 0:
                        log.warning(
                            f"gemini_sdk: timeout exceeded before tool "
                            f"{fc['name']} (elapsed {int(now_tool - start_time)}s)"
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
                    name = fc["name"]
                    args = dict(fc["args"])

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
                    log.info(f"gemini_sdk: tool {name}({json.dumps(args, default=str)[:80]})")
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
                            "tool_call_id": f"call_{turn}_{function_calls.index(fc)}",
                            "_tool_name": name,
                            "content": tool_content,
                        }
                    )

                cb.on_status("thinking")
                continue  # Next turn: model sees tool results.

            # No tool calls - text-only response. Treat as final.
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
                f"gemini_sdk: hit MAX_TURNS={MAX_TURNS} without final answer (elapsed {elapsed}s, {tool_count} tools)"
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
        log.info(f"gemini_sdk: done in {elapsed}s, {tool_count} tools, {len(final_text)} chars")
        return RuntimeResult(
            text=final_text,
            history=history,
            session_id=None,
            tool_count=tool_count,
            files_written=files_written,
            exit_reason="done",
            elapsed_seconds=elapsed,
        )
