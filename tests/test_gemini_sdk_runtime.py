"""Tests for the Gemini SDK runtime adapter.

The Gemini SDK (google.generativeai) is mocked via sys.modules so these
tests run offline and do not consume API credits. Coverage spans
registration discovery, the missing-API-key path, the happy streaming
path (callback fan-out, RuntimeResult shape, history accumulation),
system prompt threading, function-call dispatch across turns, partial-
failure handling when the stream raises mid-response, the
missing-package path, tool-timeout clamping, the wall-clock deadline,
and the iteration-limit ceiling.

Mirrors the test conventions in tests/test_groq_sdk_runtime.py so the
two provider suites read the same to reviewers.
"""

from __future__ import annotations

import os
import sys
import types
from unittest.mock import MagicMock

import pytest  # noqa: F401  (pytest is the test runner; import keeps tooling happy)

# The Hermes sentinel prepends ax_cli/runtimes/hermes to sys.path in production
# so vendored runtimes can do `from tools import ...` as an absolute import.
# Replicate that here so the same import path resolves under pytest.
_HERMES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "ax_cli",
    "runtimes",
    "hermes",
)
if _HERMES_DIR not in sys.path:
    sys.path.insert(0, _HERMES_DIR)

# Importing the module triggers `@register("gemini_sdk")` at module load time,
# so the runtime is in REGISTRY regardless of which other tests in the suite
# may have already populated it (get_runtime's auto-discovery only fires when
# REGISTRY is fully empty).
from ax_cli.runtimes.hermes.runtimes import gemini_sdk  # noqa: F401, E402

# ── Helpers ────────────────────────────────────────────────────────────────


def _fake_text_part(text: str):
    """Build a duck-typed Gemini Part that carries text."""
    return types.SimpleNamespace(text=text, function_call=None)


def _fake_function_call_part(name: str, args: dict):
    """Build a duck-typed Gemini Part that carries a function_call.

    fc.args in the real SDK is a proto MapComposite; the runtime coerces
    it via `dict(fc.args)`, so any dict-coercible value works for tests.
    """
    fc = types.SimpleNamespace(name=name, args=args)
    return types.SimpleNamespace(text=None, function_call=fc)


def _fake_chunk(*parts):
    """Build a streaming chunk wrapping a single candidate with these parts."""
    content = types.SimpleNamespace(parts=list(parts))
    candidate = types.SimpleNamespace(content=content)
    return types.SimpleNamespace(candidates=[candidate])


def _install_fake_genai(monkeypatch, fake_model):
    """Swap `google.generativeai` in sys.modules so `import google.generativeai`
    returns our mock. We have to install both the parent `google` package and
    the `generativeai` submodule because the runtime does
    `import google.generativeai as genai`.
    """
    fake_generativeai = types.ModuleType("google.generativeai")
    fake_generativeai.configure = MagicMock()
    fake_generativeai.GenerativeModel = MagicMock(return_value=fake_model)

    fake_google = types.ModuleType("google")
    fake_google.generativeai = fake_generativeai

    monkeypatch.setitem(sys.modules, "google", fake_google)
    monkeypatch.setitem(sys.modules, "google.generativeai", fake_generativeai)
    return fake_generativeai


class _RecordingCallback:
    """Minimal StreamCallback implementation that records what it sees."""

    def __init__(self):
        self.deltas: list[str] = []
        self.complete: str | None = None
        self.statuses: list[str] = []
        self.tool_starts: list[tuple[str, str]] = []

    def on_text_delta(self, text: str) -> None:
        self.deltas.append(text)

    def on_text_complete(self, text: str) -> None:
        self.complete = text

    def on_tool_start(self, name: str, summary: str) -> None:
        self.tool_starts.append((name, summary))

    def on_tool_end(self, *_args, **_kwargs) -> None:
        pass

    def on_status(self, status: str) -> None:
        self.statuses.append(status)


# ── Tests ──────────────────────────────────────────────────────────────────


def test_sanitize_schema_strips_gemini_incompatible_fields():
    """Gemini's Schema proto rejects unknown fields with
    `ValueError: Unknown field for Schema: <key>`. The sanitizer must drop
    `default`, `examples`, `$ref`, `$schema`, `title`, and any other field
    not on Gemini's allow list, recursing into nested `properties` / `items` /
    `anyOf`. Regression test for the live-smoke bug caught on 2026-05-14
    where Hermes' read_file tool definition (which has `default: 1` and
    `default: 2000` on its offset/limit properties) crashed
    GenerativeModel(...) instantiation.
    """
    from ax_cli.runtimes.hermes.runtimes.gemini_sdk import _sanitize_schema_for_gemini

    dirty = {
        "type": "object",
        "title": "ReadFileArgs",  # not allowed
        "$schema": "http://json-schema.org/draft-07/schema#",  # not allowed
        "properties": {
            "path": {"type": "string", "description": "Path to read"},
            "offset": {
                "type": "integer",
                "description": "Start line (1-indexed)",
                "default": 1,  # the field that broke the live smoke
                "examples": [1, 5, 100],  # also not allowed
            },
            "lines": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"n": {"type": "integer", "default": 0}},
                },
            },
        },
        "required": ["path"],
    }

    cleaned = _sanitize_schema_for_gemini(dirty)

    # Top-level forbidden keys stripped:
    assert "title" not in cleaned
    assert "$schema" not in cleaned
    # Allowed top-level keys preserved:
    assert cleaned["type"] == "object"
    assert cleaned["required"] == ["path"]
    # Nested properties cleaned recursively:
    offset = cleaned["properties"]["offset"]
    assert "default" not in offset
    assert "examples" not in offset
    assert offset["type"] == "integer"
    assert offset["description"] == "Start line (1-indexed)"
    # Recursion through `items` and then into nested `properties`:
    nested = cleaned["properties"]["lines"]["items"]["properties"]["n"]
    assert "default" not in nested
    assert nested["type"] == "integer"


def test_gemini_sdk_does_not_misclassify_404_as_rate_limited(monkeypatch):
    """Regression test for the live-smoke bug caught on 2026-05-14: a 404
    model-not-found error contains the substring 'rate' inside the word
    'generateContent', which the original error classifier matched via
    naive `'rate' in error_str.lower()`. That caused exit_reason='rate_limited'
    when the real cause was a crash. The classifier now uses word-boundary
    regex matching so unrelated substrings don't trigger false positives."""
    from ax_cli.runtimes.hermes.runtimes import get_runtime

    monkeypatch.setenv("GOOGLE_API_KEY", "test_key")

    fake_model = MagicMock()
    fake_model.generate_content.side_effect = RuntimeError(
        "404 models/gemini-1.5-flash is not found for API version v1beta, "
        "or is not supported for generateContent. Call ListModels to see "
        "the list of available models and their supported methods."
    )
    _install_fake_genai(monkeypatch, fake_model)

    rt = get_runtime("gemini_sdk")
    result = rt.execute("hello", workdir="/tmp")

    # The error string contains "rate" as a substring of "generateContent",
    # but no whole-word "rate" or "quota". Should NOT be rate_limited.
    assert result.exit_reason == "crashed", (
        f"404-not-found misclassified as {result.exit_reason!r}; "
        "the word-boundary regex should reject substring matches"
    )


def test_gemini_sdk_does_classify_real_rate_limit_correctly(monkeypatch):
    """Companion to the misclassifier regression: an error that legitimately
    mentions rate limiting should still be picked up by the classifier."""
    from ax_cli.runtimes.hermes.runtimes import get_runtime

    monkeypatch.setenv("GOOGLE_API_KEY", "test_key")

    fake_model = MagicMock()
    fake_model.generate_content.side_effect = RuntimeError(
        "429 Resource exhausted: rate limit exceeded for model gemini-2.5-flash"
    )
    _install_fake_genai(monkeypatch, fake_model)

    rt = get_runtime("gemini_sdk")
    result = rt.execute("hello", workdir="/tmp")

    assert result.exit_reason == "rate_limited"


def test_gemini_sdk_registers_under_expected_name():
    from ax_cli.runtimes.hermes.runtimes import get_runtime

    rt = get_runtime("gemini_sdk")
    assert type(rt).__name__ == "GeminiSDKRuntime"
    assert rt.name == "gemini_sdk"


def test_gemini_sdk_returns_crashed_when_api_key_missing(monkeypatch):
    """No API key in env should short-circuit before any genai import."""
    from ax_cli.runtimes.hermes.runtimes import get_runtime

    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    rt = get_runtime("gemini_sdk")
    result = rt.execute("hello", workdir="/tmp")

    assert result.exit_reason == "crashed"
    assert "GOOGLE_API_KEY" in result.text
    assert result.elapsed_seconds == 0


def test_gemini_sdk_streams_chunks_and_accumulates_history(monkeypatch):
    """Happy path: text buffers locally, on_text_complete fires once, history grows."""
    from ax_cli.runtimes.hermes.runtimes import get_runtime

    monkeypatch.setenv("GOOGLE_API_KEY", "test_key")
    fake_model = MagicMock()
    fake_model.generate_content.return_value = iter(
        [
            _fake_chunk(_fake_text_part("Hello ")),
            _fake_chunk(_fake_text_part("world.")),
        ]
    )
    _install_fake_genai(monkeypatch, fake_model)

    rt = get_runtime("gemini_sdk")
    cb = _RecordingCallback()
    result = rt.execute(
        "Say hello.",
        workdir="/tmp",
        stream_cb=cb,
    )

    # No incremental deltas should fire; text is buffered until the turn is
    # confirmed text-only (no function_calls), matching openai_sdk.py and
    # groq_sdk.py's pattern.
    assert cb.deltas == []
    # on_text_complete fires with the assembled text.
    assert cb.complete == "Hello world."
    # RuntimeResult fields.
    assert result.exit_reason == "done"
    assert result.text == "Hello world."
    assert result.tool_count == 0
    assert result.files_written == []
    # History records the round trip: user prompt + assistant reply.
    assert len(result.history) == 2
    assert result.history[0] == {"role": "user", "content": "Say hello."}
    assert result.history[1] == {"role": "assistant", "content": "Hello world."}
    # The runtime requested streaming explicitly.
    assert fake_model.generate_content.call_args.kwargs["stream"] is True


def test_gemini_sdk_threads_system_prompt_into_model_configuration(monkeypatch):
    """The system_prompt arg should become the system_instruction kwarg on
    GenerativeModel(...). Gemini doesn't take a 'system' role in contents;
    instructions live on the model object itself."""
    from ax_cli.runtimes.hermes.runtimes import get_runtime

    monkeypatch.setenv("GOOGLE_API_KEY", "test_key")
    fake_model = MagicMock()
    fake_model.generate_content.return_value = iter([_fake_chunk(_fake_text_part("ok"))])
    fake_generativeai = _install_fake_genai(monkeypatch, fake_model)

    rt = get_runtime("gemini_sdk")
    rt.execute(
        "Question.",
        workdir="/tmp",
        system_prompt="You are a strict reviewer.",
    )

    # GenerativeModel should have been instantiated with system_instruction
    # set to our system_prompt arg.
    init_call = fake_generativeai.GenerativeModel.call_args
    assert init_call.kwargs.get("system_instruction") == "You are a strict reviewer."
    # And the user message is in the contents passed to generate_content.
    contents = fake_model.generate_content.call_args.args[0]
    assert contents[-1]["role"] == "user"
    assert contents[-1]["parts"][0]["text"] == "Question."


def test_gemini_sdk_dispatches_tool_call_and_continues_to_final_answer(monkeypatch):
    """Model emits a function_call as a Part; runtime executes it, threads
    the result into history with role=tool, and finalizes on the next turn."""
    # Production code imports `from tools import ...` (absolute) because the
    # hermes sentinel puts ax_cli/runtimes/hermes on sys.path. We do the same
    # in module setup above, so this import lands on the same module object
    # that the runtime will read.
    import tools as tools_mod

    from ax_cli.runtimes.hermes.runtimes import get_runtime

    monkeypatch.setenv("GOOGLE_API_KEY", "test_key")

    # Turn 1: a single chunk carrying a function_call part.
    turn1 = iter(
        [
            _fake_chunk(_fake_function_call_part("read_file", {"path": "/etc/hostname"})),
        ]
    )
    # Turn 2: plain text finalization.
    turn2 = iter([_fake_chunk(_fake_text_part("The hostname is foo."))])

    fake_model = MagicMock()
    fake_model.generate_content.side_effect = [turn1, turn2]
    _install_fake_genai(monkeypatch, fake_model)

    # Stub execute_tool so we do not touch the real filesystem.
    monkeypatch.setattr(
        tools_mod,
        "execute_tool",
        lambda name, args, workdir: tools_mod.ToolResult(output=f"stubbed {name}({args})"),
    )

    rt = get_runtime("gemini_sdk")
    cb = _RecordingCallback()
    result = rt.execute("Read /etc/hostname.", workdir="/tmp", stream_cb=cb)

    assert result.exit_reason == "done"
    assert result.text == "The hostname is foo."
    assert result.tool_count == 1
    # Two turns = two API calls.
    assert fake_model.generate_content.call_count == 2

    # History shape: user, assistant-with-tool-calls, tool result, final assistant.
    roles = [h.get("role") for h in result.history]
    assert roles == ["user", "assistant", "tool", "assistant"]
    # Tool call recorded in chat-completions shape on the assistant turn
    # (kept portable across providers — see _history_to_gemini_contents docs).
    assistant_with_tools = result.history[1]
    tc = assistant_with_tools["tool_calls"][0]
    assert tc["function"]["name"] == "read_file"
    # Arguments serialized as JSON in the assistant turn for portability.
    import json

    assert json.loads(tc["function"]["arguments"]) == {"path": "/etc/hostname"}
    # Tool message references _tool_name so the Gemini-contents converter
    # can populate function_response.name on the next turn.
    assert result.history[2]["_tool_name"] == "read_file"
    assert "stubbed read_file" in result.history[2]["content"]
    # Final assistant carries the visible reply.
    assert result.history[3]["content"] == "The hostname is foo."
    # Tool execution surfaces through the callback.
    assert cb.statuses == ["thinking"]
    assert cb.tool_starts == [("read_file", "Read hostname")]


def test_gemini_sdk_preserves_partial_text_on_mid_stream_error(monkeypatch):
    """If the stream raises mid-response, already-received text must not be lost."""
    from ax_cli.runtimes.hermes.runtimes import get_runtime

    monkeypatch.setenv("GOOGLE_API_KEY", "test_key")

    def explode_after_two():
        yield _fake_chunk(_fake_text_part("Partial "))
        yield _fake_chunk(_fake_text_part("reply"))
        raise RuntimeError("stream broke")

    fake_model = MagicMock()
    fake_model.generate_content.return_value = explode_after_two()
    _install_fake_genai(monkeypatch, fake_model)

    rt = get_runtime("gemini_sdk")
    cb = _RecordingCallback()
    result = rt.execute("Say hello.", workdir="/tmp", stream_cb=cb)

    # Partial text preserved in both the RuntimeResult and history.
    assert result.text == "Partial reply"
    assert result.exit_reason == "crashed"
    assert any(h.get("role") == "assistant" and h.get("content") == "Partial reply" for h in result.history)
    # Text is buffered locally during the stream and is never emitted as
    # incremental deltas, so the callback sees no on_text_delta calls even
    # though the partial text is preserved in result.text and history.
    assert cb.deltas == []


def test_gemini_sdk_handles_missing_genai_package_gracefully(monkeypatch):
    """If the `google-generativeai` SDK is not installed, return a clean
    RuntimeResult instead of letting ModuleNotFoundError kill the sentinel."""
    from ax_cli.runtimes.hermes.runtimes import get_runtime

    monkeypatch.setenv("GOOGLE_API_KEY", "test_key")
    # Force `import google.generativeai` to raise ModuleNotFoundError by
    # setting the entries in sys.modules to None (Python treats this as
    # "not importable"). We have to nuke both google and google.generativeai
    # in case a real install is on the test PATH.
    monkeypatch.setitem(sys.modules, "google", None)
    monkeypatch.setitem(sys.modules, "google.generativeai", None)

    rt = get_runtime("gemini_sdk")
    result = rt.execute("hello", workdir="/tmp")

    assert result.exit_reason == "crashed"
    # Message should mention the missing package so the operator can act.
    assert "google-generativeai" in result.text
    assert "pip install" in result.text


def test_gemini_sdk_clamps_tool_timeout_to_remaining_budget(monkeypatch):
    """A model-supplied `timeout` arg on a tool call should be clamped down
    to the wall-clock budget remaining, so a single tool cannot block the
    listener past the operator's --timeout."""
    from itertools import chain, repeat

    import tools as tools_mod

    from ax_cli.runtimes.hermes.runtimes import gemini_sdk as gemini_mod
    from ax_cli.runtimes.hermes.runtimes import get_runtime

    monkeypatch.setenv("GOOGLE_API_KEY", "test_key")

    # Fake clock: start at t=0, every later read returns t=5. With timeout=30,
    # remaining_for_tool ends up ~25s, so a model-supplied 600s timeout must
    # be clamped down to 25.
    clock = chain([0.0], repeat(5.0))
    monkeypatch.setattr(gemini_mod.time, "time", lambda: next(clock))

    # Turn 1: one bash tool call asking for a 600-second budget.
    turn1 = iter(
        [
            _fake_chunk(
                _fake_function_call_part(
                    "bash",
                    {"command": "sleep 999", "timeout": 600},
                )
            ),
        ]
    )
    # Turn 2: text-only finalization so the runtime exits cleanly.
    turn2 = iter([_fake_chunk(_fake_text_part("ok"))])

    fake_model = MagicMock()
    fake_model.generate_content.side_effect = [turn1, turn2]
    _install_fake_genai(monkeypatch, fake_model)

    captured: list[dict] = []

    def recording_execute_tool(name, args, workdir):
        captured.append({"name": name, "args": dict(args)})
        return tools_mod.ToolResult(output="stubbed")

    monkeypatch.setattr(tools_mod, "execute_tool", recording_execute_tool)

    rt = get_runtime("gemini_sdk")
    rt.execute("run it", workdir="/tmp", timeout=30)

    assert captured, "execute_tool should have been invoked"
    forwarded = captured[0]["args"]
    # The model asked for 600 but only ~25 seconds remained in the budget.
    assert forwarded["timeout"] <= 25
    # And it must still be a positive value (not zero or negative).
    assert forwarded["timeout"] >= 1


def test_gemini_sdk_returns_timeout_when_deadline_exceeded(monkeypatch):
    """When wall-clock exceeds the timeout budget before the first turn can
    open a stream, the runtime should return exit_reason='timeout' rather
    than blocking the sentinel past its configured per-invocation budget."""
    from itertools import chain, repeat

    from ax_cli.runtimes.hermes.runtimes import gemini_sdk as gemini_mod
    from ax_cli.runtimes.hermes.runtimes import get_runtime

    monkeypatch.setenv("GOOGLE_API_KEY", "test_key")

    # Fake clock. First call (captures start_time) returns 0; every later
    # call returns 2, which is already past the 1-second timeout.
    clock = chain([0.0], repeat(2.0))
    monkeypatch.setattr(gemini_mod.time, "time", lambda: next(clock))

    fake_model = MagicMock()
    # Should never be called because the deadline check fires first.
    fake_model.generate_content.side_effect = AssertionError("API should not be called once deadline has passed")
    _install_fake_genai(monkeypatch, fake_model)

    rt = get_runtime("gemini_sdk")
    result = rt.execute("any prompt", workdir="/tmp", timeout=1)

    assert result.exit_reason == "timeout"
    assert fake_model.generate_content.call_count == 0
    assert "timed out" in result.text.lower()


def test_gemini_sdk_returns_iteration_limit_when_max_turns_exhausted(monkeypatch):
    """If the model keeps producing tool calls and never finalizes, the runtime
    should exit with exit_reason='iteration_limit' rather than a misleading 'done'."""
    import tools as tools_mod

    from ax_cli.runtimes.hermes.runtimes import get_runtime
    from ax_cli.runtimes.hermes.runtimes.gemini_sdk import MAX_TURNS

    monkeypatch.setenv("GOOGLE_API_KEY", "test_key")

    counter = {"n": 0}

    def one_turn_with_function_call(*_args, **_kwargs):
        counter["n"] += 1
        return iter(
            [
                _fake_chunk(_fake_function_call_part("bash", {"command": "ls"})),
            ]
        )

    fake_model = MagicMock()
    fake_model.generate_content.side_effect = one_turn_with_function_call
    _install_fake_genai(monkeypatch, fake_model)
    monkeypatch.setattr(
        tools_mod,
        "execute_tool",
        lambda name, args, workdir: tools_mod.ToolResult(output="stubbed"),
    )

    rt = get_runtime("gemini_sdk")
    result = rt.execute("loop forever", workdir="/tmp")

    assert result.exit_reason == "iteration_limit"
    assert result.tool_count == MAX_TURNS
    assert fake_model.generate_content.call_count == MAX_TURNS
    # User-visible message should reflect the bounded-loop exit.
    assert "turn limit" in result.text.lower()


def test_gemini_sdk_appends_truncation_marker_when_tool_output_exceeds_cap(monkeypatch):
    """Tool output longer than TOOL_OUTPUT_CAP must be clipped AND get a
    '[output truncated]' marker appended so the model can tell content was
    clipped. Without the marker the model may reason as if it has the full
    output (e.g. assume a large file was fully read). Mirrors groq_sdk.py
    behavior (lines 411-427)."""
    import tools as tools_mod

    from ax_cli.runtimes.hermes.runtimes import get_runtime
    from ax_cli.runtimes.hermes.runtimes.gemini_sdk import TOOL_OUTPUT_CAP

    monkeypatch.setenv("GOOGLE_API_KEY", "test_key")

    turn1 = iter(
        [
            _fake_chunk(_fake_function_call_part("read_file", {"path": "/big.txt"})),
        ]
    )
    turn2 = iter([_fake_chunk(_fake_text_part("ok"))])
    fake_model = MagicMock()
    fake_model.generate_content.side_effect = [turn1, turn2]
    _install_fake_genai(monkeypatch, fake_model)

    # Output 2x the cap so we definitely hit the truncation path.
    big_output = "A" * (TOOL_OUTPUT_CAP * 2)
    monkeypatch.setattr(
        tools_mod,
        "execute_tool",
        lambda name, args, workdir: tools_mod.ToolResult(output=big_output),
    )

    rt = get_runtime("gemini_sdk")
    result = rt.execute("Read it.", workdir="/tmp")

    tool_msg = next(h for h in result.history if h.get("role") == "tool")
    content = tool_msg["content"]
    # Marker appended.
    assert content.endswith("\n[output truncated]")
    # Truncated to the cap (plus the marker).
    assert len(content) == TOOL_OUTPUT_CAP + len("\n[output truncated]")
    # Original bytes are preserved up to the cap.
    assert content.startswith("A" * TOOL_OUTPUT_CAP)


def test_gemini_sdk_does_not_append_marker_when_tool_output_under_cap(monkeypatch):
    """Marker must NOT appear when output is at or below the cap. Regression
    guard against accidentally appending to every tool message (which would
    confuse the model on small outputs)."""
    import tools as tools_mod

    from ax_cli.runtimes.hermes.runtimes import get_runtime

    monkeypatch.setenv("GOOGLE_API_KEY", "test_key")

    turn1 = iter(
        [
            _fake_chunk(_fake_function_call_part("read_file", {"path": "/tiny.txt"})),
        ]
    )
    turn2 = iter([_fake_chunk(_fake_text_part("ok"))])
    fake_model = MagicMock()
    fake_model.generate_content.side_effect = [turn1, turn2]
    _install_fake_genai(monkeypatch, fake_model)

    monkeypatch.setattr(
        tools_mod,
        "execute_tool",
        lambda name, args, workdir: tools_mod.ToolResult(output="small output"),
    )

    rt = get_runtime("gemini_sdk")
    result = rt.execute("Read it.", workdir="/tmp")

    tool_msg = next(h for h in result.history if h.get("role") == "tool")
    assert tool_msg["content"] == "small output"
    assert "[output truncated]" not in tool_msg["content"]


def test_gemini_sdk_returns_auth_error_on_permission_denied(monkeypatch):
    """A 'permission denied' / 401 / 403 from generate_content must surface
    as exit_reason='auth_error' so operators can rotate GOOGLE_API_KEY rather
    than see a generic 'crashed'. Mirrors groq_sdk.py / leapfrog_sdk.py
    auth_error path. Tests the string-matching classifier defensively because
    google.api_core.exceptions subclasses are inconsistently surfaced through
    the streaming code path (see inline comment in gemini_sdk.py)."""
    from ax_cli.runtimes.hermes.runtimes import get_runtime

    monkeypatch.setenv("GOOGLE_API_KEY", "test_key")

    fake_model = MagicMock()
    fake_model.generate_content.side_effect = RuntimeError(
        "403 Permission denied: API key invalid or revoked"
    )
    _install_fake_genai(monkeypatch, fake_model)

    rt = get_runtime("gemini_sdk")
    result = rt.execute("hello", workdir="/tmp")

    assert result.exit_reason == "auth_error"
    assert "GOOGLE_API_KEY" in result.text


def test_gemini_sdk_returns_auth_error_on_401_unauthenticated(monkeypatch):
    """Companion to the 403 test: 401 / 'unauthenticated' wording must also
    map to auth_error."""
    from ax_cli.runtimes.hermes.runtimes import get_runtime

    monkeypatch.setenv("GOOGLE_API_KEY", "test_key")

    fake_model = MagicMock()
    fake_model.generate_content.side_effect = RuntimeError(
        "401 Unauthenticated: request missing valid authentication credentials"
    )
    _install_fake_genai(monkeypatch, fake_model)

    rt = get_runtime("gemini_sdk")
    result = rt.execute("hello", workdir="/tmp")

    assert result.exit_reason == "auth_error"


def test_gemini_sdk_auth_error_classifier_does_not_swallow_unrelated_errors(monkeypatch):
    """Regression guard: the auth-error classifier must NOT match unrelated
    errors that happen to contain digits or substrings. A 500 internal error
    with no auth-related wording should still be 'crashed', not 'auth_error'."""
    from ax_cli.runtimes.hermes.runtimes import get_runtime

    monkeypatch.setenv("GOOGLE_API_KEY", "test_key")

    fake_model = MagicMock()
    fake_model.generate_content.side_effect = RuntimeError(
        "500 Internal Server Error: backend unavailable"
    )
    _install_fake_genai(monkeypatch, fake_model)

    rt = get_runtime("gemini_sdk")
    result = rt.execute("hello", workdir="/tmp")

    assert result.exit_reason == "crashed"
