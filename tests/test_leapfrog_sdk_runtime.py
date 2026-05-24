"""Tests for the LeapfrogAI SDK runtime adapter.

The `openai` Python SDK is mocked via sys.modules so these tests run
offline and do not need a live LeapfrogAI deployment. Coverage spans
registration discovery, missing-credential paths (both LEAPFROG_API_KEY
and LEAPFROG_BASE_URL), the happy streaming path (callback fan-out,
RuntimeResult shape, history accumulation), system prompt threading,
tool-call dispatch across turns, partial-failure handling when the
stream raises mid-response, the missing-package path, tool-timeout
clamping, the wall-clock deadline, the iteration-limit ceiling, and
operator-supplied base_url propagation.

Mirrors the test conventions in tests/test_gemini_sdk_runtime.py so the
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

# Importing the module triggers `@register("leapfrog_sdk")` at module load time,
# so the runtime is in REGISTRY regardless of which other tests in the suite
# may have already populated it (get_runtime's auto-discovery only fires when
# REGISTRY is fully empty).
from ax_cli.runtimes.hermes.runtimes import leapfrog_sdk  # noqa: F401, E402

# ── Helpers ────────────────────────────────────────────────────────────────


def _text_delta(text: str | None):
    """Build a duck-typed chat-completions stream chunk carrying text content
    in choices[0].delta.content."""
    delta = types.SimpleNamespace(content=text, tool_calls=None)
    choice = types.SimpleNamespace(delta=delta)
    return types.SimpleNamespace(choices=[choice])


def _tool_call_delta(*, index: int, call_id: str = "", name: str = "", arguments: str = ""):
    """Build a duck-typed chunk carrying a tool_call fragment.

    Real OpenAI streams spread a single tool call across multiple chunks,
    each with `id` / `function.name` / `function.arguments` on a different
    chunk. The runtime accumulates by `index`. These helpers let tests
    build either a one-chunk-per-tool-call shape (simple) or a
    fragmented-across-chunks shape (more realistic).
    """
    fn = types.SimpleNamespace(name=name, arguments=arguments)
    tc = types.SimpleNamespace(index=index, id=call_id, function=fn)
    delta = types.SimpleNamespace(content=None, tool_calls=[tc])
    choice = types.SimpleNamespace(delta=delta)
    return types.SimpleNamespace(choices=[choice])


# Stand-in typed exception classes for the openai SDK. These mirror the
# names + minimal shape (status_code, message) the runtime's typed `except`
# blocks expect. We define them locally instead of `import openai` so the
# test suite can run in environments where the openai package isn't
# installed (e.g. CI images, the operator-qa harness).
class _FakeAPIStatusError(Exception):
    def __init__(self, message="", *, status_code=400):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class _FakeAPITimeoutError(_FakeAPIStatusError):
    def __init__(self, message="timeout"):
        super().__init__(message, status_code=408)


class _FakeRateLimitError(_FakeAPIStatusError):
    def __init__(self, message="rate limit exceeded"):
        super().__init__(message, status_code=429)


class _FakeAuthenticationError(_FakeAPIStatusError):
    def __init__(self, message="invalid api key"):
        super().__init__(message, status_code=401)


class _FakePermissionDeniedError(_FakeAPIStatusError):
    def __init__(self, message="permission denied"):
        super().__init__(message, status_code=403)


class _FakeInternalServerError(_FakeAPIStatusError):
    def __init__(self, message="server error"):
        super().__init__(message, status_code=500)


def _install_fake_openai(monkeypatch, fake_client):
    """Swap the `openai` module in sys.modules so `from openai import OpenAI,
    APIStatusError, APITimeoutError, AuthenticationError, InternalServerError,
    PermissionDeniedError, RateLimitError` inside the runtime returns our
    mock's OpenAI constructor and stand-in typed exception classes.

    Tests that want to trigger a typed exception path raise the stand-in
    class (e.g. _FakeRateLimitError) from
    fake_client.chat.completions.create.side_effect so the runtime's typed
    `except` blocks dispatch correctly. Tests that want to trigger the
    catch-all `except Exception` path raise something outside this hierarchy
    (e.g. RuntimeError, ConnectionError).
    """
    fake_openai = types.ModuleType("openai")
    fake_openai.OpenAI = MagicMock(return_value=fake_client)
    fake_openai.APIStatusError = _FakeAPIStatusError
    fake_openai.APITimeoutError = _FakeAPITimeoutError
    fake_openai.AuthenticationError = _FakeAuthenticationError
    fake_openai.InternalServerError = _FakeInternalServerError
    fake_openai.PermissionDeniedError = _FakePermissionDeniedError
    fake_openai.RateLimitError = _FakeRateLimitError
    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    return fake_openai


def _set_credentials(monkeypatch):
    monkeypatch.setenv("LEAPFROG_API_KEY", "test_key")
    monkeypatch.setenv("LEAPFROG_BASE_URL", "https://leapfrog.test.mil/v1")


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


def test_leapfrog_sdk_registers_under_expected_name():
    from ax_cli.runtimes.hermes.runtimes import get_runtime

    rt = get_runtime("leapfrog_sdk")
    assert type(rt).__name__ == "LeapfrogSDKRuntime"
    assert rt.name == "leapfrog_sdk"


def test_leapfrog_sdk_returns_crashed_when_api_key_missing(monkeypatch):
    """No LEAPFROG_API_KEY in env should short-circuit before any openai import."""
    from ax_cli.runtimes.hermes.runtimes import get_runtime

    monkeypatch.delenv("LEAPFROG_API_KEY", raising=False)
    monkeypatch.setenv("LEAPFROG_BASE_URL", "https://leapfrog.test.mil/v1")

    rt = get_runtime("leapfrog_sdk")
    result = rt.execute("hello", workdir="/tmp")

    assert result.exit_reason == "crashed"
    assert "LEAPFROG_API_KEY" in result.text
    assert result.elapsed_seconds == 0


def test_leapfrog_sdk_returns_crashed_when_base_url_missing(monkeypatch):
    """LEAPFROG_BASE_URL is required because LeapfrogAI deployments are private
    — each operator has their own endpoint URL. Distinct from openai_sdk
    (hardcoded ChatGPT backend) and gemini_sdk (public Google endpoint)."""
    from ax_cli.runtimes.hermes.runtimes import get_runtime

    monkeypatch.setenv("LEAPFROG_API_KEY", "test_key")
    monkeypatch.delenv("LEAPFROG_BASE_URL", raising=False)

    rt = get_runtime("leapfrog_sdk")
    result = rt.execute("hello", workdir="/tmp")

    assert result.exit_reason == "crashed"
    assert "LEAPFROG_BASE_URL" in result.text
    # Error message should explain WHY the URL is required so operators can act.
    assert "private" in result.text.lower() or "endpoint" in result.text.lower()


def test_leapfrog_sdk_uses_operator_supplied_base_url(monkeypatch):
    """OpenAI(base_url=...) must be called with the LEAPFROG_BASE_URL env value.
    Regression guard: hardcoding the OpenAI default would silently route
    LeapfrogAI traffic to api.openai.com — a security incident."""
    from ax_cli.runtimes.hermes.runtimes import get_runtime

    monkeypatch.setenv("LEAPFROG_API_KEY", "operator_key_value")
    monkeypatch.setenv("LEAPFROG_BASE_URL", "https://leapfrog.example.mil/v1")

    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = iter([_text_delta("ok")])
    fake_openai = _install_fake_openai(monkeypatch, fake_client)

    rt = get_runtime("leapfrog_sdk")
    rt.execute("hello", workdir="/tmp")

    # The runtime must instantiate OpenAI with the operator-supplied URL and key.
    init_kwargs = fake_openai.OpenAI.call_args.kwargs
    assert init_kwargs["api_key"] == "operator_key_value"
    assert init_kwargs["base_url"] == "https://leapfrog.example.mil/v1"


def test_leapfrog_sdk_streams_chunks_and_accumulates_history(monkeypatch):
    """Happy path: text buffers locally, on_text_complete fires once, history grows."""
    from ax_cli.runtimes.hermes.runtimes import get_runtime

    _set_credentials(monkeypatch)
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = iter(
        [
            _text_delta("Hello "),
            _text_delta("world."),
        ]
    )
    _install_fake_openai(monkeypatch, fake_client)

    rt = get_runtime("leapfrog_sdk")
    cb = _RecordingCallback()
    result = rt.execute("Say hello.", workdir="/tmp", stream_cb=cb)

    # No incremental deltas; text is buffered until the turn is confirmed
    # text-only (matches openai_sdk.py and gemini_sdk.py).
    assert cb.deltas == []
    assert cb.complete == "Hello world."
    assert result.exit_reason == "done"
    assert result.text == "Hello world."
    assert result.tool_count == 0
    assert result.files_written == []
    # History records the round trip: user prompt + assistant reply.
    assert len(result.history) == 2
    assert result.history[0] == {"role": "user", "content": "Say hello."}
    assert result.history[1] == {"role": "assistant", "content": "Hello world."}
    # The runtime requested streaming explicitly.
    assert fake_client.chat.completions.create.call_args.kwargs["stream"] is True


def test_leapfrog_sdk_threads_system_prompt_into_messages(monkeypatch):
    """system_prompt should become the first {role: system} message on each
    chat.completions.create call. Unlike Gemini (which takes system_instruction
    as a model kwarg), OpenAI-compat APIs encode the instruction in the
    messages list."""
    from ax_cli.runtimes.hermes.runtimes import get_runtime

    _set_credentials(monkeypatch)
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = iter([_text_delta("ok")])
    _install_fake_openai(monkeypatch, fake_client)

    rt = get_runtime("leapfrog_sdk")
    rt.execute(
        "Question.",
        workdir="/tmp",
        system_prompt="You are a strict reviewer.",
    )

    call_kwargs = fake_client.chat.completions.create.call_args.kwargs
    messages = call_kwargs["messages"]
    assert messages[0] == {"role": "system", "content": "You are a strict reviewer."}
    assert messages[-1] == {"role": "user", "content": "Question."}


def test_leapfrog_sdk_dispatches_tool_call_and_continues_to_final_answer(monkeypatch):
    """Model emits a tool_call fragment stream; runtime accumulates, executes,
    threads the result back as role=tool, and finalizes on the next turn."""
    import tools as tools_mod

    from ax_cli.runtimes.hermes.runtimes import get_runtime

    _set_credentials(monkeypatch)

    # Turn 1: tool_call streamed across 3 fragments (id, name, arguments).
    # Real OpenAI streams split these — test the accumulator handles it.
    turn1 = iter(
        [
            _tool_call_delta(index=0, call_id="call_abc"),
            _tool_call_delta(index=0, name="read_file"),
            _tool_call_delta(index=0, arguments='{"path": '),
            _tool_call_delta(index=0, arguments='"/etc/hostname"}'),
        ]
    )
    # Turn 2: plain text finalization.
    turn2 = iter([_text_delta("The hostname is foo.")])

    fake_client = MagicMock()
    fake_client.chat.completions.create.side_effect = [turn1, turn2]
    _install_fake_openai(monkeypatch, fake_client)

    monkeypatch.setattr(
        tools_mod,
        "execute_tool",
        lambda name, args, workdir: tools_mod.ToolResult(output=f"stubbed {name}({args})"),
    )

    rt = get_runtime("leapfrog_sdk")
    cb = _RecordingCallback()
    result = rt.execute("Read /etc/hostname.", workdir="/tmp", stream_cb=cb)

    assert result.exit_reason == "done"
    assert result.text == "The hostname is foo."
    assert result.tool_count == 1
    assert fake_client.chat.completions.create.call_count == 2

    # History: user, assistant-with-tool-calls, tool result, final assistant.
    roles = [h.get("role") for h in result.history]
    assert roles == ["user", "assistant", "tool", "assistant"]
    assistant_with_tools = result.history[1]
    tc = assistant_with_tools["tool_calls"][0]
    assert tc["function"]["name"] == "read_file"
    # Arguments serialized as JSON in the assistant turn.
    import json

    assert json.loads(tc["function"]["arguments"]) == {"path": "/etc/hostname"}
    # Tool message uses the same call_id as the assistant tool_call (so the
    # model can pair them, per OpenAI's tool_call_id contract).
    assert result.history[2]["tool_call_id"] == "call_abc"
    assert "stubbed read_file" in result.history[2]["content"]
    assert result.history[3]["content"] == "The hostname is foo."
    assert cb.statuses == ["thinking"]
    assert cb.tool_starts == [("read_file", "Read hostname")]


def test_leapfrog_sdk_preserves_partial_text_on_mid_stream_error(monkeypatch):
    """If the stream raises mid-response, already-received text must not be lost."""
    from ax_cli.runtimes.hermes.runtimes import get_runtime

    _set_credentials(monkeypatch)

    def explode_after_two():
        yield _text_delta("Partial ")
        yield _text_delta("reply")
        raise RuntimeError("stream broke")

    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = explode_after_two()
    _install_fake_openai(monkeypatch, fake_client)

    rt = get_runtime("leapfrog_sdk")
    cb = _RecordingCallback()
    result = rt.execute("Say hello.", workdir="/tmp", stream_cb=cb)

    assert result.text == "Partial reply"
    assert result.exit_reason == "crashed"
    assert any(
        h.get("role") == "assistant" and h.get("content") == "Partial reply" for h in result.history
    )
    # Buffered streaming means no incremental deltas reach the callback.
    assert cb.deltas == []


def test_leapfrog_sdk_handles_missing_openai_package_gracefully(monkeypatch):
    """If the `openai` SDK is not installed, return a clean RuntimeResult
    instead of letting ModuleNotFoundError kill the sentinel."""
    from ax_cli.runtimes.hermes.runtimes import get_runtime

    _set_credentials(monkeypatch)
    # Force `from openai import OpenAI` to raise ModuleNotFoundError by
    # setting the entry in sys.modules to None.
    monkeypatch.setitem(sys.modules, "openai", None)

    rt = get_runtime("leapfrog_sdk")
    result = rt.execute("hello", workdir="/tmp")

    assert result.exit_reason == "crashed"
    assert "openai" in result.text.lower()
    assert "pip install" in result.text


def test_leapfrog_sdk_clamps_tool_timeout_to_remaining_budget(monkeypatch):
    """A model-supplied `timeout` arg on a tool call should be clamped down
    to the wall-clock budget remaining, so a single tool cannot block the
    listener past the operator's --timeout."""
    from itertools import chain, repeat

    import tools as tools_mod

    from ax_cli.runtimes.hermes.runtimes import get_runtime
    from ax_cli.runtimes.hermes.runtimes import leapfrog_sdk as leapfrog_mod

    _set_credentials(monkeypatch)

    # Fake clock: start at t=0, every later read returns t=5. With timeout=30,
    # remaining_for_tool ends up ~25s, so a model-supplied 600s timeout must
    # be clamped down to 25.
    clock = chain([0.0], repeat(5.0))
    monkeypatch.setattr(leapfrog_mod.time, "time", lambda: next(clock))

    # Turn 1: one bash tool call (single chunk for simplicity) asking 600s budget.
    turn1 = iter(
        [
            _tool_call_delta(index=0, call_id="call_bash"),
            _tool_call_delta(index=0, name="bash"),
            _tool_call_delta(
                index=0,
                arguments='{"command": "sleep 999", "timeout": 600}',
            ),
        ]
    )
    # Turn 2: text-only finalization so the runtime exits cleanly.
    turn2 = iter([_text_delta("ok")])

    fake_client = MagicMock()
    fake_client.chat.completions.create.side_effect = [turn1, turn2]
    _install_fake_openai(monkeypatch, fake_client)

    captured: list[dict] = []

    def recording_execute_tool(name, args, workdir):
        captured.append({"name": name, "args": dict(args)})
        return tools_mod.ToolResult(output="stubbed")

    monkeypatch.setattr(tools_mod, "execute_tool", recording_execute_tool)

    rt = get_runtime("leapfrog_sdk")
    rt.execute("run it", workdir="/tmp", timeout=30)

    assert captured, "execute_tool should have been invoked"
    forwarded = captured[0]["args"]
    # The model asked for 600 but only ~25 seconds remained in the budget.
    assert forwarded["timeout"] <= 25
    # And it must still be a positive value (not zero or negative).
    assert forwarded["timeout"] >= 1


def test_leapfrog_sdk_returns_timeout_when_deadline_exceeded(monkeypatch):
    """When wall-clock exceeds the timeout budget before the first turn can
    open a stream, the runtime should return exit_reason='timeout' rather
    than blocking the sentinel past its configured per-invocation budget."""
    from itertools import chain, repeat

    from ax_cli.runtimes.hermes.runtimes import get_runtime
    from ax_cli.runtimes.hermes.runtimes import leapfrog_sdk as leapfrog_mod

    _set_credentials(monkeypatch)

    # Fake clock. First call (captures start_time) returns 0; every later
    # call returns 2, which is already past the 1-second timeout.
    clock = chain([0.0], repeat(2.0))
    monkeypatch.setattr(leapfrog_mod.time, "time", lambda: next(clock))

    fake_client = MagicMock()
    fake_client.chat.completions.create.side_effect = AssertionError(
        "API should not be called once deadline has passed"
    )
    _install_fake_openai(monkeypatch, fake_client)

    rt = get_runtime("leapfrog_sdk")
    result = rt.execute("any prompt", workdir="/tmp", timeout=1)

    assert result.exit_reason == "timeout"
    assert fake_client.chat.completions.create.call_count == 0
    assert "timed out" in result.text.lower()


def test_leapfrog_sdk_returns_iteration_limit_when_max_turns_exhausted(monkeypatch):
    """If the model keeps producing tool calls and never finalizes, the runtime
    should exit with exit_reason='iteration_limit' rather than a misleading 'done'."""
    import tools as tools_mod

    from ax_cli.runtimes.hermes.runtimes import get_runtime
    from ax_cli.runtimes.hermes.runtimes.leapfrog_sdk import MAX_TURNS

    _set_credentials(monkeypatch)

    counter = {"n": 0}

    def one_turn_with_tool_call(*_args, **_kwargs):
        counter["n"] += 1
        return iter(
            [
                _tool_call_delta(index=0, call_id=f"call_{counter['n']}"),
                _tool_call_delta(index=0, name="bash"),
                _tool_call_delta(index=0, arguments='{"command": "ls"}'),
            ]
        )

    fake_client = MagicMock()
    fake_client.chat.completions.create.side_effect = one_turn_with_tool_call
    _install_fake_openai(monkeypatch, fake_client)
    monkeypatch.setattr(
        tools_mod,
        "execute_tool",
        lambda name, args, workdir: tools_mod.ToolResult(output="stubbed"),
    )

    rt = get_runtime("leapfrog_sdk")
    result = rt.execute("loop forever", workdir="/tmp")

    assert result.exit_reason == "iteration_limit"
    assert result.tool_count == MAX_TURNS
    assert fake_client.chat.completions.create.call_count == MAX_TURNS
    assert "turn limit" in result.text.lower()


def test_leapfrog_sdk_returns_rate_limited_on_RateLimitError(monkeypatch):
    """openai.RateLimitError must map to exit_reason='rate_limited'. Replaces
    the previous string-classifier test now that the runtime uses typed
    exceptions (Avrohom's PR #41 review)."""
    from ax_cli.runtimes.hermes.runtimes import get_runtime

    _set_credentials(monkeypatch)

    fake_client = MagicMock()
    fake_client.chat.completions.create.side_effect = _FakeRateLimitError(
        "rate limit exceeded for model llama-3.3-70b-instruct"
    )
    _install_fake_openai(monkeypatch, fake_client)

    rt = get_runtime("leapfrog_sdk")
    result = rt.execute("hello", workdir="/tmp")

    assert result.exit_reason == "rate_limited"


def test_leapfrog_sdk_returns_timeout_on_APITimeoutError(monkeypatch):
    """openai.APITimeoutError (connection / read timeout from httpx-backed
    SDK) must map to exit_reason='timeout'."""
    from ax_cli.runtimes.hermes.runtimes import get_runtime

    _set_credentials(monkeypatch)

    fake_client = MagicMock()
    fake_client.chat.completions.create.side_effect = _FakeAPITimeoutError(
        "read timeout after 30s"
    )
    _install_fake_openai(monkeypatch, fake_client)

    rt = get_runtime("leapfrog_sdk")
    result = rt.execute("hello", workdir="/tmp")

    assert result.exit_reason == "timeout"


def test_leapfrog_sdk_returns_auth_error_on_AuthenticationError(monkeypatch):
    """openai.AuthenticationError (401) must map to exit_reason='auth_error'
    so operators can rotate LEAPFROG_API_KEY rather than seeing a generic
    'crashed'. Mirrors groq_sdk.py auth_error contract."""
    from ax_cli.runtimes.hermes.runtimes import get_runtime

    _set_credentials(monkeypatch)

    fake_client = MagicMock()
    fake_client.chat.completions.create.side_effect = _FakeAuthenticationError(
        "invalid api key"
    )
    _install_fake_openai(monkeypatch, fake_client)

    rt = get_runtime("leapfrog_sdk")
    result = rt.execute("hello", workdir="/tmp")

    assert result.exit_reason == "auth_error"
    assert "LEAPFROG_API_KEY" in result.text


def test_leapfrog_sdk_returns_auth_error_on_PermissionDeniedError(monkeypatch):
    """openai.PermissionDeniedError (403) must also map to exit_reason='auth_error'.
    Companion to the 401 test — both are operator-actionable credential failures."""
    from ax_cli.runtimes.hermes.runtimes import get_runtime

    _set_credentials(monkeypatch)

    fake_client = MagicMock()
    fake_client.chat.completions.create.side_effect = _FakePermissionDeniedError(
        "deployment ACL denied request"
    )
    _install_fake_openai(monkeypatch, fake_client)

    rt = get_runtime("leapfrog_sdk")
    result = rt.execute("hello", workdir="/tmp")

    assert result.exit_reason == "auth_error"


def test_leapfrog_sdk_returns_server_error_on_InternalServerError(monkeypatch):
    """openai.InternalServerError (5xx from the LeapfrogAI deployment) must
    map to exit_reason='server_error' so operators know retry is plausible."""
    from ax_cli.runtimes.hermes.runtimes import get_runtime

    _set_credentials(monkeypatch)

    fake_client = MagicMock()
    fake_client.chat.completions.create.side_effect = _FakeInternalServerError(
        "backend unavailable"
    )
    _install_fake_openai(monkeypatch, fake_client)

    rt = get_runtime("leapfrog_sdk")
    result = rt.execute("hello", workdir="/tmp")

    assert result.exit_reason == "server_error"


def test_leapfrog_sdk_returns_api_error_on_other_APIStatusError(monkeypatch):
    """Other 4xx status errors not matched by the more specific typed-exception
    catches (e.g. 400 BadRequest, 404 NotFound) must map to exit_reason='api_error'
    with the status surfaced in the user-visible text."""
    from ax_cli.runtimes.hermes.runtimes import get_runtime

    _set_credentials(monkeypatch)

    fake_client = MagicMock()
    fake_client.chat.completions.create.side_effect = _FakeAPIStatusError(
        "model not found", status_code=404
    )
    _install_fake_openai(monkeypatch, fake_client)

    rt = get_runtime("leapfrog_sdk")
    result = rt.execute("hello", workdir="/tmp")

    assert result.exit_reason == "api_error"
    assert "404" in result.text


def test_leapfrog_sdk_returns_crashed_on_unexpected_exception(monkeypatch):
    """Errors outside the openai SDK's typed exception hierarchy (network
    adapter bugs, connection refused before an APIConnectionError, etc.)
    must still hit the catch-all and report exit_reason='crashed'."""
    from ax_cli.runtimes.hermes.runtimes import get_runtime

    _set_credentials(monkeypatch)

    fake_client = MagicMock()
    fake_client.chat.completions.create.side_effect = RuntimeError(
        "socket: connection refused before SDK could wrap it"
    )
    _install_fake_openai(monkeypatch, fake_client)

    rt = get_runtime("leapfrog_sdk")
    result = rt.execute("hello", workdir="/tmp")

    assert result.exit_reason == "crashed"


def test_leapfrog_sdk_appends_truncation_marker_when_tool_output_exceeds_cap(monkeypatch):
    """Tool output longer than TOOL_OUTPUT_CAP must be clipped AND get a
    '[output truncated]' marker appended so the model can tell content was
    clipped. Without the marker the model may reason as if it has the full
    output (e.g. assume a large file was fully read). Mirrors groq_sdk.py
    behavior (lines 411-427)."""
    import tools as tools_mod

    from ax_cli.runtimes.hermes.runtimes import get_runtime
    from ax_cli.runtimes.hermes.runtimes.leapfrog_sdk import TOOL_OUTPUT_CAP

    _set_credentials(monkeypatch)

    # Turn 1: tool_call requesting a read.
    turn1 = iter(
        [
            _tool_call_delta(index=0, call_id="call_big"),
            _tool_call_delta(index=0, name="read_file"),
            _tool_call_delta(index=0, arguments='{"path": "/big.txt"}'),
        ]
    )
    # Turn 2: text-only finalization.
    turn2 = iter([_text_delta("ok")])

    fake_client = MagicMock()
    fake_client.chat.completions.create.side_effect = [turn1, turn2]
    _install_fake_openai(monkeypatch, fake_client)

    big_output = "A" * (TOOL_OUTPUT_CAP * 2)
    monkeypatch.setattr(
        tools_mod,
        "execute_tool",
        lambda name, args, workdir: tools_mod.ToolResult(output=big_output),
    )

    rt = get_runtime("leapfrog_sdk")
    result = rt.execute("Read it.", workdir="/tmp")

    tool_msg = next(h for h in result.history if h.get("role") == "tool")
    content = tool_msg["content"]
    assert content.endswith("\n[output truncated]")
    assert len(content) == TOOL_OUTPUT_CAP + len("\n[output truncated]")
    assert content.startswith("A" * TOOL_OUTPUT_CAP)


def test_leapfrog_sdk_does_not_append_marker_when_tool_output_under_cap(monkeypatch):
    """Marker must NOT appear when output is at or below the cap. Regression
    guard against accidentally appending to every tool message (which would
    confuse the model on small outputs)."""
    import tools as tools_mod

    from ax_cli.runtimes.hermes.runtimes import get_runtime

    _set_credentials(monkeypatch)

    turn1 = iter(
        [
            _tool_call_delta(index=0, call_id="call_tiny"),
            _tool_call_delta(index=0, name="read_file"),
            _tool_call_delta(index=0, arguments='{"path": "/tiny.txt"}'),
        ]
    )
    turn2 = iter([_text_delta("ok")])

    fake_client = MagicMock()
    fake_client.chat.completions.create.side_effect = [turn1, turn2]
    _install_fake_openai(monkeypatch, fake_client)

    monkeypatch.setattr(
        tools_mod,
        "execute_tool",
        lambda name, args, workdir: tools_mod.ToolResult(output="small output"),
    )

    rt = get_runtime("leapfrog_sdk")
    result = rt.execute("Read it.", workdir="/tmp")

    tool_msg = next(h for h in result.history if h.get("role") == "tool")
    assert tool_msg["content"] == "small output"
    assert "[output truncated]" not in tool_msg["content"]
