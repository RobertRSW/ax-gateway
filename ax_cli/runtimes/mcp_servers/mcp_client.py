"""Synchronous MCP stdio client — drive an MCP server from a Python process.

Spawns the server as a subprocess, exchanges JSON-RPC over stdin/stdout
(protocol version 2025-11-25), and exposes `initialize()` / `list_tools()`
/ `call_tool()` as plain method calls. The complementary piece to
`stdio_server.py` — same protocol, opposite side of the pipe.

Used by `langchain_adapter.py` to wrap MCP tools as LangChain `BaseTool`s
so the LangGraph `ToolNode` can call them transparently. Could also drive
the servers from tests or one-off scripts.

Lifecycle: instantiate, call `initialize()` once at startup, then
`call_tool()` per invocation. `close()` shuts the subprocess down cleanly.
The class is a context manager for the common single-script case.

Synchronous on purpose. The bridge's LangChain tool layer is sync (each
`@tool` call blocks), so async would only add complexity. If a future
caller needs async, wrap this in `asyncio.to_thread`.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any

DEFAULT_TIMEOUT_S = 30.0


class McpClientError(Exception):
    """Raised on protocol or transport errors."""


class McpToolError(Exception):
    """Raised when a tool call comes back with `isError: true`."""


@dataclass
class McpToolSpec:
    """Mirrors the server-side ToolSpec but without the handler."""

    name: str
    description: str
    input_schema: dict[str, Any] = field(default_factory=dict)


class McpStdioClient:
    """Drive a single MCP stdio server subprocess."""

    def __init__(
        self,
        command: list[str],
        *,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        debug: bool = False,
    ) -> None:
        self._command = list(command)
        self._env_override = env
        self._cwd = cwd
        self._timeout_s = timeout_s
        self._debug = debug
        self._proc: subprocess.Popen[bytes] | None = None
        self._next_id = 0
        self._id_lock = threading.Lock()
        self._io_lock = threading.Lock()
        self._initialized = False

    # ── lifecycle ──────────────────────────────────────────────────────────

    def __enter__(self) -> McpStdioClient:
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def start(self) -> None:
        if self._proc is not None:
            return
        env = dict(os.environ)
        if self._env_override:
            env.update(self._env_override)
        self._log(f"spawning: {' '.join(self._command)}")
        self._proc = subprocess.Popen(
            self._command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            cwd=self._cwd,
            bufsize=0,
        )

    def close(self) -> None:
        proc = self._proc
        if proc is None:
            return
        try:
            if proc.stdin and not proc.stdin.closed:
                proc.stdin.close()
            proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                proc.kill()
        finally:
            self._proc = None
            self._initialized = False

    # ── protocol ───────────────────────────────────────────────────────────

    def initialize(self) -> dict[str, Any]:
        """Send `initialize`; return the server's response (protocolVersion / serverInfo / etc.)."""
        if self._proc is None:
            self.start()
        result = self._request("initialize", {})
        self._initialized = True
        return result

    def list_tools(self) -> list[McpToolSpec]:
        """Return the server's tool catalog as `McpToolSpec`s."""
        if not self._initialized:
            self.initialize()
        result = self._request("tools/list", {})
        tools_raw = result.get("tools") or []
        return [
            McpToolSpec(
                name=str(t.get("name") or ""),
                description=str(t.get("description") or ""),
                input_schema=dict(t.get("inputSchema") or {}),
            )
            for t in tools_raw
        ]

    def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """Invoke `name` with `arguments`; return the first text content block.

        Raises `McpToolError` if the server set `isError: true`. Tool servers
        in this repo return all results as a single JSON-encoded text block,
        which is what LangChain BaseTool consumers will get back verbatim.
        """
        if not self._initialized:
            self.initialize()
        result = self._request("tools/call", {"name": name, "arguments": arguments})
        content = result.get("content") or []
        text_parts = [
            str(block.get("text") or "") for block in content if isinstance(block, dict) and block.get("type") == "text"
        ]
        joined = "\n".join(text_parts)
        if result.get("isError"):
            raise McpToolError(joined or f"tool {name} failed (no detail)")
        return joined

    # ── transport ──────────────────────────────────────────────────────────

    def _next_request_id(self) -> int:
        with self._id_lock:
            self._next_id += 1
            return self._next_id

    def _request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        request_id = self._next_request_id()
        payload = {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}
        with self._io_lock:
            self._send(payload)
            response = self._read_until(request_id)
        if "error" in response:
            err = response["error"]
            raise McpClientError(f"{method} returned error: {err.get('message')} (code={err.get('code')})")
        return response.get("result") or {}

    def _send(self, payload: dict[str, Any]) -> None:
        proc = self._proc
        if proc is None or proc.stdin is None or proc.stdin.closed:
            raise McpClientError("server subprocess is not running")
        line = (json.dumps(payload, separators=(",", ":")) + "\n").encode("utf-8")
        try:
            proc.stdin.write(line)
            proc.stdin.flush()
        except BrokenPipeError as e:
            self._capture_stderr_into_error(e)

    def _read_until(self, expected_id: int) -> dict[str, Any]:
        """Read lines from the server's stdout until we get the matching id."""
        proc = self._proc
        if proc is None or proc.stdout is None:
            raise McpClientError("server subprocess is not running")
        deadline = time.monotonic() + self._timeout_s
        while True:
            if time.monotonic() > deadline:
                raise McpClientError(f"timed out waiting for response to id={expected_id}")
            raw = proc.stdout.readline()
            if not raw:
                stderr_text = self._drain_stderr()
                raise McpClientError(
                    "server subprocess closed stdout" + (f" (stderr: {stderr_text})" if stderr_text else "")
                )
            try:
                payload = json.loads(raw.decode("utf-8").strip())
            except (UnicodeDecodeError, json.JSONDecodeError) as e:
                self._log(f"unparseable line from server: {e!r}: {raw!r}")
                continue
            if not isinstance(payload, dict):
                continue
            if payload.get("id") == expected_id:
                return payload
            self._log(f"discarding out-of-order payload id={payload.get('id')}")

    def _drain_stderr(self) -> str:
        proc = self._proc
        if proc is None or proc.stderr is None:
            return ""
        try:
            proc.stderr.flush()
            return proc.stderr.read1(4096).decode("utf-8", errors="replace")
        except Exception:
            return ""

    def _capture_stderr_into_error(self, exc: Exception) -> None:
        stderr_text = self._drain_stderr()
        raise McpClientError(
            f"write to server failed: {exc}" + (f" (stderr: {stderr_text})" if stderr_text else "")
        ) from exc

    def _log(self, message: str) -> None:
        if self._debug:
            print(f"[mcp-client] {message}", file=sys.stderr, flush=True)
