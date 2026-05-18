# Gateway-managed LangGraph template

A one-shot bridge that routes an inbound aX mention through a
LangGraph `StateGraph` and prints the reply on stdout. Designed for
`ax gateway agents add --template langgraph`.

The bridge runs in three tiers, picked at runtime by what is
installed and configured. The same Gateway lifecycle events fire on
every tier, so an operator's activity feed looks consistent whether
the bridge is wired to a real LLM or a stub.

## Execution tiers

1. **Real LLM path.** If `langgraph` and `groq` are both importable
   and `GROQ_API_KEY` is set, the bridge builds a one-node StateGraph
   whose node streams a Groq chat completion. Chunks accumulate into
   the final reply, and the bridge emits throttled `activity` events
   (~1s heartbeat with a rolling preview) so the activity feed stays
   live during the call.

2. **Stub graph path.** If `langgraph` is importable but Groq is not
   configured (no `GROQ_API_KEY`, or the SDK is not installed), the
   bridge builds the same one-node StateGraph but wires it to a
   synthetic ack node that does not call any LLM. This proves the
   LangGraph wiring end-to-end without requiring credentials, useful
   in CI and local development.

3. **String fallback path.** If `langgraph` itself is not installed,
   the bridge returns a plain string template. The same lifecycle
   events still fire, so a Gateway operator sees the round trip even
   in the most stripped-down environment.

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `GROQ_API_KEY` | (unset) | Routes the bridge onto the real LLM path. Without it, the bridge falls back to the stub graph. |
| `AX_BRIDGE_LLM_MODEL` | `llama-3.3-70b-versatile` | Overrides the Groq model. Useful for swapping to a smaller/cheaper model or in response to a Groq deprecation. |
| `AX_BRIDGE_SYSTEM_PROMPT` | `Reply concisely.` | Replaces the trailing instruction on the system prompt. The leading agent-name framing (`You are @<agent>, ...`) is always emitted automatically. |
| `AX_GATEWAY_AGENT_NAME` | `langgraph-bot` | Name the bridge replies as. Falls back to `AX_AGENT_NAME` then to the default. |
| `AX_MENTION_CONTENT` | (unset) | Prompt source. Also accepted as positional argv or on stdin. |

## Register with Gateway

```bash
ax gateway agents add --template langgraph --name my-langgraph-bot
```

The template advertises the bridge at
`examples/gateway_langgraph/langgraph_bridge.py` and runs through the
shared `exec` runtime adapter (same precedent as the Ollama template).

## Local validation

Run the bridge directly against an inbound prompt to verify wiring:

```bash
cd ~/path/to/ax-gateway
set -a; source ../.env; set +a   # GROQ_API_KEY in there for the real LLM path
AX_GATEWAY_AGENT_NAME=langgraph-bot \
AX_MENTION_CONTENT="Reply in one short sentence, what is the speed of light in km/s?" \
.venv/bin/python examples/gateway_langgraph/langgraph_bridge.py
```

Without `GROQ_API_KEY`, the bridge takes the stub graph path and
echoes the prompt through the synthetic ack node, still emitting the
full lifecycle event sequence.

## Lifecycle events

The bridge prints `AX_GATEWAY_EVENT <json>` lines to stdout. Gateway
parses them and routes them to the activity feed.

| Status | When | Detail keys |
|---|---|---|
| `processing` (start) | Bridge begins routing the prompt. | `message` |
| `processing` (LLM call) | `Calling Groq (<model>)` fires before the stream is opened. | `message` |
| `processing` (first token) | `Groq is responding (<model>)` fires when the first streamed chunk arrives. | `message` |
| `activity` (heartbeat) | Throttled ~1s preview of accumulated text during streaming. | `activity` |
| `completed` | Final status. `used_llm` reports which path ran. `stub` is kept for back-compat with the pre-LLM-validation schema. | `duration_ms`, `used_llm`, `stub` |
| `error` | Uncaught exception. | `error_message` |

## Follow-ups

The intentional scope for the initial cut. Items here are explicit
follow-ups, not gaps.

- Multi-node graphs with branching/conditional edges.
- Tool-call telemetry mapped to Gateway tool bubbles.
- Forwarding LangGraph's own streaming events (per-node state
  transitions) as `activity` events, not just the LLM token stream.
- Provider abstraction so the LLM tier is not Groq-specific.
