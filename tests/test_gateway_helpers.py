"""Tests for helper functions in ax_cli.gateway and ax_cli.commands.gateway.

Covers pure-logic helpers and state-management utilities that do not require
subprocess calls or live backend connections. Filesystem-touching helpers
use the clean_env / tmp_path fixtures from conftest so tests never interact
with a real ~/.ax directory.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from ax_cli import gateway as gw
from ax_cli.commands import gateway as gw_cmd

# ---------------------------------------------------------------------------
# ax_cli.gateway — pure helpers
# ---------------------------------------------------------------------------


class TestPhaseForEvent:
    """phase_for_event: maps activity event names to supervisor phases."""

    def test_known_event(self):
        from ax_cli.gateway import phase_for_event

        assert phase_for_event("message_received") == "received"
        assert phase_for_event("reply_sent") == "reply"
        assert phase_for_event("runtime_activity") == "working"
        assert phase_for_event("tool_started") == "tool"

    def test_unknown_event_returns_none(self):
        from ax_cli.gateway import phase_for_event

        assert phase_for_event("totally_unknown") is None

    def test_none_or_empty_returns_none(self):
        from ax_cli.gateway import phase_for_event

        assert phase_for_event(None) is None
        assert phase_for_event("") is None


class TestNormalizedControlled:
    """_normalized_controlled: validates / normalizes against a controlled set."""

    def test_exact_match(self):
        from ax_cli.gateway import _normalized_controlled

        assert _normalized_controlled("hosted", {"hosted", "attached"}, fallback="attached") == "hosted"

    def test_case_insensitive_match(self):
        from ax_cli.gateway import _normalized_controlled

        assert _normalized_controlled("HOSTED", {"hosted", "attached"}, fallback="attached") == "hosted"

    def test_fallback_on_unknown(self):
        from ax_cli.gateway import _normalized_controlled

        assert _normalized_controlled("bogus", {"hosted", "attached"}, fallback="attached") == "attached"

    def test_none_value_uses_fallback(self):
        from ax_cli.gateway import _normalized_controlled

        assert _normalized_controlled(None, {"hosted", "attached"}, fallback="attached") == "attached"

    def test_empty_string_uses_fallback(self):
        from ax_cli.gateway import _normalized_controlled

        assert _normalized_controlled("", {"hosted", "attached"}, fallback="attached") == "attached"


class TestNormalizedControlledList:
    """_normalized_controlled_list: validates lists of controlled values."""

    def test_comma_separated_string(self):
        from ax_cli.gateway import _normalized_controlled_list

        result = _normalized_controlled_list(
            "direct_message,mailbox_poll",
            {"direct_message", "mailbox_poll", "manual_check"},
            fallback=["direct_message"],
        )
        assert result == ["direct_message", "mailbox_poll"]

    def test_list_input(self):
        from ax_cli.gateway import _normalized_controlled_list

        result = _normalized_controlled_list(
            ["direct_message", "mailbox_poll"],
            {"direct_message", "mailbox_poll"},
            fallback=["direct_message"],
        )
        assert result == ["direct_message", "mailbox_poll"]

    def test_unknown_items_dropped(self):
        from ax_cli.gateway import _normalized_controlled_list

        result = _normalized_controlled_list(
            "direct_message,bogus",
            {"direct_message", "mailbox_poll"},
            fallback=["direct_message"],
        )
        assert result == ["direct_message"]

    def test_empty_returns_fallback(self):
        from ax_cli.gateway import _normalized_controlled_list

        result = _normalized_controlled_list("", {"direct_message"}, fallback=["direct_message"])
        assert result == ["direct_message"]

    def test_deduplicates(self):
        from ax_cli.gateway import _normalized_controlled_list

        result = _normalized_controlled_list(
            ["direct_message", "direct_message"],
            {"direct_message"},
            fallback=["direct_message"],
        )
        assert result == ["direct_message"]


class TestNormalizedOptionalControlled:
    """_normalized_optional_controlled: returns None for empty/unknown."""

    def test_returns_value_when_valid(self):
        from ax_cli.gateway import _normalized_optional_controlled

        assert _normalized_optional_controlled("verified", {"verified", "drifted"}) == "verified"

    def test_returns_none_for_empty(self):
        from ax_cli.gateway import _normalized_optional_controlled

        assert _normalized_optional_controlled("", {"verified", "drifted"}) is None
        assert _normalized_optional_controlled(None, {"verified", "drifted"}) is None

    def test_returns_none_for_unknown(self):
        from ax_cli.gateway import _normalized_optional_controlled

        assert _normalized_optional_controlled("bogus", {"verified", "drifted"}) is None

    def test_case_insensitive(self):
        from ax_cli.gateway import _normalized_optional_controlled

        assert _normalized_optional_controlled("VERIFIED", {"verified", "drifted"}) == "verified"


class TestNormalizedStringList:
    """_normalized_string_list: splits strings / normalizes lists."""

    def test_comma_string(self):
        from ax_cli.gateway import _normalized_string_list

        assert _normalized_string_list("a,b,c", fallback=["x"]) == ["a", "b", "c"]

    def test_list_input(self):
        from ax_cli.gateway import _normalized_string_list

        assert _normalized_string_list(["a", "b"], fallback=["x"]) == ["a", "b"]

    def test_empty_returns_fallback(self):
        from ax_cli.gateway import _normalized_string_list

        assert _normalized_string_list("", fallback=["x"]) == ["x"]
        assert _normalized_string_list(None, fallback=["x"]) == ["x"]

    def test_strips_whitespace(self):
        from ax_cli.gateway import _normalized_string_list

        assert _normalized_string_list(" a , b ", fallback=["x"]) == ["a", "b"]


class TestBoolWithFallback:
    """_bool_with_fallback: coerces various inputs to bool."""

    def test_native_bool(self):
        from ax_cli.gateway import _bool_with_fallback

        assert _bool_with_fallback(True, fallback=False) is True
        assert _bool_with_fallback(False, fallback=True) is False

    def test_truthy_strings(self):
        from ax_cli.gateway import _bool_with_fallback

        for value in ("true", "1", "yes", "y", "on", "TRUE", "Yes"):
            assert _bool_with_fallback(value, fallback=False) is True

    def test_falsy_strings(self):
        from ax_cli.gateway import _bool_with_fallback

        for value in ("false", "0", "no", "n", "off", "FALSE", "No"):
            assert _bool_with_fallback(value, fallback=True) is False

    def test_unknown_uses_fallback(self):
        from ax_cli.gateway import _bool_with_fallback

        assert _bool_with_fallback("maybe", fallback=True) is True
        assert _bool_with_fallback(None, fallback=False) is False


class TestOverrideFields:
    """_override_fields: extracts operator override field names from snapshot."""

    def test_nested_user_overrides_dict(self):
        from ax_cli.gateway import _override_fields

        snapshot = {"user_overrides": {"operator": {"placement": "hosted", "reply_mode": "silent"}}}
        result = _override_fields(snapshot, domain="operator")
        assert result == {"placement", "reply_mode"}

    def test_direct_domain_overrides(self):
        from ax_cli.gateway import _override_fields

        snapshot = {"operator_overrides": {"activation": "persistent"}}
        result = _override_fields(snapshot, domain="operator")
        assert result == {"activation"}

    def test_list_form(self):
        from ax_cli.gateway import _override_fields

        snapshot = {"user_overrides": {"asset": ["asset_class", "intake_model"]}}
        result = _override_fields(snapshot, domain="asset")
        assert result == {"asset_class", "intake_model"}

    def test_empty_snapshot(self):
        from ax_cli.gateway import _override_fields

        assert _override_fields({}, domain="operator") == set()


class TestIsSystemAgent:
    """_is_system_agent: identifies infra agents (service accounts, switchboard)."""

    def test_service_account_template(self):
        from ax_cli.gateway import _is_system_agent

        assert _is_system_agent({"template_id": "service_account"}) is True

    def test_inbox_template(self):
        from ax_cli.gateway import _is_system_agent

        assert _is_system_agent({"template_id": "inbox"}) is True

    def test_switchboard_name(self):
        from ax_cli.gateway import _is_system_agent

        assert _is_system_agent({"name": "switchboard-abc123"}) is True

    def test_normal_agent(self):
        from ax_cli.gateway import _is_system_agent

        assert _is_system_agent({"name": "my-agent", "template_id": "echo_test"}) is False

    def test_empty_entry(self):
        from ax_cli.gateway import _is_system_agent

        assert _is_system_agent({}) is False


class TestIsPassiveRuntime:
    """_is_passive_runtime: identifies runtime types that queue instead of process."""

    def test_inbox(self):
        from ax_cli.gateway import _is_passive_runtime

        assert _is_passive_runtime("inbox") is True

    def test_passive(self):
        from ax_cli.gateway import _is_passive_runtime

        assert _is_passive_runtime("passive") is True

    def test_monitor(self):
        from ax_cli.gateway import _is_passive_runtime

        assert _is_passive_runtime("monitor") is True

    def test_case_insensitive(self):
        from ax_cli.gateway import _is_passive_runtime

        assert _is_passive_runtime("INBOX") is True

    def test_active_runtimes(self):
        from ax_cli.gateway import _is_passive_runtime

        assert _is_passive_runtime("echo") is False
        assert _is_passive_runtime("exec") is False
        assert _is_passive_runtime("hermes_plugin") is False

    def test_none(self):
        from ax_cli.gateway import _is_passive_runtime

        assert _is_passive_runtime(None) is False


class TestTemplateOperatorDefaults:
    """_template_operator_defaults: returns operator profile for known templates."""

    def test_echo_test(self):
        from ax_cli.gateway import _template_operator_defaults

        result = _template_operator_defaults("echo_test", None)
        assert result["placement"] == "hosted"
        assert result["activation"] == "persistent"

    def test_pass_through(self):
        from ax_cli.gateway import _template_operator_defaults

        result = _template_operator_defaults("pass_through", None)
        assert result["placement"] == "mailbox"
        assert result["activation"] == "attach_only"

    def test_claude_code_channel(self):
        from ax_cli.gateway import _template_operator_defaults

        result = _template_operator_defaults("claude_code_channel", None)
        assert result["placement"] == "attached"
        assert result["activation"] == "attach_only"

    def test_unknown_falls_to_runtime_type(self):
        from ax_cli.gateway import _template_operator_defaults

        result = _template_operator_defaults(None, "echo")
        assert result["placement"] == "hosted"

    def test_unknown_both_falls_to_exec(self):
        from ax_cli.gateway import _template_operator_defaults

        result = _template_operator_defaults(None, "completely_unknown")
        assert result["placement"] == "hosted"
        assert result["activation"] == "persistent"

    def test_case_insensitive(self):
        from ax_cli.gateway import _template_operator_defaults

        result = _template_operator_defaults("ECHO_TEST", None)
        assert result["placement"] == "hosted"


class TestTemplateAssetDefaults:
    """_template_asset_defaults: returns asset descriptor defaults."""

    def test_echo_test(self):
        from ax_cli.gateway import _template_asset_defaults

        result = _template_asset_defaults("echo_test", None)
        assert result["asset_class"] == "interactive_agent"
        assert result["intake_model"] == "live_listener"
        assert "reply" in result["capabilities"]

    def test_service_account(self):
        from ax_cli.gateway import _template_asset_defaults

        result = _template_asset_defaults("service_account", None)
        assert result["asset_class"] == "service_account"
        assert result["schedulable"] is True
        assert result["externally_triggered"] is True

    def test_pass_through(self):
        from ax_cli.gateway import _template_asset_defaults

        result = _template_asset_defaults("pass_through", None)
        assert result["intake_model"] == "polling_mailbox"
        assert result["worker_model"] == "agent_check_in"

    def test_returns_copies(self):
        from ax_cli.gateway import _template_asset_defaults

        a = _template_asset_defaults("echo_test", None)
        b = _template_asset_defaults("echo_test", None)
        a["tags"].append("mutated")
        assert "mutated" not in b["tags"]


class TestAssetTypeLabel:
    """_asset_type_label: human-readable label for asset class + intake model."""

    def test_live_listener(self):
        from ax_cli.gateway import _asset_type_label

        assert _asset_type_label(asset_class="interactive_agent", intake_model="live_listener") == "Live Listener"

    def test_on_demand(self):
        from ax_cli.gateway import _asset_type_label

        assert _asset_type_label(asset_class="interactive_agent", intake_model="launch_on_send") == "On-Demand Agent"

    def test_pass_through(self):
        from ax_cli.gateway import _asset_type_label

        assert (
            _asset_type_label(asset_class="interactive_agent", intake_model="polling_mailbox") == "Pass-through Agent"
        )

    def test_inbox_worker(self):
        from ax_cli.gateway import _asset_type_label

        assert _asset_type_label(asset_class="background_worker", intake_model="queue_accept") == "Inbox Worker"

    def test_service_account(self):
        from ax_cli.gateway import _asset_type_label

        assert _asset_type_label(asset_class="service_account", intake_model="notification_source") == "Service Account"

    def test_service_proxy(self):
        from ax_cli.gateway import _asset_type_label

        assert _asset_type_label(asset_class="service_proxy", intake_model="anything") == "Service / Tool Proxy"

    def test_fallback(self):
        from ax_cli.gateway import _asset_type_label

        assert _asset_type_label(asset_class="unknown", intake_model="unknown") == "Connected Asset"


class TestOutputLabel:
    """_output_label: human label from the first return path."""

    def test_inline_reply(self):
        from ax_cli.gateway import _output_label

        assert _output_label(["inline_reply"]) == "Reply"

    def test_summary_post(self):
        from ax_cli.gateway import _output_label

        assert _output_label(["summary_post"]) == "Summary"

    def test_silent(self):
        from ax_cli.gateway import _output_label

        assert _output_label(["silent"]) == "Silent"

    def test_empty_defaults_to_reply(self):
        from ax_cli.gateway import _output_label

        assert _output_label([]) == "Reply"


class TestDeriveMode:
    """_derive_mode: maps operator profile to mode string."""

    def test_mailbox_placement(self):
        from ax_cli.gateway import _derive_mode

        assert _derive_mode({"placement": "mailbox", "activation": "queue_worker"}) == "INBOX"

    def test_persistent_activation(self):
        from ax_cli.gateway import _derive_mode

        assert _derive_mode({"placement": "hosted", "activation": "persistent"}) == "LIVE"

    def test_attach_only(self):
        from ax_cli.gateway import _derive_mode

        assert _derive_mode({"placement": "attached", "activation": "attach_only"}) == "LIVE"

    def test_on_demand(self):
        from ax_cli.gateway import _derive_mode

        assert _derive_mode({"placement": "hosted", "activation": "on_demand"}) == "ON-DEMAND"


class TestDerivePresence:
    """_derive_presence: maps mode + liveness + work_state to presence."""

    def test_setup_error(self):
        from ax_cli.gateway import _derive_presence

        assert _derive_presence(mode="LIVE", liveness="setup_error", work_state="idle") == "ERROR"

    def test_blocked(self):
        from ax_cli.gateway import _derive_presence

        assert _derive_presence(mode="LIVE", liveness="connected", work_state="blocked") == "BLOCKED"

    def test_stale(self):
        from ax_cli.gateway import _derive_presence

        assert _derive_presence(mode="LIVE", liveness="stale", work_state="idle") == "STALE"

    def test_offline_live_mode(self):
        from ax_cli.gateway import _derive_presence

        assert _derive_presence(mode="LIVE", liveness="offline", work_state="idle") == "OFFLINE"

    def test_working(self):
        from ax_cli.gateway import _derive_presence

        assert _derive_presence(mode="LIVE", liveness="connected", work_state="working") == "WORKING"

    def test_queued(self):
        from ax_cli.gateway import _derive_presence

        assert _derive_presence(mode="INBOX", liveness="connected", work_state="queued") == "QUEUED"

    def test_idle(self):
        from ax_cli.gateway import _derive_presence

        assert _derive_presence(mode="LIVE", liveness="connected", work_state="idle") == "IDLE"


class TestDeriveReply:
    """_derive_reply: maps reply_mode to display string."""

    def test_interactive(self):
        from ax_cli.gateway import _derive_reply

        assert _derive_reply("interactive") == "REPLY"

    def test_silent(self):
        from ax_cli.gateway import _derive_reply

        assert _derive_reply("silent") == "SILENT"

    def test_background(self):
        from ax_cli.gateway import _derive_reply

        assert _derive_reply("background") == "SUMMARY"

    def test_summary_only(self):
        from ax_cli.gateway import _derive_reply

        assert _derive_reply("summary_only") == "SUMMARY"


class TestDeriveLiveness:
    """_derive_liveness: maps snapshot state + age to liveness tuple."""

    def test_running_connected(self):
        from ax_cli.gateway import _derive_liveness

        liveness, connected = _derive_liveness({}, raw_state="running", last_seen_age=10)
        assert liveness == "connected"
        assert connected is True

    def test_running_stale(self):
        from ax_cli.gateway import RUNTIME_STALE_AFTER_SECONDS, _derive_liveness

        liveness, connected = _derive_liveness(
            {}, raw_state="running", last_seen_age=int(RUNTIME_STALE_AFTER_SECONDS) + 10
        )
        assert liveness == "stale"
        assert connected is False

    def test_running_no_heartbeat(self):
        from ax_cli.gateway import _derive_liveness

        liveness, connected = _derive_liveness({}, raw_state="running", last_seen_age=None)
        assert liveness == "stale"
        assert connected is False

    def test_error_state(self):
        from ax_cli.gateway import _derive_liveness

        liveness, connected = _derive_liveness({}, raw_state="error", last_seen_age=5)
        assert liveness == "setup_error"
        assert connected is False

    def test_stopped(self):
        from ax_cli.gateway import _derive_liveness

        liveness, connected = _derive_liveness({}, raw_state="stopped", last_seen_age=None)
        assert liveness == "offline"
        assert connected is False

    def test_starting(self):
        from ax_cli.gateway import _derive_liveness

        liveness, connected = _derive_liveness({}, raw_state="starting", last_seen_age=None)
        assert liveness == "stale"
        assert connected is False

    def test_sse_disconnected_overrides_fresh_heartbeat(self):
        from ax_cli.gateway import _derive_liveness

        liveness, connected = _derive_liveness({"sse_connected": False}, raw_state="running", last_seen_age=5)
        assert liveness == "stale"
        assert connected is False

    def test_sse_connected_true_does_not_affect_normal_logic(self):
        from ax_cli.gateway import _derive_liveness

        liveness, connected = _derive_liveness({"sse_connected": True}, raw_state="running", last_seen_age=5)
        assert liveness == "connected"
        assert connected is True

    def test_sse_connected_none_does_not_affect_normal_logic(self):
        from ax_cli.gateway import _derive_liveness

        liveness, connected = _derive_liveness({"sse_connected": None}, raw_state="running", last_seen_age=5)
        assert liveness == "connected"
        assert connected is True


class TestLooksLikeSetupError:
    """_looks_like_setup_error: detects setup errors from snapshot + state."""

    def test_error_state(self):
        from ax_cli.gateway import _looks_like_setup_error

        assert _looks_like_setup_error({}, "error") is True

    def test_repo_not_found_in_last_error(self):
        from ax_cli.gateway import _looks_like_setup_error

        assert _looks_like_setup_error({"last_error": "Repo not found at /path"}, "running") is True

    def test_stderr_in_preview(self):
        from ax_cli.gateway import _looks_like_setup_error

        assert _looks_like_setup_error({"last_reply_preview": "(stderr: something)"}, "running") is True

    def test_clean_running(self):
        from ax_cli.gateway import _looks_like_setup_error

        assert _looks_like_setup_error({}, "running") is False


class TestDoctorHasFailed:
    """_doctor_has_failed: checks doctor result for failures."""

    def test_no_result(self):
        from ax_cli.gateway import _doctor_has_failed

        assert _doctor_has_failed({}) is False
        assert _doctor_has_failed({"last_doctor_result": None}) is False

    def test_failed_status(self):
        from ax_cli.gateway import _doctor_has_failed

        assert _doctor_has_failed({"last_doctor_result": {"status": "failed"}}) is True

    def test_error_status(self):
        from ax_cli.gateway import _doctor_has_failed

        assert _doctor_has_failed({"last_doctor_result": {"status": "error"}}) is True

    def test_failed_check(self):
        from ax_cli.gateway import _doctor_has_failed

        result = {"status": "ok", "checks": [{"name": "send", "status": "failed"}]}
        assert _doctor_has_failed({"last_doctor_result": result}) is True

    def test_all_ok(self):
        from ax_cli.gateway import _doctor_has_failed

        result = {"status": "ok", "checks": [{"name": "send", "status": "ok"}]}
        assert _doctor_has_failed({"last_doctor_result": result}) is False


class TestDoctorSummary:
    """_doctor_summary: extracts human-readable summary from doctor result."""

    def test_no_result(self):
        from ax_cli.gateway import _doctor_summary

        assert _doctor_summary({}) == ""

    def test_summary_field(self):
        from ax_cli.gateway import _doctor_summary

        assert _doctor_summary({"last_doctor_result": {"summary": "All good"}}) == "All good"

    def test_failed_checks_summary(self):
        from ax_cli.gateway import _doctor_summary

        result = {"checks": [{"name": "auth", "status": "failed"}, {"name": "send", "status": "ok"}]}
        summary = _doctor_summary({"last_doctor_result": result})
        assert "auth" in summary
        assert "send" not in summary


class TestDeriveReachability:
    """_derive_reachability: maps snapshot + mode + liveness to reachability."""

    def test_setup_error(self):
        from ax_cli.gateway import _derive_reachability

        assert (
            _derive_reachability(snapshot={}, mode="LIVE", liveness="setup_error", activation="persistent")
            == "unavailable"
        )

    def test_inbox_mode(self):
        from ax_cli.gateway import _derive_reachability

        assert (
            _derive_reachability(snapshot={}, mode="INBOX", liveness="connected", activation="queue_worker")
            == "queue_available"
        )

    def test_live_connected(self):
        from ax_cli.gateway import _derive_reachability

        assert (
            _derive_reachability(snapshot={}, mode="LIVE", liveness="connected", activation="persistent") == "live_now"
        )

    def test_attach_required(self):
        from ax_cli.gateway import _derive_reachability

        assert (
            _derive_reachability(snapshot={}, mode="LIVE", liveness="offline", activation="attach_only")
            == "attach_required"
        )

    def test_on_demand(self):
        from ax_cli.gateway import _derive_reachability

        assert (
            _derive_reachability(snapshot={}, mode="ON-DEMAND", liveness="offline", activation="on_demand")
            == "launch_available"
        )

    def test_blocked_by_attestation(self):
        from ax_cli.gateway import _derive_reachability

        snapshot = {"attestation_state": "drifted"}
        assert (
            _derive_reachability(snapshot=snapshot, mode="LIVE", liveness="connected", activation="persistent")
            == "unavailable"
        )

    def test_blocked_by_approval(self):
        from ax_cli.gateway import _derive_reachability

        snapshot = {"approval_state": "pending"}
        assert (
            _derive_reachability(snapshot=snapshot, mode="LIVE", liveness="connected", activation="persistent")
            == "unavailable"
        )

    def test_sse_disconnected_for_attach_only_with_broken_sse(self):
        from ax_cli.gateway import _derive_reachability

        snapshot = {"sse_connected": False}
        assert (
            _derive_reachability(snapshot=snapshot, mode="LIVE", liveness="stale", activation="attach_only")
            == "sse_disconnected"
        )

    def test_attach_required_when_sse_connected_not_set(self):
        from ax_cli.gateway import _derive_reachability

        assert (
            _derive_reachability(snapshot={}, mode="LIVE", liveness="stale", activation="attach_only")
            == "attach_required"
        )

    def test_non_channel_agent_not_affected_by_sse_connected_false(self):
        from ax_cli.gateway import _derive_reachability

        snapshot = {"sse_connected": False}
        assert (
            _derive_reachability(snapshot=snapshot, mode="LIVE", liveness="connected", activation="persistent")
            == "live_now"
        )


class TestDeriveWorkState:
    """_derive_work_state: maps liveness + current_status to work state."""

    def test_setup_error_blocked(self):
        from ax_cli.gateway import _derive_work_state

        assert _derive_work_state({}, liveness="setup_error") == "blocked"

    def test_working_status(self):
        from ax_cli.gateway import _derive_work_state

        assert _derive_work_state({"current_status": "processing"}, liveness="connected") == "working"

    def test_blocked_by_attestation(self):
        from ax_cli.gateway import _derive_work_state

        assert _derive_work_state({"attestation_state": "drifted"}, liveness="connected") == "blocked"

    def test_idle(self):
        from ax_cli.gateway import _derive_work_state

        assert _derive_work_state({"current_status": "idle"}, liveness="connected") == "idle"

    def test_queued_for_mailbox(self):
        from ax_cli.gateway import _derive_work_state

        profile = {"placement": "mailbox", "activation": "queue_worker"}
        snapshot = {"current_status": "queued", "backlog_depth": 3}
        assert _derive_work_state(snapshot, liveness="connected", profile=profile) == "queued"


class TestDeriveConfidence:
    """_derive_confidence: maps snapshot to (level, reason, detail) triple."""

    def test_setup_blocked(self):
        from ax_cli.gateway import _derive_confidence

        level, reason, detail = _derive_confidence(
            {"last_error": "boom"}, mode="LIVE", liveness="setup_error", reachability="unavailable"
        )
        assert level == "BLOCKED"
        assert reason == "setup_blocked"

    def test_live_connected(self):
        from ax_cli.gateway import _derive_confidence

        level, reason, detail = _derive_confidence({}, mode="LIVE", liveness="connected", reachability="live_now")
        assert level == "HIGH"
        assert reason == "live_now"

    def test_inbox_queue(self):
        from ax_cli.gateway import _derive_confidence

        level, reason, detail = _derive_confidence(
            {}, mode="INBOX", liveness="connected", reachability="queue_available"
        )
        assert level == "HIGH"
        assert reason == "queue_available"

    def test_on_demand_launch(self):
        from ax_cli.gateway import _derive_confidence

        level, reason, detail = _derive_confidence(
            {}, mode="ON-DEMAND", liveness="offline", reachability="launch_available"
        )
        assert level == "MEDIUM"
        assert reason == "launch_available"

    def test_blocked_by_identity(self):
        from ax_cli.gateway import _derive_confidence

        level, reason, detail = _derive_confidence(
            {"identity_status": "unknown_identity"}, mode="LIVE", liveness="connected", reachability="live_now"
        )
        assert level == "BLOCKED"
        assert reason == "identity_unbound"

    def test_sse_disconnected_returns_low(self):
        from ax_cli.gateway import _derive_confidence

        level, reason, detail = _derive_confidence(
            {}, mode="LIVE", liveness="stale", reachability="sse_disconnected"
        )
        assert level == "LOW"
        assert reason == "sse_disconnected"
        assert "SSE subscription is down" in detail

    def test_attach_required_still_returns_low(self):
        from ax_cli.gateway import _derive_confidence

        level, reason, detail = _derive_confidence(
            {}, mode="LIVE", liveness="stale", reachability="attach_required"
        )
        assert level == "LOW"
        assert reason == "attach_required"


# ---------------------------------------------------------------------------
# Payload / hash / encoding helpers
# ---------------------------------------------------------------------------


class TestPayloadHash:
    """_payload_hash: deterministic sha256 of a JSON payload."""

    def test_deterministic(self):
        from ax_cli.gateway import _payload_hash

        payload = {"a": 1, "b": "two"}
        assert _payload_hash(payload) == _payload_hash(payload)

    def test_key_order_independent(self):
        from ax_cli.gateway import _payload_hash

        assert _payload_hash({"a": 1, "b": 2}) == _payload_hash({"b": 2, "a": 1})

    def test_starts_with_sha256(self):
        from ax_cli.gateway import _payload_hash

        assert _payload_hash({"x": 1}).startswith("sha256:")


class TestB64UrlEncodeDecode:
    """_b64url_encode / _b64url_decode round-trip."""

    def test_round_trip(self):
        from ax_cli.gateway import _b64url_decode, _b64url_encode

        original = b"hello world, this is a test payload!"
        encoded = _b64url_encode(original)
        decoded = _b64url_decode(encoded)
        assert decoded == original

    def test_no_padding(self):
        from ax_cli.gateway import _b64url_encode

        encoded = _b64url_encode(b"test")
        assert "=" not in encoded


class TestWithoutNone:
    """_without_none: strips None and empty-string values from dicts."""

    def test_removes_none(self):
        from ax_cli.gateway import _without_none

        assert _without_none({"a": 1, "b": None, "c": ""}) == {"a": 1}

    def test_preserves_zero_and_false(self):
        from ax_cli.gateway import _without_none

        assert _without_none({"a": 0, "b": False}) == {"a": 0, "b": False}


class TestNormalizedBaseUrl:
    """_normalized_base_url: strips trailing slashes, handles None."""

    def test_strips_trailing_slash(self):
        from ax_cli.gateway import _normalized_base_url

        assert _normalized_base_url("https://paxai.app/") == "https://paxai.app"

    def test_none_returns_empty(self):
        from ax_cli.gateway import _normalized_base_url

        assert _normalized_base_url(None) == ""

    def test_strips_whitespace(self):
        from ax_cli.gateway import _normalized_base_url

        assert _normalized_base_url("  https://paxai.app  ") == "https://paxai.app"


class TestEnvironmentLabelForBaseUrl:
    """_environment_label_for_base_url: maps base_url to environment label."""

    def test_prod(self):
        from ax_cli.gateway import _environment_label_for_base_url

        assert _environment_label_for_base_url("https://paxai.app") == "prod"

    def test_dev(self):
        from ax_cli.gateway import _environment_label_for_base_url

        assert _environment_label_for_base_url("https://dev.paxai.app") == "dev"

    def test_localhost(self):
        from ax_cli.gateway import _environment_label_for_base_url

        assert _environment_label_for_base_url("http://localhost:8080") == "local"

    def test_127_0_0_1(self):
        from ax_cli.gateway import _environment_label_for_base_url

        assert _environment_label_for_base_url("http://127.0.0.1:5000") == "local"

    def test_custom_host(self):
        from ax_cli.gateway import _environment_label_for_base_url

        assert _environment_label_for_base_url("https://custom.example.com") == "custom.example.com"

    def test_empty(self):
        from ax_cli.gateway import _environment_label_for_base_url

        assert _environment_label_for_base_url("") == "unknown"
        assert _environment_label_for_base_url(None) == "unknown"


class TestParseIso8601:
    """_parse_iso8601: parses ISO timestamps, returns None on failure."""

    def test_valid_utc(self):
        from ax_cli.gateway import _parse_iso8601

        dt = _parse_iso8601("2026-05-14T12:00:00+00:00")
        assert dt is not None
        assert dt.year == 2026
        assert dt.month == 5

    def test_z_suffix(self):
        from ax_cli.gateway import _parse_iso8601

        dt = _parse_iso8601("2026-05-14T12:00:00Z")
        assert dt is not None

    def test_none_returns_none(self):
        from ax_cli.gateway import _parse_iso8601

        assert _parse_iso8601(None) is None

    def test_empty_returns_none(self):
        from ax_cli.gateway import _parse_iso8601

        assert _parse_iso8601("") is None

    def test_garbage_returns_none(self):
        from ax_cli.gateway import _parse_iso8601

        assert _parse_iso8601("not-a-date") is None


class TestAgeSeconds:
    """_age_seconds: computes age in seconds from a timestamp."""

    def test_recent_timestamp(self):
        from ax_cli.gateway import _age_seconds

        now = datetime.now(timezone.utc)
        past = (now - timedelta(seconds=60)).isoformat()
        age = _age_seconds(past, now=now)
        assert age == 60

    def test_none_returns_none(self):
        from ax_cli.gateway import _age_seconds

        assert _age_seconds(None) is None

    def test_future_clamped_to_zero(self):
        from ax_cli.gateway import _age_seconds

        now = datetime.now(timezone.utc)
        future = (now + timedelta(seconds=60)).isoformat()
        age = _age_seconds(future, now=now)
        assert age == 0


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


class TestRedactedPath:
    """_redacted_path: replaces home directory with ~."""

    def test_home_path(self):
        from ax_cli.gateway import _redacted_path

        home = str(Path.home())
        result = _redacted_path(f"{home}/test/file")
        assert result is not None
        assert result.startswith("~")
        assert "test/file" in result

    def test_non_home_path(self):
        from ax_cli.gateway import _redacted_path

        result = _redacted_path("/tmp/some/file")
        assert result is not None
        # Should be an absolute path since it's not under home
        assert result.startswith("/")

    def test_empty_returns_none(self):
        from ax_cli.gateway import _redacted_path

        assert _redacted_path("") is None
        assert _redacted_path(None) is None


class TestCommandExecutablePath:
    """_command_executable_path: extracts the executable from a command string."""

    def test_simple_command(self):
        from ax_cli.gateway import _command_executable_path

        result = _command_executable_path("python3 -m myapp")
        assert result is not None
        # Should resolve python3 via which or return the raw name
        assert "python" in result.lower() or result == "python3"

    def test_env_prefix(self):
        from ax_cli.gateway import _command_executable_path

        result = _command_executable_path("env FOO=bar python3")
        assert result is not None
        assert "python" in result.lower() or result == "python3"

    def test_empty_returns_none(self):
        from ax_cli.gateway import _command_executable_path

        assert _command_executable_path(None) is None
        assert _command_executable_path("") is None


# ---------------------------------------------------------------------------
# Space cache helpers
# ---------------------------------------------------------------------------


class TestSpaceCacheRows:
    """_space_cache_rows: normalizes diverse space list formats."""

    def test_dict_items(self):
        from ax_cli.gateway import _space_cache_rows

        rows = _space_cache_rows(
            [
                {"space_id": "aaa", "name": "Alpha"},
                {"id": "bbb", "space_name": "Beta"},
            ]
        )
        assert len(rows) == 2
        assert rows[0]["space_id"] == "aaa"
        assert rows[0]["name"] == "Alpha"
        assert rows[1]["space_id"] == "bbb"
        assert rows[1]["name"] == "Beta"

    def test_deduplicates_by_space_id(self):
        from ax_cli.gateway import _space_cache_rows

        rows = _space_cache_rows(
            [
                {"space_id": "aaa", "name": "Alpha"},
                {"space_id": "aaa", "name": "Alpha Again"},
            ]
        )
        assert len(rows) == 1

    def test_non_list_returns_empty(self):
        from ax_cli.gateway import _space_cache_rows

        assert _space_cache_rows(None) == []
        assert _space_cache_rows("bogus") == []

    def test_skips_non_dict_items(self):
        from ax_cli.gateway import _space_cache_rows

        rows = _space_cache_rows([{"space_id": "aaa", "name": "A"}, "bad", 123])
        assert len(rows) == 1


class TestSpaceNameFromCacheLocal:
    """_space_name_from_cache: looks up space name from allowed_spaces list."""

    def test_found(self):
        from ax_cli.gateway import _space_name_from_cache

        spaces = [{"space_id": "aaa", "name": "Alpha"}]
        assert _space_name_from_cache(spaces, "aaa") == "Alpha"

    def test_not_found(self):
        from ax_cli.gateway import _space_name_from_cache

        spaces = [{"space_id": "aaa", "name": "Alpha"}]
        assert _space_name_from_cache(spaces, "bbb") is None

    def test_uuid_name_skipped(self):
        from ax_cli.gateway import _space_name_from_cache

        uid = "12345678-1234-1234-1234-123456789abc"
        spaces = [{"space_id": uid, "name": uid}]
        assert _space_name_from_cache(spaces, uid) is None

    def test_empty_space_id(self):
        from ax_cli.gateway import _space_name_from_cache

        assert _space_name_from_cache([], None) is None
        assert _space_name_from_cache([], "") is None


class TestLooksLikeSpaceUuid:
    """looks_like_space_uuid: validates UUID-4 shape."""

    def test_valid_uuid(self):
        from ax_cli.gateway import looks_like_space_uuid

        assert looks_like_space_uuid("12345678-1234-1234-1234-123456789abc") is True

    def test_invalid_string(self):
        from ax_cli.gateway import looks_like_space_uuid

        assert looks_like_space_uuid("not-a-uuid") is False

    def test_non_string(self):
        from ax_cli.gateway import looks_like_space_uuid

        assert looks_like_space_uuid(12345) is False
        assert looks_like_space_uuid(None) is False

    def test_whitespace_stripped(self):
        from ax_cli.gateway import looks_like_space_uuid

        assert looks_like_space_uuid("  12345678-1234-1234-1234-123456789abc  ") is True


class TestSpaceIdAllowed:
    """_space_id_allowed: checks if a space_id is in the allowed list."""

    def test_allowed(self):
        from ax_cli.gateway import _space_id_allowed

        spaces = [{"space_id": "aaa"}, {"space_id": "bbb"}]
        assert _space_id_allowed(spaces, "aaa") is True

    def test_not_allowed(self):
        from ax_cli.gateway import _space_id_allowed

        spaces = [{"space_id": "aaa"}]
        assert _space_id_allowed(spaces, "bbb") is False

    def test_empty(self):
        from ax_cli.gateway import _space_id_allowed

        assert _space_id_allowed([], "aaa") is False
        assert _space_id_allowed([], None) is False


# ---------------------------------------------------------------------------
# Registry helpers (find / upsert / remove entries, bindings)
# ---------------------------------------------------------------------------


class TestFindAgentEntry:
    """find_agent_entry: finds agent by name in registry."""

    def test_found(self):
        from ax_cli.gateway import find_agent_entry

        registry = {"agents": [{"name": "alpha"}, {"name": "beta"}]}
        result = find_agent_entry(registry, "alpha")
        assert result is not None
        assert result["name"] == "alpha"

    def test_case_insensitive(self):
        from ax_cli.gateway import find_agent_entry

        registry = {"agents": [{"name": "Alpha"}]}
        assert find_agent_entry(registry, "alpha") is not None

    def test_not_found(self):
        from ax_cli.gateway import find_agent_entry

        registry = {"agents": [{"name": "alpha"}]}
        assert find_agent_entry(registry, "gamma") is None

    def test_empty_registry(self):
        from ax_cli.gateway import find_agent_entry

        assert find_agent_entry({}, "alpha") is None
        assert find_agent_entry({"agents": []}, "alpha") is None


class TestFindAgentEntryByRef:
    """find_agent_entry_by_ref: finds agent by row number, name, or id prefix."""

    def test_by_row_number(self):
        from ax_cli.gateway import find_agent_entry_by_ref

        registry = {"agents": [{"name": "alpha"}, {"name": "beta"}]}
        assert find_agent_entry_by_ref(registry, "1")["name"] == "alpha"
        assert find_agent_entry_by_ref(registry, "#2")["name"] == "beta"

    def test_by_name(self):
        from ax_cli.gateway import find_agent_entry_by_ref

        registry = {"agents": [{"name": "alpha"}, {"name": "beta"}]}
        assert find_agent_entry_by_ref(registry, "beta")["name"] == "beta"

    def test_by_install_id(self):
        from ax_cli.gateway import find_agent_entry_by_ref

        iid = str(uuid.uuid4())
        registry = {"agents": [{"name": "alpha", "install_id": iid}]}
        assert find_agent_entry_by_ref(registry, iid)["name"] == "alpha"

    def test_by_id_prefix(self):
        from ax_cli.gateway import find_agent_entry_by_ref

        iid = "abcdef12-3456-7890-abcd-ef1234567890"
        registry = {"agents": [{"name": "alpha", "install_id": iid}]}
        assert find_agent_entry_by_ref(registry, "abcdef12")["name"] == "alpha"

    def test_empty_returns_none(self):
        from ax_cli.gateway import find_agent_entry_by_ref

        assert find_agent_entry_by_ref({"agents": []}, "") is None
        assert find_agent_entry_by_ref({"agents": []}, None) is None

    def test_out_of_range_index(self):
        from ax_cli.gateway import find_agent_entry_by_ref

        registry = {"agents": [{"name": "alpha"}]}
        assert find_agent_entry_by_ref(registry, "5") is None


class TestUpsertAgentEntry:
    """upsert_agent_entry: inserts or updates agent entries by name."""

    def test_insert_new(self):
        from ax_cli.gateway import upsert_agent_entry

        registry = {"agents": []}
        result = upsert_agent_entry(registry, {"name": "alpha", "runtime_type": "echo"})
        assert result["name"] == "alpha"
        assert len(registry["agents"]) == 1

    def test_update_existing(self):
        from ax_cli.gateway import upsert_agent_entry

        registry = {"agents": [{"name": "alpha", "runtime_type": "echo", "extra": "keep"}]}
        result = upsert_agent_entry(registry, {"name": "alpha", "runtime_type": "exec"})
        assert result["runtime_type"] == "exec"
        assert result["extra"] == "keep"
        assert len(registry["agents"]) == 1

    def test_case_insensitive_upsert(self):
        from ax_cli.gateway import upsert_agent_entry

        registry = {"agents": [{"name": "Alpha"}]}
        result = upsert_agent_entry(registry, {"name": "alpha", "updated": True})
        assert result["updated"] is True
        assert len(registry["agents"]) == 1


class TestRemoveAgentEntry:
    """remove_agent_entry: removes agent by name, returns it."""

    def test_remove_existing(self):
        from ax_cli.gateway import remove_agent_entry

        registry = {"agents": [{"name": "alpha"}, {"name": "beta"}]}
        removed = remove_agent_entry(registry, "alpha")
        assert removed is not None
        assert removed["name"] == "alpha"
        assert len(registry["agents"]) == 1

    def test_remove_missing_returns_none(self):
        from ax_cli.gateway import remove_agent_entry

        registry = {"agents": [{"name": "alpha"}]}
        assert remove_agent_entry(registry, "gamma") is None
        assert len(registry["agents"]) == 1


class TestFindBinding:
    """find_binding: looks up a binding by asset_id / install_id / gateway_id."""

    def test_find_by_asset_id(self):
        from ax_cli.gateway import find_binding

        registry = {"bindings": [{"asset_id": "a1", "install_id": "i1"}]}
        result = find_binding(registry, asset_id="a1")
        assert result is not None
        assert result["asset_id"] == "a1"

    def test_find_by_install_id(self):
        from ax_cli.gateway import find_binding

        registry = {"bindings": [{"asset_id": "a1", "install_id": "i1"}]}
        assert find_binding(registry, install_id="i1") is not None

    def test_not_found(self):
        from ax_cli.gateway import find_binding

        registry = {"bindings": [{"asset_id": "a1"}]}
        assert find_binding(registry, asset_id="a2") is None

    def test_initializes_missing_lists(self):
        from ax_cli.gateway import find_binding

        registry = {}
        find_binding(registry, asset_id="x")
        assert "bindings" in registry
        assert "identity_bindings" in registry
        assert "approvals" in registry


class TestUpsertBinding:
    """upsert_binding: inserts or merges binding by install_id."""

    def test_insert_new(self):
        from ax_cli.gateway import upsert_binding

        registry = {"bindings": [], "identity_bindings": [], "approvals": []}
        binding = {"install_id": "i1", "asset_id": "a1"}
        result = upsert_binding(registry, binding)
        assert result["install_id"] == "i1"
        assert len(registry["bindings"]) == 1

    def test_merge_existing(self):
        from ax_cli.gateway import upsert_binding

        registry = {
            "bindings": [{"install_id": "i1", "asset_id": "a1", "extra": "keep"}],
            "identity_bindings": [],
            "approvals": [],
        }
        result = upsert_binding(registry, {"install_id": "i1", "new_field": "added"})
        assert result["extra"] == "keep"
        assert result["new_field"] == "added"
        assert len(registry["bindings"]) == 1


class TestFindIdentityBinding:
    """find_identity_binding: looks up identity bindings with env matching."""

    def test_find_by_install_id(self):
        from ax_cli.gateway import find_identity_binding

        registry = {
            "identity_bindings": [{"install_id": "i1", "environment": {"base_url": "https://paxai.app"}}],
            "bindings": [],
            "approvals": [],
        }
        result = find_identity_binding(registry, install_id="i1")
        assert result is not None

    def test_base_url_filter(self):
        from ax_cli.gateway import find_identity_binding

        registry = {
            "identity_bindings": [
                {"install_id": "i1", "environment": {"base_url": "https://paxai.app"}},
                {"install_id": "i1", "environment": {"base_url": "https://dev.paxai.app"}},
            ],
            "bindings": [],
            "approvals": [],
        }
        result = find_identity_binding(registry, install_id="i1", base_url="https://dev.paxai.app")
        assert result is not None
        assert result["environment"]["base_url"] == "https://dev.paxai.app"


class TestNormalizeAllowedSpacesPayload:
    """_normalize_allowed_spaces_payload: handles various backend response formats."""

    def test_dict_with_spaces_key(self):
        from ax_cli.gateway import _normalize_allowed_spaces_payload

        payload = {"spaces": [{"space_id": "aaa", "name": "Alpha"}]}
        result = _normalize_allowed_spaces_payload(payload)
        assert len(result) == 1
        assert result[0]["space_id"] == "aaa"

    def test_dict_with_items_key(self):
        from ax_cli.gateway import _normalize_allowed_spaces_payload

        payload = {"items": [{"space_id": "bbb", "name": "Beta"}]}
        result = _normalize_allowed_spaces_payload(payload)
        assert len(result) == 1

    def test_dict_with_results_key(self):
        from ax_cli.gateway import _normalize_allowed_spaces_payload

        payload = {"results": [{"space_id": "ccc", "name": "Gamma"}]}
        result = _normalize_allowed_spaces_payload(payload)
        assert len(result) == 1

    def test_raw_list(self):
        from ax_cli.gateway import _normalize_allowed_spaces_payload

        payload = [{"space_id": "ddd", "name": "Delta"}]
        result = _normalize_allowed_spaces_payload(payload)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Apply entry current space
# ---------------------------------------------------------------------------


class TestApplyEntryCurrentSpace:
    """apply_entry_current_space: updates space fields on an agent entry."""

    def test_sets_space_fields(self):
        from ax_cli.gateway import apply_entry_current_space

        entry: dict[str, Any] = {"name": "alpha"}
        result = apply_entry_current_space(entry, "space-1", space_name="My Space")
        assert result["space_id"] == "space-1"
        assert result["active_space_id"] == "space-1"
        assert result["active_space_name"] == "My Space"
        assert result["default_space_id"] == "space-1"

    def test_empty_space_id_noop(self):
        from ax_cli.gateway import apply_entry_current_space

        entry: dict[str, Any] = {"name": "alpha", "space_id": "old"}
        result = apply_entry_current_space(entry, "")
        assert result.get("space_id") == "old"

    def test_preserves_allowed_spaces(self):
        from ax_cli.gateway import apply_entry_current_space

        entry: dict[str, Any] = {
            "name": "alpha",
            "allowed_spaces": [{"space_id": "other", "name": "Other", "is_default": True}],
        }
        result = apply_entry_current_space(entry, "new-space", space_name="New")
        assert len(result["allowed_spaces"]) == 2
        ids = {row["space_id"] for row in result["allowed_spaces"]}
        assert "new-space" in ids
        assert "other" in ids


# ---------------------------------------------------------------------------
# Approval helpers
# ---------------------------------------------------------------------------


class TestApprovalStatus:
    """_approval_status: normalizes approval status, maps denied->rejected."""

    def test_pending(self):
        from ax_cli.gateway import _approval_status

        assert _approval_status({"status": "pending"}) == "pending"

    def test_approved(self):
        from ax_cli.gateway import _approval_status

        assert _approval_status({"status": "approved"}) == "approved"

    def test_denied_maps_to_rejected(self):
        from ax_cli.gateway import _approval_status

        assert _approval_status({"status": "denied"}) == "rejected"

    def test_empty(self):
        from ax_cli.gateway import _approval_status

        assert _approval_status({}) == ""


class TestFindApprovalById:
    """_find_approval_by_id: looks up approval by its ID."""

    def test_found(self):
        from ax_cli.gateway import _find_approval_by_id

        registry = {
            "approvals": [{"approval_id": "appr-1", "status": "pending"}],
            "bindings": [],
            "identity_bindings": [],
        }
        result = _find_approval_by_id(registry, "appr-1")
        assert result is not None

    def test_not_found(self):
        from ax_cli.gateway import _find_approval_by_id

        registry = {"approvals": [], "bindings": [], "identity_bindings": []}
        assert _find_approval_by_id(registry, "appr-99") is None


class TestFindApprovalForSignature:
    """_find_approval_for_signature: finds most recent non-archived approval."""

    def test_finds_most_recent(self):
        from ax_cli.gateway import _find_approval_for_signature

        registry = {
            "approvals": [
                {"candidate_signature": "sig1", "status": "pending", "requested_at": "2026-01-01"},
                {"candidate_signature": "sig1", "status": "pending", "requested_at": "2026-05-01"},
            ],
            "bindings": [],
            "identity_bindings": [],
        }
        result = _find_approval_for_signature(registry, "sig1")
        assert result is not None
        assert result["requested_at"] == "2026-05-01"

    def test_skips_archived(self):
        from ax_cli.gateway import _find_approval_for_signature

        registry = {
            "approvals": [
                {"candidate_signature": "sig1", "status": "archived", "requested_at": "2026-01-01"},
            ],
            "bindings": [],
            "identity_bindings": [],
        }
        assert _find_approval_for_signature(registry, "sig1") is None


# ---------------------------------------------------------------------------
# Asset ID / binding type / launch spec helpers
# ---------------------------------------------------------------------------


class TestAssetIdForEntry:
    """_asset_id_for_entry: resolves agent_id > asset_id > name."""

    def test_agent_id(self):
        from ax_cli.gateway import _asset_id_for_entry

        assert _asset_id_for_entry({"agent_id": "aid1"}) == "aid1"

    def test_asset_id_fallback(self):
        from ax_cli.gateway import _asset_id_for_entry

        assert _asset_id_for_entry({"asset_id": "asid1"}) == "asid1"

    def test_name_fallback(self):
        from ax_cli.gateway import _asset_id_for_entry

        assert _asset_id_for_entry({"name": "myagent"}) == "myagent"

    def test_empty(self):
        from ax_cli.gateway import _asset_id_for_entry

        assert _asset_id_for_entry({}) == ""


class TestBindingTypeForEntry:
    """_binding_type_for_entry: determines binding type from activation/runtime."""

    def test_attach_only(self):
        from ax_cli.gateway import _binding_type_for_entry

        assert _binding_type_for_entry({"activation": "attach_only"}) == "attached_session"

    def test_queue_worker(self):
        from ax_cli.gateway import _binding_type_for_entry

        assert _binding_type_for_entry({"activation": "queue_worker"}) == "queue_worker"

    def test_inbox_runtime(self):
        from ax_cli.gateway import _binding_type_for_entry

        assert _binding_type_for_entry({"runtime_type": "inbox"}) == "queue_worker"

    def test_default(self):
        from ax_cli.gateway import _binding_type_for_entry

        assert _binding_type_for_entry({}) == "local_runtime"


class TestLaunchSpecForEntry:
    """_launch_spec_for_entry: builds launch spec dict from entry."""

    def test_full_entry(self):
        from ax_cli.gateway import _launch_spec_for_entry

        entry = {
            "runtime_type": "exec",
            "template_id": "echo_test",
            "exec_command": "python3 handler.py",
            "workdir": "/home/user/project",
            "ollama_model": "llama3",
        }
        spec = _launch_spec_for_entry(entry)
        assert spec["runtime_type"] == "exec"
        assert spec["template_id"] == "echo_test"
        assert spec["command"] == "python3 handler.py"
        assert spec["workdir"] == "/home/user/project"
        assert spec["ollama_model"] == "llama3"

    def test_empty_entry(self):
        from ax_cli.gateway import _launch_spec_for_entry

        spec = _launch_spec_for_entry({})
        assert spec["runtime_type"] is None
        assert spec["command"] is None

    def test_model_from_hermes_model(self):
        from ax_cli.gateway import _launch_spec_for_entry

        entry = {"hermes_model": "claude-sonnet-4-20250514"}
        spec = _launch_spec_for_entry(entry)
        assert spec["model"] == "claude-sonnet-4-20250514"


# ---------------------------------------------------------------------------
# Infer operator profile / asset descriptor
# ---------------------------------------------------------------------------


class TestInferOperatorProfile:
    """infer_operator_profile: derives profile from snapshot + defaults."""

    def test_defaults_for_echo(self):
        from ax_cli.gateway import infer_operator_profile

        profile = infer_operator_profile({"runtime_type": "echo"})
        assert profile["placement"] == "hosted"
        assert profile["activation"] == "persistent"

    def test_overrides_applied(self):
        from ax_cli.gateway import infer_operator_profile

        snapshot = {
            "runtime_type": "echo",
            "placement": "mailbox",
            "user_overrides": {"operator": {"placement": True}},
        }
        profile = infer_operator_profile(snapshot)
        assert profile["placement"] == "mailbox"

    def test_overrides_not_applied_without_marker(self):
        from ax_cli.gateway import infer_operator_profile

        # Without user_overrides, the snapshot value is ignored in favor of defaults
        snapshot = {"runtime_type": "echo", "placement": "mailbox"}
        profile = infer_operator_profile(snapshot)
        assert profile["placement"] == "hosted"  # default for echo


class TestInferAssetDescriptor:
    """infer_asset_descriptor: builds full asset descriptor from snapshot."""

    def test_echo_test_defaults(self):
        from ax_cli.gateway import infer_asset_descriptor

        descriptor = infer_asset_descriptor({"template_id": "echo_test"})
        assert descriptor["asset_class"] == "interactive_agent"
        assert descriptor["intake_model"] == "live_listener"
        assert descriptor["type_label"] == "Live Listener"
        assert descriptor["output_label"] == "Reply"

    def test_service_account_defaults(self):
        from ax_cli.gateway import infer_asset_descriptor

        descriptor = infer_asset_descriptor({"template_id": "service_account"})
        assert descriptor["asset_class"] == "service_account"
        assert descriptor["type_label"] == "Service Account"

    def test_primary_fields(self):
        from ax_cli.gateway import infer_asset_descriptor

        descriptor = infer_asset_descriptor({"template_id": "echo_test", "name": "my-echo"})
        assert descriptor["display_name"] == "my-echo"
        assert descriptor["primary_trigger_source"] == "direct_message"
        assert descriptor["primary_return_path"] == "inline_reply"


# ---------------------------------------------------------------------------
# Gateway environment
# ---------------------------------------------------------------------------


class TestGatewayEnvironment:
    """gateway_environment: normalizes env var to environment label."""

    def test_no_env_returns_none(self, monkeypatch):
        from ax_cli.gateway import gateway_environment

        monkeypatch.delenv("AX_GATEWAY_ENV", raising=False)
        monkeypatch.delenv("AX_USER_ENV", raising=False)
        monkeypatch.delenv("AX_ENV", raising=False)
        assert gateway_environment() is None

    def test_default_returns_none(self, monkeypatch):
        from ax_cli.gateway import gateway_environment

        monkeypatch.setenv("AX_GATEWAY_ENV", "default")
        assert gateway_environment() is None

    def test_user_returns_none(self, monkeypatch):
        from ax_cli.gateway import gateway_environment

        monkeypatch.setenv("AX_GATEWAY_ENV", "user")
        assert gateway_environment() is None

    def test_normal_env(self, monkeypatch):
        from ax_cli.gateway import gateway_environment

        monkeypatch.setenv("AX_GATEWAY_ENV", "staging")
        assert gateway_environment() == "staging"

    def test_normalizes_special_chars(self, monkeypatch):
        from ax_cli.gateway import gateway_environment

        monkeypatch.setenv("AX_GATEWAY_ENV", "My Env!@#")
        result = gateway_environment()
        assert result is not None
        assert " " not in result
        assert "!" not in result


# ---------------------------------------------------------------------------
# Filesystem-touching helpers (use tmp_path from conftest)
# ---------------------------------------------------------------------------


class TestGatewayDir:
    """gateway_dir: returns/creates the gateway state directory."""

    def test_uses_env_var(self, monkeypatch, tmp_path):
        from ax_cli.gateway import gateway_dir

        custom = tmp_path / "custom-gateway"
        monkeypatch.setenv("AX_GATEWAY_DIR", str(custom))
        result = gateway_dir()
        assert result == custom
        assert result.exists()

    def test_default_under_config(self, monkeypatch, tmp_path):
        from ax_cli.gateway import gateway_dir

        monkeypatch.delenv("AX_GATEWAY_DIR", raising=False)
        monkeypatch.delenv("AX_GATEWAY_ENV", raising=False)
        monkeypatch.delenv("AX_USER_ENV", raising=False)
        monkeypatch.delenv("AX_ENV", raising=False)
        result = gateway_dir()
        assert result.exists()
        assert "gateway" in str(result)


class TestSessionAndRegistryPaths:
    """Verify path derivations off gateway_dir."""

    def test_session_path(self, monkeypatch, tmp_path):
        from ax_cli.gateway import gateway_dir, session_path

        monkeypatch.setenv("AX_GATEWAY_DIR", str(tmp_path / "gw"))
        assert session_path() == gateway_dir() / "session.json"

    def test_registry_path(self, monkeypatch, tmp_path):
        from ax_cli.gateway import gateway_dir, registry_path

        monkeypatch.setenv("AX_GATEWAY_DIR", str(tmp_path / "gw"))
        assert registry_path() == gateway_dir() / "registry.json"


class TestLoadSaveGatewaySession:
    """load_gateway_session / save_gateway_session round-trip."""

    def test_default_empty(self, monkeypatch, tmp_path):
        from ax_cli.gateway import load_gateway_session

        monkeypatch.setenv("AX_GATEWAY_DIR", str(tmp_path / "gw"))
        assert load_gateway_session() == {}

    def test_round_trip(self, monkeypatch, tmp_path):
        from ax_cli.gateway import load_gateway_session, save_gateway_session

        monkeypatch.setenv("AX_GATEWAY_DIR", str(tmp_path / "gw"))
        save_gateway_session({"token": "abc", "base_url": "https://paxai.app"})
        loaded = load_gateway_session()
        assert loaded["token"] == "abc"
        assert loaded["base_url"] == "https://paxai.app"
        assert "saved_at" in loaded


class TestLoadSaveGatewayRegistry:
    """load_gateway_registry / save_gateway_registry round-trip."""

    def test_default_registry(self, monkeypatch, tmp_path):
        from ax_cli.gateway import load_gateway_registry

        monkeypatch.setenv("AX_GATEWAY_DIR", str(tmp_path / "gw"))
        registry = load_gateway_registry()
        assert "gateway" in registry
        assert "agents" in registry
        assert "bindings" in registry
        assert registry["gateway"]["gateway_id"]

    def test_round_trip(self, monkeypatch, tmp_path):
        from ax_cli.gateway import load_gateway_registry, save_gateway_registry, upsert_agent_entry

        monkeypatch.setenv("AX_GATEWAY_DIR", str(tmp_path / "gw"))
        registry = load_gateway_registry()
        upsert_agent_entry(registry, {"name": "test-agent", "runtime_type": "echo"})
        save_gateway_registry(registry)
        loaded = load_gateway_registry()
        assert len(loaded["agents"]) == 1
        assert loaded["agents"][0]["name"] == "test-agent"


class TestReconcileCorruptSpaceIds:
    """reconcile_corrupt_space_ids: heals agents where space_id is a name, not UUID."""

    def test_heals_from_active_space_id(self):
        from ax_cli.gateway import reconcile_corrupt_space_ids

        uid = "12345678-1234-1234-1234-123456789abc"
        registry = {"agents": [{"name": "a", "space_id": "My Space Name", "active_space_id": uid}]}
        count = reconcile_corrupt_space_ids(registry)
        assert count == 1
        assert registry["agents"][0]["space_id"] == uid

    def test_leaves_valid_uuid_alone(self):
        from ax_cli.gateway import reconcile_corrupt_space_ids

        uid = "12345678-1234-1234-1234-123456789abc"
        registry = {"agents": [{"name": "a", "space_id": uid}]}
        count = reconcile_corrupt_space_ids(registry)
        assert count == 0

    def test_heals_from_allowed_spaces(self):
        from ax_cli.gateway import reconcile_corrupt_space_ids

        uid = "12345678-1234-1234-1234-123456789abc"
        registry = {
            "agents": [
                {
                    "name": "a",
                    "space_id": "not-a-uuid",
                    "allowed_spaces": [{"space_id": uid}],
                }
            ]
        }
        count = reconcile_corrupt_space_ids(registry)
        assert count == 1
        assert registry["agents"][0]["space_id"] == uid


class TestLoadSaveSpaceCache:
    """load_space_cache / save_space_cache / upsert_space_cache_entry."""

    def test_empty_by_default(self, monkeypatch, tmp_path):
        from ax_cli.gateway import load_space_cache

        monkeypatch.setenv("AX_GATEWAY_DIR", str(tmp_path / "gw"))
        assert load_space_cache() == []

    def test_save_and_load(self, monkeypatch, tmp_path):
        from ax_cli.gateway import load_space_cache, save_space_cache

        monkeypatch.setenv("AX_GATEWAY_DIR", str(tmp_path / "gw"))
        spaces = [{"id": "aaa", "name": "Alpha", "slug": "alpha"}]
        save_space_cache(spaces)
        loaded = load_space_cache()
        assert len(loaded) == 1
        assert loaded[0]["name"] == "Alpha"

    def test_save_empty_is_noop(self, monkeypatch, tmp_path):
        from ax_cli.gateway import load_space_cache, save_space_cache

        monkeypatch.setenv("AX_GATEWAY_DIR", str(tmp_path / "gw"))
        save_space_cache([])
        assert load_space_cache() == []

    def test_upsert_new_entry(self, monkeypatch, tmp_path):
        from ax_cli.gateway import load_space_cache, save_space_cache, upsert_space_cache_entry

        monkeypatch.setenv("AX_GATEWAY_DIR", str(tmp_path / "gw"))
        uid = "12345678-1234-1234-1234-123456789abc"
        save_space_cache([{"id": "existing", "name": "Existing"}])
        upsert_space_cache_entry(uid, name="New Space", slug="new-space")
        loaded = load_space_cache()
        assert len(loaded) == 2

    def test_upsert_updates_existing(self, monkeypatch, tmp_path):
        from ax_cli.gateway import load_space_cache, save_space_cache, upsert_space_cache_entry

        monkeypatch.setenv("AX_GATEWAY_DIR", str(tmp_path / "gw"))
        uid = "12345678-1234-1234-1234-123456789abc"
        save_space_cache([{"id": uid, "name": "Old Name"}])
        upsert_space_cache_entry(uid, name="New Name")
        loaded = load_space_cache()
        assert len(loaded) == 1
        assert loaded[0]["name"] == "New Name"


class TestLookupSpaceInCache:
    """lookup_space_in_cache: resolves by UUID, slug, or name."""

    def test_by_id(self, monkeypatch, tmp_path):
        from ax_cli.gateway import lookup_space_in_cache, save_space_cache

        monkeypatch.setenv("AX_GATEWAY_DIR", str(tmp_path / "gw"))
        uid = "12345678-1234-1234-1234-123456789abc"
        save_space_cache([{"id": uid, "name": "Alpha", "slug": "alpha"}])
        result = lookup_space_in_cache(uid)
        assert result is not None

    def test_by_slug(self, monkeypatch, tmp_path):
        from ax_cli.gateway import lookup_space_in_cache, save_space_cache

        monkeypatch.setenv("AX_GATEWAY_DIR", str(tmp_path / "gw"))
        save_space_cache([{"id": "aaa", "name": "Alpha", "slug": "alpha"}])
        result = lookup_space_in_cache("alpha")
        assert result is not None

    def test_by_name(self, monkeypatch, tmp_path):
        from ax_cli.gateway import lookup_space_in_cache, save_space_cache

        monkeypatch.setenv("AX_GATEWAY_DIR", str(tmp_path / "gw"))
        save_space_cache([{"id": "aaa", "name": "Alpha Space", "slug": "alpha"}])
        result = lookup_space_in_cache("Alpha Space")
        assert result is not None

    def test_not_found(self, monkeypatch, tmp_path):
        from ax_cli.gateway import lookup_space_in_cache, save_space_cache

        monkeypatch.setenv("AX_GATEWAY_DIR", str(tmp_path / "gw"))
        save_space_cache([{"id": "aaa", "name": "Alpha"}])
        assert lookup_space_in_cache("nope") is None

    def test_empty_returns_none(self, monkeypatch, tmp_path):
        from ax_cli.gateway import lookup_space_in_cache

        monkeypatch.setenv("AX_GATEWAY_DIR", str(tmp_path / "gw"))
        assert lookup_space_in_cache("") is None

    def test_ambiguous_name_returns_none(self, monkeypatch, tmp_path):
        """Regression for #47: two cached rows with the same name must not
        silently resolve to the first match. Returning None forces the caller
        to live-fetch and hit the resolver's ambiguity branch."""
        from ax_cli.gateway import lookup_space_in_cache, save_space_cache

        monkeypatch.setenv("AX_GATEWAY_DIR", str(tmp_path / "gw"))
        save_space_cache(
            [
                {"id": "11111111-1111-1111-1111-111111111111", "name": "Demo Team", "slug": "demo-team-1"},
                {"id": "22222222-2222-2222-2222-222222222222", "name": "Demo Team", "slug": "demo-team-2"},
            ]
        )
        assert lookup_space_in_cache("Demo Team") is None

    def test_ambiguous_slug_collision_returns_none(self, monkeypatch, tmp_path):
        """Same guard for slug collisions (unlikely in practice since the
        backend disambiguates slugs, but the function should still be safe
        if upstream ever returns duplicates)."""
        from ax_cli.gateway import lookup_space_in_cache, save_space_cache

        monkeypatch.setenv("AX_GATEWAY_DIR", str(tmp_path / "gw"))
        save_space_cache(
            [
                {"id": "11111111-1111-1111-1111-111111111111", "name": "Alpha", "slug": "dup"},
                {"id": "22222222-2222-2222-2222-222222222222", "name": "Beta", "slug": "dup"},
            ]
        )
        assert lookup_space_in_cache("dup") is None

    def test_unique_name_among_many_still_resolves(self, monkeypatch, tmp_path):
        """An ambiguity-aware lookup must still resolve names that are
        unique within the cache, even when other rows exist."""
        from ax_cli.gateway import lookup_space_in_cache, save_space_cache

        monkeypatch.setenv("AX_GATEWAY_DIR", str(tmp_path / "gw"))
        save_space_cache(
            [
                {"id": "11111111-1111-1111-1111-111111111111", "name": "Demo Team", "slug": "demo-team-1"},
                {"id": "22222222-2222-2222-2222-222222222222", "name": "Demo Team", "slug": "demo-team-2"},
                {"id": "33333333-3333-3333-3333-333333333333", "name": "ax-gateway", "slug": "ax-gateway"},
            ]
        )
        result = lookup_space_in_cache("ax-gateway")
        assert result is not None
        assert result["id"] == "33333333-3333-3333-3333-333333333333"

    def test_uuid_still_resolves_when_name_is_ambiguous(self, monkeypatch, tmp_path):
        """A UUID is unambiguous by construction. Even if the cache contains
        multiple rows sharing the same name, looking up by UUID must still
        short-circuit to the exact row."""
        from ax_cli.gateway import lookup_space_in_cache, save_space_cache

        monkeypatch.setenv("AX_GATEWAY_DIR", str(tmp_path / "gw"))
        target_uuid = "22222222-2222-2222-2222-222222222222"
        save_space_cache(
            [
                {"id": "11111111-1111-1111-1111-111111111111", "name": "Demo Team", "slug": "demo-team-1"},
                {"id": target_uuid, "name": "Demo Team", "slug": "demo-team-2"},
            ]
        )
        result = lookup_space_in_cache(target_uuid)
        assert result is not None
        assert result["id"] == target_uuid


class TestPendingMessages:
    """load_agent_pending_messages / save_agent_pending_messages."""

    def test_empty_default(self, monkeypatch, tmp_path):
        from ax_cli.gateway import load_agent_pending_messages

        monkeypatch.setenv("AX_GATEWAY_DIR", str(tmp_path / "gw"))
        assert load_agent_pending_messages("agent1") == []

    def test_save_and_load(self, monkeypatch, tmp_path):
        from ax_cli.gateway import load_agent_pending_messages, save_agent_pending_messages

        monkeypatch.setenv("AX_GATEWAY_DIR", str(tmp_path / "gw"))
        items = [{"message_id": "m1", "content": "hello"}]
        save_agent_pending_messages("agent1", items)
        loaded = load_agent_pending_messages("agent1")
        assert len(loaded) == 1
        assert loaded[0]["message_id"] == "m1"


class TestLoadGatewayManagedAgentToken:
    """load_gateway_managed_agent_token: validates token and rejects bootstrap PATs."""

    def test_valid_agent_token(self, monkeypatch, tmp_path):
        from ax_cli.gateway import load_gateway_managed_agent_token

        monkeypatch.setenv("AX_GATEWAY_DIR", str(tmp_path / "gw"))
        token_file = tmp_path / "token"
        token_file.write_text("axp_a_valid.token\n")
        entry = {"token_file": str(token_file), "agent_id": "aid1"}
        result = load_gateway_managed_agent_token(entry)
        assert result == "axp_a_valid.token"

    def test_rejects_user_pat(self, monkeypatch, tmp_path):
        from ax_cli.gateway import load_gateway_managed_agent_token

        token_file = tmp_path / "token"
        token_file.write_text("axp_u_user.pat\n")
        entry = {"token_file": str(token_file), "agent_id": "aid1"}
        with pytest.raises(ValueError, match="user bootstrap PAT"):
            load_gateway_managed_agent_token(entry)

    def test_missing_file(self, tmp_path):
        from ax_cli.gateway import load_gateway_managed_agent_token

        entry = {"token_file": str(tmp_path / "nonexistent"), "agent_id": "aid1"}
        with pytest.raises(ValueError, match="missing"):
            load_gateway_managed_agent_token(entry)

    def test_empty_file(self, tmp_path):
        from ax_cli.gateway import load_gateway_managed_agent_token

        token_file = tmp_path / "token"
        token_file.write_text("")
        entry = {"token_file": str(token_file), "agent_id": "aid1"}
        with pytest.raises(ValueError, match="empty"):
            load_gateway_managed_agent_token(entry)

    def test_missing_agent_id(self, tmp_path):
        from ax_cli.gateway import load_gateway_managed_agent_token

        token_file = tmp_path / "token"
        token_file.write_text("axp_a_valid.token\n")
        entry = {"token_file": str(token_file)}
        with pytest.raises(ValueError, match="agent_id"):
            load_gateway_managed_agent_token(entry)


class TestFormatDaemonLogLine:
    """_format_daemon_log_line: prepends ISO timestamp to a log line."""

    def test_has_timestamp(self):
        from ax_cli.gateway import _format_daemon_log_line

        line = _format_daemon_log_line("test message")
        assert "test message" in line
        # Should start with an ISO date-like prefix
        assert line[0:4].isdigit()  # year
        assert "T" in line[:30]

    def test_empty_message(self):
        from ax_cli.gateway import _format_daemon_log_line

        line = _format_daemon_log_line("")
        # Still has timestamp
        assert len(line) > 0


class TestRecordAndLoadGatewayActivity:
    """record_gateway_activity / load_recent_gateway_activity round-trip."""

    def test_record_and_load(self, monkeypatch, tmp_path):
        from ax_cli.gateway import load_recent_gateway_activity, record_gateway_activity

        monkeypatch.setenv("AX_GATEWAY_DIR", str(tmp_path / "gw"))
        record_gateway_activity("test_event", agent_name="alpha")
        items = load_recent_gateway_activity(limit=10)
        assert len(items) == 1
        assert items[0]["event"] == "test_event"

    def test_agent_filter(self, monkeypatch, tmp_path):
        from ax_cli.gateway import load_recent_gateway_activity, record_gateway_activity

        monkeypatch.setenv("AX_GATEWAY_DIR", str(tmp_path / "gw"))
        entry_a = {"name": "alpha", "agent_id": "a1"}
        entry_b = {"name": "beta", "agent_id": "b1"}
        record_gateway_activity("ev1", entry=entry_a)
        record_gateway_activity("ev2", entry=entry_b)
        items = load_recent_gateway_activity(limit=10, agent_name="alpha")
        assert len(items) == 1
        assert items[0]["agent_name"] == "alpha"

    def test_phase_attached(self, monkeypatch, tmp_path):
        from ax_cli.gateway import load_recent_gateway_activity, record_gateway_activity

        monkeypatch.setenv("AX_GATEWAY_DIR", str(tmp_path / "gw"))
        record_gateway_activity("reply_sent")
        items = load_recent_gateway_activity(limit=10)
        assert items[0].get("phase") == "reply"


# ---------------------------------------------------------------------------
# Gateway runtime timeout error
# ---------------------------------------------------------------------------


class TestGatewayRuntimeTimeoutError:
    """GatewayRuntimeTimeoutError: structured timeout exception."""

    def test_basic(self):
        from ax_cli.gateway import GatewayRuntimeTimeoutError

        exc = GatewayRuntimeTimeoutError(30)
        assert exc.timeout_seconds == 30
        assert exc.runtime_type is None
        assert "30s" in str(exc)

    def test_with_runtime_type(self):
        from ax_cli.gateway import GatewayRuntimeTimeoutError

        exc = GatewayRuntimeTimeoutError(60, runtime_type="hermes")
        assert exc.runtime_type == "hermes"
        assert "hermes" in str(exc)


# ---------------------------------------------------------------------------
# ParseGatewayExecEvent
# ---------------------------------------------------------------------------


class TestParseGatewayExecEvent:
    """_parse_gateway_exec_event: extracts structured events from runtime output."""

    def test_valid_event(self):
        from ax_cli.gateway import GATEWAY_EVENT_PREFIX, _parse_gateway_exec_event

        payload = {"status": "working", "detail": "processing"}
        line = f"{GATEWAY_EVENT_PREFIX}{json.dumps(payload)}"
        result = _parse_gateway_exec_event(line)
        assert result is not None
        assert result["status"] == "working"

    def test_non_event_line(self):
        from ax_cli.gateway import _parse_gateway_exec_event

        assert _parse_gateway_exec_event("just regular output") is None

    def test_malformed_json(self):
        from ax_cli.gateway import GATEWAY_EVENT_PREFIX, _parse_gateway_exec_event

        assert _parse_gateway_exec_event(f"{GATEWAY_EVENT_PREFIX}not json") is None

    def test_empty_payload(self):
        from ax_cli.gateway import GATEWAY_EVENT_PREFIX, _parse_gateway_exec_event

        assert _parse_gateway_exec_event(f"{GATEWAY_EVENT_PREFIX}") is None


# ---------------------------------------------------------------------------
# Ollama model helpers
# ---------------------------------------------------------------------------


class TestOllamaModelRows:
    """_ollama_model_rows: parses Ollama /api/tags response."""

    def test_valid_models(self):
        from ax_cli.gateway import _ollama_model_rows

        payload = {
            "models": [
                {"name": "llama3:latest", "details": {"family": "llama", "parameter_size": "8B"}},
                {"name": "nomic-embed-text", "details": {"family": "nomic", "families": ["bert"]}},
            ]
        }
        rows = _ollama_model_rows(payload)
        assert len(rows) == 2
        assert rows[0]["name"] == "llama3:latest"
        assert rows[0]["is_embedding"] is False
        assert rows[1]["is_embedding"] is True

    def test_empty_models(self):
        from ax_cli.gateway import _ollama_model_rows

        assert _ollama_model_rows({}) == []
        assert _ollama_model_rows({"models": []}) == []

    def test_skips_invalid_items(self):
        from ax_cli.gateway import _ollama_model_rows

        payload = {"models": ["not a dict", None, {"name": "valid"}]}
        rows = _ollama_model_rows(payload)
        assert len(rows) == 1

    def test_cloud_detection(self):
        from ax_cli.gateway import _ollama_model_rows

        payload = {"models": [{"name": "gpt4:cloud", "remote_host": "api.openai.com"}]}
        rows = _ollama_model_rows(payload)
        assert rows[0]["is_cloud"] is True


class TestRecommendedOllamaModel:
    """_recommended_ollama_model: picks the best local non-embedding model."""

    def test_prefers_local_chat(self):
        from ax_cli.gateway import _recommended_ollama_model

        rows = [
            {"name": "llama3:latest", "is_cloud": False, "is_embedding": False, "modified_at": "2026-01-01"},
            {"name": "embed-text", "is_cloud": False, "is_embedding": True, "modified_at": "2026-05-01"},
        ]
        assert _recommended_ollama_model(rows) == "llama3:latest"

    def test_empty_returns_none(self):
        from ax_cli.gateway import _recommended_ollama_model

        assert _recommended_ollama_model([]) is None


# ---------------------------------------------------------------------------
# Hermes setup status
# ---------------------------------------------------------------------------


class TestHermesSetupStatus:
    """hermes_setup_status: checks if hermes checkout exists."""

    def test_non_hermes_template(self):
        from ax_cli.gateway import hermes_setup_status

        result = hermes_setup_status({"template_id": "echo_test"})
        assert result["ready"] is True

    def test_hermes_plugin_runtime(self):
        from ax_cli.gateway import hermes_setup_status

        result = hermes_setup_status({"template_id": "hermes", "runtime_type": "hermes_plugin"})
        assert result["ready"] is True

    def test_hermes_not_found(self, monkeypatch, tmp_path):
        from ax_cli.gateway import hermes_setup_status

        monkeypatch.delenv("HERMES_REPO_PATH", raising=False)
        # Force candidates to only contain paths that definitely don't exist.
        fake_path = tmp_path / "definitely-not-here" / "hermes-agent"
        monkeypatch.setattr(
            "ax_cli.gateway._hermes_repo_candidates",
            lambda entry=None: [fake_path],
        )
        entry = {
            "template_id": "hermes",
            "runtime_type": "hermes_sentinel",
        }
        result = hermes_setup_status(entry)
        assert result["ready"] is False
        assert "not found" in result["summary"].lower()


# ---------------------------------------------------------------------------
# HideAfterStaleSeconds
# ---------------------------------------------------------------------------


class TestHideAfterStaleSeconds:
    """_hide_after_stale_seconds: resolves threshold from env > registry > default."""

    def test_default(self, monkeypatch):
        from ax_cli.gateway import RUNTIME_HIDDEN_AFTER_SECONDS, _hide_after_stale_seconds

        monkeypatch.delenv("AX_GATEWAY_HIDE_AFTER_STALE_SECONDS", raising=False)
        assert _hide_after_stale_seconds() == RUNTIME_HIDDEN_AFTER_SECONDS

    def test_env_override(self, monkeypatch):
        from ax_cli.gateway import _hide_after_stale_seconds

        monkeypatch.setenv("AX_GATEWAY_HIDE_AFTER_STALE_SECONDS", "300")
        assert _hide_after_stale_seconds() == 300.0

    def test_registry_override(self, monkeypatch):
        from ax_cli.gateway import _hide_after_stale_seconds

        monkeypatch.delenv("AX_GATEWAY_HIDE_AFTER_STALE_SECONDS", raising=False)
        registry = {"gateway": {"hide_after_stale_seconds": 600}}
        assert _hide_after_stale_seconds(registry) == 600.0


# ---------------------------------------------------------------------------
# ax_cli.commands.gateway — command-layer helpers
# ---------------------------------------------------------------------------


class TestRegistryRefForAgent:
    """_registry_ref_for_agent: returns #N ref for an agent in the registry."""

    def test_by_identity(self):
        from ax_cli.commands.gateway import _registry_ref_for_agent

        agents = [{"name": "alpha"}, {"name": "beta"}]
        registry = {"agents": agents}
        assert _registry_ref_for_agent(registry, agents[0]) == "#1"
        assert _registry_ref_for_agent(registry, agents[1]) == "#2"

    def test_by_name(self):
        from ax_cli.commands.gateway import _registry_ref_for_agent

        agents = [{"name": "alpha"}, {"name": "beta"}]
        registry = {"agents": agents}
        assert _registry_ref_for_agent(registry, {"name": "beta"}) == "#2"

    def test_by_install_id(self):
        from ax_cli.commands.gateway import _registry_ref_for_agent

        iid = str(uuid.uuid4())
        agents = [{"name": "alpha", "install_id": iid}]
        registry = {"agents": agents}
        assert _registry_ref_for_agent(registry, {"install_id": iid}) == "#1"

    def test_not_found(self):
        from ax_cli.commands.gateway import _registry_ref_for_agent

        registry = {"agents": [{"name": "alpha"}]}
        assert _registry_ref_for_agent(registry, {"name": "gamma"}) is None


class TestWithRegistryRefs:
    """_with_registry_refs: annotates an agent dict with registry ref + code."""

    def test_adds_ref(self):
        from ax_cli.commands.gateway import _with_registry_refs

        agents = [{"name": "alpha", "install_id": "abc12345-6789"}]
        registry = {"agents": agents}
        result = _with_registry_refs(registry, agents[0])
        assert result["registry_ref"] == "#1"
        assert result["registry_index"] == 1
        assert result["registry_code"] == "abc12345"

    def test_no_ref(self):
        from ax_cli.commands.gateway import _with_registry_refs

        registry = {"agents": [{"name": "alpha"}]}
        result = _with_registry_refs(registry, {"name": "unknown"})
        assert "registry_ref" not in result


class TestLocalFingerprintVerification:
    """_local_fingerprint_verification: best-effort OS cross-check."""

    def test_missing_pid(self):
        from ax_cli.commands.gateway import _local_fingerprint_verification

        result = _local_fingerprint_verification({})
        assert result["status"] == "unverified"
        assert result["reason"] == "missing_pid"

    def test_invalid_pid(self):
        from ax_cli.commands.gateway import _local_fingerprint_verification

        result = _local_fingerprint_verification({"pid": "not-a-number"})
        assert result["status"] == "unverified"

    def test_procfs_unavailable(self):
        from ax_cli.commands.gateway import _local_fingerprint_verification

        # On macOS, /proc doesn't exist
        result = _local_fingerprint_verification({"pid": "99999"})
        assert result["status"] == "unavailable"


class TestFindLocalOriginCollision:
    """_find_local_origin_collision: detects duplicate agent registrations."""

    def test_no_collision(self):
        from ax_cli.commands.gateway import _find_local_origin_collision

        registry = {
            "agents": [
                {"name": "alpha", "local_fingerprint": {"exe_path": "/usr/bin/a", "cwd": "/home/a", "user": "u"}}
            ]
        }
        fingerprint = {"exe_path": "/usr/bin/b", "cwd": "/home/b", "user": "u"}
        result = _find_local_origin_collision(registry, fingerprint=fingerprint, requested_name="beta")
        assert result is None

    def test_collision_found(self):
        from ax_cli.commands.gateway import _find_local_origin_collision

        fp = {"exe_path": "/usr/bin/a", "cwd": "/home/a", "user": "u"}
        registry = {"agents": [{"name": "alpha", "local_fingerprint": fp}]}
        result = _find_local_origin_collision(registry, fingerprint=fp, requested_name="beta")
        assert result is not None
        assert result["name"] == "alpha"

    def test_same_name_no_collision(self):
        from ax_cli.commands.gateway import _find_local_origin_collision

        fp = {"exe_path": "/usr/bin/a", "cwd": "/home/a", "user": "u"}
        registry = {"agents": [{"name": "alpha", "local_fingerprint": fp}]}
        result = _find_local_origin_collision(registry, fingerprint=fp, requested_name="alpha")
        assert result is None


class TestNormalizeTimeoutSeconds:
    """_normalize_timeout_seconds: validates and normalizes timeout values."""

    def test_valid(self):
        from ax_cli.commands.gateway import _normalize_timeout_seconds

        assert _normalize_timeout_seconds(30) == 30

    def test_none(self):
        from ax_cli.commands.gateway import _normalize_timeout_seconds

        assert _normalize_timeout_seconds(None) is None

    def test_zero_raises(self):
        from ax_cli.commands.gateway import _normalize_timeout_seconds

        with pytest.raises(ValueError, match="at least 1"):
            _normalize_timeout_seconds(0)

    def test_negative_raises(self):
        from ax_cli.commands.gateway import _normalize_timeout_seconds

        with pytest.raises(ValueError, match="at least 1"):
            _normalize_timeout_seconds(-5)


class TestAgentRowSpaceIds:
    """_agent_row_space_ids: collects all distinct space IDs from agent rows."""

    def test_collects_ids(self):
        from ax_cli.commands.gateway import _agent_row_space_ids

        registry = {
            "agents": [
                {"name": "a", "space_id": "s1"},
                {"name": "b", "space_id": "s2"},
                {"name": "c", "space_id": "s1"},
            ]
        }
        result = _agent_row_space_ids(registry)
        assert result == {"s1", "s2"}

    def test_empty(self):
        from ax_cli.commands.gateway import _agent_row_space_ids

        assert _agent_row_space_ids({"agents": []}) == set()


class TestSpaceListFromResponse:
    """_space_list_from_response: normalizes backend space list response."""

    def test_dict_with_spaces_key(self):
        from ax_cli.commands.gateway import _space_list_from_response

        raw = {"spaces": [{"id": "s1", "name": "S1"}]}
        result = _space_list_from_response(raw)
        assert len(result) == 1

    def test_raw_list(self):
        from ax_cli.commands.gateway import _space_list_from_response

        raw = [{"id": "s1", "name": "S1"}]
        result = _space_list_from_response(raw)
        assert len(result) == 1

    def test_skips_non_dict(self):
        from ax_cli.commands.gateway import _space_list_from_response

        raw = {"spaces": [{"id": "s1"}, "bad", 123]}
        result = _space_list_from_response(raw)
        assert len(result) == 1


class TestGatewaySessionChallengeEnabled:
    """_gateway_session_challenge_enabled: reads opt-in env var."""

    def test_not_set(self, monkeypatch):
        from ax_cli.commands.gateway import _gateway_session_challenge_enabled

        monkeypatch.delenv("AX_GATEWAY_SESSION_CHALLENGE", raising=False)
        assert _gateway_session_challenge_enabled() is False

    def test_truthy(self, monkeypatch):
        from ax_cli.commands.gateway import _gateway_session_challenge_enabled

        for val in ("1", "true", "yes", "on"):
            monkeypatch.setenv("AX_GATEWAY_SESSION_CHALLENGE", val)
            assert _gateway_session_challenge_enabled() is True

    def test_falsy(self, monkeypatch):
        from ax_cli.commands.gateway import _gateway_session_challenge_enabled

        for val in ("0", "false", "no", "off", ""):
            monkeypatch.setenv("AX_GATEWAY_SESSION_CHALLENGE", val)
            assert _gateway_session_challenge_enabled() is False


class TestFindLocalSessionRecord:
    """_find_local_session_record: looks up session by ID in registry."""

    def test_found(self):
        from ax_cli.commands.gateway import _find_local_session_record

        registry = {"local_sessions": [{"session_id": "sess-1", "status": "active"}]}
        result = _find_local_session_record(registry, "sess-1")
        assert result is not None
        assert result["session_id"] == "sess-1"

    def test_not_found(self):
        from ax_cli.commands.gateway import _find_local_session_record

        registry = {"local_sessions": []}
        assert _find_local_session_record(registry, "sess-1") is None

    def test_empty_id(self):
        from ax_cli.commands.gateway import _find_local_session_record

        registry = {"local_sessions": [{"session_id": "sess-1"}]}
        assert _find_local_session_record(registry, "") is None


class TestSpaceCacheWith:
    """_space_cache_with: ensures a space_id is present in cache rows."""

    def test_adds_missing(self):
        from ax_cli.commands.gateway import _space_cache_with

        rows = _space_cache_with([], "s1", name="Space One")
        assert len(rows) == 1
        assert rows[0]["space_id"] == "s1"
        assert rows[0]["name"] == "Space One"
        assert rows[0]["is_default"] is True

    def test_preserves_existing(self):
        from ax_cli.commands.gateway import _space_cache_with

        existing = [{"space_id": "s1", "name": "S1", "is_default": True}]
        rows = _space_cache_with(existing, "s2", name="S2")
        assert len(rows) == 2

    def test_deduplicates(self):
        from ax_cli.commands.gateway import _space_cache_with

        existing = [{"space_id": "s1", "name": "S1"}]
        rows = _space_cache_with(existing, "s1")
        assert len(rows) == 1


class TestGatewayTestSenderName:
    """_gateway_test_sender_name: derives the per-space switchboard name."""

    def test_format(self):
        from ax_cli.commands.gateway import _gateway_test_sender_name

        result = _gateway_test_sender_name("12345678-1234-1234-1234-123456789abc")
        assert result.startswith("switchboard-")
        assert len(result) > len("switchboard-")

    def test_short_space_id(self):
        from ax_cli.commands.gateway import _gateway_test_sender_name

        result = _gateway_test_sender_name("abc")
        assert result.startswith("switchboard-")


# ---------------------------------------------------------------------------
# File SHA256
# ---------------------------------------------------------------------------


class TestFileSha256:
    """_file_sha256: computes sha256 hash of a file."""

    def test_hash(self, tmp_path):
        from ax_cli.gateway import _file_sha256

        f = tmp_path / "test.txt"
        f.write_text("hello world")
        result = _file_sha256(f)
        assert result.startswith("sha256:")
        assert len(result) > 10

    def test_deterministic(self, tmp_path):
        from ax_cli.gateway import _file_sha256

        f = tmp_path / "test.txt"
        f.write_text("hello world")
        assert _file_sha256(f) == _file_sha256(f)


class TestSafeFileSha256:
    """_safe_file_sha256: returns None on errors instead of raising."""

    def test_existing_file(self, tmp_path):
        from ax_cli.gateway import _safe_file_sha256

        f = tmp_path / "test.txt"
        f.write_text("hello")
        result = _safe_file_sha256(f)
        assert result is not None
        assert result.startswith("sha256:")

    def test_nonexistent_file(self, tmp_path):
        from ax_cli.gateway import _safe_file_sha256

        assert _safe_file_sha256(tmp_path / "nonexistent") is None

    def test_none_path(self):
        from ax_cli.gateway import _safe_file_sha256

        assert _safe_file_sha256(None) is None


class TestHostFingerprint:
    """_host_fingerprint: returns a hashed host identifier."""

    def test_format(self):
        from ax_cli.gateway import _host_fingerprint

        result = _host_fingerprint()
        assert result.startswith("host:")
        assert len(result) > 5

    def test_deterministic(self):
        from ax_cli.gateway import _host_fingerprint

        assert _host_fingerprint() == _host_fingerprint()


class TestExternalRuntimeConnected:
    """_external_runtime_connected: checks if external runtime has a fresh heartbeat."""

    def test_connected(self):
        from ax_cli.gateway import _external_runtime_connected

        snapshot = {"external_runtime_state": "connected"}
        assert _external_runtime_connected(snapshot, last_seen_age=5) is True

    def test_stale(self):
        from ax_cli.gateway import RUNTIME_STALE_AFTER_SECONDS, _external_runtime_connected

        snapshot = {"external_runtime_state": "connected"}
        assert _external_runtime_connected(snapshot, last_seen_age=int(RUNTIME_STALE_AFTER_SECONDS) + 100) is False

    def test_disconnected_state(self):
        from ax_cli.gateway import _external_runtime_connected

        snapshot = {"external_runtime_state": "disconnected"}
        assert _external_runtime_connected(snapshot, last_seen_age=5) is False


class TestExternalRuntimeExpected:
    """_external_runtime_expected: detects externally-managed runtimes."""

    def test_external_managed_flag(self):
        from ax_cli.gateway import _external_runtime_expected

        assert _external_runtime_expected({"external_runtime_managed": True}) is True

    def test_external_kind(self):
        from ax_cli.gateway import _external_runtime_expected

        assert _external_runtime_expected({"external_runtime_kind": "hermes_adapter"}) is True

    def test_external_instance_id(self):
        from ax_cli.gateway import _external_runtime_expected

        assert _external_runtime_expected({"external_runtime_instance_id": "inst-1"}) is True

    def test_not_expected(self):
        from ax_cli.gateway import _external_runtime_expected

        assert _external_runtime_expected({}) is False


class TestGatewayPickupActivity:
    """_gateway_pickup_activity: generates pickup activity description."""

    def test_passive_runtime(self):
        from ax_cli.gateway import _gateway_pickup_activity

        assert "Queued" in _gateway_pickup_activity("inbox", 0)

    def test_passive_with_backlog(self):
        from ax_cli.gateway import _gateway_pickup_activity

        result = _gateway_pickup_activity("inbox", 3)
        assert "3 pending" in result

    def test_active_runtime(self):
        from ax_cli.gateway import _gateway_pickup_activity

        assert "Picked up" in _gateway_pickup_activity("echo", 0)

    def test_active_with_backlog(self):
        from ax_cli.gateway import _gateway_pickup_activity

        result = _gateway_pickup_activity("echo", 2)
        assert "2 pending" in result


# ---------------------------------------------------------------------------
# Sentinel / Hermes plugin classification helpers
# ---------------------------------------------------------------------------


class TestIsSentinelCliRuntime:
    """_is_sentinel_cli_runtime: matches sentinel/claude/codex CLI runtimes."""

    def test_sentinel_cli(self):
        from ax_cli.gateway import _is_sentinel_cli_runtime

        assert _is_sentinel_cli_runtime("sentinel_cli") is True
        assert _is_sentinel_cli_runtime("claude_cli") is True
        assert _is_sentinel_cli_runtime("codex_cli") is True
        assert _is_sentinel_cli_runtime("echo") is False


class TestIsHermesSentinelRuntime:
    def test_matches(self):
        from ax_cli.gateway import _is_hermes_sentinel_runtime

        assert _is_hermes_sentinel_runtime("hermes_sentinel") is True
        assert _is_hermes_sentinel_runtime("hermes_sdk") is True
        assert _is_hermes_sentinel_runtime("hermes_plugin") is False


class TestIsHermesPluginRuntime:
    def test_matches(self):
        from ax_cli.gateway import _is_hermes_plugin_runtime

        assert _is_hermes_plugin_runtime("hermes_plugin") is True
        assert _is_hermes_plugin_runtime("hermes_sentinel") is False


# ---------------------------------------------------------------------------
# Unique classes migrated from test_gateway_coverage.py
# ---------------------------------------------------------------------------


class TestAttachedSessionLogIsReady:
    def test_no_path(self):
        assert gw._attached_session_log_is_ready(None) is False
        assert gw._attached_session_log_is_ready("") is False

    def test_nonexistent_path(self):
        assert gw._attached_session_log_is_ready("/tmp/nonexistent_log_12345.txt") is False

    def test_with_listening_marker(self, tmp_path):
        log = tmp_path / "test.log"
        log.write_text("Starting up\nListening for channel messages\nReady")
        assert gw._attached_session_log_is_ready(str(log)) is True

    def test_with_ax_channel_marker(self, tmp_path):
        log = tmp_path / "test.log"
        log.write_text("Starting up\nConnected to ax-channel\nReady")
        assert gw._attached_session_log_is_ready(str(log)) is True

    def test_without_marker(self, tmp_path):
        log = tmp_path / "test.log"
        log.write_text("Starting up\nLoading model\n")
        assert gw._attached_session_log_is_ready(str(log)) is False


class TestB64UrlEncoding:
    def test_roundtrip(self):
        original = b"hello world! This is a test."
        encoded = gw._b64url_encode(original)
        decoded = gw._b64url_decode(encoded)
        assert decoded == original

    def test_no_padding_in_encoded(self):
        encoded = gw._b64url_encode(b"test")
        assert "=" not in encoded


class TestComposeAgentSystemPrompt:
    def test_with_operator_prompt(self):
        entry = {"system_prompt": "You are a helpful bot.", "name": "test", "base_url": "https://paxai.app"}
        result = gw._compose_agent_system_prompt(entry)
        assert "You are a helpful bot" in result
        assert "aX environment context" in result

    def test_skip_environment(self):
        entry = {
            "system_prompt": "Just the prompt.",
            "system_prompt_skip_environment": "true",
        }
        result = gw._compose_agent_system_prompt(entry)
        assert result == "Just the prompt."
        assert "aX environment context" not in result

    def test_no_prompt_still_has_environment(self):
        entry = {"name": "test"}
        result = gw._compose_agent_system_prompt(entry)
        assert result is not None
        assert "aX environment context" in result


class TestEntryRequiresOperatorApproval:
    def test_pass_through_requires(self):
        assert gw._entry_requires_operator_approval({"template_id": "pass_through"}) is True

    def test_explicit_flag(self):
        assert gw._entry_requires_operator_approval({"requires_approval": True}) is True

    def test_echo_does_not_require(self):
        assert gw._entry_requires_operator_approval({"template_id": "echo_test"}) is False


class TestGenerateSessionChallengeCode:
    def test_returns_string(self):
        code = gw_cmd._generate_session_challenge_code()
        assert isinstance(code, str)
        assert len(code) > 0

    def test_uppercase(self):
        code = gw_cmd._generate_session_challenge_code()
        assert code == code.upper()

    def test_unique(self):
        codes = {gw_cmd._generate_session_challenge_code() for _ in range(10)}
        assert len(codes) > 1


class TestHermesPluginHome:
    def test_explicit_home(self):
        result = gw._hermes_plugin_home({"hermes_home": "/custom/hermes"})
        assert str(result) == "/custom/hermes"

    def test_default_under_workdir(self):
        result = gw._hermes_plugin_home({"workdir": "/agent/work"})
        assert str(result) == "/agent/work/.hermes"


class TestHermesPluginWorkdir:
    def test_explicit_workdir(self):
        result = gw._hermes_plugin_workdir({"workdir": "/custom/path"})
        assert str(result) == "/custom/path"

    def test_default_workdir(self):
        result = gw._hermes_plugin_workdir({"name": "test-agent"})
        assert "test-agent" in str(result)


class TestHermesRepoCandidates:
    def test_with_entry_path(self):
        candidates = gw._hermes_repo_candidates({"hermes_repo_path": "/custom/hermes"})
        assert Path("/custom/hermes") in candidates

    def test_with_env_var(self, monkeypatch):
        monkeypatch.setenv("HERMES_REPO_PATH", "/env/hermes")
        candidates = gw._hermes_repo_candidates({})
        assert Path("/env/hermes") in candidates

    def test_deduplicates(self):
        candidates = gw._hermes_repo_candidates({"hermes_repo_path": str(Path.home() / "hermes-agent")})
        paths = [str(c) for c in candidates]
        assert len(paths) == len(set(paths))

    def test_home_fallback_always_included(self, monkeypatch):
        monkeypatch.delenv("HERMES_REPO_PATH", raising=False)
        candidates = gw._hermes_repo_candidates({})
        assert Path.home() / "hermes-agent" in candidates


class TestHermesSentinelModel:
    def test_hermes_model_field(self):
        assert gw._hermes_sentinel_model({"hermes_model": "codex:gpt-4"}) == "codex:gpt-4"

    def test_sentinel_model_field(self):
        assert gw._hermes_sentinel_model({"sentinel_model": "my-model"}) == "my-model"

    def test_runtime_model(self):
        assert gw._hermes_sentinel_model({"runtime_model": "rt-model"}) == "rt-model"

    def test_default_from_env(self, monkeypatch):
        monkeypatch.delenv("AX_GATEWAY_HERMES_MODEL", raising=False)
        result = gw._hermes_sentinel_model({})
        assert result


class TestLoadGatewaySessionOrExit:
    def test_exits_when_no_session(self, monkeypatch, tmp_path):
        import typer

        monkeypatch.setenv("AX_CONFIG_DIR", str(tmp_path / "config"))
        with pytest.raises(typer.Exit):
            gw_cmd._load_gateway_session_or_exit()

    def test_returns_session_when_exists(self, monkeypatch, tmp_path):
        monkeypatch.setenv("AX_CONFIG_DIR", str(tmp_path / "config"))
        gw.save_gateway_session({"token": "axp_u_test", "base_url": "https://paxai.app"})
        session = gw_cmd._load_gateway_session_or_exit()
        assert session["token"] == "axp_u_test"


class TestLoadGatewayUserClient:
    def test_no_session_exits(self, monkeypatch, tmp_path):
        import typer

        monkeypatch.setenv("AX_CONFIG_DIR", str(tmp_path / "config"))
        with pytest.raises(typer.Exit):
            gw_cmd._load_gateway_user_client()

    def test_missing_token_exits(self, monkeypatch, tmp_path):
        import typer

        monkeypatch.setenv("AX_CONFIG_DIR", str(tmp_path / "config"))
        gw.save_gateway_session({"base_url": "https://paxai.app"})
        with pytest.raises(typer.Exit):
            gw_cmd._load_gateway_user_client()

    def test_non_user_token_exits(self, monkeypatch, tmp_path):
        import typer

        monkeypatch.setenv("AX_CONFIG_DIR", str(tmp_path / "config"))
        gw.save_gateway_session({"token": "axp_a_agent.token", "base_url": "https://paxai.app"})
        with pytest.raises(typer.Exit):
            gw_cmd._load_gateway_user_client()

    def test_valid_session_returns_client(self, monkeypatch, tmp_path):
        monkeypatch.setenv("AX_CONFIG_DIR", str(tmp_path / "config"))
        gw.save_gateway_session({"token": "axp_u_test.token", "base_url": "https://paxai.app"})
        client = gw_cmd._load_gateway_user_client()
        assert client is not None
        client.close()


class TestGatewaySessionStalenessWarning:
    """Regression for #74: warn when the gateway session predates the user
    login PAT. Operator signal that a PAT rotation (or `ax login` against a
    different env) has left the gateway session stale before the next
    /auth/exchange 401 lands as a raw traceback (#73)."""

    def _make_user_toml(self, monkeypatch, tmp_path, *, mtime: float) -> None:
        monkeypatch.setenv("AX_CONFIG_DIR", str(tmp_path / "config"))
        from ax_cli.config import _user_config_path

        user_p = _user_config_path()
        user_p.parent.mkdir(parents=True, exist_ok=True)
        user_p.write_text('token = "axp_u_user.token"\nbase_url = "https://paxai.app"\n')
        os.utime(user_p, (mtime, mtime))

    def _make_session(self, *, mtime: float) -> None:
        gw.save_gateway_session({"token": "axp_u_session.token", "base_url": "https://paxai.app"})
        os.utime(gw.session_path(), (mtime, mtime))

    def test_warns_when_session_older_than_user_toml(self, monkeypatch, tmp_path, capsys):
        # session.json is older than user.toml — classic "rotated PAT, ran
        # `ax login`, forgot `ax gateway login`" shape.
        self._make_user_toml(monkeypatch, tmp_path, mtime=2_000_000.0)
        self._make_session(mtime=1_000_000.0)
        gw_cmd._load_gateway_user_client()
        stderr = capsys.readouterr().err
        assert "gateway session is older than your user login" in stderr
        assert "ax gateway login" in stderr

    def test_no_warning_when_session_newer_than_user_toml(self, monkeypatch, tmp_path, capsys):
        # Fresh `ax gateway login` after `ax login` — no warning.
        self._make_user_toml(monkeypatch, tmp_path, mtime=1_000_000.0)
        self._make_session(mtime=2_000_000.0)
        gw_cmd._load_gateway_user_client()
        stderr = capsys.readouterr().err
        assert "older than your user login" not in stderr

    def test_no_warning_when_user_toml_missing(self, monkeypatch, tmp_path, capsys):
        # Named env / never logged in to user.toml — silently skip, never
        # crash the gateway command itself.
        monkeypatch.setenv("AX_CONFIG_DIR", str(tmp_path / "config"))
        self._make_session(mtime=1_000_000.0)
        gw_cmd._load_gateway_user_client()
        stderr = capsys.readouterr().err
        assert "older than your user login" not in stderr


class TestLocalOriginSignature:
    def test_excludes_agent_name(self):
        fp = {"exe_path": "/usr/bin/python3", "cwd": "/home/user", "user": "testuser"}
        sig1 = gw_cmd._local_origin_signature(fp)
        assert sig1.startswith("sha256:")


class TestLocalProcessFingerprint:
    def test_returns_expected_keys(self, monkeypatch):
        monkeypatch.setattr(gw, "_file_sha256", lambda p: "sha256:abc")
        fp = gw_cmd._local_process_fingerprint(agent_name="test-agent")
        assert fp["agent_name"] == "test-agent"
        assert "pid" in fp
        assert "cwd" in fp
        assert "exe_path" in fp
        assert "user" in fp
        assert "platform" in fp


class TestLocalTrustSignature:
    def test_deterministic(self):
        fp = {"exe_path": "/usr/bin/python3", "cwd": "/home/user", "user": "testuser"}
        sig1 = gw_cmd._local_trust_signature("agent", fp)
        sig2 = gw_cmd._local_trust_signature("agent", fp)
        assert sig1 == sig2
        assert sig1.startswith("sha256:")


class TestPidIsAlive:
    def test_zero_pid(self):
        assert gw._pid_is_alive(0) is False

    def test_none_pid(self):
        assert gw._pid_is_alive(None) is False

    def test_negative_pid(self):
        assert gw._pid_is_alive(-1) is False

    def test_non_numeric(self):
        assert gw._pid_is_alive("abc") is False

    def test_current_pid_is_alive(self):
        assert gw._pid_is_alive(os.getpid()) is True


class TestSentinelModel:
    def test_model_field(self):
        assert gw._sentinel_model({"model": "gpt-4"}, "claude") == "gpt-4"

    def test_runtime_specific_field(self):
        assert gw._sentinel_model({"claude_model": "claude-3"}, "claude") == "claude-3"

    def test_none_when_unset(self):
        assert gw._sentinel_model({}, "claude") is None


class TestSentinelRuntimeName:
    def test_default_claude(self):
        assert gw._sentinel_runtime_name({}) == "claude"

    def test_codex_cli_runtime_type(self):
        assert gw._sentinel_runtime_name({"runtime_type": "codex_cli"}) == "codex"

    def test_configured_codex(self):
        assert gw._sentinel_runtime_name({"sentinel_runtime": "codex_cli"}) == "codex"

    def test_configured_claude(self):
        assert gw._sentinel_runtime_name({"sentinel_runtime": "claude_cli"}) == "claude"


class TestSentinelSessionKey:
    def test_agent_scope(self):
        entry = {"space_id": "s1", "name": "bot"}
        key = gw._sentinel_session_key(entry, None, "msg-1")
        assert "s1" in key and "bot" in key

    def test_message_scope(self):
        entry = {"sentinel_session_scope": "message"}
        key = gw._sentinel_session_key(entry, None, "msg-42")
        assert key == "msg-42"

    def test_thread_scope_with_parent(self):
        entry = {"sentinel_session_scope": "thread"}
        data = {"parent_id": "thread-1"}
        key = gw._sentinel_session_key(entry, data, "msg-42")
        assert key == "thread-1"

    def test_thread_scope_no_parent(self):
        entry = {"sentinel_session_scope": "thread"}
        key = gw._sentinel_session_key(entry, {}, "msg-42")
        assert key == "msg-42"


class TestSentinelSessionScope:
    def test_default_agent(self):
        assert gw._sentinel_session_scope({}) == "agent"

    def test_thread_scope(self):
        assert gw._sentinel_session_scope({"sentinel_session_scope": "thread"}) == "thread"

    def test_message_scope(self):
        assert gw._sentinel_session_scope({"session_scope": "message"}) == "message"

    def test_invalid_scope_defaults_to_agent(self):
        assert gw._sentinel_session_scope({"sentinel_session_scope": "invalid"}) == "agent"


class TestSentinelToolSummary:
    def test_read_file(self):
        assert "Reading" in gw._sentinel_tool_summary("read", {"file_path": "/tmp/test.py"})

    def test_write_file(self):
        assert "Writing" in gw._sentinel_tool_summary("write", {"file_path": "/tmp/out.py"})

    def test_edit_file(self):
        assert "Editing" in gw._sentinel_tool_summary("edit", {"file_path": "/tmp/fix.py"})

    def test_bash(self):
        assert "Running" in gw._sentinel_tool_summary("bash", {"command": "ls -la"})

    def test_grep(self):
        assert "Searching" in gw._sentinel_tool_summary("grep", {"pattern": "TODO"})

    def test_glob(self):
        assert "Finding" in gw._sentinel_tool_summary("glob", {"pattern": "*.py"})

    def test_unknown_tool(self):
        assert "Using my_tool" in gw._sentinel_tool_summary("my_tool", {})

    def test_read_no_path(self):
        assert "Reading file" in gw._sentinel_tool_summary("read", {})

    def test_bash_no_command(self):
        assert "Running command" in gw._sentinel_tool_summary("bash", {})


class TestSpaceNameFromCache:
    def test_found(self):
        spaces = [{"space_id": "s1", "name": "My Space"}]
        assert gw._space_name_from_cache(spaces, "s1") == "My Space"

    def test_not_found(self):
        spaces = [{"space_id": "s1", "name": "My Space"}]
        assert gw._space_name_from_cache(spaces, "s2") is None

    def test_empty_space_id(self):
        assert gw._space_name_from_cache([], None) is None
        assert gw._space_name_from_cache([], "") is None


class TestSummarizeSentinelCommand:
    def test_apply_patch(self):
        assert "Applying patch" in gw._summarize_sentinel_command("apply_patch file.diff")

    def test_grep_command(self):
        assert "Searching" in gw._summarize_sentinel_command("rg pattern src/")

    def test_cat_command(self):
        assert "Reading" in gw._summarize_sentinel_command("cat /tmp/file.txt")

    def test_pytest_command(self):
        assert "Running tests" in gw._summarize_sentinel_command("pytest tests/")

    def test_uv_run(self):
        assert "Running tests" in gw._summarize_sentinel_command("uv run pytest")

    def test_generic_command(self):
        result = gw._summarize_sentinel_command("echo hello")
        assert "Running:" in result

    def test_long_command_truncated(self):
        long_cmd = "echo " + "a" * 200
        result = gw._summarize_sentinel_command(long_cmd)
        assert result.endswith("...")


class TestUpstreamRateLimitedError:
    def test_basic(self):
        import httpx

        request = httpx.Request("GET", "https://paxai.app/api/v1/spaces")
        response = httpx.Response(429, request=request, headers={"retry-after": "30"})
        exc = httpx.HTTPStatusError("429", request=request, response=response)
        rate_err = gw_cmd.UpstreamRateLimitedError(exc, retries_attempted=3)
        assert rate_err.retries_attempted == 3
        assert rate_err.retry_after_seconds == 30
        assert "3 retries" in str(rate_err)

    def test_no_retry_after_header(self):
        import httpx

        request = httpx.Request("GET", "https://paxai.app")
        response = httpx.Response(429, request=request)
        exc = httpx.HTTPStatusError("429", request=request, response=response)
        rate_err = gw_cmd.UpstreamRateLimitedError(exc, retries_attempted=2)
        assert rate_err.retry_after_seconds is None
