"""Local reminder policy runner.

This is intentionally a CLI-first dogfood loop. It stores reminder policy
state in a local JSON file, then emits Activity Stream reminder cards through
the existing ``ax alerts`` metadata contract when policies become due.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Optional

import httpx
import typer

from ..config import get_client, resolve_agent_name, resolve_space_id
from ..output import JSON_OPTION, console, print_json, print_table
from .alerts import (
    _build_alert_metadata,
    _fetch_task_snapshot,
    _format_mention_content,
    _normalize_severity,
    _resolve_target_from_task,
    _strip_at,
    _task_lifecycle,
    _validate_timestamp,
)

app = typer.Typer(name="reminders", help="Local task reminder policy runner", no_args_is_help=True)

STALE_AFTER_DAYS = 7


def _now() -> _dt.datetime:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0)


def _iso(value: _dt.datetime) -> str:
    return value.astimezone(_dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _parse_iso(value: str) -> _dt.datetime:
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = _dt.datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=_dt.timezone.utc)
    return parsed.astimezone(_dt.timezone.utc)


def _default_policy_file() -> Path:
    env_path = os.environ.get("AX_REMINDERS_FILE")
    if env_path:
        return Path(env_path).expanduser()

    cwd = Path.cwd()
    for parent in [cwd, *cwd.parents]:
        ax_dir = parent / ".ax"
        if ax_dir.is_dir():
            return ax_dir / "reminders.json"
    return Path.home() / ".ax" / "reminders.json"


def _policy_file(path: str | None) -> Path:
    return Path(path).expanduser() if path else _default_policy_file()


_LOOP_MODES = ("auto", "draft", "manual")
_DEFAULT_PRIORITY = 50


def _empty_store() -> dict[str, Any]:
    return {"version": 2, "policies": [], "drafts": []}


def _normalize_mode(value: str | None) -> str:
    text = (value or "auto").strip().lower()
    if text not in _LOOP_MODES:
        raise typer.BadParameter(f"--mode must be one of: {', '.join(_LOOP_MODES)}")
    return text


def _normalize_priority(value: int | None) -> int:
    if value is None:
        return _DEFAULT_PRIORITY
    if value < 0 or value > 100:
        raise typer.BadParameter("--priority must be between 0 and 100 (lower = higher priority)")
    return int(value)


def _short_draft_id() -> str:
    return f"draft-{uuid.uuid4().hex[:10]}"


def _load_store(path: Path) -> dict[str, Any]:
    if not path.exists():
        return _empty_store()
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        typer.echo(f"Error: reminder policy file is not valid JSON: {path} ({exc})", err=True)
        raise typer.Exit(1)
    if not isinstance(data, dict):
        typer.echo(f"Error: reminder policy file must contain a JSON object: {path}", err=True)
        raise typer.Exit(1)
    data.setdefault("version", 1)
    data.setdefault("policies", [])
    data.setdefault("drafts", [])
    if not isinstance(data["policies"], list):
        typer.echo(f"Error: reminders policies must be a list: {path}", err=True)
        raise typer.Exit(1)
    if not isinstance(data["drafts"], list):
        typer.echo(f"Error: reminders drafts must be a list: {path}", err=True)
        raise typer.Exit(1)
    return data


def _save_store(path: Path, store: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(store, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)
    path.chmod(0o600)


def _short_id() -> str:
    return f"rem-{uuid.uuid4().hex[:10]}"


def _find_policy(store: dict[str, Any], policy_id: str) -> dict[str, Any]:
    matches = [
        p for p in store.get("policies", []) if isinstance(p, dict) and str(p.get("id", "")).startswith(policy_id)
    ]
    if not matches:
        typer.echo(f"Error: reminder policy not found: {policy_id}", err=True)
        raise typer.Exit(1)
    if len(matches) > 1:
        typer.echo(f"Error: reminder policy id is ambiguous: {policy_id}", err=True)
        raise typer.Exit(1)
    return matches[0]


def _parse_optional_iso(value: Any) -> _dt.datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return _parse_iso(value)
    except Exception:
        return None


def _pause_until(first_at: str | None, minutes: int | None) -> str | None:
    if first_at:
        return _validate_timestamp(first_at, flag="--resume-at")
    if minutes is not None:
        if minutes < 1:
            raise typer.BadParameter("--minutes must be at least 1")
        return _iso(_now() + _dt.timedelta(minutes=minutes))
    return None


def _is_completed(policy: dict[str, Any]) -> bool:
    try:
        return int(policy.get("fired_count", 0)) >= int(policy.get("max_fires", 1))
    except Exception:
        return False


def _is_paused(policy: dict[str, Any], *, now: _dt.datetime | None = None) -> bool:
    if not policy.get("paused", False):
        return False
    resume_at = _parse_optional_iso(policy.get("resume_at") or policy.get("snooze_until"))
    if now is not None and resume_at and resume_at <= now:
        _resume_policy(policy, now=now, automatic=True)
        return False
    return True


def _resume_policy(policy: dict[str, Any], *, now: _dt.datetime, automatic: bool = False) -> None:
    policy["paused"] = False
    policy["paused_reason"] = None
    policy["resume_at"] = None
    policy["snooze_until"] = None
    policy["resumed_at"] = _iso(now)
    policy["resumed_by"] = "auto_resume_at" if automatic else "operator"
    policy["updated_at"] = _iso(now)


def _policy_state(policy: dict[str, Any], *, now: _dt.datetime) -> str:
    if _is_paused(policy, now=now):
        return "paused"
    if _is_completed(policy):
        return "completed"
    if not policy.get("enabled", True):
        return "disabled"
    next_fire = _parse_optional_iso(policy.get("next_fire_at"))
    if next_fire and next_fire < now - _dt.timedelta(days=STALE_AFTER_DAYS):
        return "stale"
    if next_fire and next_fire <= now:
        return "due"
    return "active"


def _policy_sort_key(policy: dict[str, Any], *, now: _dt.datetime | None = None) -> tuple[int, int, _dt.datetime, str]:
    state_rank = {"due": 0, "active": 1, "paused": 2, "stale": 3, "disabled": 4, "completed": 5}
    now = now or _now()
    state = _policy_state(policy, now=now)
    priority = int(policy.get("priority", _DEFAULT_PRIORITY))
    next_fire = _parse_optional_iso(policy.get("next_fire_at")) or _dt.datetime.max.replace(tzinfo=_dt.timezone.utc)
    if state == "paused":
        next_fire = _parse_optional_iso(policy.get("resume_at") or policy.get("snooze_until")) or next_fire
    return (state_rank.get(state, 99), priority, next_fire, str(policy.get("id") or ""))


def _grouped_policy_payload(store: dict[str, Any], *, now: _dt.datetime) -> dict[str, Any]:
    policies = [p for p in store.get("policies", []) if isinstance(p, dict)]
    ordered = sorted(policies, key=lambda p: _policy_sort_key(p, now=now))
    groups: dict[str, list[dict[str, Any]]] = {
        "due": [],
        "active": [],
        "paused": [],
        "disabled": [],
        "completed": [],
        "stale": [],
    }
    for policy in ordered:
        state = _policy_state(policy, now=now)
        view = dict(policy)
        view["state"] = state
        groups.setdefault(state, []).append(view)
    return {"policies": ordered, "groups": groups, "summary": {k: len(v) for k, v in groups.items()}}


def _policy_rows(store: dict[str, Any], *, now: _dt.datetime | None = None) -> list[dict[str, Any]]:
    rows = []
    now = now or _now()
    policies = [p for p in store.get("policies", []) if isinstance(p, dict)]
    for policy in sorted(policies, key=lambda p: _policy_sort_key(p, now=now)):
        state = _policy_state(policy, now=now)
        rows.append(
            {
                "id": policy.get("id", ""),
                "state": state,
                "priority": int(policy.get("priority", _DEFAULT_PRIORITY)),
                "mode": str(policy.get("mode", "auto")),
                "enabled": policy.get("enabled", True),
                "task": policy.get("source_task_id", ""),
                "target": policy.get("target") or "(task default)",
                "next_fire": policy.get("next_fire_at", ""),
                "resume_at": policy.get("resume_at") or policy.get("snooze_until") or "",
                "fires": f"{policy.get('fired_count', 0)}/{policy.get('max_fires', '-')}",
                "reason": policy.get("paused_reason") if state == "paused" else policy.get("reason", ""),
            }
        )
    return rows


@app.command("add")
def add(
    source_task: str = typer.Argument(..., help="Task ID to remind about"),
    reason: str = typer.Option("Please review this task.", "--reason", "-r", help="Reminder text"),
    target: Optional[str] = typer.Option(None, "--target", "-t", help="@agent/user; default resolves from task"),
    first_at: Optional[str] = typer.Option(None, "--first-at", help="First fire time, ISO-8601 UTC"),
    first_in: int = typer.Option(5, "--first-in-minutes", help="Minutes from now for first fire"),
    cadence: int = typer.Option(5, "--cadence-minutes", help="Minutes between recurring fires"),
    max_fires: int = typer.Option(1, "--max-fires", help="Maximum reminder fires before disabling"),
    severity: str = typer.Option("info", "--severity", "-s", help="info | warn | critical"),
    expected_response: Optional[str] = typer.Option(None, "--expected-response", help="What response is expected"),
    priority: int = typer.Option(_DEFAULT_PRIORITY, "--priority", help="Queue priority 0-100 (lower = higher)"),
    mode: str = typer.Option("auto", "--mode", help="auto | draft | manual"),
    space_id: Optional[str] = typer.Option(None, "--space-id", help="Override default space"),
    policy_file: Optional[str] = typer.Option(None, "--file", help="Reminder policy JSON file"),
    as_json: bool = JSON_OPTION,
) -> None:
    """Add a local reminder policy.

    The policy is local state. Use ``ax reminders run`` to fire due policies.
    Mode controls firing behavior:
      auto   — fire immediately when due (default)
      draft  — prepare draft, queue for HITL review via ``ax reminders drafts``
      manual — never auto-fire; only fired by explicit ``run --force``
    """
    if max_fires < 1:
        raise typer.BadParameter("--max-fires must be at least 1")
    if cadence < 1:
        raise typer.BadParameter("--cadence-minutes must be at least 1")
    if first_in < 0:
        raise typer.BadParameter("--first-in-minutes cannot be negative")
    normalized_priority = _normalize_priority(priority)
    normalized_mode = _normalize_mode(mode)

    first_at = _validate_timestamp(first_at, flag="--first-at")
    next_fire = _parse_iso(first_at) if first_at else _now() + _dt.timedelta(minutes=first_in)

    # Offline-first: if --space-id is provided, skip the network round-trip
    # entirely. Otherwise resolve via the configured client.
    if space_id:
        resolved_space = space_id
    else:
        try:
            client = get_client()
            resolved_space = resolve_space_id(client, explicit=None)
        except Exception as exc:
            typer.echo(
                f"Error: Space ID not resolvable: {exc}. Pass --space-id to add offline, "
                "or configure a default via `ax profile`.",
                err=True,
            )
            raise typer.Exit(2)

    path = _policy_file(policy_file)
    store = _load_store(path)
    policy = {
        "id": _short_id(),
        "enabled": True,
        "space_id": resolved_space,
        "source_task_id": source_task,
        "reason": reason,
        "target": _strip_at(target),
        "severity": _normalize_severity(severity),
        "expected_response": expected_response,
        "priority": normalized_priority,
        "mode": normalized_mode,
        "cadence_seconds": cadence * 60,
        "next_fire_at": _iso(next_fire),
        "max_fires": max_fires,
        "fired_count": 0,
        "fired_keys": [],
        "created_at": _iso(_now()),
        "updated_at": _iso(_now()),
    }
    store["policies"].append(policy)
    _save_store(path, store)

    if as_json:
        print_json({"policy": policy, "file": str(path)})
        return

    console.print(f"[bold cyan]Reminder policy added[/bold cyan] {policy['id']}")
    console.print(f"[bold]file[/bold]: {path}")
    console.print(f"[bold]next_fire_at[/bold]: {policy['next_fire_at']}")


@app.command("list")
def list_policies(
    policy_file: Optional[str] = typer.Option(None, "--file", help="Reminder policy JSON file"),
    as_json: bool = JSON_OPTION,
) -> None:
    """List local reminder policies."""
    path = _policy_file(policy_file)
    store = _load_store(path)
    now = _now()
    payload = _grouped_policy_payload(store, now=now)
    if as_json:
        print_json({"file": str(path), **payload})
        return
    rows = _policy_rows(store, now=now)
    if not rows:
        console.print(f"No reminder policies in {path}")
        return
    console.print(
        "Reminder policy groups: " + ", ".join(f"{key}={value}" for key, value in payload["summary"].items() if value)
    )
    print_table(
        ["ID", "Pri", "Mode", "State", "Enabled", "Task", "Target", "Next Fire", "Resume At", "Fires", "Reason"],
        rows,
        keys=[
            "id",
            "priority",
            "mode",
            "state",
            "enabled",
            "task",
            "target",
            "next_fire",
            "resume_at",
            "fires",
            "reason",
        ],
    )


@app.command("disable")
def disable(
    policy_id: str = typer.Argument(..., help="Policy ID or unique prefix"),
    policy_file: Optional[str] = typer.Option(None, "--file", help="Reminder policy JSON file"),
    as_json: bool = JSON_OPTION,
) -> None:
    """Disable a local reminder policy."""
    path = _policy_file(policy_file)
    store = _load_store(path)
    policy = _find_policy(store, policy_id)
    policy["enabled"] = False
    policy["updated_at"] = _iso(_now())
    _save_store(path, store)
    if as_json:
        print_json({"policy": policy, "file": str(path)})
        return
    console.print(f"Disabled reminder policy {policy['id']}")


@app.command("pause")
def pause(
    policy_id: str = typer.Argument(..., help="Policy ID or unique prefix"),
    reason: str = typer.Option("Paused by operator.", "--reason", "-r", help="Why this reminder is not actionable"),
    resume_at: Optional[str] = typer.Option(None, "--resume-at", help="Optional ISO-8601 auto-resume time"),
    minutes: Optional[int] = typer.Option(None, "--minutes", help="Snooze/pause for N minutes"),
    paused_by: Optional[str] = typer.Option(None, "--paused-by", help="Operator or agent pausing the policy"),
    policy_file: Optional[str] = typer.Option(None, "--file", help="Reminder policy JSON file"),
    as_json: bool = JSON_OPTION,
) -> None:
    """Pause a reminder without permanently disabling it."""
    path = _policy_file(policy_file)
    store = _load_store(path)
    policy = _find_policy(store, policy_id)
    now = _now()
    until = _pause_until(resume_at, minutes)
    policy["paused"] = True
    policy["paused_reason"] = reason
    policy["paused_by"] = paused_by or "operator"
    policy["paused_at"] = _iso(now)
    policy["resume_at"] = until
    policy["snooze_until"] = until
    policy["updated_at"] = _iso(now)
    _save_store(path, store)
    if as_json:
        print_json({"policy": policy, "file": str(path), "state": "paused"})
        return
    console.print(f"Paused reminder policy {policy['id']}")


@app.command("snooze")
def snooze(
    policy_id: str = typer.Argument(..., help="Policy ID or unique prefix"),
    minutes: int = typer.Option(30, "--minutes", "-m", help="Minutes to pause before auto-resume"),
    reason: str = typer.Option("Snoozed by operator.", "--reason", "-r", help="Why this reminder is being snoozed"),
    policy_file: Optional[str] = typer.Option(None, "--file", help="Reminder policy JSON file"),
    as_json: bool = JSON_OPTION,
) -> None:
    """Temporarily pause a reminder until a future time."""
    pause(
        policy_id,
        reason=reason,
        resume_at=None,
        minutes=minutes,
        paused_by="operator",
        policy_file=policy_file,
        as_json=as_json,
    )


@app.command("resume")
def resume(
    policy_id: str = typer.Argument(..., help="Policy ID or unique prefix"),
    fire_in: int = typer.Option(0, "--fire-in-minutes", help="Set next fire to N minutes from now on resume"),
    policy_file: Optional[str] = typer.Option(None, "--file", help="Reminder policy JSON file"),
    as_json: bool = JSON_OPTION,
) -> None:
    """Resume a paused reminder policy."""
    if fire_in < 0:
        raise typer.BadParameter("--fire-in-minutes cannot be negative")
    path = _policy_file(policy_file)
    store = _load_store(path)
    policy = _find_policy(store, policy_id)
    if int(policy.get("fired_count", 0)) >= int(policy.get("max_fires", 1)):
        typer.echo(f"Error: policy {policy['id']} has reached max_fires; create a new policy", err=True)
        raise typer.Exit(1)
    disabled_reason = str(policy.get("disabled_reason") or "")
    if disabled_reason.startswith("source task"):
        typer.echo(f"Error: source task is terminal; refusing to resume {policy['id']}", err=True)
        raise typer.Exit(1)
    now = _now()
    _resume_policy(policy, now=now)
    policy["enabled"] = True
    policy["next_fire_at"] = _iso(now + _dt.timedelta(minutes=fire_in))
    _save_store(path, store)
    if as_json:
        print_json({"policy": policy, "file": str(path), "state": _policy_state(policy, now=now)})
        return
    console.print(f"Resumed reminder policy {policy['id']}")


def _groom_report(store: dict[str, Any], *, now: _dt.datetime, check_tasks: bool) -> dict[str, Any]:
    client = get_client() if check_tasks else None
    items = []
    for policy in [p for p in store.get("policies", []) if isinstance(p, dict)]:
        reasons: list[str] = []
        state = _policy_state(policy, now=now)
        if state in {"disabled", "completed"}:
            reasons.append(f"state:{state}")
        next_fire = _parse_optional_iso(policy.get("next_fire_at"))
        if policy.get("next_fire_at") and not next_fire:
            reasons.append("invalid_next_fire_at")
        elif (
            next_fire
            and next_fire < now - _dt.timedelta(days=STALE_AFTER_DAYS)
            and state not in {"disabled", "completed"}
        ):
            reasons.append("stale_next_fire_at")
        if policy.get("paused") and not policy.get("paused_reason"):
            reasons.append("paused_without_reason")
        source_task = str(policy.get("source_task_id") or "")
        if check_tasks and source_task and client is not None:
            lifecycle = _task_lifecycle(client, source_task)
            if lifecycle and lifecycle.get("is_terminal"):
                reasons.append(f"source_task_terminal:{lifecycle.get('status')}")
            elif lifecycle is None:
                reasons.append("source_task_unresolved")
        elif not source_task:
            reasons.append("no_source_task")
        recommendation = "keep"
        if any(r.startswith("source_task_terminal") for r in reasons) or state == "completed":
            recommendation = "disable_or_remove_completed"
        elif state == "paused":
            recommendation = "resume_when_actionable_or_disable_if_junk"
        elif "stale_next_fire_at" in reasons or "source_task_unresolved" in reasons:
            recommendation = "review_stale_or_orphaned"
        elif state == "disabled":
            recommendation = "remove_if_no_longer_needed"
        if reasons:
            items.append(
                {
                    "policy_id": policy.get("id"),
                    "state": state,
                    "source_task_id": source_task,
                    "reasons": reasons,
                    "recommendation": recommendation,
                }
            )
    return {
        "summary": {
            "checked": len([p for p in store.get("policies", []) if isinstance(p, dict)]),
            "needs_attention": len(items),
        },
        "items": items,
        "hygiene": [
            "Close or disable reminders for completed work.",
            "Pause blocked/noisy reminders with a reason and resume_at when possible.",
            "Resume reminders when work is actionable again; keep next_fire_at near the next useful check-in.",
            "Use list groups to groom due, paused, disabled, completed, and stale reminders regularly.",
        ],
    }


@app.command("groom")
def groom(
    check_tasks: bool = typer.Option(
        True, "--check-tasks/--no-check-tasks", help="Fetch source tasks to identify terminal/orphaned reminders"
    ),
    apply: bool = typer.Option(False, "--apply", help="Disable completed/source-terminal reminder policies"),
    policy_file: Optional[str] = typer.Option(None, "--file", help="Reminder policy JSON file"),
    as_json: bool = JSON_OPTION,
) -> None:
    """Report noisy, stale, completed, or orphaned reminder policies."""
    path = _policy_file(policy_file)
    store = _load_store(path)
    now = _now()
    report = _groom_report(store, now=now, check_tasks=check_tasks)
    changed: list[str] = []
    if apply:
        attention = {item["policy_id"]: item for item in report["items"]}
        for policy in [p for p in store.get("policies", []) if isinstance(p, dict)]:
            item = attention.get(policy.get("id"))
            if not item:
                continue
            if item["recommendation"] == "disable_or_remove_completed":
                policy["enabled"] = False
                policy["paused"] = False
                policy["disabled_reason"] = ",".join(item["reasons"])
                policy["updated_at"] = _iso(now)
                changed.append(str(policy.get("id")))
        if changed:
            _save_store(path, store)
    report["file"] = str(path)
    report["changed"] = changed
    if as_json:
        print_json(report)
        return
    console.print(
        f"Reminder grooming: {report['summary']['needs_attention']} need attention / {report['summary']['checked']} checked"
    )
    if report["items"]:
        print_table(
            ["Policy", "State", "Task", "Reasons", "Recommendation"],
            report["items"],
            keys=["policy_id", "state", "source_task_id", "reasons", "recommendation"],
        )
    console.print("Hygiene:")
    for item in report["hygiene"]:
        console.print(f"  - {item}")


def _build_fire_payload(client: Any, policy: dict[str, Any], *, now: _dt.datetime) -> dict[str, Any] | None:
    """Build target/reason/content/metadata for a due policy.

    Returns None if the policy must be skipped (e.g. source task is terminal -
    side-effect: marks the policy disabled). Otherwise returns a dict with
    keys: target, target_resolved_from, content, metadata, channel.
    """
    source_task = str(policy.get("source_task_id") or "")
    reason = str(policy.get("reason") or "Please review this task.")
    target = _strip_at(policy.get("target"))
    target_resolved_from = None

    lifecycle = _task_lifecycle(client, source_task) if source_task else None

    if lifecycle and lifecycle.get("is_terminal"):
        policy["enabled"] = False
        policy["disabled_reason"] = f"source task {source_task} is {lifecycle.get('status')}"
        policy["updated_at"] = _iso(now)
        policy["_skip_reason"] = f"source_task_terminal:{lifecycle.get('status')}"
        return None

    if lifecycle and lifecycle.get("is_pending_review"):
        review_target = lifecycle.get("review_owner") or lifecycle.get("creator_name")
        if review_target:
            target = review_target
            target_resolved_from = "review_owner" if lifecycle.get("review_owner") else "creator_fallback"
            reason = f"[pending review] {reason}"
        elif not target:
            target, target_resolved_from = (lifecycle.get("assignee_name"), "assignee")
    elif source_task and not target:
        if lifecycle and lifecycle.get("assignee_name"):
            target, target_resolved_from = lifecycle["assignee_name"], "assignee"
        elif lifecycle and lifecycle.get("creator_name"):
            target, target_resolved_from = lifecycle["creator_name"], "creator"
        else:
            target, target_resolved_from = _resolve_target_from_task(client, source_task)

    try:
        triggered_by = resolve_agent_name(client=client)
    except Exception:
        triggered_by = None

    task_snapshot = (
        lifecycle.get("snapshot")
        if lifecycle and lifecycle.get("snapshot")
        else (_fetch_task_snapshot(client, source_task) if source_task else None)
    )

    fired_at = _iso(now)
    metadata = _build_alert_metadata(
        kind="reminder",
        severity=str(policy.get("severity") or "info"),
        target=target,
        reason=reason,
        source_task_id=source_task,
        due_at=policy.get("due_at"),
        remind_at=fired_at,
        expected_response=policy.get("expected_response"),
        response_required=True,
        evidence=policy.get("evidence"),
        triggered_by_agent=triggered_by,
        title=policy.get("title"),
        task_snapshot=task_snapshot,
    )
    metadata["reminder_policy"] = {
        "policy_id": policy.get("id"),
        "fire_key": policy.get("_current_fire_key"),
        "cadence_seconds": policy.get("cadence_seconds"),
        "fired_count": policy.get("fired_count", 0) + 1,
        "max_fires": policy.get("max_fires"),
        "target_resolved_from": target_resolved_from,
        "mode": str(policy.get("mode", "auto")),
    }

    return {
        "target": target,
        "target_resolved_from": target_resolved_from,
        "content": _format_mention_content(target, reason, "reminder"),
        "metadata": metadata,
        "channel": str(policy.get("channel") or "main"),
        "fired_at": fired_at,
    }


def _fire_policy(
    client: Any,
    policy: dict[str, Any],
    *,
    now: _dt.datetime,
    drafts: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Fire a due policy according to its mode.

    - auto:   send immediately (existing behavior)
    - draft:  build payload, append to drafts, do NOT send
    - manual: skip (manual policies should not appear in due_policies; safety net)
    """
    payload = _build_fire_payload(client, policy, now=now)
    if payload is None:
        # Skipped — _build_fire_payload populated _skip_reason on the policy.
        skip_reason = str(policy.pop("_skip_reason", "skipped"))
        return {
            "policy_id": policy.get("id"),
            "skipped": True,
            "reason": skip_reason,
            "source_task_id": str(policy.get("source_task_id") or ""),
            "fired_at": None,
        }

    mode = str(policy.get("mode", "auto"))

    if mode == "manual":
        return {
            "policy_id": policy.get("id"),
            "skipped": True,
            "reason": "manual_mode",
            "fired_at": None,
        }

    if mode == "draft":
        if drafts is None:
            return {
                "policy_id": policy.get("id"),
                "error": "draft mode requires a drafts store; pass --file to a v2 store",
            }
        draft = {
            "id": _short_draft_id(),
            "policy_id": policy.get("id"),
            "fire_key": policy.get("_current_fire_key"),
            "created_at": payload["fired_at"],
            "target": payload["target"],
            "target_resolved_from": payload["target_resolved_from"],
            "content": payload["content"],
            "metadata": payload["metadata"],
            "channel": payload["channel"],
            "space_id": str(policy.get("space_id") or ""),
            "status": "pending",
        }
        drafts.append(draft)
        return {
            "policy_id": policy.get("id"),
            "draft_id": draft["id"],
            "drafted": True,
            "target": payload["target"],
            "fired_at": payload["fired_at"],
        }

    # mode == "auto"
    try:
        result = client.send_message(
            str(policy.get("space_id")),
            payload["content"],
            channel=payload["channel"],
            metadata=payload["metadata"],
            message_type="reminder",
        )
    except (httpx.ConnectError, httpx.ReadError) as exc:
        # Offline-first: backend unreachable. Auto-degrade to draft so the
        # fire is not lost. Operator dispatches via `ax reminders drafts send`
        # once connectivity returns.
        if drafts is None:
            return {
                "policy_id": policy.get("id"),
                "error": f"network: {exc}",
            }
        draft = {
            "id": _short_draft_id(),
            "policy_id": policy.get("id"),
            "fire_key": policy.get("_current_fire_key"),
            "created_at": payload["fired_at"],
            "target": payload["target"],
            "target_resolved_from": payload["target_resolved_from"],
            "content": payload["content"],
            "metadata": payload["metadata"],
            "channel": payload["channel"],
            "space_id": str(policy.get("space_id") or ""),
            "status": "pending",
            "auto_degraded": True,
            "auto_degrade_reason": str(exc),
        }
        drafts.append(draft)
        return {
            "policy_id": policy.get("id"),
            "draft_id": draft["id"],
            "drafted": True,
            "auto_degraded": True,
            "target": payload["target"],
            "fired_at": payload["fired_at"],
        }
    message = result.get("message", result) if isinstance(result, dict) else {}
    return {
        "policy_id": policy.get("id"),
        "message_id": message.get("id"),
        "target": payload["target"],
        "target_resolved_from": payload["target_resolved_from"],
        "fired_at": payload["fired_at"],
    }


def _due_policies(store: dict[str, Any], *, now: _dt.datetime, include_manual: bool = False) -> list[dict[str, Any]]:
    """Return enabled, due policies in priority queue order.

    Manual-mode policies are excluded by default; pass ``include_manual=True``
    to include them (e.g. for an explicit ``run --force <id>`` path).
    """
    due = []
    for policy in store.get("policies", []):
        if not isinstance(policy, dict) or not policy.get("enabled", True):
            continue
        if not include_manual and str(policy.get("mode", "auto")) == "manual":
            continue
        if _is_paused(policy, now=now):
            continue
        if int(policy.get("fired_count", 0)) >= int(policy.get("max_fires", 1)):
            policy["enabled"] = False
            policy["updated_at"] = _iso(now)
            continue
        try:
            next_fire = _parse_iso(str(policy.get("next_fire_at")))
        except Exception:
            policy["enabled"] = False
            policy["disabled_reason"] = "invalid next_fire_at"
            policy["updated_at"] = _iso(now)
            continue
        if next_fire <= now:
            fire_key = f"{policy.get('id')}:{policy.get('next_fire_at')}"
            if fire_key in set(policy.get("fired_keys") or []):
                continue
            policy["_current_fire_key"] = fire_key
            due.append(policy)
    due.sort(key=lambda policy: _policy_sort_key(policy, now=now))
    return due


def _advance_policy(
    policy: dict[str, Any],
    *,
    now: _dt.datetime,
    message_id: str | None,
    draft_id: str | None = None,
) -> None:
    """Advance a policy after a successful fire (auto-sent or drafted).

    Drafted fires DO advance fired_count and next_fire_at — drafts are
    real fires from the loop's perspective. The HITL send/cancel does not
    re-tick the policy.
    """
    fire_key = str(policy.pop("_current_fire_key", ""))
    fired_keys = list(policy.get("fired_keys") or [])
    if fire_key:
        fired_keys.append(fire_key)
    policy["fired_keys"] = fired_keys[-50:]
    policy["fired_count"] = int(policy.get("fired_count", 0)) + 1
    policy["last_fired_at"] = _iso(now)
    policy["last_message_id"] = message_id
    policy["last_draft_id"] = draft_id
    policy["updated_at"] = _iso(now)

    max_fires = int(policy.get("max_fires", 1))
    if policy["fired_count"] >= max_fires:
        policy["enabled"] = False
        policy["disabled_reason"] = "max_fires reached"
        return
    cadence_seconds = int(policy.get("cadence_seconds", 300))
    policy["next_fire_at"] = _iso(now + _dt.timedelta(seconds=cadence_seconds))


@app.command("run")
def run(
    once: bool = typer.Option(False, "--once", help="Run one due-policy pass and exit"),
    watch: bool = typer.Option(False, "--watch", help="Keep running due-policy passes"),
    interval: int = typer.Option(30, "--interval", help="Seconds between watch passes"),
    policy_file: Optional[str] = typer.Option(None, "--file", help="Reminder policy JSON file"),
    as_json: bool = JSON_OPTION,
) -> None:
    """Fire due local reminder policies.

    Use ``--once`` for cron-like execution. Use ``--watch`` for dogfood loops.
    """
    if not once and not watch:
        once = True
    if interval < 1:
        raise typer.BadParameter("--interval must be at least 1 second")

    path = _policy_file(policy_file)
    all_results: list[dict[str, Any]] = []
    client = get_client()

    while True:
        store = _load_store(path)
        now = _now()
        pass_results: list[dict[str, Any]] = []
        drafts_list = store.setdefault("drafts", [])
        for policy in _due_policies(store, now=now):
            try:
                result = _fire_policy(client, policy, now=now, drafts=drafts_list)
            except httpx.HTTPStatusError as exc:
                result = {
                    "policy_id": policy.get("id"),
                    "error": f"{exc.response.status_code} {exc.response.text[:200]}",
                }
            except (httpx.ConnectError, httpx.ReadError) as exc:
                result = {"policy_id": policy.get("id"), "error": str(exc)}
            if not result.get("error") and not result.get("skipped"):
                _advance_policy(
                    policy,
                    now=now,
                    message_id=result.get("message_id"),
                    draft_id=result.get("draft_id"),
                )
            pass_results.append(result)
            all_results.append(result)
        _save_store(path, store)

        if once:
            if as_json:
                print_json({"file": str(path), "fired": all_results})
            elif pass_results:
                rows = []
                for item in pass_results:
                    if item.get("error"):
                        status = f"error: {item['error'][:40]}"
                    elif item.get("skipped"):
                        status = f"skipped ({item.get('reason', '')})"
                    elif item.get("drafted"):
                        status = f"drafted: {item.get('draft_id')}"
                    elif item.get("message_id"):
                        status = f"sent: {item['message_id']}"
                    else:
                        status = "fired"
                    rows.append(
                        {
                            "policy_id": item.get("policy_id"),
                            "status": status,
                            "target": item.get("target"),
                            "fired_at": item.get("fired_at"),
                        }
                    )
                print_table(
                    ["Policy", "Status", "Target", "Fired At"],
                    rows,
                    keys=["policy_id", "status", "target", "fired_at"],
                )
            else:
                console.print(f"No due reminders in {path}")
            return

        if pass_results and not as_json:
            for item in pass_results:
                if item.get("error"):
                    console.print(f"[red]{item['policy_id']}[/red]: {item['error']}")
                elif item.get("skipped"):
                    reason = item.get("reason") or "skipped"
                    console.print(f"[yellow]{item['policy_id']}[/yellow] skipped ({reason})")
                elif item.get("drafted"):
                    console.print(
                        f"[cyan]{item['policy_id']}[/cyan] drafted "
                        f"draft={item.get('draft_id')} target={item.get('target')}"
                    )
                else:
                    console.print(
                        f"[green]{item['policy_id']}[/green] fired "
                        f"message={item.get('message_id')} target={item.get('target')}"
                    )
        time.sleep(interval)


# ---- Status: online/offline + queue snapshot ------------------------------


def _probe_online(timeout: float = 2.0) -> tuple[bool, str | None]:
    """Cheap online probe. Returns (is_online, reason_if_offline)."""
    try:
        client = get_client()
    except Exception as exc:
        return False, f"client unavailable: {exc}"
    base = getattr(client, "_base_url", None) or getattr(client, "base_url", None)
    if not base:
        return False, "no base_url configured"
    try:
        # Most ax backends respond on /health quickly. Use a short timeout.
        resp = httpx.get(f"{str(base).rstrip('/')}/health", timeout=timeout)
        if resp.status_code < 500:
            return True, None
        return False, f"backend status {resp.status_code}"
    except (httpx.ConnectError, httpx.ReadError, httpx.TimeoutException) as exc:
        return False, f"network: {exc}"
    except Exception as exc:
        return False, str(exc)


@app.command("status")
def status(
    policy_file: Optional[str] = typer.Option(None, "--file", help="Reminder policy JSON file"),
    skip_probe: bool = typer.Option(False, "--skip-probe", help="Skip online probe (faster, offline assumed)"),
    as_json: bool = JSON_OPTION,
) -> None:
    """Show online/offline status, queue depth, and pending drafts count.

    Helps an operator answer "can my reminders fire now, and what's queued
    for me to review?" — the offline-first contract surface.
    """
    path = _policy_file(policy_file)
    store = _load_store(path)

    policies = [p for p in store.get("policies", []) if isinstance(p, dict)]
    enabled_policies = [p for p in policies if p.get("enabled", True)]
    drafts = [d for d in store.get("drafts", []) if isinstance(d, dict)]
    pending_drafts = [d for d in drafts if d.get("status") == "pending"]
    auto_degraded = [d for d in pending_drafts if d.get("auto_degraded") is True]

    # Find next-due policy (in priority queue order, ignoring whether it's currently due)
    sorted_enabled = sorted(enabled_policies, key=_policy_sort_key)
    next_due = sorted_enabled[0] if sorted_enabled else None

    if skip_probe:
        is_online, offline_reason = False, "probe skipped"
    else:
        is_online, offline_reason = _probe_online()

    snapshot = {
        "online": is_online,
        "offline_reason": offline_reason,
        "file": str(path),
        "policies_total": len(policies),
        "policies_enabled": len(enabled_policies),
        "policies_paused_or_disabled": len(policies) - len(enabled_policies),
        "drafts_pending": len(pending_drafts),
        "drafts_auto_degraded": len(auto_degraded),
        "next_due": (
            {
                "id": next_due.get("id"),
                "priority": int(next_due.get("priority", _DEFAULT_PRIORITY)),
                "mode": str(next_due.get("mode", "auto")),
                "target": next_due.get("target") or "(task default)",
                "next_fire_at": next_due.get("next_fire_at"),
            }
            if next_due
            else None
        ),
    }

    if as_json:
        print_json(snapshot)
        return

    state_label = "[bold green]ONLINE[/bold green]" if is_online else "[bold yellow]OFFLINE[/bold yellow]"
    console.print(f"State: {state_label}")
    if not is_online and offline_reason:
        console.print(f"  reason: {offline_reason}")
    console.print(f"Store: {path}")
    console.print(f"Policies: {len(enabled_policies)} enabled / {len(policies)} total")
    console.print(
        f"Drafts: {len(pending_drafts)} pending" + (f" ({len(auto_degraded)} auto-degraded)" if auto_degraded else "")
    )
    if next_due:
        console.print(
            f"Next: {next_due.get('id')} priority={int(next_due.get('priority', _DEFAULT_PRIORITY))} "
            f"mode={next_due.get('mode', 'auto')} target={next_due.get('target') or '(task)'} "
            f"fires_at={next_due.get('next_fire_at')}"
        )
    else:
        console.print("Next: (no enabled policies)")


# ---- Operator commands: cancel / update ------------------------------------


@app.command("cancel")
def cancel(
    policy_id: str = typer.Argument(..., help="Policy ID or unique prefix"),
    policy_file: Optional[str] = typer.Option(None, "--file", help="Reminder policy JSON file"),
    as_json: bool = JSON_OPTION,
) -> None:
    """Cancel a reminder policy permanently. Like ``disable`` but with explicit cancel reason."""
    path = _policy_file(policy_file)
    store = _load_store(path)
    policy = _find_policy(store, policy_id)
    policy["enabled"] = False
    policy["disabled_reason"] = "cancelled"
    policy["updated_at"] = _iso(_now())
    _save_store(path, store)
    if as_json:
        print_json({"policy": policy, "file": str(path)})
        return
    console.print(f"Cancelled reminder policy {policy['id']}")


@app.command("update")
def update_policy(
    policy_id: str = typer.Argument(..., help="Policy ID or unique prefix"),
    priority: Optional[int] = typer.Option(None, "--priority", help="New priority (0-100, lower = higher)"),
    cadence: Optional[int] = typer.Option(None, "--cadence-minutes", help="New cadence in minutes"),
    max_fires: Optional[int] = typer.Option(None, "--max-fires", help="New max-fires cap"),
    mode: Optional[str] = typer.Option(None, "--mode", help="auto | draft | manual"),
    reason: Optional[str] = typer.Option(None, "--reason", help="New reason text"),
    target: Optional[str] = typer.Option(None, "--target", help="New target @agent/user"),
    policy_file: Optional[str] = typer.Option(None, "--file", help="Reminder policy JSON file"),
    as_json: bool = JSON_OPTION,
) -> None:
    """Update fields on a reminder policy. ``--priority`` re-orders the queue."""
    path = _policy_file(policy_file)
    store = _load_store(path)
    policy = _find_policy(store, policy_id)
    if priority is not None:
        policy["priority"] = _normalize_priority(priority)
    if mode is not None:
        policy["mode"] = _normalize_mode(mode)
    if cadence is not None:
        if cadence < 1:
            raise typer.BadParameter("--cadence-minutes must be at least 1")
        policy["cadence_seconds"] = cadence * 60
    if max_fires is not None:
        if max_fires < 1:
            raise typer.BadParameter("--max-fires must be at least 1")
        policy["max_fires"] = max_fires
    if reason is not None:
        policy["reason"] = reason
    if target is not None:
        policy["target"] = _strip_at(target)
    policy["updated_at"] = _iso(_now())
    _save_store(path, store)
    if as_json:
        print_json({"policy": policy, "file": str(path)})
        return
    console.print(f"Updated reminder policy {policy['id']}")


# ---- Drafts subcommand group: list / show / edit / send / cancel ----------

drafts_app = typer.Typer(name="drafts", help="HITL drafts queued by draft-mode policies", no_args_is_help=True)
app.add_typer(drafts_app, name="drafts")


def _find_draft(store: dict[str, Any], draft_id: str) -> dict[str, Any]:
    matches = [
        d
        for d in store.get("drafts", [])
        if isinstance(d, dict) and str(d.get("id", "")).startswith(draft_id) and d.get("status") == "pending"
    ]
    if not matches:
        typer.echo(f"Error: pending draft not found: {draft_id}", err=True)
        raise typer.Exit(1)
    if len(matches) > 1:
        typer.echo(f"Error: draft id is ambiguous: {draft_id}", err=True)
        raise typer.Exit(1)
    return matches[0]


@drafts_app.command("list")
def drafts_list(
    policy_file: Optional[str] = typer.Option(None, "--file", help="Reminder policy JSON file"),
    as_json: bool = JSON_OPTION,
) -> None:
    """List pending HITL drafts."""
    path = _policy_file(policy_file)
    store = _load_store(path)
    pending = [d for d in store.get("drafts", []) if isinstance(d, dict) and d.get("status") == "pending"]
    if as_json:
        print_json({"file": str(path), "drafts": pending})
        return
    if not pending:
        console.print(f"No pending drafts in {path}")
        return
    rows = [
        {
            "id": d.get("id", ""),
            "policy": d.get("policy_id", ""),
            "target": d.get("target") or "(none)",
            "created_at": d.get("created_at", ""),
            "preview": (d.get("content") or "")[:60],
        }
        for d in pending
    ]
    print_table(
        ["ID", "Policy", "Target", "Created", "Preview"],
        rows,
        keys=["id", "policy", "target", "created_at", "preview"],
    )


@drafts_app.command("show")
def drafts_show(
    draft_id: str = typer.Argument(..., help="Draft ID or unique prefix"),
    policy_file: Optional[str] = typer.Option(None, "--file", help="Reminder policy JSON file"),
    as_json: bool = JSON_OPTION,
) -> None:
    """Show a pending draft's full body and metadata."""
    path = _policy_file(policy_file)
    store = _load_store(path)
    draft = _find_draft(store, draft_id)
    if as_json:
        print_json({"draft": draft, "file": str(path)})
        return
    console.print(f"[bold]{draft['id']}[/bold] (policy={draft.get('policy_id')})")
    console.print(f"[bold]target[/bold]: {draft.get('target')}")
    console.print(f"[bold]channel[/bold]: {draft.get('channel')}")
    console.print(f"[bold]created[/bold]: {draft.get('created_at')}")
    console.print()
    console.print(draft.get("content", ""))


@drafts_app.command("edit")
def drafts_edit(
    draft_id: str = typer.Argument(..., help="Draft ID or unique prefix"),
    body: Optional[str] = typer.Option(None, "--body", help="New message body"),
    target: Optional[str] = typer.Option(None, "--target", help="New target @agent/user"),
    policy_file: Optional[str] = typer.Option(None, "--file", help="Reminder policy JSON file"),
    as_json: bool = JSON_OPTION,
) -> None:
    """Edit a pending draft before sending."""
    if body is None and target is None:
        raise typer.BadParameter("--body and/or --target required")
    path = _policy_file(policy_file)
    store = _load_store(path)
    draft = _find_draft(store, draft_id)
    if target is not None:
        draft["target"] = _strip_at(target)
        # Re-mention prefix the body if it doesn't already lead with @target
        if body is None and draft.get("content"):
            existing = str(draft["content"])
            # strip the old @mention if present
            if existing.startswith("@"):
                existing_body = existing.split(" ", 1)[1] if " " in existing else ""
            else:
                existing_body = existing
            draft["content"] = _format_mention_content(draft["target"], existing_body, "reminder")
    if body is not None:
        draft["content"] = _format_mention_content(draft.get("target"), body, "reminder")
    draft["edited"] = True
    draft["updated_at"] = _iso(_now())
    _save_store(path, store)
    if as_json:
        print_json({"draft": draft, "file": str(path)})
        return
    console.print(f"Edited draft {draft['id']}")


@drafts_app.command("send")
def drafts_send(
    draft_id: str = typer.Argument(..., help="Draft ID or unique prefix"),
    policy_file: Optional[str] = typer.Option(None, "--file", help="Reminder policy JSON file"),
    as_json: bool = JSON_OPTION,
) -> None:
    """Send a pending draft via the messages API."""
    path = _policy_file(policy_file)
    store = _load_store(path)
    draft = _find_draft(store, draft_id)
    client = get_client()
    try:
        result = client.send_message(
            str(draft.get("space_id")),
            str(draft.get("content") or ""),
            channel=str(draft.get("channel") or "main"),
            metadata=draft.get("metadata") or {},
            message_type="reminder",
        )
    except httpx.HTTPStatusError as exc:
        typer.echo(f"Error: send failed: {exc.response.status_code} {exc.response.text[:200]}", err=True)
        raise typer.Exit(1)
    message = result.get("message", result) if isinstance(result, dict) else {}
    draft["status"] = "sent"
    draft["sent_at"] = _iso(_now())
    draft["message_id"] = message.get("id")
    _save_store(path, store)
    if as_json:
        print_json({"draft": draft, "message_id": message.get("id"), "file": str(path)})
        return
    console.print(f"Sent draft {draft['id']} (message {message.get('id')})")


@drafts_app.command("cancel")
def drafts_cancel(
    draft_id: str = typer.Argument(..., help="Draft ID or unique prefix"),
    policy_file: Optional[str] = typer.Option(None, "--file", help="Reminder policy JSON file"),
    as_json: bool = JSON_OPTION,
) -> None:
    """Cancel a pending draft. Does NOT re-tick the source policy."""
    path = _policy_file(policy_file)
    store = _load_store(path)
    draft = _find_draft(store, draft_id)
    draft["status"] = "cancelled"
    draft["cancelled_at"] = _iso(_now())
    _save_store(path, store)
    if as_json:
        print_json({"draft": draft, "file": str(path)})
        return
    console.print(f"Cancelled draft {draft['id']}")
