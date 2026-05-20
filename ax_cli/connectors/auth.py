"""Managed auth lifecycle for connector credentials.

Auth files live at ``gateway_dir() / "connectors" / "auth" / "<id>.env"``
with 0o600 permissions. Values are stored as ``KEY=VALUE`` lines.
"""

from __future__ import annotations

import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .errors import ConnectorAuthError

log = logging.getLogger("connectors.auth")


def _auth_dir() -> Path:
    from ax_cli.gateway import gateway_dir

    d = gateway_dir() / "connectors" / "auth"
    d.mkdir(parents=True, exist_ok=True)
    try:
        d.chmod(0o700)
    except OSError:
        pass
    return d


def _auth_path(connector_id: str) -> Path:
    return _auth_dir() / f"{connector_id}.env"


def _serialize_env(kvs: dict[str, str]) -> str:
    if not kvs:
        return ""
    lines: list[str] = ["# managed by ax gateway — do not source as shell"]
    for key, value in sorted(kvs.items()):
        if "=" in key or "\n" in key:
            raise ConnectorAuthError(key, "Key must not contain '=' or newlines")
        if "\n" in value:
            raise ConnectorAuthError(key, "Value must not contain newlines")
        lines.append(f"{key}={value}")
    return "\n".join(lines) + "\n"


def _parse_env(text: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for lineno, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            log.warning("auth env line %d: skipping malformed line (no '=')", lineno)
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if value.startswith('"') and value.endswith('"') and len(value) >= 2:
            value = value[1:-1]
        elif value.startswith("'") and value.endswith("'") and len(value) >= 2:
            value = value[1:-1]
        if key:
            result[key] = value
    return result


def write_auth(connector_id: str, connector_name: str, kvs: dict[str, str]) -> Path:
    if not kvs:
        raise ConnectorAuthError(connector_name, "No key-value pairs provided")

    content = _serialize_env(kvs)
    path = _auth_path(connector_id)

    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as tmp:
            tmp_path = Path(tmp.name)
            tmp.write(content)
            tmp.flush()
            os.fsync(tmp.fileno())
        tmp_path.chmod(0o600)
        tmp_path.replace(path)
    finally:
        if tmp_path and tmp_path.exists():
            tmp_path.unlink()

    try:
        path.chmod(0o600)
    except OSError:
        pass
    return path


def read_auth(connector_id: str, connector_name: str) -> dict[str, str]:
    path = _auth_path(connector_id)
    if not path.exists():
        raise ConnectorAuthError(connector_name, "No auth file found. Run: ax gateway connectors auth write")
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as e:
        raise ConnectorAuthError(connector_name, f"Cannot read auth file: {e}") from e
    return _parse_env(text)


def auth_status(connector_id: str, connector_name: str) -> dict[str, Any]:
    path = _auth_path(connector_id)
    if not path.exists():
        return {
            "connector": connector_name,
            "path": str(path),
            "exists": False,
            "keys": [],
        }
    try:
        text = path.read_text(encoding="utf-8")
        kvs = _parse_env(text)
        stat = path.stat()
        return {
            "connector": connector_name,
            "path": str(path),
            "exists": True,
            "keys": sorted(kvs.keys()),
            "permissions": oct(stat.st_mode & 0o777),
            "last_modified": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
            "size_bytes": stat.st_size,
        }
    except OSError as e:
        return {
            "connector": connector_name,
            "path": str(path),
            "exists": True,
            "keys": [],
            "error": str(e),
        }


def cleanup_auth(connector_id: str) -> bool:
    path = _auth_path(connector_id)
    if path.exists():
        path.unlink()
        return True
    return False
