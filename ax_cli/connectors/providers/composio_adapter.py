"""Composio HTTP adapter — no SDK dependency.

All Composio API calls go through httpx. The adapter resolves settings
from the connector config dict and the managed auth env dict.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from ..errors import ConnectorAuthError, ConnectorProviderError, classify_provider_error

log = logging.getLogger("connectors.composio")

DEFAULT_BASE_URL = "https://backend.composio.dev/api/v2"
CONNECT_TIMEOUT = 10.0
READ_TIMEOUT = 30.0


def _base_url(config: dict[str, Any]) -> str:
    url = str(config.get("composio_base_url") or DEFAULT_BASE_URL).rstrip("/")
    return url


def _api_key(auth_env: dict[str, str], connector_name: str) -> str:
    key = auth_env.get("COMPOSIO_API_KEY", "").strip()
    if not key:
        raise ConnectorAuthError(
            connector_name,
            "COMPOSIO_API_KEY not found in managed auth. "
            "Run: ax gateway connectors auth write <ref> COMPOSIO_API_KEY=<key>",
        )
    return key


def _headers(api_key: str) -> dict[str, str]:
    return {
        "x-api-key": api_key,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def _handle_error_response(resp: httpx.Response, context: str) -> None:
    if resp.is_success:
        return
    content_type = resp.headers.get("content-type", "")
    is_json = "application/json" in content_type
    body: dict[str, str] = {}
    if is_json:
        try:
            body = resp.json()
        except Exception:
            pass
    if not body:
        text = resp.text[:200].strip()
        if "<html" in text.lower() or "<!doctype" in text.lower():
            detail = f"{context}: received HTML instead of JSON (CDN/proxy intercept?)"
        else:
            detail = f"{context}: {text}" if text else f"{context}: empty response"
    else:
        detail = f"{context}: {body.get('error') or body.get('message') or str(body)}"
    request_id = body.get("requestId") if body else None
    raise classify_provider_error(
        "composio",
        detail,
        status_code=resp.status_code,
        request_id=request_id,
    )


def search_tools(
    query: str,
    auth_env: dict[str, str],
    config: dict[str, Any],
    connector_name: str,
    *,
    limit: int = 10,
) -> dict[str, Any]:
    base = _base_url(config)
    key = _api_key(auth_env, connector_name)
    params: dict[str, Any] = {"useCase": query, "limit": limit}

    try:
        resp = httpx.get(
            f"{base}/actions",
            params=params,
            headers=_headers(key),
            timeout=httpx.Timeout(connect=CONNECT_TIMEOUT, read=READ_TIMEOUT, write=READ_TIMEOUT, pool=CONNECT_TIMEOUT),
        )
    except httpx.TimeoutException as e:
        raise ConnectorProviderError("composio", f"Timeout searching tools: {e!r}") from e
    except httpx.HTTPError as e:
        raise ConnectorProviderError("composio", f"HTTP error searching tools: {e!r}") from e

    _handle_error_response(resp, "Search tools")
    return resp.json()


def execute_tool(
    tool_slug: str,
    args: dict[str, Any],
    auth_env: dict[str, str],
    config: dict[str, Any],
    connector_name: str,
) -> dict[str, Any]:
    base = _base_url(config)
    key = _api_key(auth_env, connector_name)

    body: dict[str, Any] = {"input": args}

    connected_account_id = (
        auth_env.get("COMPOSIO_CONNECTED_ACCOUNT_ID", "").strip()
        or str(config.get("connected_account_id") or "").strip()
    )
    if connected_account_id:
        body["connectedAccountId"] = connected_account_id
    else:
        entity_id = auth_env.get("COMPOSIO_ENTITY_ID", "").strip() or str(config.get("entity_id") or "default").strip()
        app_name = str(config.get("app_name") or "").strip()
        body["entityId"] = entity_id
        if app_name:
            body["appName"] = app_name

    try:
        resp = httpx.post(
            f"{base}/actions/{tool_slug}/execute",
            json=body,
            headers=_headers(key),
            timeout=httpx.Timeout(connect=CONNECT_TIMEOUT, read=READ_TIMEOUT, write=READ_TIMEOUT, pool=CONNECT_TIMEOUT),
        )
    except httpx.TimeoutException as e:
        raise ConnectorProviderError("composio", f"Timeout executing {tool_slug}: {e!r}") from e
    except httpx.HTTPError as e:
        raise ConnectorProviderError("composio", f"HTTP error executing {tool_slug}: {e!r}") from e

    _handle_error_response(resp, f"Execute {tool_slug}")
    return resp.json()
