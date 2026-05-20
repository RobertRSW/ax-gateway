"""Connector dispatch — routes a connector row to the appropriate provider adapter."""

from __future__ import annotations

from types import ModuleType
from typing import Any

from ..errors import ConnectorProviderError
from ..types import ConnectorRow
from . import composio_adapter

_ADAPTERS: dict[str, ModuleType] = {
    "composio": composio_adapter,
}


def _get_adapter(provider: str) -> ModuleType:
    adapter = _ADAPTERS.get(provider)
    if adapter is None:
        raise ConnectorProviderError(provider, f"No adapter for provider {provider!r}")
    return adapter


def execute_tool(
    connector: ConnectorRow,
    tool_slug: str,
    args: dict[str, Any],
    auth_env: dict[str, str],
) -> dict[str, Any]:
    adapter = _get_adapter(connector.provider)
    return adapter.execute_tool(
        tool_slug,
        args,
        auth_env,
        connector.config,
        connector.name,
    )


def search_tools(
    connector: ConnectorRow,
    query: str,
    auth_env: dict[str, str],
    *,
    limit: int = 10,
) -> dict[str, Any]:
    adapter = _get_adapter(connector.provider)
    return adapter.search_tools(
        query,
        auth_env,
        connector.config,
        connector.name,
        limit=limit,
    )
