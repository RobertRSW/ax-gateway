"""Outbound connectors — gateway-managed tool providers.

Public API re-exports for convenience.
"""

from .auth import auth_status, cleanup_auth, read_auth, write_auth
from .errors import (
    ConnectorAuthError,
    ConnectorAuthHTTPError,
    ConnectorNotFoundError,
    ConnectorProviderError,
    ConnectorRateLimitError,
    ConnectorTransientError,
)
from .providers.dispatch import execute_tool, search_tools
from .storage import (
    add_connector,
    connectors_registry_path,
    find_connector,
    list_connectors,
    load_connectors_registry,
    remove_connector,
    save_connectors_registry,
    update_connector,
)
from .types import ConnectorRow
from .validation import validate_new_connector

__all__ = [
    "ConnectorAuthError",
    "ConnectorAuthHTTPError",
    "ConnectorNotFoundError",
    "ConnectorProviderError",
    "ConnectorRateLimitError",
    "ConnectorTransientError",
    "ConnectorRow",
    "add_connector",
    "auth_status",
    "cleanup_auth",
    "connectors_registry_path",
    "execute_tool",
    "find_connector",
    "list_connectors",
    "load_connectors_registry",
    "read_auth",
    "remove_connector",
    "save_connectors_registry",
    "search_tools",
    "update_connector",
    "validate_new_connector",
    "write_auth",
]
