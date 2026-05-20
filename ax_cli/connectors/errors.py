"""Connector-specific exceptions."""

from __future__ import annotations


class ConnectorError(Exception):
    """Base exception for connector operations."""


class ConnectorNotFoundError(ConnectorError):
    """Raised when a connector lookup fails."""

    def __init__(self, ref: str) -> None:
        self.ref = ref
        super().__init__(f"Connector not found: {ref}")


class ConnectorAuthError(ConnectorError):
    """Raised when managed-auth operations fail."""

    def __init__(self, connector_name: str, detail: str) -> None:
        self.connector_name = connector_name
        self.detail = detail
        super().__init__(f"Auth error for connector {connector_name!r}: {detail}")


class ConnectorProviderError(ConnectorError):
    """Raised when a provider adapter call fails."""

    def __init__(
        self,
        provider: str,
        detail: str,
        *,
        status_code: int | None = None,
        request_id: str | None = None,
    ) -> None:
        self.provider = provider
        self.detail = detail
        self.status_code = status_code
        self.request_id = request_id
        parts = [f"Provider {provider!r} error"]
        if status_code is not None:
            parts.append(f"(HTTP {status_code})")
        parts.append(f": {detail}")
        if request_id:
            parts.append(f" [request_id={request_id}]")
        super().__init__("".join(parts))


class ConnectorAuthHTTPError(ConnectorProviderError):
    """HTTP 401/403 from the provider — bad or expired credentials."""


class ConnectorRateLimitError(ConnectorProviderError):
    """HTTP 429 from the provider — caller should back off and retry."""


class ConnectorTransientError(ConnectorProviderError):
    """HTTP 500/502/503/504 from the provider — transient, may succeed on retry."""


def classify_provider_error(
    provider: str,
    detail: str,
    *,
    status_code: int | None = None,
    request_id: str | None = None,
) -> ConnectorProviderError:
    """Return the most specific exception subclass for the given status code."""
    kwargs = {"status_code": status_code, "request_id": request_id}
    if status_code in (401, 403):
        return ConnectorAuthHTTPError(provider, detail, **kwargs)
    if status_code == 429:
        return ConnectorRateLimitError(provider, detail, **kwargs)
    if status_code is not None and status_code >= 500:
        return ConnectorTransientError(provider, detail, **kwargs)
    return ConnectorProviderError(provider, detail, **kwargs)
