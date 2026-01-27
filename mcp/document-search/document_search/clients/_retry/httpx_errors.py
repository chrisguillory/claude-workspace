"""Shared httpx error detection for retry logic.

Private module - import from _retry package.
"""

from __future__ import annotations

import httpx

__all__ = [
    'is_retryable_httpx_error',
]


def is_retryable_httpx_error(exc: BaseException) -> bool:
    """Check if exception is a retryable transient httpx error.

    Retries:
    - httpx.TimeoutException (all subclasses: Connect/Read/Write/PoolTimeout)
    - httpx.NetworkError (all subclasses: Connect/Read/Write/CloseError)
    - httpx.RemoteProtocolError (server sent invalid HTTP)

    Propagates (don't retry):
    - httpx.LocalProtocolError (our bug)
    - httpx.ProxyError, UnsupportedProtocol (config errors)
    """
    # Transient network errors - retry all timeouts and network issues
    if isinstance(exc, (httpx.TimeoutException, httpx.NetworkError)):
        return True

    # Server sent invalid HTTP response - transient, retry
    if isinstance(exc, httpx.RemoteProtocolError):
        return True

    return False
