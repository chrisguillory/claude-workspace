"""Retry helpers for transient network errors across API clients.

Private submodule - not exported by the package.

HTTPX Exception Hierarchy
=========================

Reference for which exceptions to catch vs propagate::

    httpx.HTTPError (base)
    ├── httpx.RequestError
    │   ├── httpx.TransportError
    │   │   ├── httpx.TimeoutException   ← RETRY (all 4 subclasses)
    │   │   │   ├── ConnectTimeout
    │   │   │   ├── ReadTimeout
    │   │   │   ├── WriteTimeout
    │   │   │   └── PoolTimeout
    │   │   ├── httpx.NetworkError       ← RETRY (all 4 subclasses)
    │   │   │   ├── ConnectError
    │   │   │   ├── ReadError
    │   │   │   ├── WriteError
    │   │   │   └── CloseError
    │   │   ├── httpx.ProtocolError
    │   │   │   ├── LocalProtocolError   ← PROPAGATE (our bug)
    │   │   │   └── RemoteProtocolError  ← RETRY (server sent invalid HTTP)
    │   │   ├── ProxyError               ← PROPAGATE (config error)
    │   │   └── UnsupportedProtocol      ← PROPAGATE (code error)
    │   ├── DecodingError                ← PROPAGATE (response malformed)
    │   └── TooManyRedirects             ← PROPAGATE (config/server error)
    ├── httpx.HTTPStatusError            ← Handle per-client (SDK wraps these)
    └── httpx.InvalidURL                 ← PROPAGATE (code error)

Retry Policy
------------
- **RETRY** = Transient network issues, worth retrying with backoff
- **PROPAGATE** = Bugs, config errors, or permanent failures (fail-fast)

Client-Specific Notes
---------------------
- **Gemini (google-genai)**: Throws httpx exceptions directly. Also retry
  ServerError with status 500/502/503/504.
- **Qdrant (qdrant-client)**: Wraps httpx in ResponseHandlingException.
  Check exc.source for the underlying error.
"""

from __future__ import annotations

from document_search.clients._retry.gemini import gemini_breaker, is_retryable_gemini_error, log_gemini_retry
from document_search.clients._retry.httpx_errors import is_retryable_httpx_error
from document_search.clients._retry.openrouter import (
    is_retryable_openrouter_error,
    log_openrouter_retry,
    openrouter_breaker,
)
from document_search.clients._retry.qdrant import is_retryable_qdrant_error, log_qdrant_retry, qdrant_breaker

__all__ = [
    'gemini_breaker',
    'is_retryable_gemini_error',
    'is_retryable_httpx_error',
    'is_retryable_openrouter_error',
    'is_retryable_qdrant_error',
    'log_gemini_retry',
    'log_openrouter_retry',
    'log_qdrant_retry',
    'openrouter_breaker',
    'qdrant_breaker',
]
