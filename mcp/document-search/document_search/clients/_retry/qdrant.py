"""Qdrant-specific retry and circuit breaker helpers.

Private module - import from _retry package.
"""

from __future__ import annotations

import logging
from collections.abc import Set

import qdrant_client.http.exceptions
import tenacity

from document_search.clients._retry.httpx_errors import is_retryable_httpx_error

__all__ = [
    'is_retryable_qdrant_error',
    'log_qdrant_retry',
]

logger = logging.getLogger(__name__)

# HTTP status codes that are transient and worth retrying
# Note: 500 excluded - for Qdrant it often indicates request issues, not transient errors
RETRYABLE_STATUS_CODES: Set[int] = {408, 502, 503, 504}


def is_retryable_qdrant_error(exc: BaseException) -> bool:
    """Check if exception is a retryable transient error from Qdrant.

    qdrant-client raises:
    - ResponseHandlingException: wraps httpx transport errors (check exc.source)
    - UnexpectedResponse: HTTP status errors (check exc.status_code)

    We retry on transient network errors and server errors (408, 502, 503, 504).
    """
    # ResponseHandlingException wraps httpx transport errors
    if isinstance(exc, qdrant_client.http.exceptions.ResponseHandlingException):
        return is_retryable_httpx_error(exc.source)

    # UnexpectedResponse for HTTP status errors (408 timeout, 5xx server errors)
    if isinstance(exc, qdrant_client.http.exceptions.UnexpectedResponse):
        return exc.status_code in RETRYABLE_STATUS_CODES

    return False


def log_qdrant_retry(retry_state: tenacity.RetryCallState) -> None:
    """Log Qdrant retry attempt with exception details."""
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    if exc is None:
        return

    exc_name = type(exc).__name__
    exc_msg = str(exc)

    # ResponseHandlingException wraps the underlying error in .source
    source_info = ''
    if isinstance(exc, qdrant_client.http.exceptions.ResponseHandlingException) and exc.source:
        source_info = f' (source: {type(exc.source).__name__}: {exc.source})'

    logger.warning(
        '[RETRY] Qdrant upsert attempt %s failed: %s: %s%s',
        retry_state.attempt_number,
        exc_name,
        exc_msg,
        source_info,
    )
