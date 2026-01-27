"""Qdrant-specific retry helpers.

Private module - import from _retry package.
"""

from __future__ import annotations

import logging

import qdrant_client.http.exceptions
import tenacity

from document_search.clients._retry.httpx_errors import is_retryable_httpx_error

__all__ = [
    'is_retryable_qdrant_error',
    'log_qdrant_retry',
]

logger = logging.getLogger(__name__)


def is_retryable_qdrant_error(exc: BaseException) -> bool:
    """Check if exception is a retryable transient error from Qdrant.

    qdrant-client wraps httpx exceptions in ResponseHandlingException.
    Check exc.source for the underlying httpx error.
    """
    if not isinstance(exc, qdrant_client.http.exceptions.ResponseHandlingException):
        return False

    return is_retryable_httpx_error(exc.source)


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
        f'[RETRY] Qdrant upsert attempt {retry_state.attempt_number} failed: {exc_name}: {exc_msg}{source_info}'
    )
