"""Retry helpers for transient network errors.

Private module - not exported by the package.
"""

from __future__ import annotations

import logging

import httpx
import tenacity
from qdrant_client.http.exceptions import ResponseHandlingException

logger = logging.getLogger(__name__)

__all__ = [
    'is_retryable_qdrant_error',
    'log_qdrant_retry',
]


def is_retryable_qdrant_error(exc: BaseException) -> bool:
    """Check if exception is a retryable transient error from Qdrant.

    qdrant-client wraps all exceptions in ResponseHandlingException.
    We only retry on transient network errors, not API errors.
    """
    if not isinstance(exc, ResponseHandlingException):
        return False

    source = exc.source
    # Retryable transient errors
    return isinstance(source, (httpx.ReadError, httpx.WriteError, httpx.ReadTimeout, httpx.WriteTimeout))


def log_qdrant_retry(retry_state: tenacity.RetryCallState) -> None:
    """Log retry attempt with exception details, including source for ResponseHandlingException."""
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    if exc is None:
        return

    exc_name = type(exc).__name__
    exc_msg = str(exc)

    # ResponseHandlingException wraps the underlying error in .source
    source_info = ''
    if isinstance(exc, ResponseHandlingException) and exc.source:
        source_info = f' (source: {type(exc.source).__name__}: {exc.source})'

    logger.warning(
        f'[RETRY] Qdrant upsert attempt {retry_state.attempt_number} failed: {exc_name}: {exc_msg}{source_info}'
    )
