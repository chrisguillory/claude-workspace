"""Gemini-specific retry helpers.

Private module - import from _retry package.
"""

from __future__ import annotations

import logging

import google.genai.errors
import tenacity

from document_search.clients._retry.httpx_errors import is_retryable_httpx_error

__all__ = [
    'is_retryable_gemini_error',
    'log_gemini_retry',
]

logger = logging.getLogger(__name__)

# Server error codes that are transient and worth retrying
RETRYABLE_STATUS_CODES = frozenset({500, 502, 503, 504})


def is_retryable_gemini_error(exc: BaseException) -> bool:
    """Check if exception is a retryable transient error from Gemini.

    google-genai throws httpx exceptions directly.
    Also retries ServerError with transient status codes (500/502/503/504).
    """
    if is_retryable_httpx_error(exc):
        return True

    # API server errors with transient status codes
    if isinstance(exc, google.genai.errors.ServerError) and exc.code in RETRYABLE_STATUS_CODES:
        return True

    return False


def log_gemini_retry(retry_state: tenacity.RetryCallState) -> None:
    """Log Gemini retry attempt with exception details."""
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    if exc is None:
        return

    exc_name = type(exc).__name__
    exc_msg = str(exc)

    logger.warning(f'[RETRY] Gemini embed attempt {retry_state.attempt_number} failed: {exc_name}: {exc_msg}')
