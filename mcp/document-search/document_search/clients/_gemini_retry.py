"""Retry helpers for transient network errors in Gemini client.

Private module - not exported by the package.
"""

from __future__ import annotations

import logging

import httpx
import tenacity
from google.genai.errors import ServerError

__all__ = [
    'is_retryable_gemini_error',
    'log_gemini_retry',
]

logger = logging.getLogger(__name__)

# Server error codes that are transient and worth retrying
RETRYABLE_STATUS_CODES = {500, 502, 503, 504}


def is_retryable_gemini_error(exc: BaseException) -> bool:
    """Check if exception is a retryable transient error from Gemini.

    We retry on:
    - httpx transient errors (ReadError, WriteError, timeouts)
    - Server errors with retryable status codes (502, 503, 504)
    """
    # Direct httpx transient errors
    if isinstance(exc, (httpx.ReadError, httpx.WriteError, httpx.ReadTimeout, httpx.WriteTimeout)):
        return True

    # API server errors with transient status codes
    if isinstance(exc, ServerError) and exc.code in RETRYABLE_STATUS_CODES:
        return True

    return False


def log_gemini_retry(retry_state: tenacity.RetryCallState) -> None:
    """Log retry attempt with exception details."""
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    if exc is None:
        return

    exc_name = type(exc).__name__
    exc_msg = str(exc)

    logger.warning(f'[RETRY] Gemini embed attempt {retry_state.attempt_number} failed: {exc_name}: {exc_msg}')
