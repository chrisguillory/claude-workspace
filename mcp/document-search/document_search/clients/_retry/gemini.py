"""Gemini-specific retry and circuit breaker helpers.

Private module - import from _retry package.
"""

from __future__ import annotations

import logging

import circuitbreaker
import google.genai.errors
import tenacity

from document_search.clients._retry.httpx_errors import is_retryable_httpx_error

__all__ = [
    'gemini_breaker',
    'is_retryable_gemini_error',
    'log_gemini_retry',
]

logger = logging.getLogger(__name__)

# Status codes that are transient and worth retrying
# 429: Rate limit exceeded (RESOURCE_EXHAUSTED)
# 500/502/503/504: Server errors
RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})

# Circuit breaker - opens after consecutive failures, hard fails until recovery
GEMINI_FAILURE_THRESHOLD = 10
GEMINI_RECOVERY_TIMEOUT = 60


def is_retryable_gemini_error(exc: BaseException) -> bool:
    """Check if exception is a retryable transient error from Gemini.

    google-genai throws httpx exceptions directly.
    Retries on:
    - httpx transport errors (timeout, network issues)
    - HTTP 429 (rate limit / RESOURCE_EXHAUSTED)
    - HTTP 500/502/503/504 (server errors)
    """
    if is_retryable_httpx_error(exc):
        return True

    # API errors with retryable status codes (ClientError for 429, ServerError for 5xx)
    if isinstance(exc, google.genai.errors.APIError) and exc.code in RETRYABLE_STATUS_CODES:
        return True

    return False


def log_gemini_retry(retry_state: tenacity.RetryCallState) -> None:
    """Log Gemini retry attempt and track 429 errors."""
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    if exc is None:
        return

    # Track 429s on client instance (args[0] is self for instance methods)
    if retry_state.args and _is_429_error(exc):
        client = retry_state.args[0]
        client.errors_429 += 1

    exc_name = type(exc).__name__
    exc_msg = str(exc)

    logger.warning(f'[RETRY] Gemini embed attempt {retry_state.attempt_number} failed: {exc_name}: {exc_msg}')


def _is_429_error(exc: BaseException | None) -> bool:
    """Check if exception is a 429 rate limit error."""
    if exc is None:
        return False
    return isinstance(exc, google.genai.errors.APIError) and exc.code == 429


def _gemini_circuit_filter(thrown_type: type, thrown_value: BaseException) -> bool:
    """Only count retryable errors toward circuit breaker."""
    return is_retryable_gemini_error(thrown_value)


gemini_breaker = circuitbreaker.CircuitBreaker(
    failure_threshold=GEMINI_FAILURE_THRESHOLD,
    recovery_timeout=GEMINI_RECOVERY_TIMEOUT,
    expected_exception=_gemini_circuit_filter,
    name='gemini',
)
