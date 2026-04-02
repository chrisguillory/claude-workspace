"""Gemini-specific retry and circuit breaker helpers.

Private module - import from _retry package.
"""

from __future__ import annotations

import logging
from typing import Literal

import circuitbreaker
import google.genai.errors
import tenacity

from document_search.clients._retry.httpx_errors import is_retryable_httpx_error

__all__ = [
    'GeminiTransientErrorCategory',
    'gemini_breaker',
    'is_retryable_gemini_error',
    'log_gemini_retry',
]

logger = logging.getLogger(__name__)

# Circuit breaker - opens after consecutive failures, hard fails until recovery
GEMINI_FAILURE_THRESHOLD = 10
GEMINI_RECOVERY_TIMEOUT = 60

type GeminiTransientErrorCategory = Literal[
    'rate_limit',  # 429 — RESOURCE_EXHAUSTED
    'bad_gateway',  # 502 — Gemini backend returned invalid response
    'timeout',  # httpx transport errors — network/timeout issues
    'server_error',  # 500, 503, 504 — Gemini infrastructure issues
]


def is_retryable_gemini_error(exc: BaseException) -> bool:
    """Check if exception is a retryable transient error from Gemini.

    Delegates to _classify_transient_error — an error is retryable
    if and only if it has a transient category.
    """
    return _classify_transient_error(exc) is not None


def log_gemini_retry(retry_state: tenacity.RetryCallState) -> None:
    """Log Gemini retry attempt and track categorized transient errors."""
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    if exc is None:
        return

    # Track categorized transient errors on client instance (args[0] is self)
    if retry_state.args:
        category = _classify_transient_error(exc)
        if category is not None:
            client = retry_state.args[0]
            client.transient_errors[category] += 1

    exc_name = type(exc).__name__
    exc_msg = str(exc)

    logger.warning(f'[RETRY] Gemini embed attempt {retry_state.attempt_number} failed: {exc_name}: {exc_msg}')


def _classify_transient_error(exc: BaseException) -> GeminiTransientErrorCategory | None:
    """Classify a Gemini error into a transient category.

    google-genai throws httpx exceptions directly.
    """
    if isinstance(exc, google.genai.errors.APIError):
        if exc.code == 429:
            return 'rate_limit'
        if exc.code == 502:
            return 'bad_gateway'
        if isinstance(exc.code, int) and exc.code in {500, 503, 504}:
            return 'server_error'
        return None

    if is_retryable_httpx_error(exc):
        return 'timeout'

    return None


def _gemini_circuit_filter(thrown_type: type, thrown_value: BaseException) -> bool:
    """Only count retryable errors toward circuit breaker."""
    return _classify_transient_error(thrown_value) is not None


gemini_breaker = circuitbreaker.CircuitBreaker(
    failure_threshold=GEMINI_FAILURE_THRESHOLD,
    recovery_timeout=GEMINI_RECOVERY_TIMEOUT,
    expected_exception=_gemini_circuit_filter,
    name='gemini',
)
