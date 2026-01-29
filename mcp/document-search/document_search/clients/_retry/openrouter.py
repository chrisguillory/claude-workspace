"""OpenRouter-specific retry and circuit breaker helpers.

Private module - import from _retry package.
"""

from __future__ import annotations

import logging

import circuitbreaker
import httpx
import tenacity

from document_search.clients._retry.httpx_errors import is_retryable_httpx_error

__all__ = [
    'is_retryable_openrouter_error',
    'log_openrouter_retry',
    'openrouter_breaker',
]

logger = logging.getLogger(__name__)

# HTTP status codes that are transient and worth retrying
# 429: Rate limit exceeded
# 500: Internal server error
# 502: Bad gateway (provider failure)
# 503: Service unavailable
# 504: Gateway timeout
RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})

# Circuit breaker - opens after consecutive failures, hard fails until recovery
OPENROUTER_FAILURE_THRESHOLD = 10
OPENROUTER_RECOVERY_TIMEOUT = 60


def is_retryable_openrouter_error(exc: BaseException) -> bool:
    """Check if exception is a retryable transient error from OpenRouter.

    Retries on:
    - httpx transport errors (timeout, network issues)
    - HTTP 429 (rate limit)
    - HTTP 500/502/503 (server/provider errors)
    """
    if is_retryable_httpx_error(exc):
        return True

    # HTTP status errors with transient codes
    if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code in RETRYABLE_STATUS_CODES:
        return True

    return False


def log_openrouter_retry(retry_state: tenacity.RetryCallState) -> None:
    """Log OpenRouter retry attempt with exception details."""
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    if exc is None:
        return

    exc_name = type(exc).__name__
    exc_msg = str(exc)

    # Include status code for HTTP errors
    if isinstance(exc, httpx.HTTPStatusError):
        exc_msg = f'HTTP {exc.response.status_code}: {exc_msg}'

    logger.warning(f'[RETRY] OpenRouter embed attempt {retry_state.attempt_number} failed: {exc_name}: {exc_msg}')


def _openrouter_circuit_filter(thrown_type: type, thrown_value: BaseException) -> bool:
    """Only count retryable errors toward circuit breaker."""
    return is_retryable_openrouter_error(thrown_value)


openrouter_breaker = circuitbreaker.CircuitBreaker(
    failure_threshold=OPENROUTER_FAILURE_THRESHOLD,
    recovery_timeout=OPENROUTER_RECOVERY_TIMEOUT,
    expected_exception=_openrouter_circuit_filter,
    name='openrouter',
)
