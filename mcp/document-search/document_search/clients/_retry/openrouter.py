"""OpenRouter-specific retry and circuit breaker helpers.

Private module - import from _retry package.
"""

from __future__ import annotations

import logging
from typing import Literal

import circuitbreaker
import httpx
import tenacity

from document_search.clients._retry.httpx_errors import is_retryable_httpx_error
from document_search.clients.openrouter_errors import OpenRouterAPIError

__all__ = [
    'OpenRouterTransientErrorCategory',
    'is_retryable_openrouter_error',
    'log_openrouter_retry',
    'openrouter_breaker',
]

logger = logging.getLogger(__name__)

# Circuit breaker - opens after consecutive failures, hard fails until recovery
OPENROUTER_FAILURE_THRESHOLD = 10
OPENROUTER_RECOVERY_TIMEOUT = 60

type OpenRouterTransientErrorCategory = Literal[
    'rate_limit',  # 429 — upstream provider or OpenRouter rate limit
    'provider_unavailable',  # 404 "No successful provider responses." — all providers failed
    'bad_gateway',  # 502 — upstream provider down or returned invalid response
    'timeout',  # 408 — provider cold start or slow inference
    'server_error',  # 500, 503, 504 — OpenRouter infrastructure issues
]


def is_retryable_openrouter_error(exc: BaseException) -> bool:
    """Check if exception is a retryable transient error from OpenRouter.

    Delegates to _classify_transient_error — an error is retryable
    if and only if it has a transient category.
    """
    return _classify_transient_error(exc) is not None


def log_openrouter_retry(retry_state: tenacity.RetryCallState) -> None:
    """Log retry attempt and track categorized transient errors."""
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

    if isinstance(exc, httpx.HTTPStatusError):
        exc_msg = f'HTTP {exc.response.status_code}: {exc_msg}'

    logger.warning('[RETRY] OpenRouter embed attempt %s failed: %s: %s', retry_state.attempt_number, exc_name, exc_msg)


def _classify_transient_error(exc: BaseException) -> OpenRouterTransientErrorCategory | None:
    """Classify an error into a transient category.

    Returns None if the error is not transient (permanent failures, unknown formats).
    Single source of truth for retryability — an error is retryable if and only if
    it has a transient category.

    Transient codes (can appear as HTTP status OR error.code inside HTTP 200 bodies):

    408 - Request timeout. Provider cold start or slow inference. Retry lets
          OpenRouter route to a warmer provider.
    429 - Rate limit exceeded. May include provider_name in error metadata
          identifying which upstream provider rate-limited.
    500 - Internal server error. Transient infrastructure issue.
    502 - Bad gateway. Upstream provider returned invalid response or is down.
          OpenRouter already attempted internal fallback before surfacing this.
    503 - Service unavailable. No provider currently meets routing requirements.
    504 - Gateway timeout. Provider didn't respond within OpenRouter's window.
    404 - Conditional. "No successful provider responses." only. OpenRouter
          exhausted its provider fallback chain — all providers were tried and
          all returned errors. Functionally a 503 but reported as 404 because
          providers exist in the routing table.

    Does NOT classify:
    - OpenRouterUnexpectedResponse (unknown format won't change on retry)
    - 404 with other messages ("No endpoints found for model") — permanent
    """
    if isinstance(exc, OpenRouterAPIError):
        if exc.code == 429:
            return 'rate_limit'
        if exc.code == 404:
            if exc.message == 'No successful provider responses.':
                return 'provider_unavailable'
            logger.debug('Non-retryable 404: %r', exc.message)
            return None
        if exc.code == 502:
            return 'bad_gateway'
        if exc.code == 408:
            return 'timeout'
        if isinstance(exc.code, int) and exc.code in {500, 503, 504}:
            return 'server_error'
        return None

    # Note: 404 is intentionally excluded here. OpenRouter's "No successful provider
    # responses" arrives as an error-in-body (HTTP 200 with code=404), not as an
    # HTTP 404 status. An actual HTTP 404 would mean the endpoint doesn't exist.
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        if code == 429:
            return 'rate_limit'
        if code == 502:
            return 'bad_gateway'
        if code == 408:
            return 'timeout'
        if code in {500, 503, 504}:
            return 'server_error'
        return None

    if is_retryable_httpx_error(exc):
        return 'timeout'

    return None


def _openrouter_circuit_filter(thrown_type: type, thrown_value: BaseException) -> bool:
    """Only count retryable errors toward circuit breaker."""
    return _classify_transient_error(thrown_value) is not None


openrouter_breaker = circuitbreaker.CircuitBreaker(
    failure_threshold=OPENROUTER_FAILURE_THRESHOLD,
    recovery_timeout=OPENROUTER_RECOVERY_TIMEOUT,
    expected_exception=_openrouter_circuit_filter,
    name='openrouter',
)
