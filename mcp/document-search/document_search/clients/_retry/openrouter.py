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

# HTTP status codes that indicate transient provider issues worth retrying.
# These can appear as HTTP status codes OR as `error.code` inside HTTP 200 bodies.
#
# 408: Request timeout. Provider cold start or slow inference. Retry lets
#      OpenRouter route to a warmer provider.
# 429: Rate limit exceeded. Back off and retry. May include provider_name
#      in error metadata identifying which upstream provider rate-limited.
# 500: Internal server error. Transient infrastructure issue.
# 502: Bad gateway. Upstream provider returned invalid response or is down.
#      OpenRouter already attempted internal fallback before surfacing this.
# 503: Service unavailable. No provider currently meets routing requirements.
# 504: Gateway timeout. Provider didn't respond within OpenRouter's window.
RETRYABLE_STATUS_CODES = frozenset({408, 429, 500, 502, 503, 504})

# Circuit breaker - opens after consecutive failures, hard fails until recovery
OPENROUTER_FAILURE_THRESHOLD = 10
OPENROUTER_RECOVERY_TIMEOUT = 60

type OpenRouterTransientErrorCategory = Literal[
    'rate_limit',  # 429 — upstream provider or OpenRouter rate limit
    'provider_unavailable',  # 404 "No successful provider responses." — all providers failed
    'provider_error',  # 502 — upstream provider down or returned invalid response
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

    logger.warning(f'[RETRY] OpenRouter embed attempt {retry_state.attempt_number} failed: {exc_name}: {exc_msg}')


def _classify_transient_error(exc: BaseException) -> OpenRouterTransientErrorCategory | None:
    """Classify an error into a transient category.

    Returns None if the error is not transient (permanent failures, unknown formats).
    Single source of truth for retryability — an error is retryable if and only if
    it has a transient category.

    Does NOT classify:
    - OpenRouterUnexpectedResponse (unknown format won't change on retry)
    - 404 with other messages ("No endpoints found for model") — permanent
    """
    if isinstance(exc, OpenRouterAPIError):
        if exc.code == 429:
            return 'rate_limit'
        # 404 with this exact message = OpenRouter exhausted its provider fallback chain.
        # All providers were tried and all returned errors. This is functionally a 503
        # (transient provider unavailability) but reported as 404 because providers exist
        # in the routing table — they just all failed this request.
        if exc.code == 404 and exc.message == 'No successful provider responses.':
            return 'provider_unavailable'
        if exc.code == 502:
            return 'provider_error'
        if exc.code == 408:
            return 'timeout'
        if isinstance(exc.code, int) and exc.code in {500, 503, 504}:
            return 'server_error'
        return None

    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        if code == 429:
            return 'rate_limit'
        if code == 502:
            return 'provider_error'
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
