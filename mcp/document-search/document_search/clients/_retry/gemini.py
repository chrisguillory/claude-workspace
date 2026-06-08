"""Gemini-specific retry and circuit breaker helpers.

Private module - import from _retry package.
"""

from __future__ import annotations

import logging
import random
from collections.abc import Mapping
from typing import Literal

import google.genai.errors
import google.protobuf.duration_pb2
import tenacity

from document_search.clients._retry.breaker import LoggingCircuitBreaker
from document_search.clients._retry.httpx_errors import is_retryable_httpx_error
from document_search.clients._retry.remote import (
    DEFAULT_STOP,
    DEFAULT_WAIT,
    RATE_LIMIT_STOP,
    RATE_LIMIT_WAIT,
    category_aware_stop,
    category_aware_wait,
)

__all__ = [
    'GeminiTransientErrorCategory',
    'gemini_breaker',
    'gemini_stop',
    'gemini_wait',
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
            client.on_transient_error(category)

    exc_name = type(exc).__name__
    exc_msg = str(exc)

    logger.warning('[RETRY] Gemini embed attempt %s failed: %s: %s', retry_state.attempt_number, exc_name, exc_msg)


def gemini_wait(retry_state: tenacity.RetryCallState) -> float:
    """Honor the server's RetryInfo.retryDelay hint; else category-aware backoff.

    A 429 carries a RetryInfo telling us how long to wait. When present, sleep that
    long (capped at the rate-limit ceiling, plus jitter to desynchronize parallel
    fan-out). Otherwise fall back to the shared category-aware profile: deep +
    jittered for rate limits, fast + jittered otherwise.
    """
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    if exc is not None:
        delay = _retry_delay_seconds(exc)
        if delay is not None:
            return min(delay, 16) + random.uniform(0, 1)

    return category_aware_wait(
        retry_state,
        classifier=_classify_transient_error,
        rate_limit_wait=RATE_LIMIT_WAIT,
        default_wait=DEFAULT_WAIT,
    )


def gemini_stop(retry_state: tenacity.RetryCallState) -> bool:
    """More attempts for a sustained rate limit than for a transient blip (no intent depth)."""
    return category_aware_stop(
        retry_state,
        classifier=_classify_transient_error,
        rate_limit_stop=RATE_LIMIT_STOP,
        default_stop=DEFAULT_STOP,
    )


def _retry_delay_seconds(exc: BaseException) -> float | None:
    """Extract the server's RetryInfo.retryDelay (seconds) from a 429, if present.

    A Gemini 429 (RESOURCE_EXHAUSTED) carries a google.rpc.RetryInfo in
    error.details telling us how long to wait. exc.details is the parsed JSON body;
    navigate to the RetryInfo entry and parse its protobuf Duration string (e.g.
    "17s"). A structural miss (no RetryInfo or unexpected shape) yields None and the
    caller falls back to category-aware backoff; a malformed Duration is left to raise.
    """
    if not (isinstance(exc, google.genai.errors.APIError) and exc.code == 429):
        return None

    details = exc.details
    if not isinstance(details, Mapping):
        return None
    error = details.get('error')
    if not isinstance(error, Mapping):
        return None
    detail_entries = error.get('details')
    if not isinstance(detail_entries, list):
        return None

    for entry in detail_entries:
        if not isinstance(entry, Mapping):
            continue
        type_url = entry.get('@type')
        if not (isinstance(type_url, str) and type_url.endswith('RetryInfo')):
            continue
        retry_delay = entry.get('retryDelay')
        if not isinstance(retry_delay, str):
            return None
        duration = google.protobuf.duration_pb2.Duration()
        duration.FromJsonString(retry_delay)
        return duration.ToTimedelta().total_seconds()

    return None


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


gemini_breaker = LoggingCircuitBreaker(
    failure_threshold=GEMINI_FAILURE_THRESHOLD,
    recovery_timeout=GEMINI_RECOVERY_TIMEOUT,
    expected_exception=_gemini_circuit_filter,
    name='gemini',
)
