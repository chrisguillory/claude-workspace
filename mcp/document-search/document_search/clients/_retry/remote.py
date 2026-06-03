"""Category-aware backoff shared by the remote embedding providers.

Private module - import from _retry package.
"""

from __future__ import annotations

from collections.abc import Callable

import tenacity
from tenacity.stop import stop_base
from tenacity.wait import wait_base

__all__ = [
    'DEFAULT_STOP',
    'DEFAULT_WAIT',
    'RATE_LIMIT_STOP',
    'RATE_LIMIT_WAIT',
    'TransientClassifier',
    'category_aware_stop',
    'category_aware_wait',
]

# Returns the provider's transient category for an exception, or None if permanent.
# Generic over the provider's category Literal so the precise type survives dispatch
# (a typo'd `== 'raate_limit'` then fails mypy comparison-overlap instead of silently passing).
type TransientClassifier[CatT: str] = Callable[[BaseException], CatT | None]

# Shared backoff profile for the remote providers. A rate-limit overload persists
# for seconds-to-minutes and is signaled without a Retry-After header (OpenRouter
# sends an HTTP-200 body error), so the backoff is blind: a deeper, jittered window
# than a transient blip, with full jitter to desynchronize parallel fan-out so a
# fleet of indexers does not retry in lockstep and re-amplify the overload. Both
# remote providers share this one profile — converging gemini onto openrouter.
RATE_LIMIT_MAX_ATTEMPTS = 6
DEFAULT_MAX_ATTEMPTS = 3

RATE_LIMIT_WAIT = tenacity.wait_random_exponential(multiplier=1, max=16)
DEFAULT_WAIT = tenacity.wait_random_exponential(multiplier=0.5, max=5)
RATE_LIMIT_STOP = tenacity.stop_after_attempt(RATE_LIMIT_MAX_ATTEMPTS)
DEFAULT_STOP = tenacity.stop_after_attempt(DEFAULT_MAX_ATTEMPTS)


def category_aware_wait[CatT: str](
    retry_state: tenacity.RetryCallState,
    *,
    classifier: TransientClassifier[CatT],
    rate_limit_wait: wait_base,
    default_wait: wait_base,
) -> float:
    """Deep + jittered backoff for rate limits, fast + jittered otherwise."""
    wait = rate_limit_wait if _is_rate_limited(retry_state, classifier) else default_wait
    return wait(retry_state)


def category_aware_stop[CatT: str](
    retry_state: tenacity.RetryCallState,
    *,
    classifier: TransientClassifier[CatT],
    rate_limit_stop: stop_base,
    default_stop: stop_base,
) -> bool:
    """More attempts for a sustained rate limit than for a transient blip."""
    stop = rate_limit_stop if _is_rate_limited(retry_state, classifier) else default_stop
    return stop(retry_state)


def _is_rate_limited[CatT: str](retry_state: tenacity.RetryCallState, classifier: TransientClassifier[CatT]) -> bool:
    """Whether the most recent attempt failed with a rate-limit overload."""
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    return exc is not None and classifier(exc) == 'rate_limit'
