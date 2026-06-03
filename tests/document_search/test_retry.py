"""Regression: the dead OpenRouter circuit breaker.

Pins the pre-existing bug where the inner tenacity.retry lacked reraise=True, so
exhaustion raised tenacity.RetryError — which the breaker filter classifies as a
non-failure (the false-negative), treating the exhausted call as a success and
resetting the counter, so the breaker never opened.

With reraise=True the concrete transient reaches the filter, the failure count
climbs, and the breaker opens at threshold. Synthetic-exception based because a
real sustained outage is hard to stage. Doubles as a guard on the
circuitbreaker/tenacity dependency floors (a reset/wrapping-semantics change trips it).
"""

from __future__ import annotations

import collections.abc

import circuitbreaker
import document_search.clients._retry.openrouter as orr
import pytest
import tenacity
from document_search.clients._retry import is_retryable_openrouter_error, openrouter_breaker
from document_search.clients._retry.openrouter import _openrouter_circuit_filter
from document_search.clients.openrouter import OpenRouterClient
from document_search.clients.openrouter_errors import OpenRouterAPIError

# Low threshold so the breaker trips after a few synthetic exhaustions (default is 10).
TEST_THRESHOLD = 3


@pytest.fixture
def instant_breaker(monkeypatch: pytest.MonkeyPatch) -> collections.abc.Iterator[None]:
    """Zero the backoff and start from a clean, low-threshold breaker."""
    monkeypatch.setattr(orr, 'RATE_LIMIT_WAIT', tenacity.wait_none())
    monkeypatch.setattr(orr, 'DEFAULT_WAIT', tenacity.wait_none())
    openrouter_breaker.reset()
    monkeypatch.setattr(openrouter_breaker, '_failure_threshold', TEST_THRESHOLD)
    yield
    openrouter_breaker.reset()


def test_filter_false_negative_on_retry_error() -> None:
    """The root cause: the filter counts the concrete transient but NOT RetryError.

    Without reraise=True, exhaustion surfaces RetryError -> filter returns False
    -> breaker treats the call as a success. This asymmetry is the dead-breaker bug.
    """
    concrete = _make_429()
    assert _openrouter_circuit_filter(type(concrete), concrete) is True
    assert is_retryable_openrouter_error(concrete) is True

    future: tenacity.Future = tenacity.Future(attempt_number=1)
    future.set_exception(concrete)
    retry_error = tenacity.RetryError(last_attempt=future)
    assert _openrouter_circuit_filter(type(retry_error), retry_error) is False


@pytest.mark.asyncio
async def test_exhausted_retryable_opens_breaker(instant_breaker: None) -> None:
    """reraise=True: an exhausted retryable error is counted; the breaker opens at threshold."""
    client = OpenRouterClient('dummy/model', dimensions=1536, api_key='dummy-key-not-used')

    async def always_429(*_args: object, **_kwargs: object) -> object:
        raise _make_429()

    # Override the bound httpx POST seam the retry wraps.
    _monkeypatch_post(client, always_429)

    try:
        # Each embed() exhausts the retry budget and re-raises the concrete 429
        # (not RetryError), which the breaker counts. At the threshold it opens.
        for _ in range(TEST_THRESHOLD):
            with pytest.raises(OpenRouterAPIError):
                await client.embed(['hi'], intent='document')

        assert openrouter_breaker.failure_count == TEST_THRESHOLD
        assert openrouter_breaker.opened is True

        # Once open, the next call fast-fails with CircuitBreakerError BEFORE the network.
        network_hit = False

        async def tripwire(*_args: object, **_kwargs: object) -> object:
            nonlocal network_hit
            network_hit = True
            raise _make_429()

        _monkeypatch_post(client, tripwire)
        with pytest.raises(circuitbreaker.CircuitBreakerError):
            await client.embed(['hi'], intent='document')
        assert network_hit is False
    finally:
        await client.close()


def _monkeypatch_post(client: OpenRouterClient, fn: collections.abc.Callable[..., object]) -> None:
    """Swap the client's bound httpx POST — the seam the inner tenacity.retry wraps."""
    client._client.post = fn  # type: ignore[method-assign,assignment]  # test seam: replace the bound httpx method with a stub


def _make_429() -> OpenRouterAPIError:
    """A retryable rate-limit error the classifier recognizes."""
    return OpenRouterAPIError(
        message='Rate limit exceeded: engine overloaded',
        code=429,
        error_type='rate_limit',
        status_code=429,
        model='dummy/model',
    )
