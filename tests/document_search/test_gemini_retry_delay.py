"""Gemini RetryInfo.retryDelay parsing and gemini_wait branch selection.

Hard-to-stage: a real 429 carrying a google.rpc.RetryInfo requires exhausting a
live quota. These synthetic APIErrors pin the parser's structural navigation and
the wait function's "honor server hint vs. fall back to backoff" branch — the
behavior the dynamic rate-limit model depends on.
"""

from __future__ import annotations

import google.genai.errors
import tenacity
from document_search.clients._retry.gemini import _retry_delay_seconds, gemini_wait

# Cap from RATE_LIMIT_WAIT(max=16) that gemini_wait applies to the server hint.
WAIT_CAP = 16


def test_well_formed_retry_info_returns_seconds() -> None:
    """A valid RetryInfo.retryDelay parses to its duration in seconds."""
    exc = _make_429(_retry_info_body(retry_delay='17s'))
    assert _retry_delay_seconds(exc) == 17.0


def test_non_dict_details_returns_none() -> None:
    """exc.details that isn't a mapping (string body) yields None."""
    assert _retry_delay_seconds(_make_429('rate limit exceeded')) is None


def test_missing_retry_info_entry_returns_none() -> None:
    """A 429 whose details list has no RetryInfo entry yields None."""
    body = {
        'error': {
            'status': 'RESOURCE_EXHAUSTED',
            'details': [{'@type': 'type.googleapis.com/google.rpc.QuotaFailure', 'violations': []}],
        }
    }
    assert _retry_delay_seconds(_make_429(body)) is None


def test_absent_retry_delay_returns_none() -> None:
    """A RetryInfo entry lacking retryDelay yields None."""
    exc = _make_429(_retry_info_body(retry_delay=None))
    assert _retry_delay_seconds(exc) is None


def test_gemini_wait_honors_server_hint() -> None:
    """gemini_wait returns at least the server delay and never exceeds the cap + 1s jitter."""
    exc = _make_429(_retry_info_body(retry_delay='5s'))
    state = tenacity.RetryCallState(retry_object=None, fn=None, args=(), kwargs={})  # type: ignore[arg-type]  # test seam: only outcome is read
    future: tenacity.Future = tenacity.Future(attempt_number=1)
    future.set_exception(exc)
    state.outcome = future

    wait = gemini_wait(state)
    assert wait >= 5.0  # honors the 5s hint
    assert wait <= WAIT_CAP + 1  # capped at 16s, +[0,1) jitter


def _make_429(details: object) -> google.genai.errors.APIError:
    """A Gemini 429 whose parsed body (exc.details) is `details`."""
    return google.genai.errors.APIError(429, details)


def _retry_info_body(*, retry_delay: str | None) -> dict[str, object]:
    """A RESOURCE_EXHAUSTED body carrying QuotaFailure + RetryInfo (Gemini's shape)."""
    retry_info: dict[str, object] = {'@type': 'type.googleapis.com/google.rpc.RetryInfo'}
    if retry_delay is not None:
        retry_info['retryDelay'] = retry_delay
    return {
        'error': {
            'code': 429,
            'status': 'RESOURCE_EXHAUSTED',
            'details': [
                {'@type': 'type.googleapis.com/google.rpc.QuotaFailure', 'violations': []},
                retry_info,
            ],
        }
    }
