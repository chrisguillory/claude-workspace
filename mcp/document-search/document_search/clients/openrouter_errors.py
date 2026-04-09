from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import httpx

logger = logging.getLogger(__name__)

__all__ = [
    'OpenRouterAPIError',
    'OpenRouterEmptyResponse',
    'OpenRouterTruncatedResponse',
    'OpenRouterUnexpectedResponse',
]


class OpenRouterAPIError(Exception):
    """Known error from OpenRouter API (has message, code).

    May arrive with any HTTP status, including 200.
    Retryable if classified as transient by _retry.openrouter.
    """

    def __init__(
        self,
        *,
        message: str,
        code: int | None,
        error_type: str | None,
        status_code: int,
        model: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.message = message
        self.code = code
        self.error_type = error_type
        self.status_code = status_code
        self.model = model
        self.metadata = metadata

        parts = [f'HTTP {status_code}']
        if code is not None:
            parts.append(f'code={code}')
        parts.append(f'model={model}')
        if error_type is not None:
            parts.append(f'type={error_type}')
        line = f'OpenRouter API error ({", ".join(parts)}): {message}'
        if metadata:
            line += f'\n  metadata: {dict(metadata)}'
        super().__init__(line)


class OpenRouterEmptyResponse(Exception):
    """OpenRouter returned HTTP 200 with an empty or whitespace-only body.

    Root cause: Cloudflare Worker committed 200 headers, then the upstream
    provider failed before generating any response content. The Worker
    cannot change the status code after headers are sent.

    Distinct from truncation (mid-body cutoff) — this is a pre-body failure.
    Transient: retrying routes to a different provider instance.

    References:
    - Cloudflare community: HTTP 200 with empty body
      https://community.cloudflare.com/t/cloudflared-sometimes-respond-http-status-200-with-empty-body/447257
    - CherryHQ/cherry-studio#13863: OpenRouter empty responses
    - chatboxai/chatbox#2688: Intermittent empty responses, 1 in 5-10
    """

    def __init__(
        self,
        *,
        raw_body_length: int,
        status_code: int,
        batch_size: int,
        model: str,
    ) -> None:
        self.raw_body_length = raw_body_length
        self.status_code = status_code
        self.batch_size = batch_size
        self.model = model
        super().__init__(
            f'Empty response from OpenRouter '
            f'(body={raw_body_length:,}b whitespace, status={status_code}, '
            f'batch_size={batch_size}, model={model})'
        )


class OpenRouterTruncatedResponse(Exception):
    """OpenRouter returned a truncated JSON response (valid HTTP, incomplete body).

    Root cause: OpenRouter runs on Cloudflare Workers, which enforce CPU time
    limits per request. When a Worker exceeds its quota while streaming a
    chunked response, Cloudflare cleanly terminates the HTTP transfer — valid
    status code, valid chunked encoding terminator — but the JSON body is
    cut off mid-content. The client receives HTTP 200 with a well-formed
    transfer that contains invalid JSON.

    This is transient: the same request succeeds on retry because it may hit
    a different Worker instance with fresh CPU budget, or the same instance
    under less load.

    Detection: JSONDecodeError where the error position is near the end of
    a large response body (truncation_ratio > 0.8), the response starts with
    valid JSON, and the HTTP status is 200.

    Mitigation: Retried by tenacity (classified as 'truncated_response').
    Triggers AIMD batch size reduction to decrease response size on future
    requests.

    References:
    - Cloudflare Workers CPU limits:
      https://developers.cloudflare.com/workers/platform/limits/#cpu-time
    - cline/cline#60: "Invalid JSON response body" confirmed OpenRouter-specific
    - blakeblackshear/frigate#17571: ~60% of embedding API calls crash
    - openai/openai-python#235: Large batch JSONDecodeError, small batches work
    - Cloudflare community on streaming truncation:
      https://community.cloudflare.com/t/streaming-response-catch-worker-exceeded-cpu-time-limit-error/372138
    """

    # Truncation signature: error position must be in the last 20% of the body.
    # Cloudflare truncation cuts near the end of multi-MB responses.
    # Garbled/corrupt responses fail early (ratio < 0.5).
    TRUNCATION_RATIO_THRESHOLD = 0.8

    def __init__(
        self,
        *,
        raw_body_length: int,
        json_error_position: int,
        truncation_ratio: float,
        status_code: int,
        batch_size: int,
        model: str,
    ) -> None:
        self.raw_body_length = raw_body_length
        self.json_error_position = json_error_position
        self.truncation_ratio = truncation_ratio
        self.status_code = status_code
        self.batch_size = batch_size
        self.model = model
        super().__init__(
            f'Truncated JSON response from OpenRouter '
            f'(body={raw_body_length:,}b, error at {json_error_position:,}b, '
            f'ratio={truncation_ratio:.2%}, batch_size={batch_size}, model={model})'
        )

    @classmethod
    def from_json_error(
        cls,
        *,
        json_error: json.JSONDecodeError,
        response: httpx.Response,
        model: str,
        batch_size: int,
    ) -> OpenRouterTruncatedResponse | None:
        """Identify Cloudflare Worker truncation from a JSONDecodeError.

        Returns an instance if the error matches the truncation signature,
        or None if it doesn't (caller should fall through to
        OpenRouterUnexpectedResponse for permanent/unknown failures).
        """
        body = response.content
        error_pos = json_error.pos
        body_length = len(body)

        if body_length == 0:
            return None

        ratio = error_pos / body_length
        first_char = body.lstrip()[:1]
        valid_start = first_char in (b'{', b'[')

        if ratio < cls.TRUNCATION_RATIO_THRESHOLD or not valid_start or response.status_code != 200:
            body_text = body.decode(errors='replace')
            logger.warning(
                '[TRUNCATION-MISS] JSONDecodeError did not match truncation signature: '
                'ratio=%.3f (threshold=%.1f), valid_start=%s, status=%d, '
                'body=%d bytes, error_pos=%d, model=%s\n'
                '  Body:\n%s',
                ratio,
                cls.TRUNCATION_RATIO_THRESHOLD,
                valid_start,
                response.status_code,
                body_length,
                error_pos,
                model,
                body_text,
            )
            return None

        return cls(
            raw_body_length=body_length,
            json_error_position=error_pos,
            truncation_ratio=ratio,
            status_code=response.status_code,
            batch_size=batch_size,
            model=model,
        )


class OpenRouterUnexpectedResponse(Exception):
    """Unknown response format — neither success nor recognized error.

    NOT retryable. Carries diagnostic context for actionable debugging.
    """

    def __init__(
        self,
        *,
        message: str,
        body_preview: str,
        body_keys: Sequence[str],
        status_code: int,
        model: str,
        batch_size: int,
    ) -> None:
        self.body_preview = body_preview
        self.body_keys = body_keys
        self.status_code = status_code
        self.model = model
        self.batch_size = batch_size
        super().__init__(
            f'{message}\n'
            f'  Status: {status_code}\n'
            f'  Model: {model}, Batch size: {batch_size}\n'
            f'  Response keys: {list(body_keys)}\n'
            f'  Body preview: {body_preview}'
        )
