from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

__all__ = [
    'OpenRouterAPIError',
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
