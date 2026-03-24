"""Tests for OpenRouter API response parsing."""

from __future__ import annotations

import base64
from collections.abc import Mapping
from typing import Any

import numpy as np
import pytest
from document_search.clients.openrouter import _decode_embedding, _parse_embedding_response
from document_search.clients.openrouter_errors import OpenRouterAPIError, OpenRouterUnexpectedResponse


def _parse(body: Mapping[str, Any]) -> Any:
    return _parse_embedding_response(body, status_code=200, model='test/model', batch_size=1)


class TestParseSuccess:
    def test_float_format(self) -> None:
        body = {
            'data': [{'embedding': [0.1, 0.2, 0.3], 'index': 0}],
            'model': 'test/model',
            'usage': {'prompt_tokens': 10, 'total_tokens': 10},
        }
        result = _parse(body)
        assert len(result.data) == 1
        assert list(result.data[0].embedding) == [0.1, 0.2, 0.3]

    def test_base64_format(self) -> None:
        vector = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        encoded = base64.b64encode(vector.tobytes()).decode()
        body = {
            'data': [{'embedding': encoded, 'index': 0}],
            'model': 'test/model',
            'usage': {'prompt_tokens': 10, 'total_tokens': 10},
        }
        result = _parse(body)
        assert isinstance(result.data[0].embedding, str)

    def test_int_in_float_array(self) -> None:
        """JSON integer 0 must not crash strict mode."""
        body = {
            'data': [{'embedding': [0, 1.5, -0.3], 'index': 0}],
            'model': 'test/model',
            'usage': {'prompt_tokens': 10, 'total_tokens': 10},
        }
        result = _parse(body)
        assert result.data[0].embedding[0] == 0.0

    def test_multiple_embeddings_preserve_order(self) -> None:
        body = {
            'data': [
                {'embedding': [0.1], 'index': 1},
                {'embedding': [0.2], 'index': 0},
            ],
            'model': 'test/model',
            'usage': {'prompt_tokens': 10, 'total_tokens': 10},
        }
        result = _parse(body)
        assert result.data[0].index == 1
        assert result.data[1].index == 0

    def test_usage_tracked(self) -> None:
        body = {
            'data': [{'embedding': [0.1], 'index': 0}],
            'model': 'test/model',
            'usage': {'prompt_tokens': 42, 'total_tokens': 100},
        }
        result = _parse(body)
        assert result.usage.total_tokens == 100
        assert result.usage.prompt_tokens == 42


class TestParseError:
    def test_error_body_raises_api_error(self) -> None:
        body = {'error': {'message': 'rate limited', 'code': 429}}
        with pytest.raises(OpenRouterAPIError) as exc_info:
            _parse(body)
        assert exc_info.value.code == 429
        assert 'rate limited' in str(exc_info.value)

    def test_error_without_code(self) -> None:
        body = {'error': {'message': 'unknown error'}}
        with pytest.raises(OpenRouterAPIError) as exc_info:
            _parse(body)
        assert exc_info.value.code is None

    def test_error_with_type(self) -> None:
        body = {'error': {'message': 'bad request', 'code': 400, 'type': 'invalid_request_error'}}
        with pytest.raises(OpenRouterAPIError) as exc_info:
            _parse(body)
        assert exc_info.value.error_type == 'invalid_request_error'


class TestParseUnknown:
    def test_unknown_format(self) -> None:
        body = {'unexpected': 'format'}
        with pytest.raises(OpenRouterUnexpectedResponse) as exc_info:
            _parse(body)
        assert 'unexpected' in exc_info.value.body_keys

    def test_empty_body(self) -> None:
        with pytest.raises(OpenRouterUnexpectedResponse):
            _parse({})

    def test_body_preview_included(self) -> None:
        body = {'something': 'weird', 'another': 'field'}
        with pytest.raises(OpenRouterUnexpectedResponse) as exc_info:
            _parse(body)
        assert 'something' in exc_info.value.body_preview


class TestDecodeEmbedding:
    def test_base64_roundtrip(self) -> None:
        original = [0.1, 0.2, 0.3]
        encoded = base64.b64encode(np.array(original, dtype=np.float32).tobytes()).decode()
        decoded = _decode_embedding(encoded)
        assert decoded == pytest.approx(original)

    def test_float_passthrough(self) -> None:
        original = [0.1, 0.2, 0.3]
        decoded = _decode_embedding(original)
        assert list(decoded) == pytest.approx(original)
