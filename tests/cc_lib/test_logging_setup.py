"""Tests for cc_lib.logging_setup -- CC_LOG parser + configure_logging."""

from __future__ import annotations

import logging

import pytest
from cc_lib.logging_setup import _parse_cc_log, configure_logging


class TestParseCcLog:
    """CC_LOG grammar edges that empirical use won't reliably expose."""

    def test_empty(self) -> None:
        assert _parse_cc_log('') == (None, {})

    def test_mixed_rules(self) -> None:
        root, per_logger = _parse_cc_log('warning,cc_lib.mcp=debug,uvicorn.access=error')
        assert (root, dict(per_logger)) == (
            'WARNING',
            {'cc_lib.mcp': 'DEBUG', 'uvicorn.access': 'ERROR'},
        )

    def test_whitespace_tolerated(self) -> None:
        root, per_logger = _parse_cc_log(' warning , cc_lib.mcp.bridge = debug ')
        assert (root, dict(per_logger)) == ('WARNING', {'cc_lib.mcp.bridge': 'DEBUG'})

    def test_levels_uppercased(self) -> None:
        _, per_logger = _parse_cc_log('foo=DeBuG')
        assert dict(per_logger) == {'foo': 'DEBUG'}

    def test_empty_segments_skipped(self) -> None:
        root, per_logger = _parse_cc_log(',,warning,,foo=info,')
        assert (root, dict(per_logger)) == ('WARNING', {'foo': 'INFO'})

    def test_duplicate_logger_last_wins(self) -> None:
        _, per_logger = _parse_cc_log('foo=info,foo=debug')
        assert dict(per_logger) == {'foo': 'DEBUG'}


class TestConfigureLogging:
    def test_env_var_sets_per_module_levels(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The dispositive integration: per-module override flows env → setLevel."""
        monkeypatch.setenv('CC_LOG', 'warning,cc_lib.mcp.bridge=debug')
        configure_logging()
        assert logging.getLogger().level == logging.WARNING
        assert logging.getLogger('cc_lib.mcp.bridge').level == logging.DEBUG


@pytest.fixture(autouse=True)
def _reset_root_logger() -> object:
    yield
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
    root.setLevel(logging.WARNING)
