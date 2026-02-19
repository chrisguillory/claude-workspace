"""Unit tests for pydantic_replace mypy plugin."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest


@pytest.mark.skipif(sys.version_info < (3, 13), reason='Plugin targets Python 3.13+')
class TestReportConfigData:
    """Verify cache invalidation metadata."""

    def test_includes_plugin_version(self) -> None:
        """Plugin version in config data ensures mypy cache invalidation."""
        from plugins.pydantic_replace import PydanticReplacePlugin

        plugin = PydanticReplacePlugin(MagicMock())
        ctx = MagicMock()
        data = plugin.report_config_data(ctx)

        assert 'pydantic_replace_version' in data
        assert data['pydantic_replace_version'] == 1

    def test_preserves_pydantic_config_data(self) -> None:
        """Verify we don't clobber parent plugin's config data."""
        from plugins.pydantic_replace import PydanticReplacePlugin

        plugin = PydanticReplacePlugin(MagicMock())
        ctx = MagicMock()
        data = plugin.report_config_data(ctx)

        # Should contain both pydantic's config and our version
        assert 'pydantic_replace_version' in data
        assert isinstance(data, dict)
