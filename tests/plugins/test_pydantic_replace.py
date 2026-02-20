"""Unit tests for pydantic_replace mypy plugin."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from mypy.options import Options
from mypy.plugin import ReportConfigContext

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.skipif(sys.version_info < (3, 13), reason='Plugin targets Python 3.13+')
class TestReportConfigData:
    """Verify cache invalidation metadata."""

    def test_includes_plugin_version(self) -> None:
        """Plugin version in config data ensures mypy cache invalidation."""
        from plugins.pydantic_replace import PydanticReplacePlugin

        plugin = PydanticReplacePlugin(_make_options())
        ctx = ReportConfigContext(id='test', path='test.py', is_check=False)
        data = plugin.report_config_data(ctx)

        assert 'pydantic_replace_version' in data
        assert data['pydantic_replace_version'] == 1

    def test_preserves_pydantic_config_data(self) -> None:
        """Verify we don't clobber parent plugin's config data."""
        from plugins.pydantic_replace import PydanticReplacePlugin

        plugin = PydanticReplacePlugin(_make_options())
        ctx = ReportConfigContext(id='test', path='test.py', is_check=False)
        data = plugin.report_config_data(ctx)

        assert data == {
            'init_forbid_extra': True,
            'init_typed': True,
            'warn_required_dynamic_aliases': False,
            'debug_dataclass_transform': False,
            'pydantic_replace_version': 1,
        }

    def test_report_config_data_returns_independent_copies(self) -> None:
        """Repeated calls return equal dicts without mutating parent state."""
        from plugins.pydantic_replace import PydanticReplacePlugin

        plugin = PydanticReplacePlugin(_make_options())
        ctx = ReportConfigContext(id='test', path='test.py', is_check=False)
        first = plugin.report_config_data(ctx)
        second = plugin.report_config_data(ctx)

        assert first == second
        assert first is not second
        # Parent's internal _plugin_data must not contain our key
        assert 'pydantic_replace_version' not in plugin._plugin_data


def _make_options() -> Options:
    """Create Options with config_file so PydanticPluginConfig reads [tool.pydantic-mypy]."""
    options = Options()
    options.config_file = str(REPO_ROOT / 'pyproject.toml')
    return options