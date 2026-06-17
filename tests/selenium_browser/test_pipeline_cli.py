"""Unit tests for the ``pipeline`` command's tool-name normalization (no browser/bridge)."""

from __future__ import annotations

import json
from typing import Any

import pytest
from selenium_browser.cli import main as cli_main
from typer.testing import CliRunner


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def sent_steps(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Capture the steps the command would POST to the bridge (bridge call stubbed)."""
    box: dict[str, Any] = {}

    def _stub(steps: object, on_error: str = 'stop') -> dict[str, Any]:
        box['steps'] = steps
        return {'status': 'completed', 'completed': 0, 'total': 0, 'elapsed_ms': 0, 'results': []}

    monkeypatch.setattr(cli_main, '_call_pipeline', _stub)
    return box


def _invoke(runner: CliRunner, payload: object) -> Any:  # noqa: ANN401 — payload shape varies per case
    return runner.invoke(cli_main.app, ['pipeline'], input=json.dumps(payload))


class TestPipelineToolNameNormalization:
    @pytest.mark.parametrize(
        ('tool_in', 'tool_out'),
        [
            ('wait-for-selector', 'wait_for_selector'),
            ('get-page-text', 'get_page_text'),
            ('navigate', 'navigate'),  # no separator — unchanged
            ('execute_javascript', 'execute_javascript'),  # already underscored — unchanged
        ],
    )
    def test_hyphenated_tool_becomes_underscored(
        self, runner: CliRunner, sent_steps: dict[str, Any], tool_in: str, tool_out: str
    ) -> None:
        result = _invoke(runner, {'steps': [{'tool': tool_in, 'params': {}}]})
        assert result.exit_code == 0, result.output
        assert sent_steps['steps'][0]['tool'] == tool_out

    def test_param_values_are_left_untouched(self, runner: CliRunner, sent_steps: dict[str, Any]) -> None:
        # Only `tool` is normalized — a hyphen inside a param value must survive.
        result = _invoke(runner, {'steps': [{'tool': 'wait-for-selector', 'params': {'css_selector': 'a-b#x-y'}}]})
        assert result.exit_code == 0, result.output
        assert sent_steps['steps'][0] == {'tool': 'wait_for_selector', 'params': {'css_selector': 'a-b#x-y'}}

    def test_bare_array_form_is_accepted(self, runner: CliRunner, sent_steps: dict[str, Any]) -> None:
        result = _invoke(runner, [{'tool': 'get-page-text'}])
        assert result.exit_code == 0, result.output
        assert sent_steps['steps'][0]['tool'] == 'get_page_text'

    def test_non_string_tool_rejected_before_dispatch(self, runner: CliRunner, sent_steps: dict[str, Any]) -> None:
        result = _invoke(runner, {'steps': [{'tool': 404}]})
        assert result.exit_code != 0
        assert 'steps' not in sent_steps  # rejected at the CLI boundary, never sent to the bridge

    def test_unknown_step_key_rejected(self, runner: CliRunner, sent_steps: dict[str, Any]) -> None:
        result = _invoke(runner, {'steps': [{'tool': 'navigate', 'bogus': 1}]})
        assert result.exit_code != 0
        assert 'steps' not in sent_steps
