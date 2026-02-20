"""Tests for the approve-compound-bash PreToolUse hook.

Validates bashlex-based compound command parsing, dangerous construct
detection, wrapper prefix stripping, prefix matching, and settings loading.
"""

from __future__ import annotations

import importlib.util
import io
import json
import sys
from collections.abc import Generator
from pathlib import Path
from types import ModuleType

import git
import pytest

REPO_ROOT = Path(git.Repo(__file__, search_parent_directories=True).working_tree_dir or '.').resolve(strict=True)


@pytest.fixture(scope='session')
def hook_module() -> Generator[ModuleType]:
    """Import approve-compound-bash.py (hyphenated filename requires importlib)."""
    path = REPO_ROOT / 'hooks' / 'approve-compound-bash.py'
    spec = importlib.util.spec_from_file_location('approve_compound_bash', path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules['approve_compound_bash'] = mod
    spec.loader.exec_module(mod)
    yield mod
    sys.modules.pop('approve_compound_bash', None)


# ---------------------------------------------------------------------------
# TestAnalyzeCommand — AST splitting correctness
# ---------------------------------------------------------------------------


class TestAnalyzeCommand:
    """Verify compound command decomposition via bashlex AST."""

    def test_simple_command(self, hook_module: ModuleType) -> None:
        result = hook_module.analyze_command('echo hello')
        assert len(result) == 1
        assert result[0].base_command == 'echo hello'

    def test_and_operator(self, hook_module: ModuleType) -> None:
        result = hook_module.analyze_command('echo "---" && git log')
        assert len(result) == 2
        assert result[0].base_command.startswith('echo')
        assert result[1].base_command.startswith('git log')

    def test_or_operator(self, hook_module: ModuleType) -> None:
        result = hook_module.analyze_command('echo test || git log')
        assert len(result) == 2

    def test_semicolon(self, hook_module: ModuleType) -> None:
        result = hook_module.analyze_command('echo test; git log')
        assert len(result) == 2

    def test_pipe(self, hook_module: ModuleType) -> None:
        result = hook_module.analyze_command('echo test | grep test')
        assert len(result) == 2

    def test_three_part_compound(self, hook_module: ModuleType) -> None:
        result = hook_module.analyze_command('git diff 2>&1 && echo "---" && git log')
        assert len(result) == 3

    def test_nested_quotes(self, hook_module: ModuleType) -> None:
        result = hook_module.analyze_command('echo \'hello "world"\' && git status')
        assert len(result) == 2

    def test_double_quotes(self, hook_module: ModuleType) -> None:
        result = hook_module.analyze_command('echo "hello world" && git log')
        assert len(result) == 2

    def test_env_var_assignment(self, hook_module: ModuleType) -> None:
        result = hook_module.analyze_command('VAR=val echo test && git status')
        assert len(result) == 2
        assert result[0].base_command == 'echo test'

    def test_newline_separated(self, hook_module: ModuleType) -> None:
        result = hook_module.analyze_command('echo test\ngit log')
        assert len(result) == 2

    def test_subshell(self, hook_module: ModuleType) -> None:
        result = hook_module.analyze_command('echo safe && (echo inner; ls)')
        assert len(result) == 3

    def test_brace_group(self, hook_module: ModuleType) -> None:
        result = hook_module.analyze_command('echo safe && { echo inner; ls; }')
        assert len(result) == 3

    def test_assignment_only(self, hook_module: ModuleType) -> None:
        """Assignment-only subcommand produces empty base_command."""
        result = hook_module.analyze_command('VAR=hello && echo test')
        assert len(result) == 2
        assert result[0].base_command == ''

    def test_if_then_visible(self, hook_module: ModuleType) -> None:
        """if/then blocks must not be silently dropped from analysis."""
        result = hook_module.analyze_command('echo safe && if true; then rm -rf /; fi')
        assert len(result) >= 2

    def test_for_loop_visible(self, hook_module: ModuleType) -> None:
        """for loops must not be silently dropped from analysis."""
        result = hook_module.analyze_command('echo safe && for i in a b; do echo $i; done')
        assert len(result) >= 2

    def test_while_loop_visible(self, hook_module: ModuleType) -> None:
        """while loops must not be silently dropped from analysis."""
        result = hook_module.analyze_command('echo safe && while true; do echo test; done')
        assert len(result) >= 2

    def test_function_definition_visible(self, hook_module: ModuleType) -> None:
        """Function definitions must not be silently dropped from analysis."""
        result = hook_module.analyze_command('f() { rm -rf /; } && echo done')
        assert len(result) >= 2


# ---------------------------------------------------------------------------
# TestDangerDetection — security-critical AST inspection
# ---------------------------------------------------------------------------


class TestDangerDetection:
    """Verify dangerous construct detection via AST node types."""

    def test_fd_to_fd_safe(self, hook_module: ModuleType) -> None:
        """2>&1 is fd duplication, not a file redirect."""
        result = hook_module.analyze_command('echo test 2>&1 && git log')
        assert not result[0].is_dangerous
        assert not result[1].is_dangerous

    def test_stderr_redirect_safe(self, hook_module: ModuleType) -> None:
        """>&2 is fd duplication."""
        result = hook_module.analyze_command('echo test >&2 && git log')
        assert not result[0].is_dangerous

    def test_file_redirect_dangerous(self, hook_module: ModuleType) -> None:
        result = hook_module.analyze_command('echo test > /tmp/file && git log')
        assert result[0].is_dangerous
        assert not result[1].is_dangerous

    def test_append_redirect_dangerous(self, hook_module: ModuleType) -> None:
        result = hook_module.analyze_command('echo test >> /tmp/file && git log')
        assert result[0].is_dangerous

    def test_input_redirect_dangerous(self, hook_module: ModuleType) -> None:
        result = hook_module.analyze_command('cat < /tmp/file && git log')
        assert result[0].is_dangerous

    def test_command_substitution_dangerous(self, hook_module: ModuleType) -> None:
        result = hook_module.analyze_command('echo $(rm -rf /) && git log')
        assert result[0].is_dangerous
        assert not result[1].is_dangerous

    def test_backtick_substitution_dangerous(self, hook_module: ModuleType) -> None:
        result = hook_module.analyze_command('echo `whoami` && git log')
        assert result[0].is_dangerous

    def test_process_substitution_dangerous(self, hook_module: ModuleType) -> None:
        result = hook_module.analyze_command('cat <(echo test) && git log')
        assert result[0].is_dangerous

    def test_here_string_safe(self, hook_module: ModuleType) -> None:
        """Here-strings (<<<) with plain data are safe."""
        result = hook_module.analyze_command('cat <<< hello && git log')
        assert not result[0].is_dangerous

    def test_here_string_with_substitution_dangerous(self, hook_module: ModuleType) -> None:
        """Here-strings (<<<) with command substitution are dangerous."""
        result = hook_module.analyze_command('cat <<< $(whoami) && git log')
        assert result[0].is_dangerous

    def test_heredoc_safe(self, hook_module: ModuleType) -> None:
        """Heredocs (<<) are data input, not file redirections."""
        result = hook_module.analyze_command('cat <<EOF && git log\nhello\nEOF')
        assert not result[0].is_dangerous

    def test_assignment_substitution_dangerous(self, hook_module: ModuleType) -> None:
        """Command substitution in assignment position is executed at runtime."""
        result = hook_module.analyze_command('X=$(whoami) echo test && git log')
        assert result[0].is_dangerous

    def test_assignment_unknown_var_dangerous(self, hook_module: ModuleType) -> None:
        """Unknown env var in assignment is dangerous (allowlist model)."""
        result = hook_module.analyze_command('VAR=hello echo test && git log')
        assert result[0].is_dangerous

    def test_eval_base_command_preserved(self, hook_module: ModuleType) -> None:
        """eval is parsed as a regular command — base_command includes 'eval'."""
        result = hook_module.analyze_command('eval "rm -rf /" && git log')
        assert 'eval' in result[0].base_command

    def test_if_block_marked_dangerous(self, hook_module: ModuleType) -> None:
        """if/then blocks are unanalyzable — must be marked dangerous."""
        result = hook_module.analyze_command('echo safe && if true; then rm -rf /; fi')
        if_results = [r for r in result if r.is_dangerous]
        assert len(if_results) >= 1

    def test_for_loop_marked_dangerous(self, hook_module: ModuleType) -> None:
        """for loops are unanalyzable — must be marked dangerous."""
        result = hook_module.analyze_command('echo safe && for i in a b; do echo $i; done')
        dangerous = [r for r in result if r.is_dangerous]
        assert len(dangerous) >= 1

    def test_while_loop_marked_dangerous(self, hook_module: ModuleType) -> None:
        """while loops are unanalyzable — must be marked dangerous."""
        result = hook_module.analyze_command('echo safe && while true; do echo test; done')
        dangerous = [r for r in result if r.is_dangerous]
        assert len(dangerous) >= 1

    def test_until_loop_marked_dangerous(self, hook_module: ModuleType) -> None:
        """until loops are unanalyzable — must be marked dangerous."""
        result = hook_module.analyze_command('echo safe && until false; do echo test; done')
        dangerous = [r for r in result if r.is_dangerous]
        assert len(dangerous) >= 1

    def test_function_definition_marked_dangerous(self, hook_module: ModuleType) -> None:
        """Function definitions are unanalyzable — must be marked dangerous."""
        result = hook_module.analyze_command('f() { rm -rf /; } && echo done')
        dangerous = [r for r in result if r.is_dangerous]
        assert len(dangerous) >= 1

    def test_control_flow_blocks_auto_approve(self, hook_module: ModuleType) -> None:
        """Compound with if block must NOT be auto-approved even if visible commands match."""
        result = hook_module.analyze_command('echo safe && if true; then rm -rf /; fi')
        prefixes = {'echo'}
        assert not all(hook_module.matches_prefix(info, prefixes) for info in result)

    # -- Parameter expansion bypass vectors --

    def test_parameter_default_with_substitution_dangerous(self, hook_module: ModuleType) -> None:
        """${x:-$(cmd)} hides command substitution inside parameter expansion."""
        result = hook_module.analyze_command('echo ${x:-$(whoami)} && git log')
        assert result[0].is_dangerous

    def test_parameter_assign_default_with_substitution_dangerous(self, hook_module: ModuleType) -> None:
        """${x:=$(cmd)} hides command substitution inside assign-default expansion."""
        result = hook_module.analyze_command('echo ${x:=$(whoami)} && git log')
        assert result[0].is_dangerous

    def test_assignment_parameter_expansion_dangerous(self, hook_module: ModuleType) -> None:
        """Parameter expansion with substitution in assignment position."""
        result = hook_module.analyze_command('X=${a:-$(whoami)} echo test && git log')
        assert result[0].is_dangerous

    # -- Compound redirect bypass vectors --

    def test_subshell_redirect_dangerous(self, hook_module: ModuleType) -> None:
        """File redirect on subshell is invisible to inner command analysis."""
        result = hook_module.analyze_command('(echo a; echo b) > /tmp/file && git log')
        assert any(r.is_dangerous for r in result)

    def test_brace_group_redirect_dangerous(self, hook_module: ModuleType) -> None:
        """File redirect on brace group is invisible to inner command analysis."""
        result = hook_module.analyze_command('{ echo a; echo b; } > /tmp/file && git log')
        assert any(r.is_dangerous for r in result)

    def test_compound_redirect_blocks_auto_approve(self, hook_module: ModuleType) -> None:
        """Compound with redirect must NOT be auto-approved even if commands match."""
        result = hook_module.analyze_command('(echo a; echo b) > /tmp/file && git log')
        prefixes = {'echo', 'git log'}
        assert not all(hook_module.matches_prefix(info, prefixes) for info in result)

    def test_compound_fd_redirect_safe(self, hook_module: ModuleType) -> None:
        """fd-to-fd redirect (2>&1) on compound is NOT a file redirect."""
        result = hook_module.analyze_command('(echo a; echo b) 2>&1 && git log')
        assert not any(r.is_dangerous for r in result)

    # -- Heredoc content scanning --

    def test_heredoc_with_command_substitution_dangerous(self, hook_module: ModuleType) -> None:
        """Unquoted heredocs expand $() at runtime — content must be scanned."""
        result = hook_module.analyze_command('cat <<EOF && git log\n$(rm -rf /)\nEOF')
        assert result[0].is_dangerous

    def test_heredoc_with_backtick_dangerous(self, hook_module: ModuleType) -> None:
        """Unquoted heredocs expand backticks at runtime."""
        result = hook_module.analyze_command('cat <<EOF && git log\n`whoami`\nEOF')
        assert result[0].is_dangerous

    def test_heredoc_plain_content_safe(self, hook_module: ModuleType) -> None:
        """Heredoc with no substitution patterns is safe."""
        result = hook_module.analyze_command('cat <<EOF && git log\nhello world\nEOF')
        assert not result[0].is_dangerous

    # -- Environment variable injection --

    def test_ld_preload_dangerous(self, hook_module: ModuleType) -> None:
        """LD_PRELOAD= loads arbitrary .so into subprocess — must be caught."""
        result = hook_module.analyze_command('LD_PRELOAD=/evil.so git log && echo done')
        assert result[0].is_dangerous

    def test_path_manipulation_dangerous(self, hook_module: ModuleType) -> None:
        """PATH= manipulation changes which binary executes."""
        result = hook_module.analyze_command('PATH=/evil git log && echo done')
        assert result[0].is_dangerous

    def test_git_dir_dangerous(self, hook_module: ModuleType) -> None:
        """GIT_DIR= points git at attacker-controlled repository."""
        result = hook_module.analyze_command('GIT_DIR=/evil/.git git log && echo done')
        assert result[0].is_dangerous

    def test_git_ssh_command_dangerous(self, hook_module: ModuleType) -> None:
        """GIT_SSH_COMMAND= executes arbitrary code on remote operations."""
        result = hook_module.analyze_command('GIT_SSH_COMMAND="rm -rf /" git fetch && echo done')
        assert result[0].is_dangerous

    def test_ifs_manipulation_dangerous(self, hook_module: ModuleType) -> None:
        """IFS= changes word splitting, can alter command interpretation."""
        result = hook_module.analyze_command('IFS=/ echo test && git log')
        assert result[0].is_dangerous

    def test_ld_library_path_dangerous(self, hook_module: ModuleType) -> None:
        """LD_LIBRARY_PATH= library search path poisoning."""
        result = hook_module.analyze_command('LD_LIBRARY_PATH=/evil git log && echo done')
        assert result[0].is_dangerous

    def test_safe_env_var_allowed(self, hook_module: ModuleType) -> None:
        """Known-safe env vars (TERM, LANG, TZ, etc.) pass through."""
        result = hook_module.analyze_command('TERM=xterm git log && echo done')
        assert not result[0].is_dangerous

    def test_locale_env_var_allowed(self, hook_module: ModuleType) -> None:
        """Locale variables are known-safe."""
        result = hook_module.analyze_command('LC_ALL=C git log && echo done')
        assert not result[0].is_dangerous

    def test_unknown_env_var_dangerous(self, hook_module: ModuleType) -> None:
        """Unknown env vars are dangerous (allowlist model, not denylist)."""
        result = hook_module.analyze_command('MY_CUSTOM_VAR=hello git log && echo done')
        assert result[0].is_dangerous

    # -- Code-execution command denylist --

    def test_eval_marked_dangerous(self, hook_module: ModuleType) -> None:
        """eval executes arbitrary code from string arguments."""
        result = hook_module.analyze_command('eval "rm -rf /" && git log')
        assert result[0].is_dangerous

    def test_source_marked_dangerous(self, hook_module: ModuleType) -> None:
        """source executes arbitrary file contents."""
        result = hook_module.analyze_command('source /tmp/evil.sh && git log')
        assert result[0].is_dangerous

    def test_dot_source_marked_dangerous(self, hook_module: ModuleType) -> None:
        """. is equivalent to source."""
        result = hook_module.analyze_command('. /tmp/evil.sh && git log')
        assert result[0].is_dangerous

    def test_bash_c_marked_dangerous(self, hook_module: ModuleType) -> None:
        """bash -c spawns shell with arbitrary command string."""
        result = hook_module.analyze_command('bash -c "rm -rf /" && echo done')
        assert result[0].is_dangerous

    def test_trap_marked_dangerous(self, hook_module: ModuleType) -> None:
        """trap defers arbitrary code execution to signal/exit."""
        result = hook_module.analyze_command('trap "rm -rf /" EXIT && echo done')
        assert result[0].is_dangerous

    def test_exec_marked_dangerous(self, hook_module: ModuleType) -> None:
        """exec replaces shell process with arbitrary command."""
        result = hook_module.analyze_command('exec git log && echo done')
        assert result[0].is_dangerous

    # -- Additional coverage: safe constructs --

    def test_simple_parameter_expansion_safe(self, hook_module: ModuleType) -> None:
        """Simple $VAR expansion (no substitution) is safe."""
        result = hook_module.analyze_command('echo $HOME && git log')
        assert not result[0].is_dangerous

    def test_assignment_unknown_var_with_parameter_dangerous(self, hook_module: ModuleType) -> None:
        """Unknown env var is dangerous regardless of safe value content."""
        result = hook_module.analyze_command('VAR=$HOME echo test && git log')
        assert result[0].is_dangerous

    def test_assignment_process_substitution_dangerous(self, hook_module: ModuleType) -> None:
        """Process substitution in assignment position is dangerous."""
        result = hook_module.analyze_command('X=<(echo test) echo done && git log')
        assert result[0].is_dangerous

    def test_here_string_process_substitution_dangerous(self, hook_module: ModuleType) -> None:
        """Here-strings (<<<) with process substitution are dangerous."""
        result = hook_module.analyze_command('cat <<< <(echo test) && git log')
        assert result[0].is_dangerous

    def test_backtick_in_parameter_expansion_dangerous(self, hook_module: ModuleType) -> None:
        """Backticks inside parameter expansion are caught."""
        result = hook_module.analyze_command('echo ${x:-`whoami`} && git log')
        assert result[0].is_dangerous


# ---------------------------------------------------------------------------
# TestResolveBaseCommand — wrapper prefix stripping
# ---------------------------------------------------------------------------


class TestResolveBaseCommand:
    """Verify wrapper command stripping (timeout, nohup, nice, time)."""

    @pytest.mark.parametrize(
        'words, expected',
        [
            (['git', 'log'], 'git log'),
            (['timeout', '5s', 'git', 'log'], 'git log'),
            (['nohup', 'git', 'log'], 'git log'),
            (['time', 'git', 'log'], 'git log'),
            (['nice', '-n', '10', 'git', 'log'], 'git log'),
            (['timeout', '5s', 'nohup', 'git', 'log'], 'git log'),
            ([], ''),
            (['timeout', '5s'], ''),
            (['nice', '-n', '10'], ''),
        ],
        ids=[
            'plain',
            'timeout',
            'nohup',
            'time',
            'nice',
            'chained',
            'empty',
            'wrapper-only-timeout',
            'wrapper-only-nice',
        ],
    )
    def test_wrapper_stripping(self, hook_module: ModuleType, words: list[str], expected: str) -> None:
        assert hook_module._resolve_base_command(words) == expected

    @pytest.mark.parametrize(
        'words, expected',
        [
            (['sudo', 'git', 'status'], 'sudo git status'),
            (['env', 'VAR=1', 'git', 'status'], 'env VAR=1 git status'),
            (['command', 'git', 'status'], 'command git status'),
        ],
        ids=['sudo', 'env', 'command'],
    )
    def test_privilege_wrappers_not_stripped(self, hook_module: ModuleType, words: list[str], expected: str) -> None:
        """sudo/env/command are NOT stripped — conservative by design."""
        assert hook_module._resolve_base_command(words) == expected

    @pytest.mark.parametrize(
        'words, expected',
        [
            (['timeout', '--signal=KILL', '5s', 'git', 'log'], '5s git log'),
            (['timeout', '-k', '5', '10', 'git', 'log'], '5 10 git log'),
            (['nice', '-5', 'git', 'log'], '-5 git log'),
            (['nice', '--adjustment=5', 'git', 'log'], '--adjustment=5 git log'),
        ],
        ids=['timeout-signal', 'timeout-kill', 'nice-bsd', 'nice-gnu'],
    )
    def test_wrapper_flag_limitations(self, hook_module: ModuleType, words: list[str], expected: str) -> None:
        """Wrapper flags produce imperfect base_command (safe direction — falls through)."""
        assert hook_module._resolve_base_command(words) == expected


# ---------------------------------------------------------------------------
# TestMatchesPrefix — prefix matching with danger gate
# ---------------------------------------------------------------------------


class TestMatchesPrefix:
    """Verify prefix matching replicates Claude Code's Bash(prefix:*) logic."""

    def test_exact_match(self, hook_module: ModuleType) -> None:
        info = hook_module.SubcommandInfo(text='echo', base_command='echo')
        assert hook_module.matches_prefix(info, {'echo'})

    def test_prefix_match(self, hook_module: ModuleType) -> None:
        info = hook_module.SubcommandInfo(text='git log --oneline', base_command='git log --oneline')
        assert hook_module.matches_prefix(info, {'git log'})

    def test_no_match(self, hook_module: ModuleType) -> None:
        info = hook_module.SubcommandInfo(text='rm -rf /', base_command='rm -rf /')
        assert not hook_module.matches_prefix(info, {'echo', 'git log'})

    def test_dangerous_blocks_match(self, hook_module: ModuleType) -> None:
        """Dangerous subcommand rejected even if prefix matches."""
        info = hook_module.SubcommandInfo(text='echo $(whoami)', base_command='echo $(whoami)', is_dangerous=True)
        assert not hook_module.matches_prefix(info, {'echo'})

    def test_prefix_boundary(self, hook_module: ModuleType) -> None:
        """Prefix must match at word boundary (space), not just string prefix."""
        info = hook_module.SubcommandInfo(text='git logistics', base_command='git logistics')
        assert not hook_module.matches_prefix(info, {'git log'})

    def test_empty_base_command_never_matches(self, hook_module: ModuleType) -> None:
        """Assignment-only subcommands (empty base_command) never match any prefix."""
        info = hook_module.SubcommandInfo(text='VAR=hello', base_command='')
        assert not hook_module.matches_prefix(info, {'echo', 'git'})

    def test_mixed_safe_dangerous_blocks_all(self, hook_module: ModuleType) -> None:
        """One dangerous subcommand blocks the entire compound."""
        result = hook_module.analyze_command('git log && echo $(whoami) && git status')
        prefixes = {'git log', 'git status', 'echo'}
        assert not all(hook_module.matches_prefix(info, prefixes) for info in result)


# ---------------------------------------------------------------------------
# TestLoadBashPrefixes — settings file parsing (isolated)
# ---------------------------------------------------------------------------


class TestLoadBashPrefixes:
    """Verify settings hierarchy reading with filesystem isolation."""

    def test_extracts_bash_prefixes(
        self, hook_module: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(Path, 'home', staticmethod(lambda: tmp_path / 'home'))
        cwd = tmp_path / 'project'
        (cwd / '.claude').mkdir(parents=True)
        (cwd / '.claude' / 'settings.json').write_text(
            json.dumps({'permissions': {'allow': ['Bash(git log:*)', 'Bash(echo:*)']}})
        )
        assert hook_module.load_bash_prefixes(str(cwd)) == {'git log', 'echo'}

    def test_ignores_non_bash_entries(
        self, hook_module: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(Path, 'home', staticmethod(lambda: tmp_path / 'home'))
        cwd = tmp_path / 'project'
        (cwd / '.claude').mkdir(parents=True)
        (cwd / '.claude' / 'settings.json').write_text(
            json.dumps({'permissions': {'allow': ['WebSearch', 'mcp__foo__bar']}})
        )
        assert hook_module.load_bash_prefixes(str(cwd)) == set()

    def test_merges_home_and_project(
        self, hook_module: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        home = tmp_path / 'home'
        (home / '.claude').mkdir(parents=True)
        (home / '.claude' / 'settings.json').write_text(json.dumps({'permissions': {'allow': ['Bash(git log:*)']}}))
        monkeypatch.setattr(Path, 'home', staticmethod(lambda: home))

        cwd = tmp_path / 'project'
        (cwd / '.claude').mkdir(parents=True)
        (cwd / '.claude' / 'settings.local.json').write_text(json.dumps({'permissions': {'allow': ['Bash(echo:*)']}}))
        assert hook_module.load_bash_prefixes(str(cwd)) == {'git log', 'echo'}

    def test_skips_missing_files(
        self, hook_module: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(Path, 'home', staticmethod(lambda: tmp_path / 'home'))
        assert hook_module.load_bash_prefixes(str(tmp_path / 'nonexistent')) == set()

    def test_skips_malformed_json(
        self, hook_module: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(Path, 'home', staticmethod(lambda: tmp_path / 'home'))
        cwd = tmp_path / 'project'
        (cwd / '.claude').mkdir(parents=True)
        (cwd / '.claude' / 'settings.json').write_text('{invalid json')
        assert hook_module.load_bash_prefixes(str(cwd)) == set()


# ---------------------------------------------------------------------------
# TestMainIntegration — full stdin/stdout flow
# ---------------------------------------------------------------------------


class TestMainIntegration:
    """Verify the main() function end-to-end with mocked stdin/stdout."""

    @staticmethod
    def _hook_input(command: str, tool_name: str = 'Bash', cwd: str = '/tmp') -> str:
        return json.dumps(
            {
                'session_id': 'test',
                'cwd': cwd,
                'transcript_path': '/tmp/transcript.jsonl',
                'hook_event_name': 'PreToolUse',
                'tool_name': tool_name,
                'tool_input': {'command': command},
                'tool_use_id': 'tu_test',
            }
        )

    def test_approves_compound(
        self,
        hook_module: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        home = tmp_path / 'home'
        (home / '.claude').mkdir(parents=True)
        (home / '.claude' / 'settings.json').write_text(
            json.dumps({'permissions': {'allow': ['Bash(git log:*)', 'Bash(echo:*)']}})
        )
        monkeypatch.setattr(Path, 'home', staticmethod(lambda: home))
        monkeypatch.setattr('sys.stdin', io.StringIO(self._hook_input('echo "---" && git log')))

        hook_module.main.__wrapped__()

        output = json.loads(capsys.readouterr().out)
        assert output['hookSpecificOutput']['permissionDecision'] == 'allow'

    def test_skips_non_bash_tool(
        self,
        hook_module: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr('sys.stdin', io.StringIO(self._hook_input('test', tool_name='Write')))

        hook_module.main.__wrapped__()
        assert capsys.readouterr().out == ''

    def test_skips_single_command(
        self,
        hook_module: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr('sys.stdin', io.StringIO(self._hook_input('git log')))

        hook_module.main.__wrapped__()
        assert capsys.readouterr().out == ''

    def test_passthrough_on_dangerous(
        self,
        hook_module: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        home = tmp_path / 'home'
        (home / '.claude').mkdir(parents=True)
        (home / '.claude' / 'settings.json').write_text(
            json.dumps({'permissions': {'allow': ['Bash(echo:*)', 'Bash(git log:*)']}})
        )
        monkeypatch.setattr(Path, 'home', staticmethod(lambda: home))
        monkeypatch.setattr('sys.stdin', io.StringIO(self._hook_input('echo $(whoami) && git log')))

        hook_module.main.__wrapped__()
        assert capsys.readouterr().out == ''

    def test_skips_empty_command(
        self,
        hook_module: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr('sys.stdin', io.StringIO(self._hook_input('')))

        hook_module.main.__wrapped__()
        assert capsys.readouterr().out == ''

    def test_skips_no_prefixes(
        self,
        hook_module: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """No Bash prefixes loaded — passthrough to built-in permissions."""
        home = tmp_path / 'home'
        (home / '.claude').mkdir(parents=True)
        (home / '.claude' / 'settings.json').write_text(json.dumps({'permissions': {'allow': []}}))
        monkeypatch.setattr(Path, 'home', staticmethod(lambda: home))
        monkeypatch.setattr(
            'sys.stdin',
            io.StringIO(
                self._hook_input(
                    'echo "---" && git log',
                    cwd=str(tmp_path / 'noproject'),
                )
            ),
        )

        hook_module.main.__wrapped__()
        assert capsys.readouterr().out == ''

    def test_passthrough_on_no_match(
        self,
        hook_module: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Safe but non-matching subcommand — passthrough to built-in permissions."""
        home = tmp_path / 'home'
        (home / '.claude').mkdir(parents=True)
        (home / '.claude' / 'settings.json').write_text(json.dumps({'permissions': {'allow': ['Bash(echo:*)']}}))
        monkeypatch.setattr(Path, 'home', staticmethod(lambda: home))
        monkeypatch.setattr('sys.stdin', io.StringIO(self._hook_input('echo test && rm -rf /')))

        hook_module.main.__wrapped__()
        assert capsys.readouterr().out == ''

    # -- ErrorBoundary integration --

    def test_parse_error_passthrough_via_boundary(
        self,
        hook_module: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Parse errors in main() passthrough safely (exit 0, no decision).

        Calls main() WITH ErrorBoundary (not __wrapped__) to verify the full
        decorator chain handles bashlex failures gracefully.
        """
        monkeypatch.setattr('sys.stdin', io.StringIO(self._hook_input('echo "unterminated')))

        with pytest.raises(SystemExit) as exc_info:
            hook_module.main()

        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert captured.out == ''
        assert 'error' in captured.err.lower()

    def test_unknown_field_passthrough_via_boundary(
        self,
        hook_module: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """extra='forbid' is intentional: unknown fields trigger safe passthrough.

        StrictModel(extra='forbid') rejects payloads with unknown fields. This
        is a deliberate design choice: when Claude Code adds new fields, the hook
        fails safely (ErrorBoundary catches ValidationError, exits 0, passes
        through to built-in permissions) rather than silently ignoring potentially
        important new fields. The tradeoff is that the hook stops auto-approving
        until the schema is updated — a loud failure that gets fixed, rather than
        a silent one that hides new behavior.
        """
        payload = json.dumps(
            {
                'session_id': 'test',
                'cwd': '/tmp',
                'transcript_path': '/tmp/transcript.jsonl',
                'hook_event_name': 'PreToolUse',
                'tool_name': 'Bash',
                'tool_input': {'command': 'echo a && echo b'},
                'tool_use_id': 'tu_test',
                'new_future_field': 'surprise',
            }
        )
        monkeypatch.setattr('sys.stdin', io.StringIO(payload))

        with pytest.raises(SystemExit) as exc_info:
            hook_module.main()

        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert captured.out == ''
        assert 'error' in captured.err.lower()

    # -- Null byte protection --

    def test_null_byte_passthrough(
        self,
        hook_module: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Null bytes cause parsing divergence — must not auto-approve."""
        home = tmp_path / 'home'
        (home / '.claude').mkdir(parents=True)
        (home / '.claude' / 'settings.json').write_text(
            json.dumps({'permissions': {'allow': ['Bash(echo:*)', 'Bash(git log:*)']}})
        )
        monkeypatch.setattr(Path, 'home', staticmethod(lambda: home))
        monkeypatch.setattr(
            'sys.stdin',
            io.StringIO(self._hook_input('echo safe\x00 && git log')),
        )

        hook_module.main.__wrapped__()
        assert capsys.readouterr().out == ''


# ---------------------------------------------------------------------------
# TestRealWorldCommands — regression cases
# ---------------------------------------------------------------------------


class TestRealWorldCommands:
    """Commands that triggered the original Claude Code permission bug."""

    @pytest.mark.parametrize(
        'cmd',
        [
            'echo "---" && git log --oneline -5',
            'git diff --cached --stat 2>&1 && echo "---" && git log --oneline -5 2>&1',
            'echo \'hello "world"\' && git status --short 2>&1',
        ],
        ids=['quoted-echo', 'three-part-with-redirects', 'nested-quotes'],
    )
    def test_safe_compound(self, hook_module: ModuleType, cmd: str) -> None:
        result = hook_module.analyze_command(cmd)
        assert len(result) >= 2
        assert all(not info.is_dangerous for info in result)


# ---------------------------------------------------------------------------
# TestParseErrors — bashlex failure modes
# ---------------------------------------------------------------------------


class TestParseErrors:
    """Verify bashlex raises on malformed input (ErrorBoundary catches in production)."""

    def test_unterminated_quote(self, hook_module: ModuleType) -> None:
        with pytest.raises(Exception):
            hook_module.analyze_command('echo "unterminated')

    def test_empty_string(self, hook_module: ModuleType) -> None:
        with pytest.raises(Exception):
            hook_module.analyze_command('')

    def test_trailing_operator(self, hook_module: ModuleType) -> None:
        with pytest.raises(Exception):
            hook_module.analyze_command('echo test &&')

    def test_arithmetic_expansion(self, hook_module: ModuleType) -> None:
        """bashlex does not support arithmetic expansion — raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            hook_module.analyze_command('echo $((1+2)) && git log')

    def test_case_statement(self, hook_module: ModuleType) -> None:
        """bashlex does not support case statements — raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            hook_module.analyze_command('case $x in a) echo a;; esac && git log')
