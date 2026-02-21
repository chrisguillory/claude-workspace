"""Tests for the approve-compound-bash PreToolUse hook.

Validates bashlex-based compound command parsing, dangerous construct
detection via ApproveCompoundBashException, prefix matching, and settings loading.
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
        assert result[0] == 'echo hello'

    def test_and_operator(self, hook_module: ModuleType) -> None:
        result = hook_module.analyze_command('echo "---" && git log')
        assert len(result) == 2
        assert result[0].startswith('echo')
        assert result[1].startswith('git log')

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
        """Assignment raises ValueError — inline assignments are dangerous."""
        with pytest.raises(hook_module.ApproveCompoundBashException):
            hook_module.analyze_command('VAR=val echo test && git status')

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
        """Assignment-only subcommand raises ValueError."""
        with pytest.raises(hook_module.ApproveCompoundBashException):
            hook_module.analyze_command('VAR=hello && echo test')

    def test_if_then_visible(self, hook_module: ModuleType) -> None:
        """if/then blocks raise ValueError — unanalyzable construct."""
        with pytest.raises(hook_module.ApproveCompoundBashException):
            hook_module.analyze_command('echo safe && if true; then rm -rf /; fi')

    def test_for_loop_visible(self, hook_module: ModuleType) -> None:
        """for loops raise ValueError — unanalyzable construct."""
        with pytest.raises(hook_module.ApproveCompoundBashException):
            hook_module.analyze_command('echo safe && for i in a b; do echo $i; done')

    def test_while_loop_visible(self, hook_module: ModuleType) -> None:
        """while loops raise ValueError — unanalyzable construct."""
        with pytest.raises(hook_module.ApproveCompoundBashException):
            hook_module.analyze_command('echo safe && while true; do echo test; done')

    def test_function_definition_visible(self, hook_module: ModuleType) -> None:
        """Function definitions raise ValueError — unanalyzable construct."""
        with pytest.raises(hook_module.ApproveCompoundBashException):
            hook_module.analyze_command('f() { rm -rf /; } && echo done')

    def test_negated_pipeline(self, hook_module: ModuleType) -> None:
        """! (negation) is a reservedword — inner command must still be extracted."""
        result = hook_module.analyze_command('! false && echo done')
        assert len(result) == 2
        assert result[0] == 'false'


# ---------------------------------------------------------------------------
# TestDangerDetection — security-critical AST inspection
# ---------------------------------------------------------------------------


class TestDangerDetection:
    """Verify dangerous construct detection raises ValueError."""

    def test_fd_to_fd_safe(self, hook_module: ModuleType) -> None:
        """2>&1 is fd duplication, not a file redirect."""
        result = hook_module.analyze_command('echo test 2>&1 && git log')
        assert len(result) == 2

    def test_stderr_redirect_safe(self, hook_module: ModuleType) -> None:
        """>&2 is fd duplication."""
        result = hook_module.analyze_command('echo test >&2 && git log')
        assert len(result) == 2

    def test_file_redirect_dangerous(self, hook_module: ModuleType) -> None:
        with pytest.raises(hook_module.ApproveCompoundBashException):
            hook_module.analyze_command('echo test > /tmp/file && git log')

    def test_append_redirect_dangerous(self, hook_module: ModuleType) -> None:
        with pytest.raises(hook_module.ApproveCompoundBashException):
            hook_module.analyze_command('echo test >> /tmp/file && git log')

    def test_input_redirect_dangerous(self, hook_module: ModuleType) -> None:
        with pytest.raises(hook_module.ApproveCompoundBashException):
            hook_module.analyze_command('cat < /tmp/file && git log')

    def test_command_substitution_dangerous(self, hook_module: ModuleType) -> None:
        with pytest.raises(hook_module.ApproveCompoundBashException):
            hook_module.analyze_command('echo $(rm -rf /) && git log')

    def test_backtick_substitution_dangerous(self, hook_module: ModuleType) -> None:
        with pytest.raises(hook_module.ApproveCompoundBashException):
            hook_module.analyze_command('echo `whoami` && git log')

    def test_process_substitution_dangerous(self, hook_module: ModuleType) -> None:
        with pytest.raises(hook_module.ApproveCompoundBashException):
            hook_module.analyze_command('cat <(echo test) && git log')

    def test_here_string_dangerous(self, hook_module: ModuleType) -> None:
        """Here-strings (<<<) are non-fd redirects — always dangerous."""
        with pytest.raises(hook_module.ApproveCompoundBashException):
            hook_module.analyze_command('cat <<< hello && git log')

    def test_here_string_with_substitution_dangerous(self, hook_module: ModuleType) -> None:
        """Here-strings (<<<) with command substitution are dangerous."""
        with pytest.raises(hook_module.ApproveCompoundBashException):
            hook_module.analyze_command('cat <<< $(whoami) && git log')

    def test_heredoc_dangerous(self, hook_module: ModuleType) -> None:
        """Heredocs (<<) are non-fd redirects — always dangerous."""
        with pytest.raises(hook_module.ApproveCompoundBashException):
            hook_module.analyze_command('cat <<EOF && git log\nhello\nEOF')

    def test_assignment_substitution_dangerous(self, hook_module: ModuleType) -> None:
        """Command substitution in assignment position is executed at runtime."""
        with pytest.raises(hook_module.ApproveCompoundBashException):
            hook_module.analyze_command('X=$(whoami) echo test && git log')

    def test_assignment_unknown_var_dangerous(self, hook_module: ModuleType) -> None:
        """Unknown env var in assignment is dangerous (allowlist model)."""
        with pytest.raises(hook_module.ApproveCompoundBashException):
            hook_module.analyze_command('VAR=hello echo test && git log')

    def test_eval_base_command_preserved(self, hook_module: ModuleType) -> None:
        """eval is parsed as a regular command — base_command includes 'eval'."""
        result = hook_module.analyze_command('eval "rm -rf /" && git log')
        assert 'eval' in result[0]

    def test_if_block_marked_dangerous(self, hook_module: ModuleType) -> None:
        """if/then blocks are unanalyzable — must raise ValueError."""
        with pytest.raises(hook_module.ApproveCompoundBashException):
            hook_module.analyze_command('echo safe && if true; then rm -rf /; fi')

    def test_for_loop_marked_dangerous(self, hook_module: ModuleType) -> None:
        """for loops are unanalyzable — must raise ValueError."""
        with pytest.raises(hook_module.ApproveCompoundBashException):
            hook_module.analyze_command('echo safe && for i in a b; do echo $i; done')

    def test_while_loop_marked_dangerous(self, hook_module: ModuleType) -> None:
        """while loops are unanalyzable — must raise ValueError."""
        with pytest.raises(hook_module.ApproveCompoundBashException):
            hook_module.analyze_command('echo safe && while true; do echo test; done')

    def test_until_loop_marked_dangerous(self, hook_module: ModuleType) -> None:
        """until loops are unanalyzable — must raise ValueError."""
        with pytest.raises(hook_module.ApproveCompoundBashException):
            hook_module.analyze_command('echo safe && until false; do echo test; done')

    def test_function_definition_marked_dangerous(self, hook_module: ModuleType) -> None:
        """Function definitions are unanalyzable — must raise ValueError."""
        with pytest.raises(hook_module.ApproveCompoundBashException):
            hook_module.analyze_command('f() { rm -rf /; } && echo done')

    # -- Parameter expansion bypass vectors --

    def test_parameter_default_with_substitution_dangerous(self, hook_module: ModuleType) -> None:
        """${x:-$(cmd)} hides command substitution inside parameter expansion."""
        with pytest.raises(hook_module.ApproveCompoundBashException):
            hook_module.analyze_command('echo ${x:-$(whoami)} && git log')

    def test_parameter_assign_default_with_substitution_dangerous(self, hook_module: ModuleType) -> None:
        """${x:=$(cmd)} hides command substitution inside assign-default expansion."""
        with pytest.raises(hook_module.ApproveCompoundBashException):
            hook_module.analyze_command('echo ${x:=$(whoami)} && git log')

    def test_assignment_parameter_expansion_dangerous(self, hook_module: ModuleType) -> None:
        """Parameter expansion with substitution in assignment position."""
        with pytest.raises(hook_module.ApproveCompoundBashException):
            hook_module.analyze_command('X=${a:-$(whoami)} echo test && git log')

    # -- Compound redirect bypass vectors --

    def test_subshell_redirect_dangerous(self, hook_module: ModuleType) -> None:
        """File redirect on subshell is invisible to inner command analysis."""
        with pytest.raises(hook_module.ApproveCompoundBashException):
            hook_module.analyze_command('(echo a; echo b) > /tmp/file && git log')

    def test_brace_group_redirect_dangerous(self, hook_module: ModuleType) -> None:
        """File redirect on brace group is invisible to inner command analysis."""
        with pytest.raises(hook_module.ApproveCompoundBashException):
            hook_module.analyze_command('{ echo a; echo b; } > /tmp/file && git log')

    def test_compound_redirect_blocks_auto_approve(self, hook_module: ModuleType) -> None:
        """Compound with redirect must raise ValueError."""
        with pytest.raises(hook_module.ApproveCompoundBashException):
            hook_module.analyze_command('(echo a; echo b) > /tmp/file && git log')

    def test_compound_fd_redirect_safe(self, hook_module: ModuleType) -> None:
        """fd-to-fd redirect (2>&1) on compound is NOT a file redirect."""
        result = hook_module.analyze_command('(echo a; echo b) 2>&1 && git log')
        assert len(result) == 3

    def test_compound_mixed_fd_then_file_redirect(self, hook_module: ModuleType) -> None:
        """fd redirect first, file redirect second — file redirect must trigger danger."""
        with pytest.raises(hook_module.ApproveCompoundBashException):
            hook_module.analyze_command('(echo a; echo b) 2>&1 >/tmp/file && git log')

    def test_compound_multiple_fd_redirects_safe(self, hook_module: ModuleType) -> None:
        """Multiple fd-to-fd redirects on compound — all safe."""
        result = hook_module.analyze_command('(echo a; echo b) 2>&1 3>&2 && git log')
        assert len(result) == 3

    # -- Heredoc content scanning --

    def test_heredoc_with_command_substitution_dangerous(self, hook_module: ModuleType) -> None:
        """Unquoted heredocs expand $() at runtime — content must be scanned."""
        with pytest.raises(hook_module.ApproveCompoundBashException):
            hook_module.analyze_command('cat <<EOF && git log\n$(rm -rf /)\nEOF')

    def test_heredoc_with_backtick_dangerous(self, hook_module: ModuleType) -> None:
        """Unquoted heredocs expand backticks at runtime."""
        with pytest.raises(hook_module.ApproveCompoundBashException):
            hook_module.analyze_command('cat <<EOF && git log\n`whoami`\nEOF')

    def test_heredoc_plain_content_dangerous(self, hook_module: ModuleType) -> None:
        """Heredocs are always dangerous — no content scanning needed."""
        with pytest.raises(hook_module.ApproveCompoundBashException):
            hook_module.analyze_command('cat <<EOF && git log\nhello world\nEOF')

    # -- Environment variable injection --

    def test_ld_preload_dangerous(self, hook_module: ModuleType) -> None:
        """LD_PRELOAD= loads arbitrary .so into subprocess — must be caught."""
        with pytest.raises(hook_module.ApproveCompoundBashException):
            hook_module.analyze_command('LD_PRELOAD=/evil.so git log && echo done')

    def test_path_manipulation_dangerous(self, hook_module: ModuleType) -> None:
        """PATH= manipulation changes which binary executes."""
        with pytest.raises(hook_module.ApproveCompoundBashException):
            hook_module.analyze_command('PATH=/evil git log && echo done')

    def test_git_dir_dangerous(self, hook_module: ModuleType) -> None:
        """GIT_DIR= points git at attacker-controlled repository."""
        with pytest.raises(hook_module.ApproveCompoundBashException):
            hook_module.analyze_command('GIT_DIR=/evil/.git git log && echo done')

    def test_git_ssh_command_dangerous(self, hook_module: ModuleType) -> None:
        """GIT_SSH_COMMAND= executes arbitrary code on remote operations."""
        with pytest.raises(hook_module.ApproveCompoundBashException):
            hook_module.analyze_command('GIT_SSH_COMMAND="rm -rf /" git fetch && echo done')

    def test_ifs_manipulation_dangerous(self, hook_module: ModuleType) -> None:
        """IFS= changes word splitting, can alter command interpretation."""
        with pytest.raises(hook_module.ApproveCompoundBashException):
            hook_module.analyze_command('IFS=/ echo test && git log')

    def test_ld_library_path_dangerous(self, hook_module: ModuleType) -> None:
        """LD_LIBRARY_PATH= library search path poisoning."""
        with pytest.raises(hook_module.ApproveCompoundBashException):
            hook_module.analyze_command('LD_LIBRARY_PATH=/evil git log && echo done')

    def test_safe_env_var_dangerous(self, hook_module: ModuleType) -> None:
        """All assignments are dangerous — no env var allowlist."""
        with pytest.raises(hook_module.ApproveCompoundBashException):
            hook_module.analyze_command('TERM=xterm git log && echo done')

    def test_locale_env_var_dangerous(self, hook_module: ModuleType) -> None:
        """All assignments are dangerous — no env var allowlist."""
        with pytest.raises(hook_module.ApproveCompoundBashException):
            hook_module.analyze_command('LC_ALL=C git log && echo done')

    def test_unknown_env_var_dangerous(self, hook_module: ModuleType) -> None:
        """All assignments are dangerous — no env var allowlist."""
        with pytest.raises(hook_module.ApproveCompoundBashException):
            hook_module.analyze_command('MY_CUSTOM_VAR=hello git log && echo done')

    # -- Code-execution commands (handled by prefix matching, not AST) --

    def test_eval_not_dangerous_at_ast_level(self, hook_module: ModuleType) -> None:
        """eval is plain words at AST level — blocked by prefix matching instead."""
        result = hook_module.analyze_command('eval "rm -rf /" && git log')
        assert len(result) == 2

    def test_source_not_dangerous_at_ast_level(self, hook_module: ModuleType) -> None:
        """source is plain words at AST level — blocked by prefix matching instead."""
        result = hook_module.analyze_command('source /tmp/evil.sh && git log')
        assert len(result) == 2

    def test_dot_source_not_dangerous_at_ast_level(self, hook_module: ModuleType) -> None:
        """. is plain words at AST level — blocked by prefix matching instead."""
        result = hook_module.analyze_command('. /tmp/evil.sh && git log')
        assert len(result) == 2

    def test_bash_c_not_dangerous_at_ast_level(self, hook_module: ModuleType) -> None:
        """bash -c is plain words at AST level — blocked by prefix matching instead."""
        result = hook_module.analyze_command('bash -c "rm -rf /" && echo done')
        assert len(result) == 2

    def test_trap_not_dangerous_at_ast_level(self, hook_module: ModuleType) -> None:
        """trap is plain words at AST level — blocked by prefix matching instead."""
        result = hook_module.analyze_command('trap "rm -rf /" EXIT && echo done')
        assert len(result) == 2

    def test_exec_not_dangerous_at_ast_level(self, hook_module: ModuleType) -> None:
        """exec is plain words at AST level — blocked by prefix matching instead."""
        result = hook_module.analyze_command('exec git log && echo done')
        assert len(result) == 2

    # -- Additional coverage: safe constructs --

    def test_tilde_expansion_safe(self, hook_module: ModuleType) -> None:
        """Tilde expansion (~, ~/path) is path resolution, not code execution."""
        result = hook_module.analyze_command('echo ~/test && git log')
        assert len(result) == 2

    def test_tilde_bare_safe(self, hook_module: ModuleType) -> None:
        """Bare ~ is safe path resolution."""
        result = hook_module.analyze_command('ls ~ && git status')
        assert len(result) == 2

    def test_simple_parameter_expansion_safe(self, hook_module: ModuleType) -> None:
        """Simple $VAR expansion (no substitution) is safe."""
        result = hook_module.analyze_command('echo $HOME && git log')
        assert len(result) == 2

    def test_assignment_unknown_var_with_parameter_dangerous(self, hook_module: ModuleType) -> None:
        """Unknown env var is dangerous regardless of safe value content."""
        with pytest.raises(hook_module.ApproveCompoundBashException):
            hook_module.analyze_command('VAR=$HOME echo test && git log')

    def test_assignment_process_substitution_dangerous(self, hook_module: ModuleType) -> None:
        """Process substitution in assignment position is dangerous."""
        with pytest.raises(hook_module.ApproveCompoundBashException):
            hook_module.analyze_command('X=<(echo test) echo done && git log')

    def test_here_string_process_substitution_dangerous(self, hook_module: ModuleType) -> None:
        """Here-strings (<<<) with process substitution are dangerous."""
        with pytest.raises(hook_module.ApproveCompoundBashException):
            hook_module.analyze_command('cat <<< <(echo test) && git log')

    def test_backtick_in_parameter_expansion_dangerous(self, hook_module: ModuleType) -> None:
        """Backticks inside parameter expansion are caught."""
        with pytest.raises(hook_module.ApproveCompoundBashException):
            hook_module.analyze_command('echo ${x:-`whoami`} && git log')

    # -- Parameter transform operators (@P, @E, @A, etc.) --

    def test_prompt_expansion_dangerous(self, hook_module: ModuleType) -> None:
        """${x@P} interprets value as PS1 prompt, executing embedded $(cmd)."""
        with pytest.raises(hook_module.ApproveCompoundBashException):
            hook_module.analyze_command('echo ${PS1@P} && git log')

    def test_assign_transform_dangerous(self, hook_module: ModuleType) -> None:
        """${x@A} produces assignment statement — could leak values."""
        with pytest.raises(hook_module.ApproveCompoundBashException):
            hook_module.analyze_command('echo ${x@A} && git log')

    def test_escape_transform_dangerous(self, hook_module: ModuleType) -> None:
        """${x@E} interprets escape sequences — fail closed on all @ operators."""
        with pytest.raises(hook_module.ApproveCompoundBashException):
            hook_module.analyze_command('echo ${x@E} && git log')


# ---------------------------------------------------------------------------
# TestWrapperCommandsPreserved — wrappers are NOT stripped (YAGNI)
# ---------------------------------------------------------------------------


class TestWrapperCommandsPreserved:
    """Verify wrapper commands (timeout, nohup, nice, time) are preserved in base_command.

    Wrapper stripping was removed (YAGNI). Users add prefixes like Bash(timeout:*)
    if needed. These tests confirm wrappers stay in base_command via analyze_command.
    """

    @pytest.mark.parametrize(
        'command, expected_base',
        [
            ('timeout 5s git log && echo done', 'timeout 5s git log'),
            ('nohup git log && echo done', 'nohup git log'),
            # 'time' is a bash reserved word — bashlex raises NotImplementedError,
            # caught by ErrorBoundary in production (passthrough). Tested separately.
            ('nice -n 10 git log && echo done', 'nice -n 10 git log'),
            ('timeout 5s nohup git log && echo done', 'timeout 5s nohup git log'),
        ],
        ids=['timeout', 'nohup', 'nice', 'chained'],
    )
    def test_wrapper_preserved(self, hook_module: ModuleType, command: str, expected_base: str) -> None:
        result = hook_module.analyze_command(command)
        assert result[0] == expected_base

    @pytest.mark.parametrize(
        'command, expected_base',
        [
            ('sudo git status && echo done', 'sudo git status'),
            ('command git status && echo done', 'command git status'),
        ],
        ids=['sudo', 'command'],
    )
    def test_privilege_wrappers_preserved(self, hook_module: ModuleType, command: str, expected_base: str) -> None:
        """sudo/command stay in base_command — handled by prefix matching."""
        result = hook_module.analyze_command(command)
        assert result[0] == expected_base

    def test_time_raises_not_implemented(self, hook_module: ModuleType) -> None:
        """bashlex doesn't support 'time' — ErrorBoundary catches this in production."""
        with pytest.raises(NotImplementedError):
            hook_module.analyze_command('time git log && echo done')

    def test_wrapper_without_prefix_no_match(self, hook_module: ModuleType) -> None:
        """Wrapped command won't match unwrapped prefix — safe direction."""
        result = hook_module.analyze_command('timeout 5s git log && echo done')
        prefixes = {'git log'}  # No 'timeout' prefix
        assert not hook_module.matches_prefix(result[0], prefixes)


# ---------------------------------------------------------------------------
# TestMatchesPrefix — prefix matching
# ---------------------------------------------------------------------------


class TestMatchesPrefix:
    """Verify prefix matching replicates Claude Code's Bash(prefix:*) logic."""

    def test_exact_match(self, hook_module: ModuleType) -> None:
        assert hook_module.matches_prefix('echo', {'echo'})

    def test_prefix_match(self, hook_module: ModuleType) -> None:
        assert hook_module.matches_prefix('git log --oneline', {'git log'})

    def test_no_match(self, hook_module: ModuleType) -> None:
        assert not hook_module.matches_prefix('rm -rf /', {'echo', 'git log'})

    def test_prefix_boundary(self, hook_module: ModuleType) -> None:
        """Prefix must match at word boundary (space), not just string prefix."""
        assert not hook_module.matches_prefix('git logistics', {'git log'})

    def test_empty_base_command_never_matches(self, hook_module: ModuleType) -> None:
        """Empty base_command never matches any prefix."""
        assert not hook_module.matches_prefix('', {'echo', 'git'})

    def test_mixed_safe_dangerous_raises(self, hook_module: ModuleType) -> None:
        """One dangerous subcommand raises ValueError for the entire compound."""
        with pytest.raises(hook_module.ApproveCompoundBashException):
            hook_module.analyze_command('git log && echo $(whoami) && git status')


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

    def test_malformed_json_raises(
        self, hook_module: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Malformed JSON bubbles up — ErrorBoundary handles in production."""
        monkeypatch.setattr(Path, 'home', staticmethod(lambda: tmp_path / 'home'))
        cwd = tmp_path / 'project'
        (cwd / '.claude').mkdir(parents=True)
        (cwd / '.claude' / 'settings.json').write_text('{invalid json')
        with pytest.raises(json.JSONDecodeError):
            hook_module.load_bash_prefixes(str(cwd))

    def test_non_string_entry_raises(
        self, hook_module: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Non-string entries bubble up TypeError — ErrorBoundary handles in production."""
        monkeypatch.setattr(Path, 'home', staticmethod(lambda: tmp_path / 'home'))
        cwd = tmp_path / 'project'
        (cwd / '.claude').mkdir(parents=True)
        (cwd / '.claude' / 'settings.json').write_text(json.dumps({'permissions': {'allow': [123, 'Bash(echo:*)']}}))
        with pytest.raises(TypeError):
            hook_module.load_bash_prefixes(str(cwd))


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
        """Dangerous command raises ValueError in __wrapped__, caught by ErrorBoundary in production."""
        home = tmp_path / 'home'
        (home / '.claude').mkdir(parents=True)
        (home / '.claude' / 'settings.json').write_text(
            json.dumps({'permissions': {'allow': ['Bash(echo:*)', 'Bash(git log:*)']}})
        )
        monkeypatch.setattr(Path, 'home', staticmethod(lambda: home))
        monkeypatch.setattr('sys.stdin', io.StringIO(self._hook_input('echo $(whoami) && git log')))

        with pytest.raises(hook_module.ApproveCompoundBashException):
            hook_module.main.__wrapped__()

    def test_skips_empty_command(
        self,
        hook_module: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr('sys.stdin', io.StringIO(self._hook_input('')))

        hook_module.main.__wrapped__()
        assert capsys.readouterr().out == ''

    def test_skips_missing_command_key(
        self,
        hook_module: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """tool_input without command key — passthrough."""
        payload = json.dumps(
            {
                'session_id': 'test',
                'cwd': '/tmp',
                'transcript_path': '/tmp/transcript.jsonl',
                'hook_event_name': 'PreToolUse',
                'tool_name': 'Bash',
                'tool_input': {},
                'tool_use_id': 'tu_test',
            }
        )
        monkeypatch.setattr('sys.stdin', io.StringIO(payload))

        hook_module.main.__wrapped__()
        assert capsys.readouterr().out == ''

    def test_skips_non_string_command(
        self,
        hook_module: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Non-string command value — passthrough (type guard)."""
        payload = json.dumps(
            {
                'session_id': 'test',
                'cwd': '/tmp',
                'transcript_path': '/tmp/transcript.jsonl',
                'hook_event_name': 'PreToolUse',
                'tool_name': 'Bash',
                'tool_input': {'command': 12345},
                'tool_use_id': 'tu_test',
            }
        )
        monkeypatch.setattr('sys.stdin', io.StringIO(payload))

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


# ---------------------------------------------------------------------------
# TestParseErrors — bashlex failure modes
# ---------------------------------------------------------------------------


class TestParseErrors:
    """Verify bashlex raises on malformed input (ErrorBoundary catches in production)."""

    def test_unterminated_quote(self, hook_module: ModuleType) -> None:
        with pytest.raises(Exception):
            hook_module.analyze_command('echo "unterminated')

    def test_arithmetic_expansion_passthrough_via_boundary(
        self,
        hook_module: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Arithmetic expansion ($(())) triggers NotImplementedError in bashlex.

        NotImplementedError is a subclass of Exception, so ErrorBoundary must
        catch it and exit 0 (passthrough), not crash with exit 1.
        """
        payload = json.dumps(
            {
                'session_id': 'test',
                'cwd': '/tmp',
                'transcript_path': '/tmp/transcript.jsonl',
                'hook_event_name': 'PreToolUse',
                'tool_name': 'Bash',
                'tool_input': {'command': 'echo $((1+2)) && git log'},
                'tool_use_id': 'tu_test',
            }
        )
        monkeypatch.setattr('sys.stdin', io.StringIO(payload))

        with pytest.raises(SystemExit) as exc_info:
            hook_module.main()

        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert captured.out == ''
        assert 'NotImplementedError' in captured.err

    def test_time_keyword_passthrough_via_boundary(
        self,
        hook_module: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """'time' keyword triggers NotImplementedError — must passthrough safely."""
        payload = json.dumps(
            {
                'session_id': 'test',
                'cwd': '/tmp',
                'transcript_path': '/tmp/transcript.jsonl',
                'hook_event_name': 'PreToolUse',
                'tool_name': 'Bash',
                'tool_input': {'command': 'time git log && echo done'},
                'tool_use_id': 'tu_test',
            }
        )
        monkeypatch.setattr('sys.stdin', io.StringIO(payload))

        with pytest.raises(SystemExit) as exc_info:
            hook_module.main()

        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert captured.out == ''
        assert 'NotImplementedError' in captured.err


# ---------------------------------------------------------------------------
# TestMultiRootCommands — newline-separated compound statements
# ---------------------------------------------------------------------------


class TestMultiRootCommands:
    """Newline-separated statements produce multiple AST roots."""

    def test_newline_separated_compounds(self, hook_module: ModuleType) -> None:
        """Each newline-separated statement is a separate top-level AST node."""
        result = hook_module.analyze_command('echo a && echo b\ngit log && git status')
        assert len(result) == 4

    def test_newline_single_commands(self, hook_module: ModuleType) -> None:
        """Newline-separated simple commands still produce multiple subcommands."""
        result = hook_module.analyze_command('echo a\ngit log')
        assert len(result) == 2

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
        """bashlex raises NotImplementedError for this case syntax."""
        with pytest.raises(NotImplementedError):
            hook_module.analyze_command('case $x in a) echo a;; esac && git log')


# ---------------------------------------------------------------------------
# TestStrengthenedAssertions — content checks, not just counts
# ---------------------------------------------------------------------------


class TestStrengthenedAssertions:
    """Strengthen weak len-only assertions to verify content."""

    def test_or_operator_content(self, hook_module: ModuleType) -> None:
        result = hook_module.analyze_command('echo test || git log')
        assert len(result) == 2
        assert result[0] == 'echo test'
        assert result[1] == 'git log'

    def test_semicolon_content(self, hook_module: ModuleType) -> None:
        result = hook_module.analyze_command('echo test; git log')
        assert len(result) == 2
        assert result[0] == 'echo test'
        assert result[1] == 'git log'

    def test_pipe_content(self, hook_module: ModuleType) -> None:
        result = hook_module.analyze_command('echo test | grep test')
        assert len(result) == 2
        assert result[0] == 'echo test'
        assert result[1] == 'grep test'

    def test_overlapping_prefixes(self, hook_module: ModuleType) -> None:
        """Both 'git' and 'git log' match 'git log --oneline'."""
        assert hook_module.matches_prefix('git log --oneline', {'git', 'git log'})


# ---------------------------------------------------------------------------
# TestSurvivingMutations — kill mutations from mutation testing review
# ---------------------------------------------------------------------------


class TestSurvivingMutations:
    """Tests targeting specific mutations that would survive the original suite."""

    def test_single_command_with_prefixes_skipped(
        self,
        hook_module: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Single command must be skipped even when prefixes are loaded.

        Isolates the len(subcommands) <= 1 guard from the empty-prefixes guard.
        """
        home = tmp_path / 'home'
        (home / '.claude').mkdir(parents=True)
        (home / '.claude' / 'settings.json').write_text(json.dumps({'permissions': {'allow': ['Bash(echo:*)']}}))
        monkeypatch.setattr(Path, 'home', staticmethod(lambda: home))
        payload = json.dumps(
            {
                'session_id': 'test',
                'cwd': str(tmp_path / 'project'),
                'transcript_path': '/tmp/transcript.jsonl',
                'hook_event_name': 'PreToolUse',
                'tool_name': 'Bash',
                'tool_input': {'command': 'echo hello'},
                'tool_use_id': 'tu_test',
            }
        )
        monkeypatch.setattr('sys.stdin', io.StringIO(payload))
        hook_module.main.__wrapped__()
        assert capsys.readouterr().out == ''

    def test_lowercase_transform_operator_dangerous(self, hook_module: ModuleType) -> None:
        """Lowercase transform operators (@a, @k, @q) must also be caught."""
        with pytest.raises(hook_module.ApproveCompoundBashException):
            hook_module.analyze_command('echo ${x@a} && git log')

    def test_empty_prefix_regex_rejected(
        self, hook_module: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Bash(:*) must not produce empty prefix."""
        monkeypatch.setattr(Path, 'home', staticmethod(lambda: tmp_path / 'home'))
        cwd = tmp_path / 'project'
        (cwd / '.claude').mkdir(parents=True)
        (cwd / '.claude' / 'settings.json').write_text(json.dumps({'permissions': {'allow': ['Bash(:*)']}}))
        assert '' not in hook_module.load_bash_prefixes(str(cwd))


# ---------------------------------------------------------------------------
# TestQuotingEdgeCases — quoting, escaping, string handling
# ---------------------------------------------------------------------------


class TestQuotingEdgeCases:
    """Verify bashlex quoting, escaping, and string handling edge cases."""

    def test_dollar_single_quote_structure(self, hook_module: ModuleType) -> None:
        """$'...' ANSI-C quoting: structure is correct regardless of word value."""
        result = hook_module.analyze_command("echo $'hello' && git log")
        assert len(result) == 2

    def test_dollar_double_quote_structure(self, hook_module: ModuleType) -> None:
        """$"..." locale translation: structure correct."""
        result = hook_module.analyze_command('echo $"hello" && git log')
        assert len(result) == 2

    def test_escaped_double_quote(self, hook_module: ModuleType) -> None:
        """Escaped quote in double-quoted string doesn't break structure."""
        result = hook_module.analyze_command('echo "say \\"hello\\"" && git log')
        assert len(result) == 2
        assert result[0].startswith('echo ')

    def test_quote_concatenation(self, hook_module: ModuleType) -> None:
        """Adjacent quoted strings are concatenated into one word."""
        result = hook_module.analyze_command('echo \'part1\'"part2" && git log')
        assert len(result) == 2

    def test_empty_quotes(self, hook_module: ModuleType) -> None:
        result = hook_module.analyze_command('echo "" && git log')
        assert len(result) == 2

    def test_glob_in_argument(self, hook_module: ModuleType) -> None:
        """Glob patterns stay as literal words in AST."""
        result = hook_module.analyze_command('echo *.txt && git log')
        assert len(result) == 2

    def test_brace_expansion(self, hook_module: ModuleType) -> None:
        result = hook_module.analyze_command('echo {a,b,c} && git log')
        assert len(result) == 2

    def test_comment_strips_trailing(self, hook_module: ModuleType) -> None:
        """Comments correctly strip everything after #."""
        result = hook_module.analyze_command('echo a && echo b # && rm -rf /')
        assert len(result) == 2

    def test_no_space_around_operator(self, hook_module: ModuleType) -> None:
        result = hook_module.analyze_command('echo test&&git log')
        assert len(result) == 2

    def test_background_operator(self, hook_module: ModuleType) -> None:
        result = hook_module.analyze_command('echo test & echo test2')
        assert len(result) == 2

    def test_double_quoted_command_name(self, hook_module: ModuleType) -> None:
        """Double-quoted command name is correctly unquoted."""
        result = hook_module.analyze_command('"echo" hello && git log')
        assert result[0] == 'echo hello'

    def test_variable_as_command_name(self, hook_module: ModuleType) -> None:
        """$VAR as command name — won't match normal prefixes."""
        result = hook_module.analyze_command('$CMD && git log')
        assert len(result) == 2
        assert result[0] == '$CMD'

    def test_double_bracket_raises(self, hook_module: ModuleType) -> None:
        """[[ ]] test syntax raises (caught by ErrorBoundary)."""
        with pytest.raises(Exception):
            hook_module.analyze_command('[[ -f /etc/passwd ]] && echo exists')

    def test_indented_heredoc_dangerous(self, hook_module: ModuleType) -> None:
        """Indented heredocs (<<-) are non-fd redirects — always dangerous."""
        with pytest.raises(hook_module.ApproveCompoundBashException):
            hook_module.analyze_command('cat <<-EOF && git log\n\thello\n\tEOF')

    def test_malformed_json_passthrough_via_boundary(
        self,
        hook_module: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Completely invalid JSON on stdin passthroughs via ErrorBoundary."""
        monkeypatch.setattr('sys.stdin', io.StringIO('not json at all'))
        with pytest.raises(SystemExit) as exc_info:
            hook_module.main()
        assert exc_info.value.code == 0
