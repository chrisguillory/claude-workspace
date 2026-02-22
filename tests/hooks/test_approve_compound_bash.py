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

import bashlex.errors
import bashlex.tokenizer
import git
import pydantic
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


class TestAnalyzeCommand:
    """Verify compound command decomposition via bashlex AST."""

    def test_simple_command(self, hook_module: ModuleType) -> None:
        assert list(hook_module.analyze_command('echo hello')) == [
            'echo hello',
        ]

    def test_and_operator(self, hook_module: ModuleType) -> None:
        assert list(hook_module.analyze_command('echo "---" && git log')) == [
            'echo ---',
            'git log',
        ]

    def test_or_operator(self, hook_module: ModuleType) -> None:
        assert list(hook_module.analyze_command('echo test || git log')) == [
            'echo test',
            'git log',
        ]

    def test_semicolon(self, hook_module: ModuleType) -> None:
        assert list(hook_module.analyze_command('echo test; git log')) == [
            'echo test',
            'git log',
        ]

    def test_pipe(self, hook_module: ModuleType) -> None:
        assert list(hook_module.analyze_command('echo test | grep test')) == [
            'echo test',
            'grep test',
        ]

    def test_three_part_compound(self, hook_module: ModuleType) -> None:
        assert list(hook_module.analyze_command('git diff 2>&1 && echo "---" && git log')) == [
            'git diff',
            'echo ---',
            'git log',
        ]

    def test_nested_quotes(self, hook_module: ModuleType) -> None:
        assert list(hook_module.analyze_command('echo \'hello "world"\' && git status')) == [
            'echo hello "world"',
            'git status',
        ]

    def test_double_quotes(self, hook_module: ModuleType) -> None:
        assert list(hook_module.analyze_command('echo "hello world" && git log')) == [
            'echo hello world',
            'git log',
        ]

    def test_env_var_assignment(self, hook_module: ModuleType) -> None:
        """Assignment raises — inline assignments are dangerous."""
        with pytest.raises(hook_module.ApproveCompoundBashException) as exc_info:
            hook_module.analyze_command('VAR=val echo test && git status')
        assert exc_info.value.args == ('inline assignment in: VAR=val echo test',)

    def test_newline_separated(self, hook_module: ModuleType) -> None:
        assert list(hook_module.analyze_command('echo test\ngit log')) == [
            'echo test',
            'git log',
        ]

    def test_subshell(self, hook_module: ModuleType) -> None:
        assert list(hook_module.analyze_command('echo safe && (echo inner; ls)')) == [
            'echo safe',
            'echo inner',
            'ls',
        ]

    def test_brace_group(self, hook_module: ModuleType) -> None:
        assert list(hook_module.analyze_command('echo safe && { echo inner; ls; }')) == [
            'echo safe',
            'echo inner',
            'ls',
        ]

    def test_assignment_only(self, hook_module: ModuleType) -> None:
        """Assignment-only subcommand raises."""
        with pytest.raises(hook_module.ApproveCompoundBashException) as exc_info:
            hook_module.analyze_command('VAR=hello && echo test')
        assert exc_info.value.args == ('inline assignment in: VAR=hello',)

    def test_if_then_visible(self, hook_module: ModuleType) -> None:
        """if/then blocks raise — unanalyzable construct."""
        with pytest.raises(hook_module.ApproveCompoundBashException) as exc_info:
            hook_module.analyze_command('echo safe && if true; then rm -rf /; fi')
        assert exc_info.value.args == ('unanalyzable construct (if): if true; then rm -rf /; fi',)

    def test_for_loop_visible(self, hook_module: ModuleType) -> None:
        """for loops raise — unanalyzable construct."""
        with pytest.raises(hook_module.ApproveCompoundBashException) as exc_info:
            hook_module.analyze_command('echo safe && for i in a b; do echo $i; done')
        assert exc_info.value.args == ('unanalyzable construct (for): for i in a b; do echo $i; done',)

    def test_while_loop_visible(self, hook_module: ModuleType) -> None:
        """while loops raise — unanalyzable construct."""
        with pytest.raises(hook_module.ApproveCompoundBashException) as exc_info:
            hook_module.analyze_command('echo safe && while true; do echo test; done')
        assert exc_info.value.args == ('unanalyzable construct (while): while true; do echo test; done',)

    def test_function_definition_visible(self, hook_module: ModuleType) -> None:
        """Function definitions raise — unanalyzable construct."""
        with pytest.raises(hook_module.ApproveCompoundBashException) as exc_info:
            hook_module.analyze_command('f() { rm -rf /; } && echo done')
        assert exc_info.value.args == ('unanalyzable construct (function): f() { rm -rf /; }',)

    def test_negated_pipeline(self, hook_module: ModuleType) -> None:
        """! (negation) is a reservedword — inner command must still be extracted."""
        assert list(hook_module.analyze_command('! false && echo done')) == [
            'false',
            'echo done',
        ]


class TestDangerDetection:
    """Verify dangerous construct detection raises ApproveCompoundBashException."""

    def test_fd_to_fd_safe(self, hook_module: ModuleType) -> None:
        """2>&1 is fd duplication, not a file redirect."""
        assert list(hook_module.analyze_command('echo test 2>&1 && git log')) == [
            'echo test',
            'git log',
        ]

    def test_stderr_redirect_safe(self, hook_module: ModuleType) -> None:
        """>&2 is fd duplication."""
        assert list(hook_module.analyze_command('echo test >&2 && git log')) == [
            'echo test',
            'git log',
        ]

    def test_file_redirect_dangerous(self, hook_module: ModuleType) -> None:
        with pytest.raises(hook_module.ApproveCompoundBashException) as exc_info:
            hook_module.analyze_command('echo test > /tmp/file && git log')
        assert exc_info.value.args == ('file redirect in: echo test > /tmp/file',)

    def test_append_redirect_dangerous(self, hook_module: ModuleType) -> None:
        with pytest.raises(hook_module.ApproveCompoundBashException) as exc_info:
            hook_module.analyze_command('echo test >> /tmp/file && git log')
        assert exc_info.value.args == ('file redirect in: echo test >> /tmp/file',)

    def test_input_redirect_dangerous(self, hook_module: ModuleType) -> None:
        with pytest.raises(hook_module.ApproveCompoundBashException) as exc_info:
            hook_module.analyze_command('cat < /tmp/file && git log')
        assert exc_info.value.args == ('file redirect in: cat < /tmp/file',)

    def test_command_substitution_dangerous(self, hook_module: ModuleType) -> None:
        with pytest.raises(hook_module.ApproveCompoundBashException) as exc_info:
            hook_module.analyze_command('echo $(rm -rf /) && git log')
        assert exc_info.value.args == ('unsafe expansion in: $(rm -rf /)',)

    def test_backtick_substitution_dangerous(self, hook_module: ModuleType) -> None:
        with pytest.raises(hook_module.ApproveCompoundBashException) as exc_info:
            hook_module.analyze_command('echo `whoami` && git log')
        assert exc_info.value.args == ('unsafe expansion in: `whoami`',)

    def test_process_substitution_dangerous(self, hook_module: ModuleType) -> None:
        with pytest.raises(hook_module.ApproveCompoundBashException) as exc_info:
            hook_module.analyze_command('cat <(echo test) && git log')
        assert exc_info.value.args == ('unsafe expansion in: <(echo test)',)

    def test_here_string_dangerous(self, hook_module: ModuleType) -> None:
        """Here-strings (<<<) are non-fd redirects — always dangerous."""
        with pytest.raises(hook_module.ApproveCompoundBashException) as exc_info:
            hook_module.analyze_command('cat <<< hello && git log')
        assert exc_info.value.args == ('file redirect in: cat <<< hello',)

    def test_here_string_with_substitution_dangerous(self, hook_module: ModuleType) -> None:
        """Here-strings (<<<) with command substitution are dangerous."""
        with pytest.raises(hook_module.ApproveCompoundBashException) as exc_info:
            hook_module.analyze_command('cat <<< $(whoami) && git log')
        assert exc_info.value.args == ('file redirect in: cat <<< $(whoami)',)

    def test_heredoc_dangerous(self, hook_module: ModuleType) -> None:
        """Heredocs (<<) are non-fd redirects — always dangerous."""
        with pytest.raises(hook_module.ApproveCompoundBashException) as exc_info:
            hook_module.analyze_command('cat <<EOF && git log\nhello\nEOF')
        assert exc_info.value.args == ('file redirect in: cat <<EOF',)

    def test_assignment_substitution_dangerous(self, hook_module: ModuleType) -> None:
        """Command substitution in assignment position is executed at runtime."""
        with pytest.raises(hook_module.ApproveCompoundBashException) as exc_info:
            hook_module.analyze_command('X=$(whoami) echo test && git log')
        assert exc_info.value.args == ('inline assignment in: X=$(whoami) echo test',)

    def test_assignment_unknown_var_dangerous(self, hook_module: ModuleType) -> None:
        """Unknown env var in assignment is dangerous (allowlist model)."""
        with pytest.raises(hook_module.ApproveCompoundBashException) as exc_info:
            hook_module.analyze_command('VAR=hello echo test && git log')
        assert exc_info.value.args == ('inline assignment in: VAR=hello echo test',)

    def test_eval_base_command_preserved(self, hook_module: ModuleType) -> None:
        """eval is parsed as a regular command — base_command includes 'eval'."""
        assert list(hook_module.analyze_command('eval "rm -rf /" && git log')) == [
            'eval rm -rf /',
            'git log',
        ]

    def test_if_block_marked_dangerous(self, hook_module: ModuleType) -> None:
        """if/then blocks are unanalyzable — must raise."""
        with pytest.raises(hook_module.ApproveCompoundBashException) as exc_info:
            hook_module.analyze_command('echo safe && if true; then rm -rf /; fi')
        assert exc_info.value.args == ('unanalyzable construct (if): if true; then rm -rf /; fi',)

    def test_for_loop_marked_dangerous(self, hook_module: ModuleType) -> None:
        """for loops are unanalyzable — must raise."""
        with pytest.raises(hook_module.ApproveCompoundBashException) as exc_info:
            hook_module.analyze_command('echo safe && for i in a b; do echo $i; done')
        assert exc_info.value.args == ('unanalyzable construct (for): for i in a b; do echo $i; done',)

    def test_while_loop_marked_dangerous(self, hook_module: ModuleType) -> None:
        """while loops are unanalyzable — must raise."""
        with pytest.raises(hook_module.ApproveCompoundBashException) as exc_info:
            hook_module.analyze_command('echo safe && while true; do echo test; done')
        assert exc_info.value.args == ('unanalyzable construct (while): while true; do echo test; done',)

    def test_until_loop_marked_dangerous(self, hook_module: ModuleType) -> None:
        """until loops are unanalyzable — must raise."""
        with pytest.raises(hook_module.ApproveCompoundBashException) as exc_info:
            hook_module.analyze_command('echo safe && until false; do echo test; done')
        assert exc_info.value.args == ('unanalyzable construct (until): until false; do echo test; done',)

    def test_function_definition_marked_dangerous(self, hook_module: ModuleType) -> None:
        """Function definitions are unanalyzable — must raise."""
        with pytest.raises(hook_module.ApproveCompoundBashException) as exc_info:
            hook_module.analyze_command('f() { rm -rf /; } && echo done')
        assert exc_info.value.args == ('unanalyzable construct (function): f() { rm -rf /; }',)

    # -- Parameter expansion bypass vectors --

    def test_parameter_default_with_substitution_dangerous(self, hook_module: ModuleType) -> None:
        """${x:-$(cmd)} hides command substitution inside parameter expansion."""
        with pytest.raises(hook_module.ApproveCompoundBashException) as exc_info:
            hook_module.analyze_command('echo ${x:-$(whoami)} && git log')
        assert exc_info.value.args == ('unsafe expansion in: ${x:-$(whoami)}',)

    def test_parameter_assign_default_with_substitution_dangerous(self, hook_module: ModuleType) -> None:
        """${x:=$(cmd)} hides command substitution inside assign-default expansion."""
        with pytest.raises(hook_module.ApproveCompoundBashException) as exc_info:
            hook_module.analyze_command('echo ${x:=$(whoami)} && git log')
        assert exc_info.value.args == ('unsafe expansion in: ${x:=$(whoami)}',)

    def test_assignment_parameter_expansion_dangerous(self, hook_module: ModuleType) -> None:
        """Parameter expansion with substitution in assignment position."""
        with pytest.raises(hook_module.ApproveCompoundBashException) as exc_info:
            hook_module.analyze_command('X=${a:-$(whoami)} echo test && git log')
        assert exc_info.value.args == ('inline assignment in: X=${a:-$(whoami)} echo test',)

    # -- Compound redirect bypass vectors --

    def test_subshell_redirect_dangerous(self, hook_module: ModuleType) -> None:
        """File redirect on subshell is invisible to inner command analysis."""
        with pytest.raises(hook_module.ApproveCompoundBashException) as exc_info:
            hook_module.analyze_command('(echo a; echo b) > /tmp/file && git log')
        assert exc_info.value.args == ('file redirect on compound: (echo a; echo b) > /tmp/file',)

    def test_brace_group_redirect_dangerous(self, hook_module: ModuleType) -> None:
        """File redirect on brace group is invisible to inner command analysis."""
        with pytest.raises(hook_module.ApproveCompoundBashException) as exc_info:
            hook_module.analyze_command('{ echo a; echo b; } > /tmp/file && git log')
        assert exc_info.value.args == ('file redirect on compound: { echo a; echo b; } > /tmp/file',)

    def test_compound_redirect_blocks_auto_approve(self, hook_module: ModuleType) -> None:
        """Compound with redirect must raise."""
        with pytest.raises(hook_module.ApproveCompoundBashException) as exc_info:
            hook_module.analyze_command('(echo a; echo b) > /tmp/file && git log')
        assert exc_info.value.args == ('file redirect on compound: (echo a; echo b) > /tmp/file',)

    def test_compound_fd_redirect_safe(self, hook_module: ModuleType) -> None:
        """fd-to-fd redirect (2>&1) on compound is NOT a file redirect."""
        assert list(hook_module.analyze_command('(echo a; echo b) 2>&1 && git log')) == [
            'echo a',
            'echo b',
            'git log',
        ]

    def test_compound_mixed_fd_then_file_redirect(self, hook_module: ModuleType) -> None:
        """fd redirect first, file redirect second — file redirect must trigger danger."""
        with pytest.raises(hook_module.ApproveCompoundBashException) as exc_info:
            hook_module.analyze_command('(echo a; echo b) 2>&1 >/tmp/file && git log')
        assert exc_info.value.args == ('file redirect on compound: (echo a; echo b) 2>&1 >/tmp/file',)

    def test_compound_multiple_fd_redirects_safe(self, hook_module: ModuleType) -> None:
        """Multiple fd-to-fd redirects on compound — all safe."""
        assert list(hook_module.analyze_command('(echo a; echo b) 2>&1 3>&2 && git log')) == [
            'echo a',
            'echo b',
            'git log',
        ]

    # -- Heredoc content scanning --

    def test_heredoc_with_command_substitution_dangerous(self, hook_module: ModuleType) -> None:
        """Unquoted heredocs expand $() at runtime — content must be scanned."""
        with pytest.raises(hook_module.ApproveCompoundBashException) as exc_info:
            hook_module.analyze_command('cat <<EOF && git log\n$(rm -rf /)\nEOF')
        assert exc_info.value.args == ('file redirect in: cat <<EOF',)

    def test_heredoc_with_backtick_dangerous(self, hook_module: ModuleType) -> None:
        """Unquoted heredocs expand backticks at runtime."""
        with pytest.raises(hook_module.ApproveCompoundBashException) as exc_info:
            hook_module.analyze_command('cat <<EOF && git log\n`whoami`\nEOF')
        assert exc_info.value.args == ('file redirect in: cat <<EOF',)

    def test_heredoc_plain_content_dangerous(self, hook_module: ModuleType) -> None:
        """Heredocs are always dangerous — no content scanning needed."""
        with pytest.raises(hook_module.ApproveCompoundBashException) as exc_info:
            hook_module.analyze_command('cat <<EOF && git log\nhello world\nEOF')
        assert exc_info.value.args == ('file redirect in: cat <<EOF',)

    # -- Environment variable injection --

    def test_ld_preload_dangerous(self, hook_module: ModuleType) -> None:
        """LD_PRELOAD= loads arbitrary .so into subprocess — must be caught."""
        with pytest.raises(hook_module.ApproveCompoundBashException) as exc_info:
            hook_module.analyze_command('LD_PRELOAD=/evil.so git log && echo done')
        assert exc_info.value.args == ('inline assignment in: LD_PRELOAD=/evil.so git log',)

    def test_path_manipulation_dangerous(self, hook_module: ModuleType) -> None:
        """PATH= manipulation changes which binary executes."""
        with pytest.raises(hook_module.ApproveCompoundBashException) as exc_info:
            hook_module.analyze_command('PATH=/evil git log && echo done')
        assert exc_info.value.args == ('inline assignment in: PATH=/evil git log',)

    def test_git_dir_dangerous(self, hook_module: ModuleType) -> None:
        """GIT_DIR= points git at attacker-controlled repository."""
        with pytest.raises(hook_module.ApproveCompoundBashException) as exc_info:
            hook_module.analyze_command('GIT_DIR=/evil/.git git log && echo done')
        assert exc_info.value.args == ('inline assignment in: GIT_DIR=/evil/.git git log',)

    def test_git_ssh_command_dangerous(self, hook_module: ModuleType) -> None:
        """GIT_SSH_COMMAND= executes arbitrary code on remote operations."""
        with pytest.raises(hook_module.ApproveCompoundBashException) as exc_info:
            hook_module.analyze_command('GIT_SSH_COMMAND="rm -rf /" git fetch && echo done')
        assert exc_info.value.args == ('inline assignment in: GIT_SSH_COMMAND="rm -rf /" git fetch',)

    def test_ifs_manipulation_dangerous(self, hook_module: ModuleType) -> None:
        """IFS= changes word splitting, can alter command interpretation."""
        with pytest.raises(hook_module.ApproveCompoundBashException) as exc_info:
            hook_module.analyze_command('IFS=/ echo test && git log')
        assert exc_info.value.args == ('inline assignment in: IFS=/ echo test',)

    def test_ld_library_path_dangerous(self, hook_module: ModuleType) -> None:
        """LD_LIBRARY_PATH= library search path poisoning."""
        with pytest.raises(hook_module.ApproveCompoundBashException) as exc_info:
            hook_module.analyze_command('LD_LIBRARY_PATH=/evil git log && echo done')
        assert exc_info.value.args == ('inline assignment in: LD_LIBRARY_PATH=/evil git log',)

    def test_safe_env_var_dangerous(self, hook_module: ModuleType) -> None:
        """All assignments are dangerous — no env var allowlist."""
        with pytest.raises(hook_module.ApproveCompoundBashException) as exc_info:
            hook_module.analyze_command('TERM=xterm git log && echo done')
        assert exc_info.value.args == ('inline assignment in: TERM=xterm git log',)

    def test_locale_env_var_dangerous(self, hook_module: ModuleType) -> None:
        """All assignments are dangerous — no env var allowlist."""
        with pytest.raises(hook_module.ApproveCompoundBashException) as exc_info:
            hook_module.analyze_command('LC_ALL=C git log && echo done')
        assert exc_info.value.args == ('inline assignment in: LC_ALL=C git log',)

    def test_unknown_env_var_dangerous(self, hook_module: ModuleType) -> None:
        """All assignments are dangerous — no env var allowlist."""
        with pytest.raises(hook_module.ApproveCompoundBashException) as exc_info:
            hook_module.analyze_command('MY_CUSTOM_VAR=hello git log && echo done')
        assert exc_info.value.args == ('inline assignment in: MY_CUSTOM_VAR=hello git log',)

    # -- Code-execution commands (handled by prefix matching, not AST) --

    def test_eval_not_dangerous_at_ast_level(self, hook_module: ModuleType) -> None:
        """eval is plain words at AST level — blocked by prefix matching instead."""
        assert list(hook_module.analyze_command('eval "rm -rf /" && git log')) == [
            'eval rm -rf /',
            'git log',
        ]

    def test_source_not_dangerous_at_ast_level(self, hook_module: ModuleType) -> None:
        """source is plain words at AST level — blocked by prefix matching instead."""
        assert list(hook_module.analyze_command('source /tmp/evil.sh && git log')) == [
            'source /tmp/evil.sh',
            'git log',
        ]

    def test_dot_source_not_dangerous_at_ast_level(self, hook_module: ModuleType) -> None:
        """. is plain words at AST level — blocked by prefix matching instead."""
        assert list(hook_module.analyze_command('. /tmp/evil.sh && git log')) == [
            '. /tmp/evil.sh',
            'git log',
        ]

    def test_bash_c_not_dangerous_at_ast_level(self, hook_module: ModuleType) -> None:
        """bash -c is plain words at AST level — blocked by prefix matching instead."""
        assert list(hook_module.analyze_command('bash -c "rm -rf /" && echo done')) == [
            'bash -c rm -rf /',
            'echo done',
        ]

    def test_trap_not_dangerous_at_ast_level(self, hook_module: ModuleType) -> None:
        """trap is plain words at AST level — blocked by prefix matching instead."""
        assert list(hook_module.analyze_command('trap "rm -rf /" EXIT && echo done')) == [
            'trap rm -rf / EXIT',
            'echo done',
        ]

    def test_exec_not_dangerous_at_ast_level(self, hook_module: ModuleType) -> None:
        """exec is plain words at AST level — blocked by prefix matching instead."""
        assert list(hook_module.analyze_command('exec git log && echo done')) == [
            'exec git log',
            'echo done',
        ]

    # -- Additional coverage: safe constructs --

    def test_tilde_expansion_safe(self, hook_module: ModuleType) -> None:
        """Tilde expansion (~, ~/path) is path resolution, not code execution."""
        assert list(hook_module.analyze_command('echo ~/test && git log')) == [
            'echo ~/test',
            'git log',
        ]

    def test_tilde_bare_safe(self, hook_module: ModuleType) -> None:
        """Bare ~ is safe path resolution."""
        assert list(hook_module.analyze_command('ls ~ && git status')) == [
            'ls ~',
            'git status',
        ]

    def test_simple_parameter_expansion_safe(self, hook_module: ModuleType) -> None:
        """Simple $VAR expansion (no substitution) is safe."""
        assert list(hook_module.analyze_command('echo $HOME && git log')) == [
            'echo $HOME',
            'git log',
        ]

    def test_assignment_unknown_var_with_parameter_dangerous(self, hook_module: ModuleType) -> None:
        """Unknown env var is dangerous regardless of safe value content."""
        with pytest.raises(hook_module.ApproveCompoundBashException) as exc_info:
            hook_module.analyze_command('VAR=$HOME echo test && git log')
        assert exc_info.value.args == ('inline assignment in: VAR=$HOME echo test',)

    def test_assignment_process_substitution_dangerous(self, hook_module: ModuleType) -> None:
        """Process substitution in assignment position is dangerous."""
        with pytest.raises(hook_module.ApproveCompoundBashException) as exc_info:
            hook_module.analyze_command('X=<(echo test) echo done && git log')
        assert exc_info.value.args == ('inline assignment in: X=<(echo test) echo done',)

    def test_here_string_process_substitution_dangerous(self, hook_module: ModuleType) -> None:
        """Here-strings (<<<) with process substitution are dangerous."""
        with pytest.raises(hook_module.ApproveCompoundBashException) as exc_info:
            hook_module.analyze_command('cat <<< <(echo test) && git log')
        assert exc_info.value.args == ('file redirect in: cat <<< <(echo test)',)

    def test_backtick_in_parameter_expansion_dangerous(self, hook_module: ModuleType) -> None:
        """Backticks inside parameter expansion are caught."""
        with pytest.raises(hook_module.ApproveCompoundBashException) as exc_info:
            hook_module.analyze_command('echo ${x:-`whoami`} && git log')
        assert exc_info.value.args == ('unsafe expansion in: ${x:-`whoami`}',)

    # -- Parameter transform operators (@P, @E, @A, etc.) --

    def test_prompt_expansion_dangerous(self, hook_module: ModuleType) -> None:
        """${x@P} interprets value as PS1 prompt, executing embedded $(cmd)."""
        with pytest.raises(hook_module.ApproveCompoundBashException) as exc_info:
            hook_module.analyze_command('echo ${PS1@P} && git log')
        assert exc_info.value.args == ('unsafe expansion in: ${PS1@P}',)

    def test_assign_transform_dangerous(self, hook_module: ModuleType) -> None:
        """${x@A} produces assignment statement — could leak values."""
        with pytest.raises(hook_module.ApproveCompoundBashException) as exc_info:
            hook_module.analyze_command('echo ${x@A} && git log')
        assert exc_info.value.args == ('unsafe expansion in: ${x@A}',)

    def test_escape_transform_dangerous(self, hook_module: ModuleType) -> None:
        """${x@E} interprets escape sequences — fail closed on all @ operators."""
        with pytest.raises(hook_module.ApproveCompoundBashException) as exc_info:
            hook_module.analyze_command('echo ${x@E} && git log')
        assert exc_info.value.args == ('unsafe expansion in: ${x@E}',)


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

    def test_time_raises_bashlex_boundary(self, hook_module: ModuleType) -> None:
        """bashlex doesn't support 'time' — translated via LibraryBoundary."""
        with pytest.raises(hook_module.BashlexBoundaryException) as exc_info:
            hook_module.analyze_command('time git log && echo done')
        assert isinstance(exc_info.value.__cause__, NotImplementedError)

    def test_wrapper_without_prefix_no_match(self, hook_module: ModuleType) -> None:
        """Wrapped command won't match unwrapped prefix — safe direction."""
        result = hook_module.analyze_command('timeout 5s git log && echo done')
        prefixes = {'git log'}  # No 'timeout' prefix
        assert not hook_module.matches_prefix(result[0], prefixes)


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
        """One dangerous subcommand raises for the entire compound."""
        with pytest.raises(hook_module.ApproveCompoundBashException) as exc_info:
            hook_module.analyze_command('git log && echo $(whoami) && git status')
        assert exc_info.value.args == ('unsafe expansion in: $(whoami)',)


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

    def test_malformed_second_file_discards_all_prefixes(
        self, hook_module: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """One corrupted settings file discards all prefixes, not just its own.

        Intentional: uncertain config state defers to built-in permissions
        rather than proceeding with partial configuration.
        """
        home = tmp_path / 'home'
        (home / '.claude').mkdir(parents=True)
        (home / '.claude' / 'settings.json').write_text(json.dumps({'permissions': {'allow': ['Bash(git log:*)']}}))
        monkeypatch.setattr(Path, 'home', staticmethod(lambda: home))

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
        """Dangerous command raises in __wrapped__, caught by ErrorBoundary in production."""
        home = tmp_path / 'home'
        (home / '.claude').mkdir(parents=True)
        (home / '.claude' / 'settings.json').write_text(
            json.dumps({'permissions': {'allow': ['Bash(echo:*)', 'Bash(git log:*)']}})
        )
        monkeypatch.setattr(Path, 'home', staticmethod(lambda: home))
        monkeypatch.setattr('sys.stdin', io.StringIO(self._hook_input('echo $(whoami) && git log')))

        with pytest.raises(hook_module.ApproveCompoundBashException) as exc_info:
            hook_module.main.__wrapped__()
        assert exc_info.value.args == ('unsafe expansion in: $(whoami)',)

    def test_skips_empty_command(
        self,
        hook_module: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr('sys.stdin', io.StringIO(self._hook_input('')))

        hook_module.main.__wrapped__()
        assert capsys.readouterr().out == ''

    def test_missing_command_key_raises(
        self,
        hook_module: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """tool_input without command key — ValidationError defers via ErrorBoundary."""
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

        with pytest.raises(pydantic.ValidationError):
            hook_module.main.__wrapped__()

    def test_non_string_command_raises(
        self,
        hook_module: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Non-string command value — ValidationError defers via ErrorBoundary."""
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

        with pytest.raises(pydantic.ValidationError):
            hook_module.main.__wrapped__()

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
        assert captured.err == (
            'approve-compound-bash hook error: '
            "BashlexBoundaryException('unexpected EOF while looking for matching \\'\"\\' (position 18)') "
            'from MatchedPairError\n'
        )

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
        assert captured.err == (
            'approve-compound-bash hook error: 1 validation error for PreToolUseHookInput\n'
            'new_future_field\n'
            '  Extra inputs are not permitted '
            "[type=extra_forbidden, input_value='surprise', input_type=str]\n"
            '    For further information visit https://errors.pydantic.dev/2.12/v/extra_forbidden\n'
        )


class TestRealWorldCommands:
    """Commands that triggered the original Claude Code permission bug."""

    def test_quoted_echo(self, hook_module: ModuleType) -> None:
        assert list(hook_module.analyze_command('echo "---" && git log --oneline -5')) == [
            'echo ---',
            'git log --oneline -5',
        ]

    def test_three_part_with_redirects(self, hook_module: ModuleType) -> None:
        assert list(
            hook_module.analyze_command('git diff --cached --stat 2>&1 && echo "---" && git log --oneline -5 2>&1')
        ) == [
            'git diff --cached --stat',
            'echo ---',
            'git log --oneline -5',
        ]

    def test_nested_quotes(self, hook_module: ModuleType) -> None:
        assert list(hook_module.analyze_command('echo \'hello "world"\' && git status --short 2>&1')) == [
            'echo hello "world"',
            'git status --short',
        ]


class TestBashlexExceptions:
    """Verify the hook handles bashlex exceptions — raises BashlexBoundaryException."""

    def test_unterminated_quote(self, hook_module: ModuleType) -> None:
        with pytest.raises(hook_module.BashlexBoundaryException) as exc_info:
            hook_module.analyze_command('echo "unterminated')
        assert isinstance(exc_info.value.__cause__, bashlex.tokenizer.MatchedPairError)

    def test_empty_string(self, hook_module: ModuleType) -> None:
        with pytest.raises(hook_module.BashlexBoundaryException) as exc_info:
            hook_module.analyze_command('')
        assert isinstance(exc_info.value.__cause__, AttributeError)

    def test_trailing_operator(self, hook_module: ModuleType) -> None:
        with pytest.raises(hook_module.BashlexBoundaryException) as exc_info:
            hook_module.analyze_command('echo test &&')
        assert isinstance(exc_info.value.__cause__, bashlex.errors.ParsingError)

    def test_arithmetic_expansion(self, hook_module: ModuleType) -> None:
        """bashlex does not support arithmetic expansion — translated via LibraryBoundary."""
        with pytest.raises(hook_module.BashlexBoundaryException) as exc_info:
            hook_module.analyze_command('echo $((1+2)) && git log')
        assert isinstance(exc_info.value.__cause__, NotImplementedError)

    def test_case_statement(self, hook_module: ModuleType) -> None:
        """bashlex raises NotImplementedError for case syntax — translated via LibraryBoundary."""
        with pytest.raises(hook_module.BashlexBoundaryException) as exc_info:
            hook_module.analyze_command('case $x in a) echo a;; esac && git log')
        assert isinstance(exc_info.value.__cause__, NotImplementedError)

    def test_double_bracket(self, hook_module: ModuleType) -> None:
        """[[ ]] test syntax raises — translated via LibraryBoundary."""
        with pytest.raises(hook_module.BashlexBoundaryException) as exc_info:
            hook_module.analyze_command('[[ -f /etc/passwd ]] && echo exists')
        assert isinstance(exc_info.value.__cause__, bashlex.errors.ParsingError)

    def test_arithmetic_expansion_passthrough_via_boundary(
        self,
        hook_module: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Arithmetic expansion ($(())) is translated to BashlexBoundaryException.

        ErrorBoundary catches it and exits 0 (passthrough), deferring to
        Claude Code's native permissions.
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
        assert (
            captured.err
            == "approve-compound-bash hook error: BashlexBoundaryException('arithmetic expansion') from NotImplementedError\n"
        )

    def test_time_keyword_passthrough_via_boundary(
        self,
        hook_module: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """'time' keyword is translated to BashlexBoundaryException — passthrough safely."""
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
        assert (
            captured.err
            == "approve-compound-bash hook error: BashlexBoundaryException('type = {time command}, token = {time}') from NotImplementedError\n"
        )


class TestMultiRootCommands:
    """Newline-separated statements produce multiple AST roots."""

    def test_newline_separated_compounds(self, hook_module: ModuleType) -> None:
        """Each newline-separated statement is a separate top-level AST node."""
        assert list(hook_module.analyze_command('echo a && echo b\ngit log && git status')) == [
            'echo a',
            'echo b',
            'git log',
            'git status',
        ]

    def test_newline_single_commands(self, hook_module: ModuleType) -> None:
        """Newline-separated simple commands still produce multiple subcommands."""
        assert list(hook_module.analyze_command('echo a\ngit log')) == [
            'echo a',
            'git log',
        ]


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
        with pytest.raises(hook_module.ApproveCompoundBashException) as exc_info:
            hook_module.analyze_command('echo ${x@a} && git log')
        assert exc_info.value.args == ('unsafe expansion in: ${x@a}',)

    def test_empty_prefix_regex_rejected(
        self, hook_module: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Bash(:*) must not produce empty prefix."""
        monkeypatch.setattr(Path, 'home', staticmethod(lambda: tmp_path / 'home'))
        cwd = tmp_path / 'project'
        (cwd / '.claude').mkdir(parents=True)
        (cwd / '.claude' / 'settings.json').write_text(json.dumps({'permissions': {'allow': ['Bash(:*)']}}))
        assert '' not in hook_module.load_bash_prefixes(str(cwd))

    def test_overlapping_prefixes(self, hook_module: ModuleType) -> None:
        """Both 'git' and 'git log' match 'git log --oneline'."""
        assert hook_module.matches_prefix('git log --oneline', {'git', 'git log'})


class TestQuotingEdgeCases:
    """Verify bashlex quoting, escaping, and string handling edge cases."""

    def test_dollar_single_quote_structure(self, hook_module: ModuleType) -> None:
        """$'...' ANSI-C quoting: structure is correct regardless of word value."""
        assert list(hook_module.analyze_command("echo $'hello' && git log")) == [
            'echo $hello',
            'git log',
        ]

    def test_dollar_double_quote_structure(self, hook_module: ModuleType) -> None:
        """$"..." locale translation: structure correct."""
        assert list(hook_module.analyze_command('echo $"hello" && git log')) == [
            'echo $hello',
            'git log',
        ]

    def test_escaped_double_quote(self, hook_module: ModuleType) -> None:
        """Escaped quote in double-quoted string doesn't break structure."""
        assert list(hook_module.analyze_command('echo "say \\"hello\\"" && git log')) == [
            'echo say "hello"',
            'git log',
        ]

    def test_quote_concatenation(self, hook_module: ModuleType) -> None:
        """Adjacent quoted strings are concatenated into one word."""
        assert list(hook_module.analyze_command('echo \'part1\'"part2" && git log')) == [
            'echo part1part2',
            'git log',
        ]

    def test_empty_quotes(self, hook_module: ModuleType) -> None:
        assert list(hook_module.analyze_command('echo "" && git log')) == [
            'echo ',
            'git log',
        ]

    def test_glob_in_argument(self, hook_module: ModuleType) -> None:
        """Glob patterns stay as literal words in AST."""
        assert list(hook_module.analyze_command('echo *.txt && git log')) == [
            'echo *.txt',
            'git log',
        ]

    def test_brace_expansion(self, hook_module: ModuleType) -> None:
        assert list(hook_module.analyze_command('echo {a,b,c} && git log')) == [
            'echo {a,b,c}',
            'git log',
        ]

    def test_comment_strips_trailing(self, hook_module: ModuleType) -> None:
        """Comments correctly strip everything after #."""
        assert list(hook_module.analyze_command('echo a && echo b # && rm -rf /')) == [
            'echo a',
            'echo b',
        ]

    def test_no_space_around_operator(self, hook_module: ModuleType) -> None:
        assert list(hook_module.analyze_command('echo test&&git log')) == [
            'echo test',
            'git log',
        ]

    def test_background_operator(self, hook_module: ModuleType) -> None:
        assert list(hook_module.analyze_command('echo test & echo test2')) == [
            'echo test',
            'echo test2',
        ]

    def test_double_quoted_command_name(self, hook_module: ModuleType) -> None:
        """Double-quoted command name is correctly unquoted."""
        assert list(hook_module.analyze_command('"echo" hello && git log')) == [
            'echo hello',
            'git log',
        ]

    def test_variable_as_command_name(self, hook_module: ModuleType) -> None:
        """$VAR as command name — won't match normal prefixes."""
        assert list(hook_module.analyze_command('$CMD && git log')) == [
            '$CMD',
            'git log',
        ]

    def test_indented_heredoc_dangerous(self, hook_module: ModuleType) -> None:
        """Indented heredocs (<<-) are non-fd redirects — always dangerous."""
        with pytest.raises(hook_module.ApproveCompoundBashException) as exc_info:
            hook_module.analyze_command('cat <<-EOF && git log\n\thello\n\tEOF')
        assert exc_info.value.args == ('file redirect in: cat <<-EOF',)

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
