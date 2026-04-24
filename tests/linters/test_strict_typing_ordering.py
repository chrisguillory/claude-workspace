"""Ordering tests for strict_typing_linter.py.

Focus: the private-base-class exemption (_collect_private_base_refs). A private
class used as the base of a public class declared later in the same file must
be exempt from the "private after public" ordering rule, because Python
requires the base to be defined before its subclass.

These tests use the subprocess runner pattern from test_strict_typing_linter.py
so they validate the linter's full pipeline, not just one function in isolation.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

TEST_DIR = Path(__file__).parent
CC_DIR = TEST_DIR.parent.parent
LINTER = CC_DIR / 'linters' / 'strict_typing_linter.py'


def run_linter(source_file: Path) -> tuple[int, str]:
    """Run linter on a file and return (exit_code, combined_output)."""
    result = subprocess.run(
        [sys.executable, str(LINTER), '--no-skip-file', '--no-config', str(source_file)],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    return result.returncode, result.stdout + result.stderr


def write_source(tmp_path: Path, name: str, source: str) -> Path:
    """Write source to a tmp file and return its path."""
    path = tmp_path / name
    path.write_text(source)
    return path


def has_ordering_violation(output: str) -> bool:
    """Check if output contains any ordering-related error."""
    return any(
        phrase in output
        for phrase in (
            'out of order',
            'should come after __all__',
            'should come after public',
            'should come after classes',
        )
    )


# -- EXEMPT: private base class of public subclass ----------------------------


def test_private_base_with_public_subclass_is_exempt(tmp_path: Path) -> None:
    """A private base class declared before its public subclasses is exempt."""
    source = '''"""Module with private base class pattern."""

from __future__ import annotations

__all__ = ['ChildA', 'ChildB']


class _Base:
    """Private base — implementation detail."""

    x: int = 0


class ChildA(_Base):
    """Public subclass."""

    a: str = ''


class ChildB(_Base):
    """Another public subclass."""

    b: str = ''
'''
    source_path = write_source(tmp_path, 'exempt_private_base.py', source)
    _, output = run_linter(source_path)
    assert not has_ordering_violation(output), f'Unexpected ordering violation:\n{output}'


def test_private_base_with_multiple_public_subclasses(tmp_path: Path) -> None:
    """Multiple public subclasses of a single private base — still exempt."""
    source = '''"""Five public subclasses sharing one private base."""

from __future__ import annotations

__all__ = ['C1', 'C2', 'C3', 'C4', 'C5']


class _Shared:
    common: int = 0


class C1(_Shared): pass
class C2(_Shared): pass
class C3(_Shared): pass
class C4(_Shared): pass
class C5(_Shared): pass
'''
    source_path = write_source(tmp_path, 'multi_subclass.py', source)
    _, output = run_linter(source_path)
    assert not has_ordering_violation(output), f'Unexpected ordering violation:\n{output}'


def test_chain_of_private_bases(tmp_path: Path) -> None:
    """Private base inheriting from another private base — both exempt."""
    source = '''"""Chain: _Grandparent -> _Parent -> Public."""

from __future__ import annotations

__all__ = ['Public']


class _Grandparent:
    g: int = 0


class _Parent(_Grandparent):
    p: int = 0


class Public(_Parent):
    pub: str = ''
'''
    source_path = write_source(tmp_path, 'chain.py', source)
    _, output = run_linter(source_path)
    assert not has_ordering_violation(output), f'Unexpected ordering violation:\n{output}'


# -- VIOLATION: private class NOT used as a base is still ordered --------------


def test_private_class_not_used_as_base_still_ordered(tmp_path: Path) -> None:
    """Private class that is NOT a base for anything must still obey ordering.

    This is the baseline that proves the exemption is scoped to bases only —
    the linter still catches genuine ordering violations.
    """
    source = '''"""Private class declared before public class, not as a base."""

from __future__ import annotations

__all__ = ['Public']


class _Helper:
    """Private helper, not used as a base anywhere."""

    x: int = 0


class Public:
    """Public class that does NOT inherit from _Helper."""

    y: str = ''
'''
    source_path = write_source(tmp_path, 'violation_private_before_public.py', source)
    _, output = run_linter(source_path)
    assert has_ordering_violation(output), (
        f'Expected ordering violation for private _Helper before public class, got:\n{output}'
    )


def test_private_class_used_only_in_annotation_still_ordered(tmp_path: Path) -> None:
    """Private class used only in a type annotation (not as base) still ordered.

    Confirms the exemption is scoped to ``.bases`` specifically, matching our
    intent. Type annotations on public classes do not grant exemption.
    """
    source = '''"""Private class used only in annotation, not as base."""

from __future__ import annotations

__all__ = ['Public']


class _Color:
    name: str = ''


class Public:
    color: '_Color' = None  # annotation only, not a base
'''
    source_path = write_source(tmp_path, 'annotation_only.py', source)
    _, output = run_linter(source_path)
    assert has_ordering_violation(output), (
        f'Expected ordering violation for private _Color used only in annotation, got:\n{output}'
    )


# -- EXEMPT: mixed private base + private referenced-as-type -------------------


def test_private_base_exempt_but_private_helper_still_ordered(tmp_path: Path) -> None:
    """In a file with both patterns, the exemption is precisely scoped.

    _Base is a base of Public → exempt.
    _Helper is used only in annotations → still flagged.
    """
    source = '''"""Mixed: one private class is exempt, another is not."""

from __future__ import annotations

__all__ = ['Public']


class _Base:
    x: int = 0


class _Helper:
    y: int = 0


class Public(_Base):
    helper: '_Helper' = None
'''
    source_path = write_source(tmp_path, 'mixed.py', source)
    _, output = run_linter(source_path)
    # Should still flag _Helper but not _Base
    assert has_ordering_violation(output), f'Expected ordering violation for _Helper (annotation-only), got:\n{output}'


# -- Regression: existing function exemption still works ----------------------


def test_existing_function_exemption_still_works(tmp_path: Path) -> None:
    """Pre-existing _collect_import_time_refs exemption still applies.

    A private function referenced by Annotated[..., BeforeValidator(_fn)]
    or as a decorator must remain exempt.
    """
    source = '''"""Private function used in decorator — existing exemption."""

from __future__ import annotations

__all__ = ['DecoratedClass']


def _my_decorator(cls):
    return cls


@_my_decorator
class DecoratedClass:
    x: int = 0
'''
    source_path = write_source(tmp_path, 'decorator.py', source)
    _, output = run_linter(source_path)
    assert not has_ordering_violation(output), f'Pre-existing decorator exemption regressed:\n{output}'


# -- Real-world pattern from cc-lib -------------------------------------------


def test_hook_input_base_pattern(tmp_path: Path) -> None:
    """Exact pattern used in cc_lib/schemas/hooks.py after the refactor."""
    source = '''"""Real-world pattern: Anthropic cz() composition."""

from __future__ import annotations

from typing import Literal

__all__ = [
    'PostToolUseHookInput',
    'PreToolUseHookInput',
    'SessionEndHookInput',
    'SessionStartHookInput',
]


class _HookInputBase:
    """Shared fields for every hook input."""

    session_id: str
    cwd: str
    transcript_path: str


class SessionStartHookInput(_HookInputBase):
    hook_event_name: Literal['SessionStart']


class SessionEndHookInput(_HookInputBase):
    hook_event_name: Literal['SessionEnd']


class PreToolUseHookInput(_HookInputBase):
    hook_event_name: Literal['PreToolUse']


class PostToolUseHookInput(_HookInputBase):
    hook_event_name: Literal['PostToolUse']
'''
    source_path = write_source(tmp_path, 'hooks_pattern.py', source)
    _, output = run_linter(source_path)
    assert not has_ordering_violation(output), f'hooks.py pattern rejected:\n{output}'


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
