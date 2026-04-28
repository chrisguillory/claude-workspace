from __future__ import annotations

from pathlib import Path

from claude_remote_bash.diagnose.system import (
    codesign_identifier,
    python_version,
    resolve_interpreter,
    which_all,
)
from claude_remote_bash.diagnose.types import Status, VectorResult

__all__ = [
    'check_interpreter',
]


_CANONICAL_INSTALL = (
    'uv tool install --force '
    '--python /opt/homebrew/opt/python@3.13/bin/python3 '
    '--editable ~/claude-workspace/mcp/claude-remote-bash'
)


def check_interpreter() -> VectorResult:
    """Walk `which -a claude-remote-bash`; report each install's Python interpreter and codesign Identifier."""
    shims = which_all('claude-remote-bash')
    if not shims:
        return VectorResult(
            name='interpreter',
            status=Status.FAIL,
            summary='claude-remote-bash not on PATH',
            detail='',
            fix_suggestion=_CANONICAL_INSTALL,
        )

    descriptions: list[str] = []
    identifiers: list[str] = []
    for shim in shims:
        desc, ident = _describe_shim(Path(shim))
        descriptions.append(desc)
        identifiers.append(ident)

    detail = '\n\n'.join(descriptions)
    distinct = {i for i in identifiers if i}

    if len(distinct) > 1:
        return VectorResult(
            name='interpreter',
            status=Status.WARN,
            summary=f'{len(shims)} installs use {len(distinct)} different codesign Identifiers (LN grant drift)',
            detail=detail,
            fix_suggestion=f'Reinstall canonically: {_CANONICAL_INSTALL}',
        )

    return VectorResult(
        name='interpreter',
        status=Status.OK,
        summary=f'{len(shims)} install(s), unified codesign Identifier',
        detail=detail,
        fix_suggestion='',
    )


def _describe_shim(shim: Path) -> tuple[str, str]:
    """Return (rendered description, codesign Identifier or empty)."""
    interp = resolve_interpreter(shim)
    if interp is None:
        return f'{shim}\n  interpreter: <could not resolve from shebang>', ''

    ident = codesign_identifier(interp)
    version = python_version(interp)
    desc = (
        f'{shim}\n'
        f'  interpreter: {interp}\n'
        f'  identifier:  {ident or "<not signed or codesign failed>"}\n'
        f'  version:     {version}'
    )
    return desc, ident
