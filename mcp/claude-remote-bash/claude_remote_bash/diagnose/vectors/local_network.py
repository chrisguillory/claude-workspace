from __future__ import annotations

import os
import plistlib
import sys
from collections.abc import Set
from pathlib import Path
from plistlib import UID

from claude_remote_bash.diagnose.system import codesign_identifier
from claude_remote_bash.diagnose.types import Status, VectorResult

__all__ = [
    'check_local_network',
]


_NE_PLIST = Path('/Library/Preferences/com.apple.networkextension.plist')


def check_local_network() -> VectorResult:
    """Compare the current interpreter's codesign Identifier to the granted rules in the NE plist."""
    if sys.platform != 'darwin':
        return VectorResult(
            name='local-network',
            status=Status.INFO,
            summary='not macOS; Local Network Privacy does not apply',
            detail='',
            fix_suggestion='',
        )

    if not _NE_PLIST.exists():
        return VectorResult(
            name='local-network',
            status=Status.INFO,
            summary=f'{_NE_PLIST} does not exist (no Local Network rules configured)',
            detail='',
            fix_suggestion='',
        )

    if not os.access(_NE_PLIST, os.R_OK):
        return VectorResult(
            name='local-network',
            status=Status.WARN,
            summary='cannot read NE plist; the calling terminal needs Full Disk Access',
            detail=f'plist: {_NE_PLIST}',
            fix_suggestion=(
                'Open System Settings → Privacy & Security → Full Disk Access and add your terminal application.'
            ),
        )

    binary = Path(sys.executable)
    binary_ident = codesign_identifier(binary)
    framework_ident = _framework_identifier(binary)
    granted = _granted_identifiers()

    matches = {i for i in (binary_ident, framework_ident) if i and i in granted}

    detail = (
        f'current interpreter:   {sys.executable}\n'
        f'binary Identifier:     {binary_ident or "<not signed>"}\n'
        f'framework Identifier:  {framework_ident or "<not inside a framework>"}\n'
        f'granted Identifiers:   {", ".join(sorted(granted)) if granted else "<none>"}'
    )

    if matches:
        return VectorResult(
            name='local-network',
            status=Status.OK,
            summary=f'matched granted Identifier: {", ".join(sorted(matches))}',
            detail=detail,
            fix_suggestion='',
        )

    if not binary_ident and not framework_ident:
        return VectorResult(
            name='local-network',
            status=Status.WARN,
            summary='current interpreter has no codesign Identifier (LN grants attach to the Identifier string)',
            detail=detail,
            fix_suggestion='Reinstall under a codesigned Python (Homebrew framework Python has a stable Identifier).',
        )

    return VectorResult(
        name='local-network',
        status=Status.FAIL,
        summary='neither binary nor framework Identifier is in the granted set',
        detail=detail,
        fix_suggestion='claude-remote-bash diagnose --request-local-network',
    )


def _framework_identifier(binary: Path) -> str:
    """If `binary` lives inside a *.framework, return that framework bundle's codesign Identifier."""
    for parent in binary.resolve().parents:
        if parent.suffix == '.framework':
            return codesign_identifier(parent)
    return ''


def _granted_identifiers() -> Set[str]:
    """Return SigningIdentifiers for every NEPathRule with MulticastPreferenceSet=True."""
    raw = plistlib.loads(_NE_PLIST.read_bytes())
    objs = raw['$objects']
    out: set[str] = set()
    for obj in objs:
        if not isinstance(obj, dict):
            continue
        if obj.get('MulticastPreferenceSet') is not True:
            continue
        sid_ref = obj.get('SigningIdentifier')
        if isinstance(sid_ref, UID):
            sid = objs[sid_ref.data]
            if isinstance(sid, str) and sid:
                out.add(sid)
    return out
