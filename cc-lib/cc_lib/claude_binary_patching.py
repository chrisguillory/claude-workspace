"""Claude Code binary patching definitions and scan logic.

Patch definitions describe same-length byte replacements in the Claude Code
Mach-O binary. Each patch locates a stable anchor string, then detects or
overwrites nearby target bytes.

Used by claude-binary-patcher (applies patches) and claude-version-manager
(detects patch status). Adding a new patch: add a PatchDef to PATCHES.

Binary Structure:
    Claude Code is a Node.js SEA compiled to Mach-O arm64 (~190MB). The
    minified JS bundle (~10MB) is duplicated in the ``__BUN`` segment from
    2.1.0+, so each patch has 2 sites. Minified identifiers (2-4 chars)
    change per build; string literals (flag names, JSX props) are stable
    and serve as anchors.

Same-Length Constraint:
    Old and new byte sequences MUST be the same length. Direct byte
    replacement preserves Mach-O segment offsets, page hashes, and code
    signing. After replacement, ad-hoc codesign restores executable validity.

    If length-changing patches are ever needed, adopt extract/modify/repack
    via Python ``lief`` (pull JS from Mach-O, modify freely, rebuild with
    corrected segment offsets).

Patches:
    statusline      Fix multi-line truncation (Ink wrap prop regression).
                    Anchor: ``statusLine?.padding`` (stable since 2.0.0).
                    Regression introduced in 2.1.51 (last clean: 2.1.50).
                    https://github.com/anthropics/claude-code/issues/28750

    session-memory  Enable background session memory extraction. Claude
                    writes summaries to session-memory/summary.md, loaded
                    at the start of future sessions for cross-session context.
                    Statsig gate ``tengu_session_memory``, default false.
                    Not publicly documented — companion to auto-memory (GA 2.1.59).
                    Flag introduced in 2.0.64 (last absent: 2.0.62).
                    Related flags: ``tengu_sm_compact``, ``tengu_sm_config``.
                    https://claudefa.st/blog/guide/mechanics/session-memory
                    https://giuseppegurgone.com/claude-memory

    remember-skill  Enable /remember skill for session memory search.
                    Statsig gate ``tengu_coral_fern``, default false.
                    Not publicly documented or mentioned in any changelog.
                    Flag introduced in 2.1.21 (last absent: 2.1.20).

Version-Agnostic Patterns:
    Gate function names (``Tq``, ``lT``, ``Jq``) change per build. Patches
    match the stable argument list instead: ``("flag_name",!1)`` → ``!0``.
    Verified across 2.1.74, 2.1.80, 2.1.81.

References:
    https://github.com/marckrenn/claude-code-changelog (feature flag tracking)
    https://gist.github.com/gastonmorixe/9c596b6de1095b6bd3b746ca3a1fd3d7

Anchor Presence Survey (2026-03-24, 22+ versions via CDN)::

    Anchor                    First version   Last absent
    statusLine?.padding       2.0.0           never absent
    tengu_session_memory      2.0.64          2.0.62
    tengu_coral_fern          2.1.21          2.1.20

Site Count Evolution::

    Version   statusline   session-memory   remember-skill
    2.0.64    0            6                0
    2.0.70    0            9                0
    2.1.0     0            9                0
    2.1.21    0            —                3
    2.1.40    0            18               3
    2.1.51    2            18               9
    2.1.74    2            18               3
    2.1.80    2            18               3
    2.1.81    2            18               3

Version Log::

    2.1.81 (2026-03-24)
        statusline: 2 sites, applied
        session-memory: 4 sites, unpatched (gate: lT)
        remember-skill: 2 sites, unpatched (gate: lT)

    2.1.80 (2026-03-24)
        statusline: 2 sites, applied
        session-memory: 4 sites (gate: Tq)
        remember-skill: 2 sites (gate: Tq)

    2.1.74 (2026-03-24)
        statusline: N/A (predates regression)
        session-memory: present (gate: Jq)
        remember-skill: present (gate: Jq)
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from cc_lib.types import CCVersion

__all__ = [
    'PATCHES',
    'PATCHES_BY_NAME',
    'PatchDef',
    'PatchScanResult',
    'scan_binary',
]


@dataclass(frozen=True, slots=True)
class PatchDef:
    """A binary patch definition: same-length byte replacement near an anchor."""

    name: str
    description: str
    anchor: bytes
    old: bytes
    new: bytes
    window: int = 200
    min_version: CCVersion | None = None
    max_version: CCVersion | None = None

    def __post_init__(self) -> None:
        if len(self.old) != len(self.new):
            raise ValueError(
                f'Patch {self.name!r}: old ({len(self.old)}B) and new ({len(self.new)}B) '
                f'must be the same length for binary replacement'
            )


PATCHES: Sequence[PatchDef] = (
    PatchDef(
        name='statusline',
        description='Restore multi-line statusline wrapping (fix truncation)',
        anchor=b'statusLine?.padding',
        old=b'wrap:"truncate"',
        new=b'wrap:"wrap"    ',
        min_version='2.1.51',
    ),
    PatchDef(
        name='session-memory',
        description='Enable background session memory extraction',
        anchor=b'tengu_session_memory',
        old=b'("tengu_session_memory",!1)',
        new=b'("tengu_session_memory",!0)',
        window=50,
        min_version='2.0.64',
    ),
    PatchDef(
        name='remember-skill',
        description='Enable /remember skill for session memory search',
        anchor=b'tengu_coral_fern',
        old=b'("tengu_coral_fern",!1)',
        new=b'("tengu_coral_fern",!0)',
        window=50,
        min_version='2.1.21',
    ),
)

PATCHES_BY_NAME: Mapping[str, PatchDef] = {p.name: p for p in PATCHES}


@dataclass(frozen=True, slots=True)
class PatchScanResult:
    """Per-patch scan outcome."""

    patch: PatchDef
    sites: Sequence[int]
    status: Literal['unpatched', 'applied', 'changed', 'missing']
    # unpatched = anchor found, old bytes found -> patch not yet applied
    # applied   = anchor found, new bytes found -> patch is applied
    # changed   = anchor found, neither old nor new bytes found -> code changed
    # missing   = anchor NOT found (binary structure changed)


def scan_binary(
    data: bytes,
    patches: Sequence[PatchDef] | None = None,
) -> Mapping[str, PatchScanResult]:
    """Scan binary data for patch status. Pure function, no I/O.

    Args:
        data: Raw binary content.
        patches: Patches to check. Defaults to all PATCHES.

    Returns:
        Mapping of patch name to scan result.
    """
    if patches is None:
        patches = PATCHES
    results: dict[str, PatchScanResult] = {}
    for patch in patches:
        old_sites = _find_sites(data, patch, patch.old)
        if old_sites:
            results[patch.name] = PatchScanResult(patch=patch, sites=old_sites, status='unpatched')
        elif _has_anchor(data, patch):
            new_sites = _find_sites(data, patch, patch.new)
            status: Literal['applied', 'changed'] = 'applied' if new_sites else 'changed'
            results[patch.name] = PatchScanResult(patch=patch, sites=(), status=status)
        else:
            results[patch.name] = PatchScanResult(patch=patch, sites=(), status='missing')
    return results


def _find_sites(data: bytes, patch: PatchDef, target: bytes) -> Sequence[int]:
    """Find all offsets where target appears within patch.window of patch.anchor.

    Bidirectional search from each anchor occurrence. The anchor may be
    embedded inside the target bytes. Results are deduplicated.
    """
    sites: dict[int, None] = {}
    pos = 0
    while True:
        pos = data.find(patch.anchor, pos)
        if pos == -1:
            break
        window_start = max(0, pos - patch.window)
        window_end = min(len(data), pos + patch.window)
        hit = data.find(target, window_start, window_end)
        if hit != -1:
            sites[hit] = None
        pos += 1
    return tuple(sites)


def _has_anchor(data: bytes, patch: PatchDef) -> bool:
    """Check whether the anchor pattern exists anywhere in the data."""
    return data.find(patch.anchor) != -1
