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

Patch Kinds:
    fix       Restores broken functionality (regression, rendering bug).
    feature   Enables gated/disabled functionality (Statsig gate flip).
    tweak     Behavioral adjustment (telemetry, config, UX change).

Patches:
    statusline      [fix] Multi-line truncation (Ink wrap prop regression).
                    Anchor: ``statusLine?.padding`` (stable since 2.0.0).
                    Regression introduced in 2.1.51 (last clean: 2.1.50).
                    https://github.com/anthropics/claude-code/issues/28750

    mcp-tool-results [fix] MCP tool result rendering. outputSchema safeParse
                    guard returns null on schema mismatch, killing the React
                    component. Patch nullifies the safeParse result instead
                    of returning null, allowing fallthrough to raw toolUseResult.
                    Anchor: ``outputSchema?.safeParse`` (stable property chain).
                    Regression introduced in 2.1.89 (last clean: 2.1.87).
                    No 2.1.88 was published.
                    Minified vars (M, P, H) may change per build — framework
                    reports ``changed`` status when they do.
                    https://github.com/anthropics/claude-code/issues/41361

    session-memory  [feature] Enable background session memory extraction. Claude
                    writes summaries to session-memory/summary.md, loaded
                    at the start of future sessions for cross-session context.
                    Statsig gate ``tengu_session_memory``, default false.
                    Not publicly documented — companion to auto-memory (GA 2.1.59).
                    Flag introduced in 2.0.64 (last absent: 2.0.62).
                    Related flags: ``tengu_sm_compact``, ``tengu_sm_config``.
                    https://claudefa.st/blog/guide/mechanics/session-memory
                    https://giuseppegurgone.com/claude-memory

    remember-skill  [feature] Enable /remember skill for session memory search.
                    Statsig gate ``tengu_coral_fern``, default false.
                    Not publicly documented or mentioned in any changelog.
                    Flag introduced in 2.1.21 (last absent: 2.1.20).

    sm-compact      [feature] Use session memory for auto-compaction instead of LLM
                    summarization. When the context window fills, Claude
                    normally calls the LLM to summarize the conversation
                    (expensive, slow). With sm-compact enabled, auto-compact
                    uses the existing session memory summary directly --
                    no extra API call, faster, and more deterministic.
                    Requires session-memory patch (gate checks both flags).
                    Statsig gate ``tengu_sm_compact``, default false.
                    Env var override: ``ENABLE_CLAUDE_CODE_SM_COMPACT=1``.
                    Config: ``tengu_sm_compact_config`` (minTokens: 10000,
                    minTextBlockMessages: 5, maxTokens: 40000).
                    Present in all versions with session-memory (2.0.64+),
                    verified in 2.1.45, 2.1.74, 2.1.80, 2.1.81.

    scratchpad      [feature] Enable session-scoped scratchpad directory. Creates
                    ``<data_dir>/<project>/<session>/scratchpad`` with auto-
                    permissions for reading and writing. Claude uses this
                    instead of ``/tmp`` for intermediate files.
                    Statsig gate ``tengu_scratch``, default false.
                    Uses different gate mechanism (no default arg) — patched
                    by replacing call ``("tengu_scratch")`` with ``||!0``.
                    Flag introduced in 2.1.45 (last absent: 2.1.44).

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
    outputSchema?.safeParse   2.1.87          2.1.86 (anchor exists in 2.1.87 but bug absent)
    tengu_session_memory      2.0.64          2.0.62
    tengu_coral_fern          2.1.21          2.1.20
    tengu_sm_compact          2.0.64          2.0.62 (co-introduced with session-memory)
    tengu_scratch             2.1.45          2.1.44

Site Count Evolution::

    Version   statusline   mcp-tool-results   session-memory   remember-skill   sm-compact
    2.0.64    0            —                  6                0                —
    2.0.70    0            —                  9                0                —
    2.1.0     0            —                  9                0                —
    2.1.21    0            —                  —                3                —
    2.1.40    0            —                  18               3                —
    2.1.45    —            —                  —                —                2
    2.1.51    2            —                  18               9                —
    2.1.74    2            —                  18               3                2
    2.1.80    2            —                  18               3                2
    2.1.81    2            —                  18               3                2
    2.1.89    —            2                  —                —                —
    2.1.90    —            2                  —                —                —

Version Log::

    2.1.90 (2026-04-02)
        mcp-tool-results: 2 sites, unpatched (vars: M, P, H)
        statusline: 2 sites, applied

    2.1.89 (2026-04-02)
        mcp-tool-results: 2 sites, unpatched (vars: M, P, H)

    2.1.81 (2026-03-24)
        statusline: 2 sites, applied
        session-memory: 4 sites, unpatched (gate: lT)
        remember-skill: 2 sites, unpatched (gate: lT)
        sm-compact: 2 sites, unpatched (gate: lT)

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
from enum import Enum
from typing import Literal

from cc_lib.types import CCVersion

__all__ = [
    'PATCHES',
    'PATCHES_BY_KIND',
    'PATCHES_BY_NAME',
    'PatchDef',
    'PatchKind',
    'PatchScanResult',
    'scan_binary',
]


class PatchKind(str, Enum):
    """Classification of what a patch does.

    FIX:     Restores broken functionality (regression, rendering bug).
             Low risk — returns to known-good behavior.
    FEATURE: Enables gated/disabled functionality (Statsig gate flip).
             Medium risk — may have unknown side effects.
    TWEAK:   Behavioral adjustment (telemetry, config, UX change).
             Variable risk — neither a fix nor a feature unlock.
    """

    FIX = 'fix'
    FEATURE = 'feature'
    TWEAK = 'tweak'


@dataclass(frozen=True, slots=True)
class PatchDef:
    """A binary patch definition: same-length byte replacement near an anchor."""

    name: str
    description: str
    kind: PatchKind
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
        kind=PatchKind.FIX,
        anchor=b'statusLine?.padding',
        old=b'wrap:"truncate"',
        new=b'wrap:"wrap"    ',
        min_version='2.1.51',
    ),
    PatchDef(
        name='mcp-tool-results',
        description='Fix MCP tool result rendering (outputSchema safeParse regression)',
        kind=PatchKind.FIX,
        anchor=b'outputSchema?.safeParse',
        old=b'if(M&&!M.success)return null;let P=M?.data??H.toolUseResult',
        new=b'if(M&&!M.success)M=null;     let P=M?.data??H.toolUseResult',
        window=100,
        min_version='2.1.89',
    ),
    PatchDef(
        name='session-memory',
        description='Enable background session memory extraction',
        kind=PatchKind.FEATURE,
        anchor=b'tengu_session_memory',
        old=b'("tengu_session_memory",!1)',
        new=b'("tengu_session_memory",!0)',
        window=50,
        min_version='2.0.64',
    ),
    PatchDef(
        name='remember-skill',
        description='Enable /remember skill for session memory search',
        kind=PatchKind.FEATURE,
        anchor=b'tengu_coral_fern',
        old=b'("tengu_coral_fern",!1)',
        new=b'("tengu_coral_fern",!0)',
        window=50,
        min_version='2.1.21',
    ),
    PatchDef(
        name='sm-compact',
        description='Use session memory for auto-compaction (no LLM summary call)',
        kind=PatchKind.FEATURE,
        anchor=b'tengu_sm_compact',
        old=b'("tengu_sm_compact",!1)',
        new=b'("tengu_sm_compact",!0)',
        window=50,
        min_version='2.0.64',
    ),
    PatchDef(
        name='scratchpad',
        description='Enable session-scoped scratchpad directory with auto-permissions',
        kind=PatchKind.FEATURE,
        anchor=b'tengu_scratch',
        old=b'("tengu_scratch")',
        new=b'||!0/*_scratch_*/',
        window=50,
        min_version='2.1.45',
    ),
)

PATCHES_BY_NAME: Mapping[str, PatchDef] = {p.name: p for p in PATCHES}
PATCHES_BY_KIND: Mapping[PatchKind, Sequence[PatchDef]] = {
    kind: tuple(p for p in PATCHES if p.kind == kind) for kind in PatchKind
}


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

    Note: window is measured from anchor START, not end. Effective forward
    reach past the anchor's last byte is ``window - len(anchor)``.
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
