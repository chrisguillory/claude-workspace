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
    fix         Restores broken functionality (regression, rendering bug).
    feature     Enables gated/disabled functionality (Statsig gate flip).
    visibility  Exposes UI-hidden content the model already receives.
    tweak       Behavioral adjustment (telemetry, config, UX change).

Patches (alphabetical by name):

    hook-ask-no-override
                    [fix] Prevent auto-mode LLM classifier from silently
                    overriding a PreToolUse hook's ``permissionDecision: "ask"``
                    when a built-in safetyCheck also returns ask (e.g., edits
                    to ``.claude/settings.json``). Claude Code's ``Ma_`` drops
                    the hook result and re-enters the permission pipeline,
                    which in auto mode ends at the classifier — binary
                    allow/deny with no prompt path. The hook's protective ask
                    is lost.
                    Fix flips the FJH-ask comparison literal so the bypass
                    branch is unreachable; control flow falls through to the
                    final ``await O(_,w,K,T,$,H)`` (6-arg) which routes to
                    the documented user-prompt path.
                    Anchor: ``ask rule/safety check requires full permission pipeline``
                    (stable diagnostic message since 2.1.109).
                    1-byte flip: ``behavior==="ask"`` → ``behavior==="xsk"``.
                    Bytes stable across 2.1.109..2.1.116.
                    Classification precedent: symmetric hook-ask vs
                    permissions.deny case fixed upstream at 2.1.101 (#39344).
                    https://github.com/anthropics/claude-code/issues/42797
                    https://github.com/anthropics/claude-code/issues/51255
                    Full analysis (published as secret gist from the PR that
                    introduced this patch):
                    https://gist.github.com/chrisguillory/8d5d401ac356b47ec078940080726b83

    mcp-array-content-to-string
                    [fix] MCP tool results without ``structuredContent``
                    (slack-mcp-server and other servers that return plain
                    text blocks) render as blank below the ⏺ tool-call line.

                    Before::

                        ⏺ slack - get_current_user (MCP)
                                                  ← blank, no result block

                    After::

                        ⏺ slack - get_current_user (MCP)
                          ⎿ [
                              {
                                "type": "text",
                                "text": "user_id,user_name,..."
                              }
                            ]

                    Patches ``transformMCPResult`` to JSON-stringify the
                    array branch so ``toolUseResult`` is always a string,
                    matching the path that already works for
                    ``structuredContent``-emitting tools. Tradeoff: renders
                    as raw JSON rather than extracted text. Full
                    investigation:
                    https://gisthost.github.io/?9018ee3bbc37a7acf95852ab25fe9100
                    https://github.com/anthropics/claude-code/issues/41361

    reject-show-comment
                    [fix] When the user rejects a tool call with a comment,
                    the comment text is silently dropped — only "Tool use
                    rejected" renders.

                    Before::

                        ⏺ Bash(open https://example.com)
                          ⎿ Tool use rejected
                                                ← user's comment gone

                    After::

                        ⏺ Bash(open https://example.com)
                          ⎿ Error: The user doesn't want to proceed with
                            this tool use. ... To tell you how to proceed,
                            the user said:
                            No, please don't run that.

                    Falsifies the prefix check in ``UserToolErrorMessage``
                    so the message falls through to the generic
                    content-renderer. Tradeoff: includes the verbose
                    system-prompt prefix that the model sees. Full
                    investigation:
                    https://gisthost.github.io/?9018ee3bbc37a7acf95852ab25fe9100

    remember-skill  [feature] Enable /remember skill for session memory search.
                    Statsig gate ``tengu_coral_fern``, default false.
                    Not publicly documented or mentioned in any changelog.
                    Flag introduced in 2.1.21 (last absent: 2.1.20).

    scratchpad      [feature] Enable session-scoped scratchpad directory. Creates
                    ``<data_dir>/<project>/<session>/scratchpad`` with auto-
                    permissions for reading and writing. Claude uses this
                    instead of ``/tmp`` for intermediate files.
                    Statsig gate ``tengu_scratch``, default false.
                    Uses different gate mechanism (no default arg) — patched
                    by replacing call ``("tengu_scratch")`` with ``||!0``.
                    Flag introduced in 2.1.45 (last absent: 2.1.44).

    session-memory  [feature] Enable background session memory extraction. Claude
                    writes summaries to session-memory/summary.md, loaded
                    at the start of future sessions for cross-session context.
                    Statsig gate ``tengu_session_memory``, default false.
                    Not publicly documented — companion to auto-memory (GA 2.1.59).
                    Flag introduced in 2.0.64 (last absent: 2.0.62).
                    Related flags: ``tengu_sm_compact``, ``tengu_sm_config``.
                    https://claudefa.st/blog/guide/mechanics/session-memory
                    https://giuseppegurgone.com/claude-memory

    show-subagent-prompt-tools-response
                    [visibility] Expand completed subagent to show prompt, tool
                    calls, and response when verbose=true. Without this patch,
                    subagent output is collapsed to a single "Done" line with
                    "(ctrl+o to expand)". With it, verbose mode shows::

                        Agent(description)
                          ├ Prompt:
                          │   The prompt sent to the agent
                          ├ Read(file.py · lines 1-5)
                          ├ Edit(file.py)
                          ├ Bash(command)
                          ├ Response:
                          │   The agent's final answer
                          └ Done (N tool uses · Xk tokens · Ys)

                    Tool calls appear as one-line summaries (tool name + args),
                    not full input/output.
                    Substitutes the four standalone ``T`` (isTranscriptMode)
                    usages in IR7's JSX body with ``K`` (the verbose param):
                    ``T&&J&&`` → ``K&&J&&`` (prompt), ``T?`` → ``K?`` (tools),
                    ``T&&j&&`` → ``K&&j&&`` (response), ``!T&&`` → ``!K&&``
                    (suppresses the "(ctrl+o to expand)" hint). The verbose
                    tree follows the verbose param directly.
                    Anchor: ``_8.createElement(IL5,{progressMessages:_,tools:q,verbose:K})``
                    (the verbose-tree React element invocation — interior
                    fragment that's stable across the substitution, so
                    patcher status detection works post-apply).
                    Strategy history: ``let T=H`` reassignment (pre-2.1.114),
                    destructured-param default flip ``T=!1`` → ``T=K``
                    (2.1.114..2.1.125), body T→K substitution (2.1.126+).
                    https://github.com/anthropics/claude-code/issues/14511
                    https://github.com/anthropics/claude-code/issues/5974

    statusline      [fix] Multi-line truncation (Ink wrap prop regression).
                    Anchor: ``statusLine?.padding`` (stable since 2.0.0).
                    Regression introduced in 2.1.51 (last clean: 2.1.50).
                    https://github.com/anthropics/claude-code/issues/28750

Version-Agnostic Patterns:
    Gate function names (``Tq``, ``lT``, ``Jq``) change per build. Patches
    match the stable argument list instead: ``("flag_name",!1)`` → ``!0``.
    Verified across 2.1.74, 2.1.80, 2.1.81.

References:
    https://github.com/marckrenn/claude-code-changelog (feature flag tracking)
    https://gist.github.com/gastonmorixe/9c596b6de1095b6bd3b746ca3a1fd3d7

Anchor Presence Survey (2026-03-24, 22+ versions via CDN; extended 2026-04-29)::

    Anchor                                                   First version   Last absent
    statusLine?.padding                                      2.0.0           never absent
    "contentArray"                                           2.1.107         2.1.106 (when contentArray type was added)
    tengu_session_memory                                     2.0.64          2.0.62
    tengu_coral_fern                                         2.1.21          2.1.20
    tengu_scratch                                            2.1.45          2.1.44
    ask rule/safety check requires full permission pipeline  2.1.109         2.1.92 (pre-diagnostic; scanned across 9 local originals)

Site Count Evolution::

    Version   statusline   mcp-array-content-to-string   session-memory   remember-skill   sm-compact
    2.0.64    0            —                             6                0                —
    2.0.70    0            —                             9                0                —
    2.1.0     0            —                             9                0                —
    2.1.21    0            —                             —                3                —
    2.1.40    0            —                             18               3                —
    2.1.45    —            —                             —                —                2
    2.1.51    2            —                             18               9                —
    2.1.74    2            —                             18               3                2
    2.1.80    2            —                             18               3                2
    2.1.81    2            —                             18               3                2
    2.1.123   2            2                             —                —                —
    2.1.126   2            2                             2                2                —

Version Log::

    2.1.126 (2026-05-03)
        Patches re-derived for new minifier scheme:
        - mcp-array-content-to-string: 2 sites, IDE-safe via q!=="ide"
          discriminator (replaces previous unconditional stringify
          which broke the JetBrains plugin's openDiff path)
        - reject-show-comment: 2 sites (e76 prefix ID, stable
          post-context anchor)
        - show-subagent-prompt-tools-response: 2 sites, strategy
          rewritten from destructured-default flip to body T→K
          substitution (default flip defeated by callers passing
          isTranscriptMode explicitly)
        New PatchKind.VISIBILITY enum value introduced;
        show-subagent-prompt-tools-response recategorized
        TWEAK → VISIBILITY.
        Full debugging writeup:
        https://gisthost.github.io/?9018ee3bbc37a7acf95852ab25fe9100

    2.1.123 (2026-04-29)
        mcp-array-content-to-string: 2 sites, clean apply.
        Replaces previous mcp-tool-results patch (downstream safeParse
        null-return) which was empirically unreachable for array-content
        MCP results — removed as dead code. Root cause was upstream in
        transformMCPResult, not in the renderer. Full debugging writeup:
        https://gisthost.github.io/?9018ee3bbc37a7acf95852ab25fe9100

    2.1.117 (2026-04-22)
        hook-ask-no-override, statusline: already applied (stable)

    2.1.114 (2026-04-18)
        show-subagent-prompt-tools-response: 2 sites, unpatched
        (function refactored to destructured param; patch rewritten)
        statusline, session-memory, remember-skill, scratchpad: clean apply

    2.1.109 (2026-04)
        sm-compact: feature removed (tengu_sm_compact flag deleted,
        consolidated into session-memory)

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
from cc_lib.utils import get_claude_workspace_config_home_dir

__all__ = [
    'ORIGINALS_DIR',
    'PATCHES',
    'PATCHES_BY_KIND',
    'PATCHES_BY_NAME',
    'PatchDef',
    'PatchKind',
    'PatchScanResult',
    'scan_binary',
]

ORIGINALS_DIR = get_claude_workspace_config_home_dir() / 'binary-patcher' / 'originals'


class PatchKind(str, Enum):
    """Classification of what a patch does.

    FIX:        Restores broken functionality (regression, rendering bug).
                Low risk — returns to known-good behavior.
    FEATURE:    Enables gated/disabled functionality (Statsig gate flip).
                Medium risk — may have unknown side effects.
    VISIBILITY: Exposes information that the model receives but the user's UI
                hides via collapse, summarization, or curation. The author's
                design intent is "tidy display"; the user wants UI parity with
                the model. Side effect: more verbose terminal output.
                Variable risk — output volume increases.
    TWEAK:      Behavioral adjustment that doesn't fit the categories above
                (telemetry, config, UX change with no clear theme).
                Variable risk — case-by-case.
    """

    FIX = 'fix'
    FEATURE = 'feature'
    VISIBILITY = 'visibility'
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
        name='hook-ask-no-override',
        description='Prevent auto-mode classifier from silently overriding a hook-emitted permission ask',
        kind=PatchKind.FIX,
        anchor=b'ask rule/safety check requires full permission pipeline',
        old=b'behavior==="ask"',
        new=b'behavior==="xsk"',
        window=200,
        min_version='2.1.109',
    ),
    PatchDef(
        name='mcp-array-content-to-string',
        description='Render MCP tool results returning content arrays without structuredContent',
        kind=PatchKind.FIX,
        anchor=b'"contentArray"',
        old=b'return{content:T,type:"contentArray",schema:Q36(Yj_(T))}',
        new=b'return{content:q!=="ide"?NH(T):T,type:"contentArray"}   ',
        window=100,
        min_version='2.1.126',
    ),
    PatchDef(
        name='reject-show-comment',
        description='Show user comment when rejecting a tool call (instead of just "Tool use rejected")',
        kind=PatchKind.FIX,
        anchor=b'){let Y;if(_[5]===Symbol.for("react.memo_cache_sen',
        old=b'T.content.startsWith(e76)',
        new=b'T.content.startsWith("Z")',
        window=100,
        min_version='2.1.126',
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
        name='scratchpad',
        description='Enable session-scoped scratchpad directory with auto-permissions',
        kind=PatchKind.FEATURE,
        anchor=b'tengu_scratch',
        old=b'("tengu_scratch")',
        new=b'||!0/*_scratch_*/',
        window=50,
        min_version='2.1.45',
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
        name='show-subagent-prompt-tools-response',
        description='Expand completed subagent to show prompt, tool calls, and response when verbose=true',
        kind=PatchKind.VISIBILITY,
        anchor=b'_8.createElement(IL5,{progressMessages:_,tools:q,verbose:K})',
        old=(
            b'!1,T&&J&&_8.createElement(M6,null,_8.createElement(FY_,{prompt:J,theme:O})),'
            b'T?_8.createElement(p1_,null,_8.createElement(IL5,{progressMessages:_,tools:q,verbose:K})):null,'
            b'T&&j&&j.length>0&&_8.createElement(M6,null,_8.createElement(XM8,{content:j,theme:O})),'
            b'_8.createElement(M6,{height:1},_8.createElement(iF,{message:X,lookups:G9H,addMargin:!1,tools:q,'
            b'commands:[],verbose:K,inProgressToolUseIDs:new Set,progressMessagesForMessage:[],shouldAnimate:!1,'
            b'shouldShowDot:!1,isTranscriptMode:!1,isStatic:!0})),'
            b'!T&&_8.createElement(v,{dimColor:!0},"  ",_8.createElement(aM,null)))'
        ),
        new=(
            b'!1,K&&J&&_8.createElement(M6,null,_8.createElement(FY_,{prompt:J,theme:O})),'
            b'K?_8.createElement(p1_,null,_8.createElement(IL5,{progressMessages:_,tools:q,verbose:K})):null,'
            b'K&&j&&j.length>0&&_8.createElement(M6,null,_8.createElement(XM8,{content:j,theme:O})),'
            b'_8.createElement(M6,{height:1},_8.createElement(iF,{message:X,lookups:G9H,addMargin:!1,tools:q,'
            b'commands:[],verbose:K,inProgressToolUseIDs:new Set,progressMessagesForMessage:[],shouldAnimate:!1,'
            b'shouldShowDot:!1,isTranscriptMode:!1,isStatic:!0})),'
            b'!K&&_8.createElement(v,{dimColor:!0},"  ",_8.createElement(aM,null)))'
        ),
        window=800,
        min_version='2.1.126',
    ),
    PatchDef(
        name='statusline',
        description='Restore multi-line statusline wrapping (fix truncation)',
        kind=PatchKind.FIX,
        anchor=b'statusLine?.padding',
        old=b'wrap:"truncate"',
        new=b'wrap:"wrap"    ',
        min_version='2.1.51',
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
