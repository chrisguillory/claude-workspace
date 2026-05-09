"""Claude Code binary patching definitions and scan logic.

Patch definitions describe same-length byte replacements in the Claude Code
Mach-O binary. Each patch locates a stable anchor string, then detects or
overwrites nearby target bytes.

Used by claude-binary-patcher (applies patches) and claude-version-manager
(detects patch status). Adding a new patch: add a PatchDef to PATCHES.

Binary Structure:
    Claude Code is a Node.js SEA compiled to Mach-O arm64 (~190-220MB).
    Minified identifiers (2-4 chars) change per build; string literals
    (flag names, JSX props) are stable and serve as anchors.

    From 2.1.0 to 2.1.137, the minified JS bundle (~10MB) was duplicated
    in the ``__BUN`` segment, so each patch had 2 sites. From 2.1.138+
    the duplication was removed (the ``__BUN`` segment is gone) and each
    patch has 1 site. ``scan_binary`` handles either count automatically
    via ``data.find()``; no patcher logic depends on a fixed count.

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

    inject-searching-past-context-prompt
                    [feature] Inject the "## Searching past context"
                    section into Claude's system prompt — explicit
                    grep-paths for the user's auto-memory directory and
                    project session transcripts. Without this section,
                    Claude is told *when* to search memory but not
                    *where*; with it, Claude has copy-paste-ready
                    commands. Patch short-circuits the gate check
                    inside ``buildSearchingPastContextSection`` (minified
                    ``iLH``) so the section always renders.

                    Verbatim prompt text injected (with paths resolved
                    for the user's home + project)::

                        ## Searching past context

                        When looking for past context:
                        1. Search topic files in your memory directory:
                        ```
                        Grep with pattern="<search term>" path="<autoMemDir>" glob="*.md"
                        ```
                        2. Session transcript logs (last resort — large files, slow):
                        ```
                        Grep with pattern="<search term>" path="<projectDir>/" glob="*.jsonl"
                        ```
                        Use narrow search terms (error messages, file paths, function names) rather than broad keywords.

                    (In ant-internal builds and REPL mode, the Grep-tool
                    invocations switch to raw ``grep -rn`` shell calls
                    against the same paths.)

                    Where this section sits in the broader memory prompt
                    (Piebald-AI prompt-archive, v2.1.126 tag-pinned, MIT;
                    shows the surrounding sections so you see what's
                    above/below the gated insertion point):
                    https://github.com/Piebald-AI/claude-code-system-prompts/blob/v2.1.126/system-prompts/system-prompt-memory-instructions.md#L24
                    Note: that file shows the placeholder
                    ``${SEARCHING_PAST_CONTEXT_INSTRUCTIONS}`` because
                    Anthropic ships the gate at default-false — what's
                    visible on most users' systems is the placeholder
                    collapsing to empty.

                    For the expanded body — the actual prompt strings
                    Claude sees when the gate is open — see the builder
                    function in the claude-code-best source mirror:
                    https://github.com/claude-code-best/claude-code/blob/main/src/memdir/memdir.ts#L375-L407

                    Statsig gate ``tengu_coral_fern``, default false in
                    Anthropic upstream. Not publicly documented by
                    Anthropic — official memory docs at
                    https://code.claude.com/docs/en/memory describe the
                    auto-memory feature but omit the search-prompt
                    injection mechanism. The flag is invisible to users
                    from official sources; visible only via the 2026-03-31
                    source leak and downstream mirrors. Note:
                    ``LOCAL_GATE_DEFAULTS`` in ``claude-code-best`` shows
                    ``tengu_coral_fern: true`` — that's the mirror's
                    override, not Anthropic's intent.

                    Why users want this: github.com/anthropics/claude-code
                    issue #51116 (open) — users asking for cross-session
                    memory persistence, unaware it already exists behind
                    this gate. Issues #48783 and #44820 surface adjacent
                    auto-memory pain.

                    Related layers (separate, NOT gated by this flag):
                    - Auto-memory *writer* (extractMemories background
                      agent) — gated by Statsig flag
                      ``tengu_passport_quail`` (no env var override).
                    - Whole auto-memory feature (read + write) — gated by
                      ``isAutoMemoryEnabled()`` via env
                      ``CLAUDE_CODE_DISABLE_AUTO_MEMORY`` (1/true → off,
                      0/false → on; default on). This IS publicly
                      documented.
                    - ``/remember`` slash command — gates on
                      ``isAutoMemoryEnabled()``, NOT on
                      ``tengu_coral_fern``.

                    Cache-override caveat: same as ``scratchpad`` and
                    ``write-session-summary`` — cached ``tengu_coral_fern: false``
                    in ``~/.claude.json`` overrides any default. The
                    short-circuit makes the cache irrelevant.

                    Requires: ``autoMemoryEnabled: true`` in ``~/.claude/settings.json``
                    (top-level, not ``env`` block). Without it ``P4()``
                    returns false and the patched ``iLH()`` is never
                    called — the patch is applied but inert.

                    Peer project: github.com/Piebald-AI/tweakcc has a UI
                    for bypassing this and ``tengu_session_memory``.
                    Flag introduced in 2.1.21 (last absent: 2.1.20).

    scratchpad      [feature] Enable session-scoped scratchpad directory. Creates
                    ``<data_dir>/<project>/<session>/scratchpad`` with auto-
                    permissions for reading and writing. Claude uses this
                    instead of ``/tmp`` for intermediate files.
                    Statsig gate ``tengu_scratch``, default false.
                    Short-circuits the gate-check function ``at()`` (which
                    is the dedup'd ``isScratchpadEnabled``/``isScratchpadGateEnabled``
                    JS site) to always return true, bypassing
                    ``getFeatureValue_CACHED_MAY_BE_STALE``'s cache lookup.
                    The standard gate-flip pattern (``!1`` → ``!0``) is
                    insufficient here because ``cachedGrowthBookFeatures``
                    in ``~/.claude.json`` typically has ``tengu_scratch:
                    false``, which short-circuits to false BEFORE the
                    patched default is consulted. The short-circuit
                    replaces the entire function body with ``return!0``
                    plus a length-padding comment.
                    Flag introduced in 2.1.45 (last absent: 2.1.44). Call
                    signature standardized to two-arg form between 2.1.114
                    and 2.1.121 (verified). The same cache-override
                    limitation applies to ``remember-skill`` (still ships
                    with the simple gate-flip; switch to short-circuit if
                    it doesn't activate). ``write-session-summary`` was
                    switched to short-circuit in 2.1.126.

    write-session-summary
                    [feature] Enable background session summary extraction.
                    Claude writes summaries to session-memory/summary.md,
                    loaded at the start of future sessions for cross-session
                    context.
                    Statsig gate ``tengu_session_memory``, default false.
                    Not publicly documented — companion to auto-memory (GA 2.1.59).
                    Flag introduced in 2.0.64 (last absent: 2.0.62).
                    Short-circuits the gate-check function ``kI3()`` (the
                    minified ``isSessionMemoryGateEnabled``) to always
                    return true, bypassing ``cachedGrowthBookFeatures``
                    which has ``tengu_session_memory: false`` and would
                    otherwise short-circuit to false BEFORE the gate's
                    patched default is consulted (same cache-override
                    pattern as ``scratchpad``).
                    Requires: ``autoCompactEnabled: true`` in ``~/.claude/settings.json``
                    (top-level, not ``env`` block). Background extraction
                    is registered as a hook on the autocompact path; with
                    autocompact off the hook is never registered and the
                    patched gate is never reached — patch is applied but inert.
                    Related flags: ``tengu_sm_compact``, ``tengu_sm_config``
                    (NOT bundled in 2.1.126; the compaction integration
                    code is not present in this build, so only ``kI3``
                    needs short-circuiting).
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

    Tracks longitudinal site counts for the patch families with
    multi-version history. Patches first introduced in a single
    version (e.g., reject-show-comment, show-subagent-prompt-tools-response
    at 2.1.126) are documented in the Version Log below, not here.

    Version   statusline   mcp-array-content-to-string   write-session-summary   inject-searching-past-context-prompt   sm-compact
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
    2.1.128   2            2                             0 (removed)      2                —
    2.1.131   2            2                             0 (removed)      2                —
    2.1.138   1            1                             0 (removed)      1                —

Empirical verification on 2.1.128 (2026-05-06)::

    All applicable patches confirmed working in fresh sessions on patched
    2.1.128. Two patches obsoleted by upstream changes:
    - write-session-summary: feature removed by Anthropic
    - reject-show-comment: rendering changed to silent in vanilla, so the
      original "Tool use rejected" bug no longer exists

Version Log::

    2.1.138 (2026-05-09)
        Major architectural change: Anthropic removed the ``__BUN``
        segment duplication. The minified JS bundle was duplicated
        in ``__BUN`` from 2.1.0..2.1.137, giving each patch 2 sites.
        In 2.1.138, ``__BUN`` is gone — the bundle lives once in
        ``__TEXT``, the binary shrunk ~12 MB (217 → 205 MB), and
        every patch now has 1 site. The patcher's ``data.find()``
        handles either count automatically.

        Five releases since 2.1.131 (2.1.132, 2.1.133, 2.1.136,
        2.1.137, 2.1.138 — all GitHub-tagged, no stealth). The
        2.1.136 changelog notes "Fixed MCP tool results being
        invisible when the server returns content blocks", which
        may be the same code path as ``mcp-array-content-to-string``;
        empirical vanilla-vs-patched comparison still pending — if
        upstream now renders correctly without the patch, the patch
        is redundant and should be marked obsolete in a follow-up.

        Patch updates:
        - hook-ask-no-override: clean apply (anchor + bytes stable
          since 2.1.109). 1 site.
        - statusline: clean apply (anchor + bytes stable since
          2.0.0). 1 site.
        - mcp-array-content-to-string: 1 site, schema-build chain
          changed from ``RD_(WcH(T))`` to ``Vf_(fnH(Y))`` (helper
          renames + local var ``T``→``Y``); JSON.stringify wrapper
          changed from ``EH`` to ``hH``.
        - inject-searching-past-context-prompt: 1 site, accessor
          renamed back ``G_`` → ``M_`` (continues to oscillate per
          release). Function holding the gate is now ``DNH``.
        - scratchpad: 1 site, function ``ve()`` renamed to
          ``XHH()``, accessor ``G_`` → ``M_``. Anchor updated.
        - show-subagent-prompt-tools-response: 1 site, every JSX
          identifier minified again (``hv5``→``vu5``, ``O8``→``w8``,
          ``G6``→``R6``, ``bw_``→``bj_``, ``N5_``→``XO_``,
          ``Nf8``→``Z08``, ``SU``→``Mg``, ``t9H``→``F7H``,
          ``Xf``→``EM``; local var ``J``→``j``). Strategy
          unchanged: 4-site T→K substitution.
        - write-session-summary: still obsolete
          (``max_version='2.1.126'``).
        - reject-show-comment: still obsolete
          (``max_version='2.1.126'``).

    2.1.131 (2026-05-06)
        Routine release with bug fixes (VS Code activation, Mantle
        endpoint auth, /clear tab title, /context grid leak, several
        UI regressions). No architectural changes affecting patches.

        Patch updates:
        - hook-ask-no-override: clean apply (anchor + bytes stable since 2.1.109).
        - statusline: clean apply (anchor + bytes stable since 2.0.0).
        - mcp-array-content-to-string: 2 sites, schema-build chain
          changed from ``$D_(TQH(T))`` to ``RD_(WcH(T))``;
          JSON.stringify alias changed from ``SH`` to ``EH``.
        - inject-searching-past-context-prompt: 2 sites, accessor
          renamed back ``Z_`` → ``G_`` (oscillates per release).
          Function holding the gate is now ``nkH``.
        - scratchpad: 2 sites, function ``Me()`` renamed to ``ve()``.
          Accessor stable as ``G_``.
        - show-subagent-prompt-tools-response: 2 sites, every JSX
          identifier minified to a different name again
          (``JN5``→``hv5``, ``q8``→``O8``, ``X6``→``G6``,
          ``Gw_``→``bw_``, ``P5_``→``N5_``, ``qf8``→``Nf8``,
          ``MU``→``SU``, ``d9H``→``t9H``, ``v``→``V``,
          ``wf``→``Xf``; local var ``P``→``X``).
          Strategy unchanged: 4-site T→K substitution.
        - write-session-summary: still obsolete
          (``max_version='2.1.126'``); ``tengu_session_memory``
          remains absent from the binary.
        - reject-show-comment: still obsolete
          (``max_version='2.1.126'``); rejection rendering remains
          silent in vanilla.

    2.1.128 (2026-05-05)
        Major upstream change: Anthropic removed the session-memory
        feature entirely. ``tengu_session_memory``, ``summary.md``,
        and the ``session-memory/`` directory are gone from the binary.
        ``/dream`` survives but now writes into auto-memory typed files
        instead. Likely driven by data-loss bugs (issues #42542, #47959,
        #50694), the April 23 caching-regression postmortem, and the
        ``Memory on Managed Agents`` architectural pivot toward
        filesystem-mounted ``/mnt/memory/``. 2.1.127 was apparently a
        failed kill build that got pulled before tagging.

        Patch updates:
        - write-session-summary: marked obsolete via
          ``max_version='2.1.126'``. The patch's underlying feature
          no longer exists in 2.1.128+.
        - hook-ask-no-override: clean apply (anchor + bytes stable).
        - statusline: clean apply (anchor + bytes stable since 2.0.0).
        - inject-searching-past-context-prompt: 2 sites, accessor
          renamed ``G_`` → ``Z_`` (Statsig accessor minified
          differently). Function holding ``iLH`` is now ``RkH``.
          Anchor stable.
        - scratchpad: 2 sites, function ``at()`` renamed to ``Me()``,
          accessor ``G_`` → ``Z_``. Anchor updated to match.
        - mcp-array-content-to-string: 2 sites, schema-build chain
          changed from ``Q36(Yj_(T))`` to ``$D_(TQH(T))``;
          JSON.stringify alias changed from ``NH`` to ``SH`` (a
          tracing-decorated wrapper at the patch site).
        - reject-show-comment: marked obsolete via
          ``max_version='2.1.126'``. Anthropic changed vanilla
          rejection rendering to silent in 2.1.128 — there's no
          "Tool use rejected" text at all; Claude's next-turn
          acknowledgment carries the UX. The original bug this
          patch addressed no longer exists. Empirically verified
          by side-by-side comparison of vanilla vs patched 2.1.128:
          both produce the same empty rendering between the tool-call
          line and Claude's next response.
        - show-subagent-prompt-tools-response: 2 sites, every
          identifier in the JSX block was minified to a different
          name (``IL5``→``JN5``, ``_8``→``q8``, ``M6``→``X6``,
          ``FY_``→``Gw_``, ``p1_``→``P5_``, ``XM8``→``qf8``,
          ``iF``→``MU``, ``G9H``→``d9H``, ``aM``→``wf``,
          local vars ``J``→``D``, ``j``→``J``, ``X``→``P``).
          Strategy unchanged: 4-site T→K substitution.

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
        - scratchpad: 2 sites, switched to short-circuit pattern
          (replace ``at()`` body with ``return!0``) because the
          simple gate-flip is defeated by ``cachedGrowthBookFeatures``.
          min_version bumped 2.1.45 → 2.1.121.
        - write-session-summary: 2 sites, switched to short-circuit
          pattern (replace ``kI3()`` body with ``return!0``) for the
          same cache-override reason as scratchpad. ``tengu_sm_compact``
          compaction integration is not bundled in 2.1.126; only
          ``kI3`` needs short-circuiting. min_version bumped
          2.0.64 → 2.1.126.
        - inject-searching-past-context-prompt: 2 sites,
          renamed from ``remember-skill`` (the old name implied
          the ``/remember`` slash command, which is actually gated
          by ``isAutoMemoryEnabled()`` not ``tengu_coral_fern``).
          Switched from simple gate-flip to short-circuit
          (replace ``if(!G_("tengu_coral_fern",!1))return[]`` with
          ``if(0/*coral_fern_gate_check*/)return[]``) for the same
          cache-override reason as scratchpad/write-session-summary.
          min_version bumped 2.1.21 → 2.1.126.
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
        statusline, write-session-summary, remember-skill, scratchpad: clean apply

    2.1.109 (2026-04)
        sm-compact: feature removed (tengu_sm_compact flag deleted,
        consolidated into write-session-summary)

    2.1.81 (2026-03-24)
        statusline: 2 sites, applied
        write-session-summary: 4 sites, unpatched (gate: lT)
        remember-skill: 2 sites, unpatched (gate: lT)
        sm-compact: 2 sites, unpatched (gate: lT)

    2.1.80 (2026-03-24)
        statusline: 2 sites, applied
        write-session-summary: 4 sites (gate: Tq)
        remember-skill: 2 sites (gate: Tq)

    2.1.74 (2026-03-24)
        statusline: N/A (predates regression)
        write-session-summary: present (gate: Jq)
        remember-skill: present (gate: Jq)
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Literal

from cc_lib.settings_env import get_cc_env_var, get_cc_setting, is_env_truthy
from cc_lib.types import CCVersion
from cc_lib.utils import get_claude_workspace_config_home_dir, version_in_range

__all__ = [
    'ORIGINALS_DIR',
    'PATCHES',
    'PATCHES_BY_KIND',
    'PATCHES_BY_NAME',
    'PatchDef',
    'PatchKind',
    'PatchScanResult',
    'RequiredSetting',
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
class RequiredSetting:
    """A top-level ``settings.json`` key + value the patch needs at runtime.

    Models the binary's actual gate logic so the patcher's INERT warnings
    don't false-positive when the user hasn't set the key but the binary
    would default to a satisfying value.

    Out-of-scope (limitations the patcher won't detect):

    - ``managed-settings.json`` (MDM precedence): the patcher only reads
      ``~/.claude/settings.json`` and ``.claude/settings.json``.
    - Runtime toggles (e.g. ``/memory off``): file-undetectable.
    - Org-policy Statsig flags (``fg_()`` etc.): server-side, not on disk.
    - ``CLAUDE_CODE_SIMPLE`` / ``CLAUDE_CODE_REMOTE`` launch modes.
    - Force-enable env vars (e.g. ``CLAUDE_CODE_DISABLE_AUTO_MEMORY=0``
      overriding ``autoMemoryEnabled: false``): rare; unmodeled.
    """

    key: str
    expected_value: object
    default_value: object
    """Value the binary uses when ``key`` is absent from settings.json.

    The patcher treats an absent key as satisfied when ``default_value``
    matches ``expected_value`` (eliminates false-positive INERT warnings
    for users who never set the key).
    """
    disable_env_vars: Sequence[str] = ()
    """Env vars whose truthy presence forces the feature off in the binary.

    Checked before settings — a truthy env var overrides any settings
    value and makes the patch inert regardless of ``key``.
    """

    def is_satisfied(self) -> bool:
        """Whether the setting + env state currently makes the patch effective."""
        for env_var in self.disable_env_vars:
            if is_env_truthy(get_cc_env_var(env_var)):
                return False
        actual = get_cc_setting(self.key)
        if actual is None:
            actual = self.default_value
        return actual == self.expected_value

    def unsatisfied_reason(self) -> str | None:
        """Human-readable reason why this requirement isn't met, or None if it is."""
        for env_var in self.disable_env_vars:
            env_value = get_cc_env_var(env_var)
            if is_env_truthy(env_value):
                return f'env var {env_var}={env_value!r} is disabling the feature (overrides settings.json)'
        actual = get_cc_setting(self.key)
        if actual is None:
            actual = self.default_value
        if actual != self.expected_value:
            return f'requires {self.key!r}: {self.expected_value!r} in ~/.claude/settings.json (currently: {actual!r})'
        return None


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
    required_setting: Sequence[RequiredSetting] = ()
    """Top-level ``settings.json`` keys the patch needs to be effective.

    Each entry's ``key`` is checked against ``~/.claude/settings.json`` (via
    ``cc_lib.settings_env.get_cc_setting``); the patcher warns when the
    actual value doesn't match ``expected_value``. Empty default — most
    patches are self-contained at the byte level.
    """

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
        old=b'return{content:Y,type:"contentArray",schema:Vf_(fnH(Y))}',
        new=b'return{content:q!=="ide"?hH(Y):Y,type:"contentArray"}   ',
        window=100,
        min_version='2.1.138',
    ),
    PatchDef(
        name='reject-show-comment',
        description=(
            'Show user comment when rejecting a tool call (instead of just "Tool use rejected"). '
            'Obsolete in 2.1.128+ — Anthropic changed vanilla rejection rendering to silent: no '
            '"Tool use rejected" text appears at all; Claude\'s next-turn acknowledgment carries '
            'the UX. The original bug this patch addressed no longer exists in 2.1.128+.'
        ),
        kind=PatchKind.FIX,
        anchor=b'){let Y;if(_[5]===Symbol.for("react.memo_cache_sen',
        old=b'T.content.startsWith(e76)',
        new=b'T.content.startsWith("Z")',
        window=100,
        min_version='2.1.126',
        max_version='2.1.126',
    ),
    PatchDef(
        name='inject-searching-past-context-prompt',
        description=(
            'Inject the "## Searching past context" system-prompt section that tells Claude how to grep '
            "the user's memory directory and session transcripts for past context"
        ),
        kind=PatchKind.FEATURE,
        anchor=b'coral_fern',
        old=b'if(!M_("tengu_coral_fern",!1))return[]',
        new=b'if(0/*coral_fern_gate_check*/)return[]',
        window=80,
        min_version='2.1.138',
        required_setting=[
            RequiredSetting(
                key='autoMemoryEnabled',
                expected_value=True,
                default_value=True,
                disable_env_vars=['CLAUDE_CODE_DISABLE_AUTO_MEMORY'],
            ),
        ],
    ),
    PatchDef(
        name='scratchpad',
        description='Enable session-scoped scratchpad directory with auto-permissions',
        kind=PatchKind.FEATURE,
        anchor=b'function XHH(){return',
        old=b'function XHH(){return M_("tengu_scratch",!1)}',
        new=b'function XHH(){return!0/*scratchpad always*/}',
        window=50,
        min_version='2.1.138',
    ),
    PatchDef(
        name='write-session-summary',
        description=(
            'Enable background extraction that writes <sid>/session-memory/summary.md for '
            'cross-session context. Obsolete in 2.1.128+ — Anthropic removed the underlying '
            'feature (tengu_session_memory and the summary.md write path) and reattached '
            '/dream to write into auto-memory typed files instead.'
        ),
        kind=PatchKind.FEATURE,
        anchor=b'function kI3(){return',
        old=b'function kI3(){return G_("tengu_session_memory",!1)}',
        new=b'function kI3(){return!0/*("tengu_session_memory")*/}',
        window=80,
        min_version='2.1.126',
        max_version='2.1.126',
        required_setting=[
            RequiredSetting(
                key='autoCompactEnabled',
                expected_value=True,
                default_value=True,
                disable_env_vars=['DISABLE_AUTO_COMPACT', 'DISABLE_COMPACT'],
            ),
        ],
    ),
    PatchDef(
        name='show-subagent-prompt-tools-response',
        description='Expand completed subagent to show prompt, tool calls, and response when verbose=true',
        kind=PatchKind.VISIBILITY,
        anchor=b'w8.createElement(vu5,{progressMessages:_,tools:q,verbose:K})',
        old=(
            b'!1,T&&D&&w8.createElement(R6,null,w8.createElement(bj_,{prompt:D,theme:O})),'
            b'T?w8.createElement(XO_,null,w8.createElement(vu5,{progressMessages:_,tools:q,verbose:K})):null,'
            b'T&&j&&j.length>0&&w8.createElement(R6,null,w8.createElement(Z08,{content:j,theme:O})),'
            b'w8.createElement(R6,{height:1},w8.createElement(Mg,{message:X,lookups:F7H,addMargin:!1,tools:q,'
            b'commands:[],verbose:K,inProgressToolUseIDs:new Set,progressMessagesForMessage:[],shouldAnimate:!1,'
            b'shouldShowDot:!1,isTranscriptMode:!1,isStatic:!0})),'
            b'!T&&w8.createElement(V,{dimColor:!0},"  ",w8.createElement(EM,null)))'
        ),
        new=(
            b'!1,K&&D&&w8.createElement(R6,null,w8.createElement(bj_,{prompt:D,theme:O})),'
            b'K?w8.createElement(XO_,null,w8.createElement(vu5,{progressMessages:_,tools:q,verbose:K})):null,'
            b'K&&j&&j.length>0&&w8.createElement(R6,null,w8.createElement(Z08,{content:j,theme:O})),'
            b'w8.createElement(R6,{height:1},w8.createElement(Mg,{message:X,lookups:F7H,addMargin:!1,tools:q,'
            b'commands:[],verbose:K,inProgressToolUseIDs:new Set,progressMessagesForMessage:[],shouldAnimate:!1,'
            b'shouldShowDot:!1,isTranscriptMode:!1,isStatic:!0})),'
            b'!K&&w8.createElement(V,{dimColor:!0},"  ",w8.createElement(EM,null)))'
        ),
        window=800,
        min_version='2.1.138',
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
    status: Literal['unpatched', 'applied', 'changed', 'missing', 'out_of_range']
    # unpatched    = anchor found, old bytes found -> patch not yet applied
    # applied      = anchor found, new bytes found -> patch is applied
    # changed      = anchor found, neither old nor new bytes found -> code changed
    # missing      = anchor NOT found (binary structure changed)
    # out_of_range = patch's [min_version, max_version] excludes current binary
    #                version (anchor + bytes not even checked)


def scan_binary(
    data: bytes,
    patches: Sequence[PatchDef] | None = None,
    current_version: CCVersion | None = None,
) -> Mapping[str, PatchScanResult]:
    """Scan binary data for patch status. Pure function, no I/O.

    Args:
        data: Raw binary content.
        patches: Patches to check. Defaults to all PATCHES.
        current_version: Binary version (e.g. ``'2.1.131'``). When provided,
            patches whose ``[min_version, max_version]`` range excludes this
            version short-circuit to status ``'out_of_range'`` without any
            byte scanning. Pass ``None`` (default) to scan every patch
            unconditionally — preserved for callers that don't have a
            version handy.

    Returns:
        Mapping of patch name to scan result.
    """
    if patches is None:
        patches = PATCHES
    results: dict[str, PatchScanResult] = {}
    for patch in patches:
        if current_version is not None and not _patch_applies_to_version(patch, current_version):
            results[patch.name] = PatchScanResult(patch=patch, sites=(), status='out_of_range')
            continue
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


def _patch_applies_to_version(patch: PatchDef, current: CCVersion) -> bool:
    """True if ``current`` falls within the patch's declared version range."""
    return version_in_range(current, patch.min_version, patch.max_version)
