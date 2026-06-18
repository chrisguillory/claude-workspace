# ruff: noqa: W505 -- the module docstring is a research reference with version-history
# tables whose long-form column headers (patch family names) are intentional; abbreviating
# would obscure what's being tracked.
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

    force-429-retry-header
                    [fix] Restore retry on the ``x-should-retry: true`` header
                    for Pro/Max OAuth users. Companion to
                    ``force-429-retry-status``. The custom ``shouldRetry`` in
                    ``services/api/withRetry.ts`` has two OAuth-tier gates
                    that disable retry for subscription accounts on shared-
                    capacity 429s. This patch handles the header-driven gate;
                    ``force-429-retry-status`` handles the status-code gate.
                    Apply both — applying only one leaves a path that still
                    fails fast.
                    Anchor: ``"x-should-retry"`` (stable, +21 bytes from
                    site).
                    13-byte same-length replacement: ``!Eq()||gQ_()`` →
                    ``!0/*Eq||gQ*/`` (always-true with comment padding).
                    Identifiers ``Eq``/``gQ_`` map to ``isClaudeAISubscriber``
                    / ``isEnterpriseSubscriber`` and re-minify each release.
                    https://github.com/anthropics/claude-code/issues/50841
                    https://github.com/anthropics/claude-code/issues/57134

    force-429-retry-status
                    [fix] Restore retry on shared-capacity 429s for Pro/Max
                    OAuth users. The custom ``shouldRetry`` in
                    ``services/api/withRetry.ts`` gates 429 retry on
                    ``!isClaudeAISubscriber()||isEnterpriseSubscriber()``,
                    so subscription accounts NEVER retry "Server is
                    temporarily limiting requests (not your usage limit) ·
                    Rate limited" — the loop bails on the first occurrence
                    and auto-mode dies. The deobfuscated source comment
                    rationale is "for Max and Pro users, should-retry is
                    true, but in several hours, so we shouldn't" — but that
                    logic only holds for user-quota 429s, not shared-
                    capacity throttles where the retry-after window is
                    seconds. Both error classes hit the same gate.

                    Effect: 429s join the existing exponential-backoff
                    retry loop (500ms doubling, capped at 32s, respects
                    ``retry-after`` header). Inherits
                    ``CLAUDE_CODE_MAX_RETRIES`` env var (default 10).
                    Surfaces the existing visible UI
                    ``"Retrying in Ns · attempt N/M"``.

                    Anchor: ``"x-should-retry"`` (stable, +278 bytes from
                    site, well within window=600).
                    37-byte same-length replacement:
                    ``if(H.status===429)return!Eq()||gQ_();`` →
                    ``if(H.status===429)return!0;/*Eq||gQ*/`` (always-true
                    with trailing comment). Identifiers ``Eq``/``gQ_`` map
                    to ``isClaudeAISubscriber`` / ``isEnterpriseSubscriber``
                    and re-minify each release.

                    Companion: ``force-429-retry-header`` handles the
                    parallel ``x-should-retry: true`` gate. Apply both
                    together; applying only one leaves a path that still
                    fails fast.

                    Note: also affects user-quota 429s (where Anthropic's
                    "wait hours" comment was correct). Mitigation: 429s
                    with a ``retry-after`` header still respect it, so a
                    quota 429 with ``retry-after: 3600`` sleeps 1 hour.
                    The retry-budget cap (``CLAUDE_CODE_MAX_RETRIES``,
                    default 10) bounds damage even with no header.

                    Future-proofing context: v2.2.1 source mirror at
                    ``claude-code-best/claude-code/src/services/api/withRetry.ts``
                    introduces ``isPersistentRetryEnabled()`` reading
                    ``CLAUDE_CODE_UNATTENDED_RETRY``, gated by Statsig
                    ``UNATTENDED_RETRY``. Source comment marks it
                    "ant-only". Not yet shipped to subscribers; not
                    present in 2.1.138. This patch is the local
                    equivalent until/unless that gate is opened.
                    https://github.com/anthropics/claude-code/issues/50841
                    https://github.com/anthropics/claude-code/issues/57134
                    https://github.com/anthropics/claude-code/issues/53915
                    https://github.com/anthropics/claude-code/issues/53922

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

    scratchpad      [feature] Enable session-scoped scratchpad directory. Creates
                    ``<data_dir>/<project>/<session>/scratchpad`` with auto-
                    permissions for reading and writing. Claude uses this
                    instead of ``/tmp`` for intermediate files.
                    Statsig gate ``tengu_scratch``, default false.
                    Under the Bun 1.4 rebuild (2.1.181) the gate is a single
                    function — ``if(<accessor>("tengu_scratch",!1))return!0;`` with
                    an ``isArtifactToolEnabled`` fallback (previously two
                    byte-identical gate functions). The patch neutralizes just the
                    gate condition (``<accessor>("tengu_scratch",!1)`` → ``!0`` plus
                    a length-padding comment) so the function returns true
                    immediately. A plain gate-flip (``!1`` → ``!0``) is defeated by
                    ``cachedGrowthBookFeatures`` in ``~/.claude.json``
                    (``tengu_scratch: false`` resolves before the patched default),
                    hence the short-circuit.
                    Anchor: ``isArtifactToolEnabled`` — the adjacent stable property
                    name in the fallback branch; the ``tengu_scratch`` call is
                    destroyed by the patch so it can't anchor detection.
                    Flag introduced in 2.1.45 (last absent: 2.1.44).

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
                    Substitutes the four standalone transcript-mode-flag usages in
                    the subagent-row JSX body with the verbose param — prompt
                    (``{flag}&&{prompt}&&`` → ``{verbose}&&{prompt}&&``), tools
                    (``{flag}?`` → ``{verbose}?``), response
                    (``{flag}&&{resp}&&`` → ``{verbose}&&{resp}&&``), and the
                    "(ctrl+o to expand)" hint (``!{flag}&&`` → ``!{verbose}&&``);
                    the verbose tree then follows the verbose param directly.
                    Anchor: ``eo.createElement(wbp,{progressMessages:t,tools:n,verbose:r})``
                    (the verbose-tree React element invocation — interior
                    fragment that's stable across the substitution, so
                    patcher status detection works post-apply).
                    Strategy history: ``let T=H`` reassignment (pre-2.1.114),
                    destructured-param default flip ``T=!1`` → ``T=K``
                    (2.1.114..2.1.125), body flag→verbose substitution (2.1.126+).
                    https://github.com/anthropics/claude-code/issues/14511
                    https://github.com/anthropics/claude-code/issues/5974

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
    ask rule/safety check requires full permission pipeline  2.1.109         2.1.92
        (pre-diagnostic; scanned across 9 local originals)

Empirical verification on 2.1.128 (2026-05-06)::

    All applicable patches confirmed working in fresh sessions on patched
    2.1.128. Two patches obsoleted by upstream changes:
    - write-session-summary: feature removed by Anthropic
    - reject-show-comment: rendering changed to silent in vanilla, so the
      original "Tool use rejected" bug no longer exists

Version Log::

    2.1.181 (2026-06-18)
        Eight releases since 2.1.172 (2.1.173-181). The 2.1.181 build upgraded the
        bundled Bun runtime to 1.4, re-minifying with a new lowercase scheme
        (locals t/e/n/r; binary shrank 223→215 MB). No changelog entry touches a
        live patch — no obsoletions.

        Patch updates:
        - force-429-retry-header / force-429-retry-status: re-derived. Subscriber-
          gate identifiers re-minified Gq/hrH → Co/oXe; the header var (_→t) and
          status param (H→e) also changed. 1 site each, applied.
        - hook-ask-no-override: clean apply (anchor + bytes stable since 2.1.109).
          1 site.
        - scratchpad: re-derived + re-anchored. Bun 1.4 consolidated the two gate
          functions into one (if(<acc>("tengu_scratch",!1))return!0; with an
          isArtifactToolEnabled fallback), so the name-independent body anchor is
          gone. Now anchors on the adjacent stable property isArtifactToolEnabled
          and neutralizes just the gate condition (<acc>("tengu_scratch",!1) → !0
          + comment). 1 site.
        - show-subagent-prompt-tools-response: re-derived. Full JSX re-map under
          the new minifier (module n8→eo, component y$3→wbp, plus others); the
          leading !1, placeholder child is gone. Control flag T→s, verbose param
          K→r; strategy unchanged (4-site flag→verbose). 1 site.

    2.1.172 (2026-06-10)
        Nine releases since 2.1.163 (2.1.164-172), including Claude Fable 5 at
        2.1.170. Routine plumbing — no changelog entry touches a live patch, so
        no obsoletions. Minified identifiers drifted; all four drifted patches
        re-derived, hook-ask applied clean.

        Patch updates:
        - force-429-retry-header / force-429-retry-status: re-derived. The
          isClaudeAISubscriber/isEnterpriseSubscriber identifiers re-minified
          Lq/PdH → Gq/hrH. 1 site each, applied.
        - hook-ask-no-override: clean apply (anchor + bytes stable since
          2.1.109). 1 site.
        - scratchpad: re-derived. Still two byte-identical gate functions
          (isScratchpadEnabled / isScratchpadGateEnabled, 2 sites) under the
          name-independent ``(){return`` anchor; accessor J_ → j_.
        - show-subagent-prompt-tools-response: re-derived. Full JSX re-map
          (module F8 → n8, component esO → y$3, plus U6→F6, FV_→DC_, iP_→CR_,
          ri8→B6q, wn→ii, MTH→k$H, N→V, C2→yW; locals f→J, J→j). Strategy
          unchanged: 4-site T→K substitution. 1 site.

    2.1.163 (2026-06-04)
        Routine bug-fix release (managed version-range settings, /plugin
        list, hook additionalContext). No changelog entry touches a live
        patch — no obsoletions. Minified identifiers drifted again, and the
        two patches deferred at 2.1.162 are now re-derived and applied.

        Patch updates:
        - force-429-retry-header / force-429-retry-status: re-derived. The
          isClaudeAISubscriber/isEnterpriseSubscriber identifiers re-minified
          Wq/odH → Lq/PdH. 1 site each, applied.
        - hook-ask-no-override: clean apply (anchor + bytes stable since
          2.1.109). 1 site.
        - scratchpad: re-derived + re-anchored. The gate is emitted as two
          byte-identical functions (isScratchpadEnabled /
          isScratchpadGateEnabled) with distinct minified names; switched to
          a name-independent body anchor (``(){return``) that hits both
          (2 sites). Accessor M_ → J_. Resolves the 2.1.162 deferral.
        - show-subagent-prompt-tools-response: re-derived. Full JSX re-map
          (module w8 → F8, component vu5 → esO; plus R6→U6, bj_→FV_,
          XO_→iP_, Z08→ri8, Mg→wn, F7H→MTH, V→N, EM→C2; locals D→f, j→J).
          Strategy unchanged: 4-site T→K substitution. 1 site. Resolves the
          2.1.162 deferral.

    2.1.162 (2026-06-04)
        Routine release train since 2.1.138 — the headline was the model jump
        (Opus 4.7→4.8 at 2.1.154), not CC plumbing. Minified identifiers drifted.

        Patch updates:
        - force-429-retry-header / force-429-retry-status: re-derived. The
          isClaudeAISubscriber/isEnterpriseSubscriber identifiers re-minified
          Eq/gQ_ → Wq/odH. 1 site each, applied + verified.
        - statusline: REMOVED (obsolete). The Ink wrap-truncation regression is
          gone upstream — wrap config no longer sits beside statusLine?.padding and
          truncation is no longer observed. It was one of the two columns in the
          Site Count Evolution table; with inject also removed, that table is
          dropped too (no active multi-version-history patches remain).
        - inject-searching-past-context-prompt: REMOVED. The tengu_coral_fern gate
          is gone from the binary (0 occurrences in 2.1.162) — nothing left to inject.
        - scratchpad: DEFERRED. Anchor missing — the gate function was renamed and
          there are now two tengu_scratch gates (Gl5, N7H) needing disambiguation.
        - show-subagent-prompt-tools-response: DEFERRED. Anchor missing — React
          module/component renamed (w8→F8, vu5→ciO); needs a JSX-block re-derive.

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
        invisible when the server returns content blocks" — that
        is exactly the bug ``mcp-array-content-to-string`` was
        created to address. Empirical vanilla-vs-patched comparison
        on 2.1.138 against ``mcp__slack__get_current_user``
        confirmed: vanilla now renders the array content as clean
        CSV text (better than the patch's raw-JSON output). The
        patch is redundant and was removed in this PR.

        Three obsolete patches removed entirely from PATCHES (kept
        in git history for anyone on older versions): policy shift
        from ``max_version`` preservation toward ideal-state cleanup
        per CLAUDE.md "ideal state over backwards compat".

        Patch updates:
        - force-429-retry-status: 1 site, ``vq()``/``$U_()`` (the
          ``isClaudeAISubscriber``/``isEnterpriseSubscriber`` gate
          identifiers) re-minified to ``Eq()``/``gQ_()``. Function
          ``fo5(H)`` renamed to ``vq3(H)``. Structure unchanged;
          re-derived bytes for new identifiers.
        - force-429-retry-header: 1 site, same identifier rename
          (``vq``/``$U_`` → ``Eq``/``gQ_``) applied to the parallel
          ``x-should-retry: true`` header gate. Companion to
          ``force-429-retry-status``.
        - hook-ask-no-override: clean apply (anchor + bytes stable
          since 2.1.109). 1 site.
        - statusline: clean apply (anchor + bytes stable since
          2.0.0). 1 site.
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
        - mcp-array-content-to-string: REMOVED. Anthropic's 2.1.136
          fix supersedes the patch; vanilla 2.1.138 renders MCP
          array content correctly.
        - write-session-summary: REMOVED. Obsolete since 2.1.128
          (Anthropic removed ``tengu_session_memory`` and the
          summary.md write path entirely). Successor exploration
          tracked as session-internal task.
        - reject-show-comment: REMOVED. Obsolete since 2.1.128
          (Anthropic silently changed vanilla rejection rendering
          to empty; the original "Tool use rejected" bug no longer
          exists).

    2.1.131 (2026-05-06)
        Routine release with bug fixes (VS Code activation, Mantle
        endpoint auth, /clear tab title, /context grid leak, several
        UI regressions). No architectural changes affecting patches.

        New patches added 2026-05-08:
        - force-429-retry-status: 2 sites, addresses the OAuth-tier
          gate in fo5() (custom shouldRetry) that disables 429 retry
          for Pro/Max subscription accounts. Anchor "x-should-retry"
          at offsets 81874259 and 206412939 (one per __BUN segment).
          Patch sites at +278 bytes from each anchor.
        - force-429-retry-header: 2 sites, companion patch addressing
          the parallel x-should-retry:true header gate. Patch sites at
          +21 bytes from each anchor. Both patches required together;
          applying only one leaves a fail-fast path.

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
        name='force-429-retry-header',
        description=(
            'Honor x-should-retry: true header for Pro/Max OAuth users. Companion to '
            'force-429-retry-status. Without this, even when the server explicitly '
            'sends x-should-retry: true with a short retry window, subscription '
            'accounts ignore it and fail-fast on shared-capacity 429s. Apply both '
            'companion patches together.'
        ),
        kind=PatchKind.FIX,
        anchor=b'"x-should-retry"',
        old=b't==="true"&&(!Co()||oXe())',
        new=b't==="true"&&(!0/*Co|oXe*/)',
        window=200,
        min_version=CCVersion('2.1.181'),
    ),
    PatchDef(
        name='force-429-retry-status',
        description=(
            'Restore retry on shared-capacity 429s for Pro/Max OAuth users. The '
            'custom shouldRetry function (minified fo5 in 2.1.131) gates 429 retry '
            'on !isClaudeAISubscriber()||isEnterpriseSubscriber(), so subscription '
            'accounts never retry "Server is temporarily limiting requests (not '
            'your usage limit) · Rate limited" — auto-mode dies on first 429. '
            'Restores retry behavior identical to API-key users. Companion: '
            'force-429-retry-header.'
        ),
        kind=PatchKind.FIX,
        anchor=b'"x-should-retry"',
        old=b'if(e.status===429)return!Co()||oXe();',
        new=b'if(e.status===429)return!0;/*Co|oXe*/',
        window=600,
        min_version=CCVersion('2.1.181'),
    ),
    PatchDef(
        name='hook-ask-no-override',
        description='Prevent auto-mode classifier from silently overriding a hook-emitted permission ask',
        kind=PatchKind.FIX,
        anchor=b'ask rule/safety check requires full permission pipeline',
        old=b'behavior==="ask"',
        new=b'behavior==="xsk"',
        window=200,
        min_version=CCVersion('2.1.109'),
    ),
    PatchDef(
        name='scratchpad',
        description='Enable session-scoped scratchpad directory with auto-permissions',
        kind=PatchKind.FEATURE,
        anchor=b'isArtifactToolEnabled',
        old=b'ut("tengu_scratch",!1)',
        new=b'!0/*scratch_force_on*/',
        window=80,
        min_version=CCVersion('2.1.181'),
    ),
    PatchDef(
        name='show-subagent-prompt-tools-response',
        description='Expand completed subagent to show prompt, tool calls, and response when verbose=true',
        kind=PatchKind.VISIBILITY,
        anchor=b'eo.createElement(wbp,{progressMessages:t,tools:n,verbose:r})',
        old=(
            b's&&p&&eo.createElement(qn,null,eo.createElement(X2t,{prompt:p,theme:o})),'
            b's?eo.createElement(LOt,null,eo.createElement(wbp,{progressMessages:t,tools:n,verbose:r})):null,'
            b's&&d&&d.length>0&&eo.createElement(qn,null,eo.createElement(jio,{content:d,theme:o})),'
            b'eo.createElement(qn,{height:1},eo.createElement(xY,{message:A,lookups:Vge,addMargin:!1,tools:n,'
            b'commands:[],verbose:r,inProgressToolUseIDs:new Set,progressMessagesForMessage:[],shouldAnimate:!1,'
            b'shouldShowDot:!1,isTranscriptMode:!1,isStatic:!0})),'
            b'!s&&eo.createElement(w,{dimColor:!0},"  ",eo.createElement(ix,null)))'
        ),
        new=(
            b'r&&p&&eo.createElement(qn,null,eo.createElement(X2t,{prompt:p,theme:o})),'
            b'r?eo.createElement(LOt,null,eo.createElement(wbp,{progressMessages:t,tools:n,verbose:r})):null,'
            b'r&&d&&d.length>0&&eo.createElement(qn,null,eo.createElement(jio,{content:d,theme:o})),'
            b'eo.createElement(qn,{height:1},eo.createElement(xY,{message:A,lookups:Vge,addMargin:!1,tools:n,'
            b'commands:[],verbose:r,inProgressToolUseIDs:new Set,progressMessagesForMessage:[],shouldAnimate:!1,'
            b'shouldShowDot:!1,isTranscriptMode:!1,isStatic:!0})),'
            b'!r&&eo.createElement(w,{dimColor:!0},"  ",eo.createElement(ix,null)))'
        ),
        window=800,
        min_version=CCVersion('2.1.181'),
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
