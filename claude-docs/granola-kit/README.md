# granola-kit — reconnaissance & migration docs

Claude-reference artifacts for migrating `~/granola-mcp/` → `claude-workspace/mcp/granola-kit/` (a workspace-pattern MCP server + CLI over Granola.ai's private API). Not shipped with the package — the package's own user-facing docs (`bleeding-edge.md`, `api-surface.md`, `possible-features.md`) are authored fresh in `mcp/granola-kit/docs/` at build time. Not auto-loaded into context; read when the task lands in the migration.

**Live plan:** `~/.claude/plans/reactive-foraging-wreath-clone-019e26cb.md` — the authoritative implementation spec (28 empirically-verified tools, validated against Granola 7.277.1). Everything here is its evidence base.

## Documents (authority order — later overrides earlier on conflict)

| Doc | What |
|---|---|
| `recon-synthesis.md` | [2026-06-01] capstone of the re-validation against Granola **7.277.1** — verdict + the full delta list against the 4.7-era recon |
| `validate-plan-findings.md` | 3-validator adversarial review: 2 BLOCKING (B1/B2) + 13 MAJOR (M1–M13), with fixes folded into the plan |
| `asar-diff-7277.md` | current-binary static analysis: endpoint→host map, identity-header builder, device-id derivation, feature flags |
| `capture-session-7277.md` | CDP-driven live mitmproxy capture: the `chat-with-documents` request + stream spec, relaunch recipe, reachability |
| `chat-stream-sample.json` | captured `chat-with-documents` stream fixture (6-of-67-chunk summary; the full fixture is extracted at impl, step 5.5) |
| `reprobe-findings.md` | authenticated re-probe of the originally-"dropped" endpoints with the correct current identity |
| `walk-endpoint-inventory.md` | full breadth walk: 47-endpoint production read surface + write-lifecycle confirmation |
| `buried-insights.md` | mined from 13 compaction summaries (dual-cache, 200-envelope, sync-in-async, drift fields) |
| `private-api-enumeration.md` | original 7.220.0 endpoint enumeration (superseded where it conflicts with the 7.277.1 pass) |
| `upstream-mcp-survey.md` | audit of Granola's official MCP (6 tools, exercised live) |
| `empirical-surface.md` | original 7.220.0 live capture (superseded on device-id, version, time-zone header, search/briefs reachability) |
