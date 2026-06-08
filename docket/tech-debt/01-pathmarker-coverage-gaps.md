---
area: claude-session
category: MAINTAINABILITY  # BUG | SECURITY | PERFORMANCE | MAINTAINABILITY | ARCHITECTURE
severity: MEDIUM           # CRITICAL | HIGH | MEDIUM | LOW
title: adjudicate the 30 PathMarker suspects check_path_markers surfaces
---

**Problem.** `check_path_markers.py` flags 30 path-named, string-typed session-model fields carrying no `PathMarker`. Each is either a real cross-machine-restore translation gap (unmarked → restore leaves source-machine paths in place) or a non-path that belongs in the checker's `EXEMPT` map. Until adjudicated, the checker stays red-by-design and the real-path subset is a latent restore bug. Manual-tool-only (not in CI/pre-commit), so it blocks nothing today.

**Area.** `claude-session` — `claude_session/schemas/session/models.py` (the fields), `scripts/check_path_markers.py` (the `EXEMPT` map)

**Fix / sketch.** Per suspect: check real JSONL values → mark `PathField`/`PathListField` if it's a translatable path, else add to `EXEMPT` with a reason; re-run to green. Likely-translate: `worktreePath`, `originalCwd`, `planFilePath`, `scriptPath`, `outputFile`/`outputFilePath`, `Glob`/`Grep.filenames`. Likely-EXEMPT: `Edit`/`WriteToolResult.originalFile` (carry file *content*, not a path — confirmed), `staleReadFileStateHint` (a message). Needs-a-look: 8× `displayPath` (UI-abbreviated vs real?), `ConnectionError.path`, `FileBackupInfo.backupFileName`, `FileHistorySnapshot.trackedFileBackups` keys, the three `SessionMetadata` derived path lists.

<sub>Claude Code session <code>75624f87-7eba-4526-9cb3-e645b1c47461</code></sub>
