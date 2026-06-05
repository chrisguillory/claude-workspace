# Session Patcher Redesign

## Why we're doing this

The patcher destroyed 6 days of conversation history (Apr 23–29) on session
`47658c6f` because the `consecutive-messages` detector had fail-open
deny-list logic — any synthetic record not on a hardcoded whitelist was
treated as corruption. That detector is already deleted.

But cleaning up that one detector exposed a deeper architectural issue:
the remaining detectors share mutable analyzer state and depend on
undocumented execution order. `_detect_duplicate_uuid` literally calls
`analyzer._uuid_index.pop(u)` for each duplicate. `orphan-sidechain` and
`stale-parent` then read the mutated index. The `_DETECTOR_ORDER` map
encodes this temporal coupling — change the order and orphan-sidechain
silently produces wrong results.

The deleted detector was one symptom. Shared mutable state and implicit
ordering are the disease. The patcher needs to evolve cleanly as Claude
Code introduces new corruption modes; today's architecture makes that
fragile.

## Root cause

The current `fix` flow:

1. Backup file
2. Build one `SessionAnalyzer` from file
3. Run all detectors against this **shared, mutated** analyzer
4. Compose a single combined `PatchPlan` (merged rewires + min `truncate_to`
   + image redactions)
5. Apply all changes atomically in one write
6. Verify

Three problems live inside this flow:

- **Shared mutable state.** `_detect_duplicate_uuid` mutates the analyzer.
  Other detectors read the mutated state. Order is load-bearing but the
  coupling is invisible at call sites.
- **Combined-plan composition is fragile.** `oversized-image` truncates
  records; `orphan-sidechain` rewires parent pointers. Their fix data is
  computed from the same pre-mutation analyzer but applied together.
  If oversized-image's truncation removes a record that orphan-sidechain
  rewired *to*, the combined plan is incoherent.
- **`_simulate_rewire` exists to paper over composition risk.** It walks
  the chain post-rewire to verify the chain still reaches root. This
  exists *because* the combined plan can't be trusted otherwise.

## New model: DAG of self-contained patches, file as truth

Each patch is a self-contained transformation:

- **Detector** — pure function of file state → result
- **Fix** — pure transformation of file → file
- **Dependencies** — declared explicitly by name: which other patches, if
  applicable, must run before me

The runner is a class (encapsulated, not module-level globals). On each
iteration it walks the DAG: pick a patch whose deps have all completed,
re-read the file, build a fresh analyzer, run detector, apply fix if
detected. Repeat until an iteration completes with no fixes applied
(convergence) or the iteration bound is exceeded (failure).

```
backup file once
loop:
    for each patch in topological order, alphabetical tiebreak:
        re-read file fresh, build new SessionAnalyzer
        run detector
        if detected: apply fix to file (write to disk)
        mark patch completed for this iteration
    if no fix was applied this iteration: converged → exit
verify final state
```

### Determinism

Within the DAG, multiple patches may have their dependencies satisfied
simultaneously (e.g., `duplicate-uuid` and `oversized-image` both have
no deps). **Tiebreak by alphabetical order on patch name.** Two runs
on the same input produce identical sequences of operations. No surprises.

### Convergence

A single criterion: an iteration completes with **zero fixes applied**.
Strict and loose collapse to the same definition under the DAG semantics
— if every detector ran (which they do, since each iteration is a full
topological sweep) and none returned `detected`, the file is healthy.

### Iteration bound

Pure safety against bugs. Idempotent fixes converge in one iteration;
real-world DAG depth is small (≤2 today). The bound exists to catch
non-idempotent bugs (a patch whose fix re-triggers its own detection),
not to limit useful work. Set generously — `MAX_ITERATIONS = 10` in
practice gives ~5x headroom for any realistic patch DAG.

We don't optimize for iteration cost. The patcher runs rarely. Correctness
and ideal architecture beat speed.

## Concrete dependency graph (current 4 detectors)

```
duplicate-uuid       (no deps; rewires parents away from duplicates)
       │
       ├──→ orphan-sidechain   (deps: duplicate-uuid, oversized-image)
       └──→ stale-parent       (deps: duplicate-uuid, oversized-image)

oversized-image      (no deps; redacts oversized images, truncates dim-error tail)
       │
       ├──→ orphan-sidechain   (truncation may remove records that other parents pointed to)
       └──→ stale-parent       (same — truncation can affect chain walk)
```

Declaration form:

| Patch | `deps` |
|---|---|
| `duplicate-uuid` | `()` |
| `oversized-image` | `()` |
| `orphan-sidechain` | `("duplicate-uuid", "oversized-image")` |
| `stale-parent` | `("duplicate-uuid", "oversized-image")` |

The two no-dep patches run in alphabetical order: `duplicate-uuid` first,
then `oversized-image`. The two dependents run after both have completed.

## Code changes

### `SessionPatchDef`

Add two fields:

- `detector: Callable[[SessionAnalyzer, SessionPatchDef], SessionScanResult]`
- `deps: tuple[str, ...] = ()` — names of patches that must run before this one

Move `PATCHES` definition to after detector functions so `detector=_detect_duplicate_uuid`
references resolve. Same applies to other detector functions.

### `SessionAnalyzer`

- **Stop mutating `_uuid_index` from anywhere outside the constructor.**
  Today `_detect_duplicate_uuid` calls `.pop()` on it. Delete that block.
- **`_build_index` excludes duplicate UUIDs from `_uuid_index` at build
  time.** This is needed because `find_rewire_target` and `walk_chain`
  read `_uuid_index` to choose valid targets and walk chains; if duplicates
  were left in the raw index, `find_rewire_target` could recommend
  rewiring to a duplicate UUID, which preserves the ambiguity. Single
  source of truth: post-construction, `_uuid_index` contains only unique
  valid UUIDs. Duplicate metadata still lives in `_duplicate_uuids` for
  `_detect_duplicate_uuid` to consume.
- After this change, `_uuid_index` is immutable post-construction. No
  detector mutates analyzer state, period.

### Detector simplifications

- `_detect_duplicate_uuid`: delete the `for u in _duplicate_uuids: _uuid_index.pop()` block. Detector becomes a pure read of `_records` and `_duplicate_uuids`.
- `_detect_orphan_sidechain`: delete the `if parent in analyzer._duplicate_uuids: continue` skip. Under the DAG, by the time orphan-sidechain runs, duplicate-uuid has already rewired all parents that pointed to duplicates — they're no longer in the file. The skip becomes dead weight.
- `_detect_stale_parent`: walks the chain on a fresh post-fix file. No code change needed beyond removing reliance on the mutation having happened.
- `_detect_oversized_image`: unchanged.

### `PatchRunner` class (new)

Encapsulated runner replaces module-level scaffolding. Lives in the same file as the rest of the patcher.

```python
class PatchRunner:
    """Drives DAG-ordered patch application against a session file.

    Each iteration is a topological sweep: every patch whose dependencies
    completed this iteration runs its detector against a fresh re-read of
    the file. If detected, the fix is applied (file mutated) before the
    next patch runs. The loop terminates when an iteration applies zero
    fixes (convergence) or MAX_ITERATIONS is reached (bug — failure).
    """

    MAX_ITERATIONS = 10

    def __init__(self, session: SessionFile, patches: Sequence[SessionPatchDef] = PATCHES) -> None:
        self._session = session
        self._patches = patches

    def run(self) -> RunResult:
        for iteration in range(self.MAX_ITERATIONS):
            applied_any = self._run_one_pass()
            if not applied_any:
                return RunResult(iterations=iteration + 1, ...)
        raise SessionPatchError(f"did not converge after {self.MAX_ITERATIONS} iterations")

    def _run_one_pass(self) -> bool:
        completed: set[str] = set()
        applied_any = False
        while True:
            patch = self._next_runnable(completed)
            if patch is None:
                return applied_any
            applied = self._run_patch(patch)
            applied_any = applied_any or applied
            completed.add(patch.name)

    def _next_runnable(self, completed: set[str]) -> SessionPatchDef | None:
        # Alphabetical tiebreak among patches with all deps completed.
        candidates = [
            p for p in self._patches
            if p.name not in completed
            and all(dep in completed for dep in p.deps)
        ]
        return min(candidates, key=lambda p: p.name) if candidates else None

    def _run_patch(self, patch: SessionPatchDef) -> bool:
        analyzer = SessionAnalyzer(self._session.read_lines())
        result = (
            patch.detector(analyzer, patch, sidechain_resolver=self._session.sidechain_resolver())
            if patch.name == 'orphan-sidechain'
            else patch.detector(analyzer, patch)
        )
        if result.status != 'detected':
            return False
        _apply_single_patch(self._session, result.fix_data)
        return True
```

(Sketch — final code may differ in details. The shape is the contract.)

### `_apply_patch` becomes `_apply_single_patch`

Today it composes rewires + redactions + truncates. After the redesign,
each call applies one patch's fix data. Dispatch on `fix_data` type:

- `FixDataType.Rewire` (and subclasses `DuplicateUuid`, `OrphanSidechain`) → apply rewire_map
- `FixDataType.RedactImageAndTruncate` → apply redactions and truncation
- (No `FixDataType.Truncate` — already deleted earlier in this session.)

### `fix` CLI command

Becomes a thin wrapper:

```python
@app.command()
def fix(session_id: str | None = ...) -> None:
    session = SessionFile.find(session_id)
    if session.is_active():
        raise ActiveSessionError(session.session_id)
    backup_mgr.create(session)
    runner = PatchRunner(session)
    result = runner.run()
    print(result.summary())
```

### `check` command

Stays simple. `check` is a snapshot — it shows what corruption is present
right now in the file. Inter-patch effects (e.g., "applying duplicate-uuid
would unblock detection of new orphans") are out of scope for `check`;
that's an apply-time concern. Build one analyzer, run all detectors,
report findings. Same as today, modulo the SessionPatchDef field changes.

The user can mentally apply any single reported patch in isolation. If
they want to see post-application state, they apply it via `fix` and
re-run `check`.

## Restore + mutation gates (folded in — no longer deferred)

The `restore` command currently blindly stomps the active session file.
The `fix` command has an `is_active()` gate, but `is_active()` itself is
racy: it shells out to `lsof` against the JSONL path, which only catches
sessions during their narrow write windows. A session that's resumed
and idle awaiting input shows as `is_active()=False` even though Claude
will append to it on the next user message.

Ideal state — a robust gate applied to every mutation path:

### Replace `is_active()` with PID-based check

`cc_lib.session_tracker` already maintains a per-session `claude_pid`
record in `~/.claude-workspace/sessions.json`. Use it.

```python
def is_active(self) -> bool:
    """Check if a Claude process is currently running on this session.

    Looks up the session's tracked claude_pid in sessions.json and verifies
    the process exists via psutil.pid_exists(). Replaces the prior
    lsof-on-JSONL approach which only caught sessions during their narrow
    write windows.

    Returns False for orphaned PIDs (Claude exited without updating
    sessions.json) — at worst this means a stale gate; the user will
    discover the staleness when their fix actually runs cleanly.
    """
    pid = session_tracker.get_claude_pid(self.session_id)
    if pid is None:
        return False
    return psutil.pid_exists(pid)
```

If `cc_lib.session_tracker` doesn't expose `get_claude_pid` directly, add
that as part of this work — it's a single-function read of the tracked
metadata.

### Apply gate to `restore`

```python
@app.command()
def restore(session_id: str | None = ..., force: bool = False) -> None:
    backup_path, meta = backup_mgr.find(session_id)
    target_path = Path(meta['original_path'])
    if target_path.exists():
        live_session = SessionFile(target_path)
        if live_session.is_active():
            raise ActiveSessionError(live_session.session_id)
        # Also catch the "appended after backup" hazard
        if not force and target_path.stat().st_mtime > backup_path.stat().st_mtime:
            live_lines = sum(1 for _ in target_path.open())
            backup_lines = sum(1 for _ in backup_path.open())
            raise SessionPatchError(
                f"Live session has {live_lines} lines vs backup {backup_lines} "
                f"({live_lines - backup_lines:+d}); "
                f"restore would discard appended content. Use --force to override."
            )
    backup_mgr.restore_to(target_path, backup_path)
```

### `--force` flag on `restore`

Bypasses the mtime hazard check. The `is_active()` gate is NOT bypassable
by `--force` — there's no use case for restoring on top of a live Claude.
If the user really wants to, they kill Claude first.

## What's deleted (with explicit rationale, not deferred to "later")

- **`_DETECTOR_ORDER`** — replaced by `deps` field on `SessionPatchDef`. The map's role (encode execution order) is subsumed by the DAG.
- **`_DETECTORS`** — replaced by `detector` field on `SessionPatchDef`. The map's role (lookup detector by name) is no longer needed when each `SessionPatchDef` carries its own.
- **`build_patch_plan`** — only caller is the current monolithic `fix`. Under DAG-driven `fix`, each patch's fix data is applied directly; no combined plan to compose. Confirmed by inspection: no other callers in the file or in any cc_lib consumer (verified via grep before deletion).
- **`PatchPlan` class** — sole purpose was to carry `build_patch_plan`'s output. Removed alongside it.
- **`_simulate_rewire`** — sole purpose was to verify that a *combined* rewire plan still produces a chain reaching root. Under per-patch apply, each rewire is applied to the actual file before the next detector runs; the next detector sees real chain state, not a simulation. Chain integrity is naturally maintained by the file-as-truth model. The function's docstring already acknowledged it's "not triggerable with current detectors" — confirming it was vestigial.
- **The `if patch.name == 'orphan-sidechain'` branch in scan/runner** — orphan-sidechain takes a `sidechain_resolver` kwarg; cleaner to express this as a per-patch invocation hook in `SessionPatchDef`, but that's a stretch goal. For now keep the branch in the runner.

## What this does NOT change

- **Which patches exist.** Same 4 detectors. No new corruption types.
- **What each fix does.** duplicate-uuid still rewires parent pointers around duplicates; oversized-image still redacts images and truncates the synthetic-error tail. Internal logic of fixes is unchanged.
- **Detect-only patches stay detect-only.** `stale-parent` reports findings without applying anything. Under the DAG model, a detect-only patch returns `status='detected'` with `fix_data=None` (or a sentinel); the runner's `_run_patch` interprets "no fix data" as "report but apply nothing."
- **No new fixes.** When Claude Code introduces a new corruption mode, we add a new patch then. Empirical-only — wait until a real session breaks, prove the failure mode, then handle it.
- **Backup format.** `.meta.json` + `.jsonl` pair, unchanged.

## Phased implementation

Each phase ends in a self-contained, working state. Verification at each
step against:

- Currently-active session `47658c6f` (post-fix state, 4,481+ records as of last check)
- Historical session `9c9c7b52` (preserved in backups)

### Phase 1 — Data model expansion

Add `detector` and `deps` fields to `SessionPatchDef`. Move `PATCHES`
tuple to after detector function definitions. Populate each entry's
`detector=...` and `deps=...`. No behavior change yet — `scan()` still
uses `_DETECTORS`/`_DETECTOR_ORDER` maps. Field additions verified by
inspection that all entries compile.

### Phase 2 — `_uuid_index` cleaning at build time

Modify `_build_index` to exclude duplicate UUIDs from `_uuid_index`.
Delete the `.pop()` block in `_detect_duplicate_uuid`. Verify with
`check` on `47658c6f` and `9c9c7b52`: same detection results as before
the change (mutation was happening downstream of detection; removing it
plus pre-cleaning the index produces equivalent state for downstream
detectors).

### Phase 3 — `PatchRunner` class introduced

Implement `PatchRunner` as described above. `fix` command switches to
use it. Old `build_patch_plan` and `_apply_patch` remain temporarily —
`PatchRunner._run_patch` calls a new `_apply_single_patch` that handles
one fix data instance.

Verify: run `fix` against a copy of the broken `47658c6f` backup state.
Result should match what the current monolithic `fix` would produce
(post-deletion of `consecutive-messages`). Same final record count, same
records, same content.

### Phase 4 — Restore + mutation gates

Replace `is_active()` with the PID-based version. Add `is_active()` and
mtime-hazard gates to `restore`. Add `--force` flag. Verify by simulation
on a fixture (don't actually run on the live `47658c6f`).

### Phase 5 — Detector simplifications

Delete the `if parent in analyzer._duplicate_uuids: continue` skip in
`_detect_orphan_sidechain` (no longer reachable: by the time orphan-sidechain
runs, parents have been rewired). Verify no regression on `47658c6f` and
`9c9c7b52` checks.

### Phase 6 — Dead code removal

Delete `_DETECTOR_ORDER`, `_DETECTORS`, `build_patch_plan`, `PatchPlan`,
`_simulate_rewire`. Update `scan()` to use `p.detector` and `p.deps`
directly (or replace `scan()` with a thinner wrapper around the runner's
detection-only mode for `check`). Final cleanup pass on docstrings that
referenced the deleted machinery.

### Phase 7 — Verification

Full pass:
- `claude-session-patcher check 47658c6f` → HEALTHY (same as today)
- `claude-session-patcher check 9c9c7b52` → reports current detected items (same as today)
- Synthetic broken fixture: induce duplicate UUIDs + oversized image + orphan parents in a controlled `/tmp` JSONL, run `fix`, verify all three are repaired correctly with the DAG order applied.
- Inspect resulting code: no `_uuid_index.pop` calls anywhere except the constructor's own building; no `_DETECTOR_ORDER`; no `_simulate_rewire`; `PatchRunner` is the only orchestrator.

## Approval checkpoint

This is the working plan. If anything's off, point at it and we adjust
before any code changes. Once you sign off, I execute Phases 1–7 in order
and report back at the end of each phase.
