# claude-remote-audio

> Multi-Mac audio topology orchestration over [`roc-vad`](https://github.com/roc-streaming/roc-vad) + [`roc-toolkit`](https://github.com/roc-streaming/roc-toolkit).

A CLI that orchestrates a small hub-and-peer audio mesh across a fleet of Macs. One Mac (the **hub**) broadcasts a physical microphone to every peer's `Claude Remote Mic` virtual device (Flow 1), and aggregates each peer's `Claude Remote Speaker` audio back to a chosen output (Flow 2). All operations are declarative + idempotent — re-running the command converges current state to the declared state. Self-healing is "re-run apply."

---

## Architecture

```mermaid
graph LR
    Hub["<b>Hub Mac</b><br/>physical mic<br/>+ roc-send / roc-recv"]
    Peer["<b>Peer Macs</b><br/>Claude Remote Mic<br/>+ Claude Remote Speaker"]
    Hub -- "Flow 1: mic broadcast (RTP/UDP)" --> Peer
    Peer -- "Flow 2: audio aggregation (RTP/UDP)" --> Hub
```

| Role | Where it lives |
|---|---|
| `claude-remote-audio` CLI | Any Mac in the mesh |
| `roc-send` + `roc-recv` processes | The hub Mac |
| `Claude Remote Mic` + `Claude Remote Speaker` virtual devices | Every Mac in the mesh (provided by `roc-vad`) |

The CLI is stateless and runs only when invoked. Cross-Mac orchestration uses a pre-installed dispatch daemon (see prerequisites below); this package owns no long-lived process of its own.

---

## Installation

```bash
uv tool install --editable ~/claude-workspace/mcp/claude-remote-audio
claude-remote-audio install-completions   # zsh/bash TAB on --hub, --target, --input, --output
```

Prerequisites on each Mac in the mesh:
- `roc-vad` driver — install via the upstream script (`sudo` required, run in a local Terminal):
  ```bash
  sudo /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/roc-streaming/roc-vad/HEAD/install.sh)"
  sudo killall coreaudiod
  ```
- `roc-toolkit` CLIs (`roc-send`, `roc-recv`) at `/usr/local/bin/`
- `switchaudio-osx` — `brew install switchaudio-osx`
- `claude-remote-bash-daemon` registered as a launchd service (`claude-remote-bash-daemon install-service` on each Mac)

---

## Quickstart

`--target` accepts a host alias, a comma-separated list, a configured group name (e.g. `mac-mesh`), or a literal `ip:port`. Groups are the most ergonomic form for steady-state operations. `--hub` defaults to the local machine — pass it explicitly only when invoking from a Mac that isn't the intended hub.

Bring the whole topology up — hub broadcast + peer aggregation, end to end:

```bash
claude-remote-audio apply --target mac-mesh --hub M5 \
  --input "DJI MIC MINI" --output "Chris's AirPods Max"
```

Same operation with an explicit host list:

```bash
claude-remote-audio apply --target M5,M2,M3,M4 --hub M5 \
  --input "DJI MIC MINI" --output "Chris's AirPods Max"
```

Recover after the hub's speaker stops working, without changing peers or the mic:

```bash
claude-remote-audio apply --target M5 --hub M5 --output "Chris's AirPods Max"
```

Read-only probe of current state (no `--input` / `--output` → no mutations):

```bash
claude-remote-audio apply --target mac-mesh --hub M5
```

See `claude-remote-audio apply --help` for full flag semantics. `--format json` emits machine-readable output for scripting.

---

## Dependency stack + stability

### Current state

```
[ User ]
   │
[ claude-remote-audio orchestrator (Python) ]                ← our code
   │
   ├─ [ claude-remote-bash (CRB) daemon + client ]           ← our code
   │
   ├─ [ roc-toolkit 0.4.0 (Jun 2024 release; master active 2026-03) ]
   │     └─ [ libsox 14.4.2 (statically linked, deprecated 2014) ]  ← LOAD-BEARING DEPRECATED
   │           └─ [ macOS Core Audio HAL ]
   │
   ├─ [ roc-vad 0.0.4 (Mar 2025) ]                           ← HAL plugin, slow but maintained
   │
   ├─ [ sox CLI (external pipeline for mono devices) ]       ← same libsox 14.4.2
   │
   ├─ [ Swift CLIs: claude-coreaudio-volume, claude-tcc-probe ]  ← our code
   │
   ├─ [ blueutil, switchaudio-osx ]                          ← stable thin wrappers
   │
   └─ [ AppleScript / System Events ]                        ← used only for the Sound-popover rescue
```

### Stability matrix

| Layer                                             | Version                                         | Maintained                             | Stability         | Evidence / notes                                                                                                                                                   |
|---------------------------------------------------|-------------------------------------------------|----------------------------------------|-------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **libsox 14.4.2** (linked into roc-toolkit)       | 14.4.2 (2014)                                   | **No — abandoned upstream since 2015** | **Deprecated**    | Drives 12-byte truncation (W1), same-name collision (W2), no channel negotiation (W3). `sox_ng 14.8.0` (released 2026-05-18) fixes all three.                      |
| **roc-toolkit**                                   | 0.4.0 (Jun 2024 release; master HEAD ~Mar 2026) | Yes (slow)                             | Evolving          | No `-c N` channel flag. No `--bind-source` on `roc-recv`. Next major (`rocd`, Rust daemon) is 12–24 months out — too far to wait.                                  |
| **roc-vad**                                       | 0.0.4 (Mar 2025)                                | Yes (slow)                             | Evolving          | HAL plugin, not kext. No `disconnect` verb → we use `del+add` to retire slots (orchestrator.py `_ensure_peer_speaker`).                                            |
| **switchaudio-osx**, **blueutil**, our Swift CLIs | brew current / source                           | Yes                                    | Stable            | Tiny surface, predictable behavior.                                                                                                                                |
| **macOS Core Audio HAL**                          | macOS 26.4                                      | Apple                                  | Evolving — quirky | Source of the rare "HAL wedge" failure (recovery: reboot). Default-device drift.                                                                                   |
| **macOS TCC**                                     | macOS 26.4                                      | Apple                                  | Hostile-by-design | Attributes to *responsible app* (terminal/iTerm), not immediate caller. Drives `claude-tcc-probe`, the bundle plan (#47), the Automation TCC heuristic complexity. |
| **macOS Continuity / Bluetooth handoff**          | macOS 26.4                                      | Apple                                  | Fragile           | Drives `bluetooth.steal` (cross-Mac exclusive routing). Cannot be controlled programmatically beyond connect/disconnect.                                           |
| **macOS Application Firewall**                    | macOS 26.4                                      | Apple                                  | Stable            | Preflight probe is sufficient.                                                                                                                                     |
| **macOS AppleScript / System Events**             | macOS 26.4                                      | Apple                                  | Fragile           | Used only for the Sound-popover rescue. Brittle row-index clicks. Replaceable with native Swift CLIs once `claude-coreaudio-resolve` ships.                        |

### Ideal-state architecture

Target state we're moving toward — listed deltas annotate the moves.

```
[ User ]
   │
[ Claude Remote Audio.app — signed bundle ]                  ← NEW: one TCC identity
   │
   ├─ [ Python orchestrator ]                                ← split into ~8 focused modules
   │
   ├─ [ Swift CLIs (bundled): coreaudio-*, tcc-*,            ← native API replaces AppleScript
   │     sound-rescue, password-prompt, … ]
   │     └─ [ Apple frameworks: CoreAudio, IOBluetooth,
   │           AppKit, AVFoundation ]                        ← Apple "stable but quirky" — wrapped in adapters
   │
   ├─ [ CRB daemon (signed binary, launchd-managed) ]
   │
   ├─ [ roc-toolkit (rebuilt against sox_ng) ]               ← DROPS the deprecated libsox 14.4.2
   │     └─ [ sox_ng 14.8.0 — modern CoreAudio API ]
   │
   ├─ [ roc-vad (matching version) ]
   │
   └─ [ blueutil, switchaudio-osx ]                          ← unchanged
```

**Key deletions from today:**
- External `sox` CLI pipeline for mono input — `-c N` injection into `roc-send` is the immediate bridge; full deletion lands when sox_ng adoption is complete
- The 12-byte input-device name preflight (W1)
- The same-name device collision preflight (W2)
- The `_diagnose_roc_send_start_failure` tier-2 channel-count fallback
- AppleScript dialog timeouts and Sound-popover rescue (replaced by Swift CLIs)
- The brittle "iTerm as responsible app" TCC chain (signed bundle owns its own consents)

---

## Workaround inventory

Load-bearing band-aids currently in the codebase. **Living document** — entries retire as their underlying limitations get fixed. Owner column tells you where the ideal-state fix lives per CLAUDE.md's "Fix the source" principle.

| #   | Workaround                                                                               | Code location                                                            | Underlying limitation                                                                                     | Owner                                                   | Ideal-state fix                                                                                               |
|-----|------------------------------------------------------------------------------------------|--------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------|---------------------------------------------------------|---------------------------------------------------------------------------------------------------------------|
| W1  | 12-byte input-device name preflight refusal                                              | `orchestrator.py` `_SOX_DEVICE_NAME_MAX_BYTES`                           | libsox 14.4.2 deprecated `kAudioDevicePropertyDeviceName` C-string truncation                             | libsox                                                  | sox_ng 14.8.0                                                                                                 |
| W2  | Same-name input/output device collision refusal                                          | `orchestrator.py` `_probe_input_device` match-count check                | libsox `findDevice` walks devices without scope filter                                                    | libsox                                                  | sox_ng 14.8.0                                                                                                 |
| W3  | Mono-input sox-pipe upmix to stereo                                                      | `orchestrator.py` `_roc_send_command` mono branch                        | `roc-send` CLI has no `-c N`; libsox can't negotiate channels down                                        | libsox + roc-toolkit                                    | Inject `-c $native_channels` via `_probe_input_device` channel count (small fix); full retirement with sox_ng |
| W4  | Microphone TCC preflight + auto-trigger prompt                                           | `orchestrator.py` `_check_microphone_tcc` + `claude-tcc-probe` Swift CLI | macOS TCC responsible-app chain attributes mic to iTerm/Cursor/Terminal                                   | macOS TCC                                               | Signed Mac app bundle (task #47)                                                                              |
| W5  | HAL-wedge diagnostic → "reboot required"                                                 | `orchestrator.py` `_diagnose_roc_send_start_failure` tier 3              | macOS coreaudiod occasional wedge; only reboot clears it                                                  | Apple Core Audio                                        | Cannot fix upstream — detect + tell user to reboot is correct ideal-state                                     |
| W6  | AppleScript Sound-menu rescue                                                            | `bluetooth.py` `engage_via_sound_menu`                                   | Apple gates the "available routes" promotion behind a real Sound-popover click                            | Apple Continuity + HAL                                  | No code fix; AppleScript-into-System-Events is the only path                                                  |
| W7  | Cross-Mac Bluetooth steal                                                                | `bluetooth.py` `steal`                                                   | macOS Continuity lets BT audio appear "connected" on multiple iCloud Macs simultaneously                  | Apple Continuity                                        | Cannot fix — disconnect-then-connect is documented unblock                                                    |
| W8  | `killall` (comm-match) instead of `pkill -f` (argv-regex) for roc-send/roc-recv teardown | `orchestrator.py` `_teardown_stale_hub_processes`, `_roc_send_command`   | Dispatch shell's argv contains literal "roc-send" → `pkill -f` self-kills the carrier shell               | Our dispatch shape (shell-script-as-message)            | Long-term: typed RPC dispatch retires this entire bug class                                                   |
| W9  | NFKC + smart-quote/NBSP fold for `--output` matching                                     | `cc_lib/utils/unicode_match.py` (`nfkc_casefold`)                        | Core Audio stores renamed devices with U+2019 (curly apostrophe); keyboards type U+0027                   | Apple                                                   | Folding is the correct fix; shared helper now lives in `cc_lib/utils/unicode_match.py` (R4)                   |
| W10 | Application Firewall preflight for roc-send/roc-recv binding                             | `orchestrator.py` `_check_application_firewall`                          | macOS firewall silently drops UDP from unauthorized binaries; daemon-spawned can't get the prompt clicked | Apple                                                   | Preflight + actionable refusal is correct ideal-state                                                         |
| W11 | roc-vad slot recreation via `del + add`                                                  | `orchestrator.py` `_ensure_peer_speaker`                                 | roc-vad has no `disconnect` verb to clear an occupied slot                                                | roc-vad                                                 | UID-preservation trick is correct given current API                                                           |
| W12 | `claude-coreaudio-volume` Swift CLI for per-device volume                                | `swift/claude-coreaudio-volume.swift`                                    | macOS has no stock CLI that names-then-controls a device's volume scalar                                  | macOS gap                                               | Building our own Swift CLI **is** the first-class fix — not a workaround                                      |
| W13 | AppleScript dialog with dual `with timeout` + `giving up after`                          | `orchestrator.py` `_run_via_applescript_dialog`                          | AppleScript has two independent timeouts that both default to seconds in non-TTY contexts                 | AppleScript spec                                        | Replace with Swift NSAlert when bundle ships                                                                  |
| W14 | `nohup ... &` detachment + post-launch `pgrep` survival check                            | `orchestrator.py` `_restart_roc_send` / `_restart_roc_recv`              | Dispatch expects bounded execution; roc-send is a daemon                                                  | Our dispatch + lack of launchd integration for roc-send | Register roc-send / roc-recv as launchd transient units; lifecycle is OS-managed                              |
| W15 | `claude-coreaudio-probe` Swift CLI for per-device HAL open                               | `swift/claude-coreaudio-probe.swift` + `orchestrator.py` Tier 1.5        | SoX/sox_ng collapses every CoreAudio input-open failure into one generic message; OSStatus is discarded   | libsox + sox_ng                                         | Building our own Swift CLI **is** the first-class fix — composable with patching sox_ng if/when needed        |

**Concentration by owner:**
- **libsox (deprecated)**: W1, W2, W3 — all retire under sox_ng adoption
- **Apple (closed, evolving)**: W4 (partial), W5, W6, W7, W9, W10, W13 — these are *correct* ideal-state for unfixable upstream
- **roc-vad / roc-toolkit (active, slow)**: W11, W14 — minor
- **Our dispatch shape**: W8, W14 — long-term typed RPC retires both
- **Gap-fillers (Swift CLIs we own)**: W12, W15 — first-class adapters at the lowest layer we own when upstream has no CLI / no symbolic-error surface
- **Us (self-inflicted, already healed)**: removed/healed entries get retired from this table; we don't keep historic band-aids around

---

## Empirical health probes

Apple-provided foundations we can't change but **can test** continuously. Each is detect-and-recover or detect-and-warn; together they form the orchestrator's preflight + post-apply assertion layer.

| Apple quirk                                                  | Probe (where + how)                                                                   | Recovery                                                                                   |
|--------------------------------------------------------------|---------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------|
| AirPods Pro BT codec drops to 24 kHz duplex                  | Read `kAudioDevicePropertyNominalSampleRate` on the active output id; alert if 24 kHz | Re-assert default input ≠ AirPods Pro; restart roc-recv; BT renegotiates A2DP within ~1 s  |
| Default-device drift (input → AirPods auto-flip)             | Poll `SwitchAudioSource -c -t input` periodically                                     | Re-assert intended input. Long-term: roc-vad registers higher-priority default eligibility |
| TCC consent revoked between applies                          | Each Swift CLI probes its own framework before action; structured error if denied     | Surface `ResolvableApplyError` with explicit Settings path                                 |
| BT device disconnected silently                              | `blueutil --is-connected <MAC>`                                                       | Reconnect via `blueutil --connect` or AppleScript Sound-popover engagement                 |
| HAL wedge (rare, unrecoverable in-process)                   | `sox -d -n trim 0 0.1` returns 0 when HAL is healthy                                  | Tell user to reboot (only known recovery)                                                  |
| Application Firewall blocks bind                             | `socketfilterfw --getappblocked` per binary at preflight                              | Surface `ResolvableApplyError` with click-through Settings instructions                    |
| Dual Wi-Fi+Ethernet on same subnet causes packet duplication | Detect via interface enumeration on hub                                               | Bind roc-recv `-s rtp://<specific-IP>:<port>` instead of `0.0.0.0` (under investigation)   |

---

## Roadmap to ideal state

Lives here so we don't lose track of the layered move. Each item is independently shippable; ordering is by leverage (impact ÷ effort), not by dependency chain.

### Tier 1 — in flight

1. **R2 — Inject `-c $native_channels` into `_roc_send_command`**. ~15 LOC. Closes the regressed mono channel-count handling (#21). Validates `_probe_input_device`'s reliability for the bigger sox_ng adoption. Removes journey-residue prose ("task #21 regressed", "commit 970040d8") from user-facing errors.
2. **R4 — Extract `_nfkc_casefold` to `cc_lib/unicode_match.py`**. ~10 LOC + 1 file. Retires duplicated helper across `orchestrator.py` and `bluetooth.py`. Honors CLAUDE.md DRY.
3. **R5 — Strip journey-residue from error prose**. Editorial pass. No commit SHAs, no internal task numbers, no apologetic "this is regressed" language in user-visible strings.
4. **R1 — Maintain a patch series against roc-toolkit master + sox_ng**. The lever. Since we already validate every change empirically on the full mesh, "tracking upstream" buys us nothing while costing us features we could add ourselves. We fork-via-patches: pin to a tested roc-toolkit master SHA, apply our patches at build time, install our binaries.
   - **Bootstrap pulls roc-toolkit master pinned to a specific SHA** (not the 0.4.0 release, not a moving target). Closes 18 months of upstream development.
   - **Swap bundled libsox for sox_ng 14.8.0** in roc-toolkit's scons 3rdparty config. Deletes W1, W2.
   - **Add our patches** (each retires a specific workaround):
     - `-c N` channel-count flag to `roc-send` → retires W3 (sox upmix pipe deleted entirely)
     - `--bind-source IP` flag to `roc-recv` → fixes task #29 dual-interface chipmunk at the source
   - **OSStatus surfacing for input-open failures** is handled out-of-tree via `claude-coreaudio-probe` (W15) — Swift CLI at the lowest layer we own. Composable with a future sox_ng patch if one ships; we don't block on upstream cadence.
   - **Patches live at** `mcp/claude-remote-audio/patches/*.patch`. Reviewable diffs in our PRs. Self-pruning when upstream lands the same fix (patch starts conflicting → drop it).
   - **Risk**: initial engineering bounded by roc-toolkit's C++ codebase + scons build complexity. Mitigation: empirical mesh validation per patch. If a patch breaks something, drop it; we still have the floor of "upstream master + sox_ng" without our additions.

### Tier 2 — after Tier 1 lands

5. **R3 — Split `orchestrator.py` into 8 focused modules**: `plan.py`, `prereqs.py`, `hub.py`, `peer.py`, `rocvad.py`, `roc_cli.py`, `tcc.py`, `diagnose.py`. Pure refactor; do AFTER sox_ng deletes the dead preflights so the split is cleaner.
6. **Sign daemon as Mac app bundle (#47)**. Collapses TCC complexity. Drops the "iTerm as responsible app" chain. Persistent consent across rebuilds.
7. **Swift CLIs replace remaining AppleScript** (#34 sound-popover rescue, password dialog, claude-coreaudio-resolve #51). Once last AppleScript use is gone, we can drop Automation TCC handling entirely.
8. **Apply absorbs recovery primitives** (#60 input drift, #62 BT-duplex recovery) — no separate `recover` subcommand.

### Tier 3 — our own house

9. CRB roadmap: #19 (interface-change reaction), #23 (mid-flight IP flux), #36 (dispatch timeout propagation), #37/#38 (cancel protocol), #41 (BluetoothError migration), #43 (TIMEOUT marker), #59 (launchd auto-start).

### Perhaps Never

- **Wait for `rocd`** (the Rust roc-toolkit successor). 12–24 months out per upstream. We unblock ourselves.
- **Fork SoX (the original)**. sox_ng exists and is maintained.
- **Build our own RTP stack**. Massively negative ROI.
- **Try to control Apple's BT codec mode programmatically**. Opaque by design.
- **Maintain workaround duplication** ("if a third caller emerges"). Move to shared module on second caller per CLAUDE.md DRY.
- **Document-as-workaround** for code-fixable issues. Per CLAUDE.md "Ideal State": fix the source.