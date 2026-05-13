# claude-remote-bash filesystem feature (NFSv3) — plan

## Goal

Mount peer Macs' filesystems on the hub Mac (M3) so Claude Code's Edit tool — which requires a real filesystem path — can operate on remote files transparently. `Edit("/Users/chris/remote/m2/path/to/file.py", ...)` should just work.

Today's gap: `claude-remote-bash` lets us *execute commands* on peer Macs (M2/M4/M5), but to edit a remote file we have to round-trip text via shell-quoted `sed` or `cat`-and-overwrite. That loses the diff viewer, loses Edit-tool semantics, loses PyCharm's preview-on-save, and is fragile against quoting bugs.

## Strategy (after breadth → depth survey)

**NFSv3 userspace server inside the claude-remote-bash daemon, mounted natively on M3 via `mount_nfs`.** Specifically:

- Daemon spawns `crb-nfsd` (Rust binary built from `xetdata/nfsserve`) per export, bound to a fresh `127.0.0.1:<random>` on the peer.
- M3's `crb mount <host>` opens a forward tunnel through the existing PSK-authed claude-remote-bash control channel and runs `mount_nfs 127.0.0.1:<local_port>:/ ~/.crb/host/<host>`.
- All NFS traffic flows through the existing authed tunnel; the peer's NFS server is never exposed to the LAN.

Why this approach beat the alternatives (full survey at the bottom):

| Approach | Verdict |
|---|---|
| **NFSv3 userspace + native macOS NFS client** ← pick | No FUSE on M3, survives M4 MDM (userland TCP), reuses daemon auth/mDNS |
| SFTP server + SSHFS via FUSE-T | Solid fallback, but FUSE-T third-party install on M3 + SSHFS small-file latency |
| WebDAV server + `mount_webdav` | Cleanest protocol fit, but Apple's WebDAV client has reliability bugs for small writes |
| Native SMB on peers | Blocked by M4 MDM |
| macFUSE kext | Requires Reduced Security boot on Apple Silicon |
| Pure-Python NFS server | Multi-week side quest; nfsserve already exists and is BSD-3-Clause |
| rclone `serve nfs` | Requires `--vfs-cache-mode full`, write-back layer breaks read-after-write |

## Architecture

```
M3 (hub)                                    M2/M4/M5 (peer)
┌─────────────────────────┐                ┌─────────────────────────┐
│ Claude Code (Edit tool) │                │ crb-nfsd (Rust)         │
│         │               │                │   127.0.0.1:<P>         │
│         ▼               │                │         ▲               │
│ ~/remote/m2/foo.py      │                │         │ NFS RPC       │
│         │               │                │ ┌───────┴───────┐       │
│ kernel NFS client       │                │ │ raw bridge    │       │
│   127.0.0.1:<lp>        │                │ │ (tunnel sink) │       │
│         │ NFS RPC       │                │ └───────▲───────┘       │
│ ┌───────┴───────┐       │   nonce-       │         │               │
│ │ local         │   authed claim conn   │         │               │
│ │ forwarder     │◀═══════════════════════════════╤ │               │
│ └───────────────┘       │                │       │ │               │
│         │               │   PSK control │ ┌─────┴─┴───┐ daemon     │
│ ┌───────┴───────┐       │   channel     │ │           │ (existing)│
│ │ crb mount     │═══════════════════════│═│ daemon    │            │
│ │   supervisor  │       │  (long-lived) │ │           │            │
│ └───────────────┘       │                │ └───────────┘            │
└─────────────────────────┘                └─────────────────────────┘
```

Two TCP connections per active mount:
- **Control channel**: existing PSK-authed claude-remote-bash; long-lived for the mount's lifetime. Issues `RequestTunnelGrant` / `TunnelGrant` exchange.
- **Tunnel connection**: opened fresh on each kernel-NFS-client connect. First message is `TunnelClaim(nonce)` against a nonce from a prior `TunnelGrant`. After daemon's `ClaimOk`, the connection becomes a raw byte pipe — daemon side bridges to its local `crb-nfsd` socket.

## Protocol extension (additive — no break to existing message flow)

> **Module layout note (post PR #133/#135 on 2026-05-12).** The protocol Pydantic models moved from `models.py` to `schemas/protocol.py`, with `schemas/__init__.py` re-exporting public names. Discovery has its own `schemas/discovery.py`. The new tunnel messages land in a new `schemas/tunnel.py` (mirrors the discovery split) — keeps `schemas/protocol.py` focused on auth+execute, makes the tunnel surface easy to remove if ever needed. `schemas/__init__.py` re-exports the new types.

New messages in `claude_remote_bash/schemas/tunnel.py`:

```python
class RequestTunnelGrant(ClosedModel):
    """Client → Daemon (on PSK-authed control channel).

    Asks daemon to spin up an NFS export and issue a single-use nonce.
    """
    type: Literal['tunnel_grant_request'] = 'tunnel_grant_request'
    id: str
    service: Literal['nfs']
    root: str | None = None       # absolute path on peer; defaults to daemon config
    readonly: bool = False

class TunnelGrant(ClosedModel):
    """Daemon → Client. Nonce expires in N seconds and is single-use."""
    type: Literal['tunnel_grant'] = 'tunnel_grant'
    id: str
    nonce: str
    nfs_port_hint: int            # informational
    expires_in_seconds: int = 10

class TunnelClaim(ClosedModel):
    """Client → Daemon on a *new* TCP connection (not the control channel)."""
    type: Literal['tunnel_claim'] = 'tunnel_claim'
    nonce: str

class ClaimOk(ClosedModel):
    """Daemon → Client. After this message the connection is raw bytes."""
    type: Literal['tunnel_claim_ok'] = 'tunnel_claim_ok'
```

Daemon's `handle_client` dispatch grows one new branch: if the first message on a fresh connection is `TunnelClaim` (instead of `AuthRequest`), look up nonce, validate, send `ClaimOk`, then `asyncio.gather(copy(reader, nfs_writer), copy(nfs_reader, writer))` until either side closes. Nonces are re-issued per claim (request fresh from control channel on each new local TCP accept).

The PSK auth path is unchanged. No multiplexing.

## CLI surface

```bash
# Mount peer M2 at ~/remote/m2 (creates dir if missing)
crb mount m2

# Custom mountpoint / remote root / read-only
crb mount m2 --remote-path /Users/chris/monorepo --mountpoint ~/remote/m2-mono --readonly

# Unmount
crb unmount m2

# List active mounts
crb mounts
```

What `crb mount` does internally:

1. mDNS lookup to find peer's claude-remote-bash daemon
2. Open control connection, PSK auth (existing)
3. `RequestTunnelGrant(service='nfs', root=...)`
4. Receive `TunnelGrant(nonce=...)` — and trigger the daemon to spawn `crb-nfsd --listen 127.0.0.1:0 --root <root>` if not already running for this `(root, readonly)` tuple
5. Allocate free local loopback port `lp` (`socket.socket().bind(('127.0.0.1', 0))` → close)
6. Spawn supervisor process: listen on `127.0.0.1:lp`, on each accept open new connection to peer daemon and present `TunnelClaim(nonce)` — re-requesting nonce per accept
7. Run `mount_nfs -o vers=3,tcp,port=<lp>,mountport=<lp>,nolocks,actimeo=0,rdirplus,rsize=65536,wsize=65536,soft,timeo=50,retrans=2,deadtimeout=30,intr 127.0.0.1:/ ~/.crb/host/<host>`
8. Write entry to `~/.claude-workspace/claude-remote-bash/mounts.json`
9. Print mountpoint, exit (supervisor stays in background)

`mount_nfs` is NOT setuid on macOS (verified `/sbin/mount_nfs` is `-rwxr-xr-x`); non-root mounts succeed as long as `resvport` is omitted and the mount-point dir is owned by the user. So **no sudo needed**.

## Critical `mount_nfs` flags (Edit-tool correctness)

| Flag | Why |
|---|---|
| `vers=3` | nfsserve is v3-only; skip v4 negotiation |
| `tcp` | TCP (not UDP) — required by nfsserve and our tunnel |
| `port=N,mountport=N` | Same port for NFS + MOUNT (nfsserve serves both); avoids portmapper |
| `nolocks` | Disable NLM; nfsserve doesn't implement lock manager |
| **`actimeo=0`** | **Critical for Edit tool. Disables attribute cache. Without this, second Read after Write returns stale content because the page cache uses cached mtime as the validation key.** |
| `rdirplus` | Use READDIRPLUS for `ls -l` — saves a round-trip per file |
| `rsize=65536,wsize=65536` | 64KB blocks; reduces RPC count |
| `soft`, `timeo=50`, `retrans=2`, `deadtimeout=30`, `intr` | Reliable failure when peer goes offline; max ~15s hang on first failed op, mount marked dead after 30s |

**Cost of `actimeo=0`**: every `open()`/`stat()` re-fetches attrs. On a 10 GbE LAN that's ~1–2ms per call. For an Edit operation (~10 stat/open/read/write cycles) total overhead is 10–20ms. Negligible for editor workloads.

## MDM constraint on M4

Verified empirically in this session:
- M4's daemon binds inbound TCP and accepts our connections (we've used `claude-remote-bash execute --host M4` successfully throughout).
- `socketfilterfw --getblockall` on M4: disabled.
- MDM restrictions disable Apple's *Sharing* services (smbd, nfsd, afpd) — they do NOT block arbitrary user-space TCP listeners.

So option β (forward tunnel through PSK-authed control channel) works on M4 without modification. The reverse-tunnel fallback isn't needed.

## MVP scope (v0.1)

In:
- Rust `crb-nfsd` binary (~600 LOC adapted from upstream `mirrorfs.rs`): NFSv3 ops + MOUNTv3 over TCP, CLI `--listen IP:PORT --root PATH [--readonly]`.
- Python protocol additions: 4 new message types, additive to discriminated union.
- Daemon: tunnel grant state, raw-bridge dispatch, per-export `crb-nfsd` lifecycle.
- CLI: `crb mount`, `crb unmount`, `crb mounts`.
- Config: `nfs_export_root` (default `~$USER`), `nfs_path_allowlist`, `nfs_export_readonly`.

Out (deferred to v0.2+):
- Auto-mount on daemon discovery / boot-time launchd jobs
- `crb mount --all` (mount every discovered peer)
- Per-file ACLs / per-host different roots
- Reverse-tunnel architecture (only if M4 ever truly blocks inbound)
- Performance tuning (rsize/wsize benchmarks)
- NFSv4 / Kerberos
- Symlink CREATE (nfsserve doesn't support it; investigate workaround if needed for repo structure)
- MCP-server exposure of mount/unmount tools (CLI-only for v0.1)

Effort estimate: ~5 working days of focused work.

## Files (anticipated touch points)

| File | Change |
|---|---|
| `tools/crb-nfsd/` (NEW) | Rust binary, `cargo init` + nfsserve dep + adapted mirrorfs |
| `mcp/claude-remote-bash/claude_remote_bash/models.py` | Add 4 new message types to discriminated union |
| `mcp/claude-remote-bash/claude_remote_bash/daemon.py` | Tunnel grant state, `_dispatch_loop` branch for `RequestTunnelGrant`, claim-connection handler |
| `mcp/claude-remote-bash/claude_remote_bash/tunnel.py` (NEW) | Forward-tunnel client helpers (grant request, raw bridge, port allocator) |
| `mcp/claude-remote-bash/claude_remote_bash/mount.py` (NEW) | `crb mount` orchestration: supervisor process, mount_nfs invocation, mounts registry |
| `mcp/claude-remote-bash/claude_remote_bash/cli/main.py` | New `mount`/`unmount`/`mounts` subcommands |
| `mcp/claude-remote-bash/claude_remote_bash/auth.py` | Daemon config: NFS export root + allowlist |
| `~/.claude-workspace/claude-remote-bash/mounts.json` (runtime) | Per-mount registry on hub |
| Workspace install: `cargo install --path tools/crb-nfsd` (one-time, per machine) |

## Auth boundary (explicit)

- **PSK auth on control channel**: existing model, unchanged.
- **Nonce auth on tunnel connection**: 256-bit `secrets.token_hex(32)`, single-use, 10-second expiry. Issued only through PSK-authed control channel.
- **NFS data**: zero auth on the wire, but the wire is loopback-on-peer → tunnel → loopback-on-hub. Anyone with shell access on M3 during an active mount session can connect to the loopback port and access the NFS export. M3 is a personal Mac with one user; this is acceptable.
- **Peer-side `crb-nfsd` bound only to `127.0.0.1`** → no LAN exposure.

## Resolved decisions

1. **NFS export root scope**: peer's `/` (full filesystem). Daemon does no scoping by default. Per-host override config knob deferred to v0.2 (only useful if you ever want a tight allowlist on M4).
2. **Home shortcut**: on `crb mount <host>` success, also create M3-local symlink `~/.crb/host/<host>.home → ~/.crb/host/<host>/Users/<peer-user>/`. Removed on `crb unmount`. Future `crb shortcut <host> <name> <peer-path>` for named additions deferred to v0.2.
3. **Read-only**: writable by default; `--readonly` flag to opt in. Daemon enforces server-side (refuses WRITE), not just client-side.
4. **Mountpoint**: `~/.crb/host/<host>` — hidden, tool-namespaced.
5. **`crb-nfsd` install**: maturin `bindings = "bin"` pattern — note this is a **new pattern in this workspace** (bm25-rs/jsonl-chunker-rs use `pyo3/extension-module`, a different maturin mode that produces a Python extension `.so`). Maturin bin packaging is well-established upstream (ruff, uv, pydantic, [tpchgen-cli](https://kevinjqliu.github.io/blog/posts/tpchgen/index.html) ship binaries inside Python wheels this way). `crb-nfsd` becomes a workspace path dep of `claude-remote-bash` via `[tool.uv.sources]`. Running `uv tool install --editable mcp/claude-remote-bash` builds the Rust binary and installs it to the tool's bin (PATH-accessible). Daemon does `shutil.which('crb-nfsd')` at startup with a fail-fast hint if missing.
6. **MCP tool exposure**: deferred — CLI-only for v0.1.
7. **Symlinks**: `nfsserve` doesn't implement `SYMLINK` CREATE. Hard-fail with clear error rather than papering over. Document scope (next section).

## Scope of the mount: read/edit-only, not general filesystem

**Use the mount for Claude tool operations on file *contents*. Use `crb execute --host <h>` for anything that touches filesystem *metadata* or runs *peer-side processes*.**

### Use the mount for

- `Read` — full file content
- `Write` — overwriting file content
- `Edit` — string-replacement read-then-write
- `Grep` — in-file content search (slower than local; still works)
- `Glob` — path-based listing via `READDIRPLUS`

### Use `crb execute` for

| Operation | Why mount fails or is wrong tool |
|---|---|
| File locking (SQLite, editor lock files) | Mount uses `nolocks`; `fcntl(F_SETLK)` no-op |
| `git` operations | Local-FS assumptions + lock-file usage; runs slow over NFS |
| Build / compile / test | Thousands of small file ops; even 10 GbE is far slower than local SSD |
| Watching file changes (FSEvents) | Peer-local writes don't fire events on M3 |
| `tar` / `rsync` bulk ops | Drive from the peer side via `execute` |
| `mmap`-based access | macOS NFS mmap has weak coherency |
| `chown` (changing owner) | Cross-machine uid mapping fragile |
| Mac resource forks, `._*` files, `.DS_Store` | Daemon refuses these server-side |
| `xattr` operations | NFSv3 has no xattrs; quarantine/Spotlight/code-sign attrs don't survive |

### Works through the mount (empirically verified in this session)

`ln -s`, `chmod` (mode bits), `mkdir`, `rmdir`, `rename`, file creation, deletion, all read/write ops. The depth-dive had flagged `SYMLINK` and a few others as `nfsserve` limitations; empirical testing against nfsserve 0.11 + `mirrorfs` example showed all of these work. Plan revised after spike at `scratch/nfsserve-spike/`.

### Footguns — work but with surprises

1. **Symlink resolution is client-side.** Peer's `/foo -> /etc/hosts` opened through the mount resolves to M3's `/etc/hosts`, not peer's. Use `crb execute --host <h> 'readlink -f <path>'` to traverse symlinks correctly.
2. **Concurrent peer-local edits** can race with M3-mount edits despite `actimeo=0`. Avoid simultaneous writers.
3. **Spotlight indexing** auto-disabled per-mount via `mdutil -i off` + `.metadata_never_index` marker at export root.
4. **Time Machine** auto-excluded per-mount via `tmutil addexclusion`.
5. **Unicode normalization** — macOS APFS does NFC; NFS doesn't normalize. Non-ASCII filenames may differ across the mount.
6. **uid/gid mapping** assumes `chris=501` on every peer. Verify with `crb execute --host <h> 'id chris'` if permissions look wrong.
7. **Sleep/wake** stales the mount; `crb unmount && crb mount` to recover. No auto-recovery in v0.1.
8. **Path length limits** — `~/.crb/host/<h>/` adds ~20 chars; combined paths can exceed `PATH_MAX` (1024) on deeply-nested peer trees.
9. **Daemon child crashes** leave the mount in EIO until daemon restarts the `crb-nfsd` child.
10. **Finder polluting peer's FS** — opening the mount in Finder writes `.DS_Store` etc. Daemon refuses these patterns server-side; mount also uses `noappledouble`.

### Where these rules live

- **This plan doc** — canonical source.
- **`crb mount --help`** output — TL;DR + pointer to this doc.
- **`tools/crb-nfsd/README.md`** — server-side refused-operations list.
- Eventual `mcp/claude-remote-bash/README.md` when written.

## Background: full breadth survey

12 paths surveyed; only the top 3 listed here. Full survey in chat history (depth-first agent output, this session).

### Picked

**F4** — NFSv3 userspace server in daemon + native macOS `mount_nfs` (THIS PLAN).

### Fallbacks (in order)

- **F3** — SFTP server via `asyncssh.SFTPServer` + SSHFS via FUSE-T on M3. Solid; only fallback if F4 turns out to have a fatal flaw.
- **F2** — WebDAV via `wsgidav` + native `mount_webdav`. Cleanest protocol fit; risky due to Apple's WebDAV client reliability bugs for small files.

### Rejected

- Native macOS sharing (SMB/AFP/NFS via `nfsd`): blocked by M4 MDM
- macFUSE kext: requires Reduced Security boot on Apple Silicon
- Pure-Python NFS server: multi-week side quest, no benefit over Rust crate
- rclone `serve nfs`: `--vfs-cache-mode full` write-back layer breaks read-after-write for Edit tool
- Custom URI schemes: doesn't meet "real filesystem path" requirement
- Syncthing/Resilio: sync-not-mount, doesn't meet "no sync" constraint
- **Tailscale Taildrive** (WebDAV-over-Tailscale mesh): peer-tier prior art for the userspace-FS-over-authed-mesh pattern, BUT chose WebDAV which has known macOS-client reliability bugs ([zero-byte file bugs](https://github.com/nextcloud/all-in-one/issues/4582), [Mountain Lion+ Range-header corruption](https://discussions.apple.com/thread/4323101)). Taildrive accepts these reliability tradeoffs; we don't. Taildrive's existence *validates* the broader userspace-FS-via-tunnel architecture.

### Fallback escape hatch (if `xetdata/nfsserve` becomes untenable)

`xetdata/nfsserve` has a small user base (XetHub/HuggingFace primarily). If it stagnates, [`willscott/go-nfs`](https://github.com/willscott/go-nfs) (Apache-2.0, 770 stars, supports SYMLINK + READDIR which nfsserve lacks) is a Go-based alternative. Trade-off: their README self-describes as "minimally tested" with known macOS Finder bugs. Pin nfsserve to a specific commit so we can fork cleanly if needed.

## Primary sources

- [xetdata/nfsserve](https://github.com/xetdata/nfsserve) — BSD-3-Clause Rust NFSv3 server, used by HuggingFace `hf-mount` and XetHub `xet mount`
- [XetHub blog: NFS > FUSE](https://xethub.com/blog/nfs-fuse-why-we-built-nfs-server-rust) — design rationale for userspace NFS over FUSE
- [rclone serve nfs docs](https://rclone.org/commands/rclone_serve_nfs/) — VFS cache requirement documented (this is why rclone serve nfs was rejected)
- macOS man pages: `mount_nfs(8)`, `mount(8)`, `socketfilterfw(8)` (all verified locally this session)
- RFC 1813 (NFSv3 protocol)
- [FUSE-T project](https://www.fuse-t.org/) — kext-less FUSE alternative (relevant to fallback F3)
- [macFUSE 5.2.0](https://macfuse.github.io/2026/04/09/macfuse-5.2.0.html) — FSKit backend for kext-less alternative (relevant to fallback F3)