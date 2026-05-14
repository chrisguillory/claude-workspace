# crb-nfsd

Userspace NFSv3 server for the claude-remote-bash filesystem-mount feature. Spawned by the `claude-remote-bash` daemon per export; accessed by macOS `mount_nfs` through a port-forward tunnel.

## Usage

```bash
crb-nfsd --root <PATH> [--listen IP:PORT] [--readonly] [--block-prefix PATH]...
```

| Flag                  | Default       | Purpose                                                                                                                      |
|-----------------------|---------------|------------------------------------------------------------------------------------------------------------------------------|
| `--listen`            | `127.0.0.1:0` | Bind address. `:0` for ephemeral port (the actual port is printed to stdout).                                                |
| `--root`              | (required)    | Local directory to serve as NFS root. Canonicalized at startup.                                                              |
| `--readonly`          | off           | Refuse all mutating ops via `VFSCapabilities::ReadOnly`.                                                                     |
| `--block-prefix PATH` | none          | Refuse to expose any path under this prefix (canonical absolute; ancestor-symlink-safe via parent-canonicalize). Repeatable. |

**Startup signal**: before entering its main loop, `crb-nfsd` writes to stdout (flushed):

```
LISTEN_PORT=<n>
READY
```

The parent process discovers the bound port and confirms readiness before issuing tunnel grants.

## Server-side refusals (always-on)

Beyond what `--readonly` and `--block-prefix` enforce, the server unconditionally refuses several attack classes:

- **Multi-component filenames** — empty, `.`, `..`, `/`-containing, or NUL-containing `filename3` arguments → `NFS3ERR_INVAL`. Closes path-traversal via NFS filename injection.
- **Suspicious symlink targets** — absolute paths, `..` segments, and NUL bytes in `SYMLINK` target arguments → `NFS3ERR_INVAL`. Relative in-tree targets remain allowed.
- **Symlink-follow at operation handlers** — `read`/`write`/`setattr` type-gate against `NF3LNK` fileids; `read`/`write`/`File::create` use `O_NOFOLLOW` to defend against leaf-substitution TOCTOU. `rename` evicts the displaced cache entry to prevent stale-handle bypasses. `dirid` arguments must resolve to `NF3DIR`.
- **macOS pollution patterns** — `.DS_Store`, `._*` (AppleDouble), `.Spotlight-V100`, `.Trashes`, `.fseventsd`, `.TemporaryItems`, `.DocumentRevisions-V100`, `Icon\r`. Keeps Finder from polluting the peer's filesystem when the mount is opened in GUI.
- **Oversize reads** — READ requests with `count > 1 MB` (the advertised `rtmax`) are refused so the server can't be coerced into multi-GB response-buffer allocations.

## Install

From the workspace root:

```bash
uv sync --all-groups --all-packages
```

This builds `crb-nfsd` via maturin (declared as a workspace path-dep of `claude-remote-bash`) and puts the binary on PATH inside the venv.

## Scope (out of scope for this binary)

- Tunnel orchestration, mDNS announce, auth, mount-point management → live in `claude-remote-bash` daemon and CLI
- NFSv4 / Kerberos → not needed; nfsserve is v3-only
- Symlink CREATE → handled by upstream's standard `symlink` VFS hook (with target validation added — see refusals above)

## License

BSD-3-Clause, inheriting from [`xetdata/nfsserve`](https://github.com/xetdata/nfsserve)'s `examples/mirrorfs.rs`. See `LICENSE-UPSTREAM` for upstream copyright (XetData 2023, Hugging Face 2025).

Local additions on top of the upstream `mirrorfs.rs` example: clap-based CLI, `--block-prefix` / `--readonly` enforcement, `validate_component` and `validate_symlink_target` helpers, symlink-follow type-gates with `O_NOFOLLOW`, macOS pollution refusal, READ count cap, and `LISTEN_PORT/READY` stdout signal.