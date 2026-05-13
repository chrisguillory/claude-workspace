# crb-nfsd

Userspace NFSv3 server for the claude-remote-bash filesystem-mount feature. Spawned by the `claude-remote-bash` daemon per export; accessed by macOS `mount_nfs` through a port-forward tunnel.

Single-binary Rust crate, adapted from [`xetdata/nfsserve`](https://github.com/xetdata/nfsserve)'s `examples/mirrorfs.rs` (BSD-3-Clause). License preserved at `LICENSE-UPSTREAM`.

## Usage

```bash
crb-nfsd --root <PATH> [--listen IP:PORT] [--readonly] [--block-prefix PATH]...
```

| Flag | Default | Purpose |
|---|---|---|
| `--listen` | `127.0.0.1:0` | Bind address. `:0` for ephemeral port (the actual port is printed to stdout). |
| `--root` | (required) | Local directory to serve as NFS root. Canonicalized at startup. |
| `--readonly` | off | Refuse all mutating ops via `VFSCapabilities::ReadOnly`. |
| `--block-prefix PATH` | none | Refuse to expose any path under this prefix. Repeatable. Used by the daemon to hide `$HOME/.crb/` and prevent recursive self-mount. |

**Startup signal**: before entering its main loop, `crb-nfsd` writes to stdout (flushed):
```
LISTEN_PORT=<n>
READY
```
This is how the parent process (the claude-remote-bash daemon) discovers the bound port and confirms the server is ready before issuing tunnel grants.

## Server-side refusals (always-on)

The server refuses operations beyond what `--block-prefix` and `--readonly` cover:

- **macOS pollution patterns** are refused on `create`, `mkdir`, `rename` (destination) and filtered out of `lookup`, `readdir`, `remove`, `rename` (source). The blocked names:
  - `.DS_Store` (Finder folder state)
  - `._*` (AppleDouble extended-attribute sidecars)
  - `.Spotlight-V100`, `.Trashes`, `.fseventsd`, `.TemporaryItems`, `.DocumentRevisions-V100`
  - `Icon\r` (macOS Custom Folder Icon — literal 0x0D byte)

  Rationale: when the mount is opened in Finder, macOS auto-writes these. Refusing them server-side prevents the mount from polluting the peer's filesystem.

- **Recursive paths** (via `--block-prefix`): canonicalized path comparison resolves symlinks under the served tree, so peer-side `~/aaa → /Users/chris/.crb` is correctly caught alongside the literal path.

## Install

Packaged via maturin `bindings = "bin"`. The wheel contains `crb_nfsd-<ver>.data/scripts/crb-nfsd` (the binary), `LICENSE-UPSTREAM`, and a CycloneDX SBOM.

A future commit on this branch will add `crb-nfsd` as a workspace path dep in `mcp/claude-remote-bash/pyproject.toml`. Once that lands, `uv tool install --editable mcp/claude-remote-bash` will build and install the binary on PATH, and the daemon will do `shutil.which('crb-nfsd')` at startup.

## Scope (out of scope for this binary)

- Tunnel orchestration, mDNS announce, auth, mount-point management → live in `claude-remote-bash` daemon and CLI
- NFSv4 / Kerberos → not needed; nfsserve is v3-only
- Symlink CREATE → handled by upstream's standard `symlink` VFS hook; no special-casing required

## License

This crate is BSD-3-Clause, inheriting from upstream `xetdata/nfsserve`. See `LICENSE-UPSTREAM` for upstream copyright (XetData 2023, Hugging Face 2025). Local modifications adapt the canonical `mirrorfs.rs` example with claude-remote-bash-specific features (CLI, `--block-prefix`, `--readonly`, pollution refusal, ready-signal stdout).
