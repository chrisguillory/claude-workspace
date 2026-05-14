//! Userspace NFSv3 server for the claude-remote-bash filesystem-mount feature.
//!
//! Serves a local directory as NFSv3 over TCP. Spawned by the claude-remote-bash
//! daemon and accessed by macOS `mount_nfs` via a port-forward tunnel.
//!
//! Adapted from `xetdata/nfsserve`'s `examples/mirrorfs.rs` (BSD-3-Clause).
//! Local additions:
//!   - clap-based CLI (`--listen`, `--root`, `--readonly`, `--block-prefix`)
//!   - `--block-prefix` enforcement in `lookup` (canonicalized via parent)
//!   - `--readonly` enforcement via `capabilities()`
//!   - macOS pollution-name refusal (lookup/readdir/create/remove/rename)
//!   - filename validation (RFC 1813 §3.3.2: single-component `filename3`)
//!   - `LISTEN_PORT=<n>\nREADY` stdout signal for parent-process port capture
//!
//! Module layout:
//!   - [`pollution`] — `validate_component` and `is_pollution_pattern` helpers.
//!   - [`fs`] — `MirrorFS` and its `NFSFileSystem` impl.
//!   - this file — CLI parsing and the tokio entry point.
//!
//! Upstream license preserved at `LICENSE-UPSTREAM`.

mod fs;
mod pollution;

use std::io::Write;
use std::path::PathBuf;

use anyhow::Context;
use clap::Parser;
use nfsserve::tcp::{NFSTcp, NFSTcpListener};

use crate::fs::MirrorFS;

/// CLI for `crb-nfsd`. Spawned by the claude-remote-bash daemon per export.
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Bind address. Use `127.0.0.1:0` for an ephemeral port; the bound port
    /// is printed to stdout as `LISTEN_PORT=<n>` before the server enters its
    /// main loop so a parent process can capture it.
    #[arg(long, default_value = "127.0.0.1:0")]
    listen: String,

    /// Local directory to serve as the NFS root.
    #[arg(long)]
    root: PathBuf,

    /// Refuse all mutating NFS operations by reporting `VFSCapabilities::ReadOnly`.
    #[arg(long, default_value_t = false)]
    readonly: bool,

    /// Refuse to expose any path under this prefix (canonical absolute path).
    /// Repeatable. Used by the daemon to hide `$HOME/.crb/` (prevents recursive
    /// self-mount loops).
    #[arg(long, value_name = "PATH")]
    block_prefix: Vec<PathBuf>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_writer(std::io::stderr)
        .init();

    let args = Args::parse();

    // Canonicalize so symlinks and relative paths don't slip past the block check.
    let root = args
        .root
        .canonicalize()
        .with_context(|| format!("--root {} could not be canonicalized", args.root.display()))?;
    // Defensive metadata check. Upstream FSMap::new() does `root.metadata().unwrap()`
    // which would panic on a race (root removed between canonicalize and FSMap init).
    std::fs::metadata(&root)
        .with_context(|| format!("--root {} metadata unreadable", root.display()))?;
    // Hard-fail if any --block-prefix can't be canonicalized — better to refuse
    // to start than to silently serve a path that should have been hidden.
    let block_prefixes: Box<[PathBuf]> = args
        .block_prefix
        .iter()
        .map(|p| {
            p.canonicalize()
                .with_context(|| format!("--block-prefix {} could not be canonicalized", p.display()))
        })
        .collect::<anyhow::Result<Vec<_>>>()?
        .into_boxed_slice();

    let fs = MirrorFS::new(root, args.readonly, block_prefixes);
    let listener = NFSTcpListener::bind(&args.listen, fs).await?;

    // Print bound port + ready signal for the parent process. Flush immediately
    // — parent reads via a line-buffered pipe and expects these promptly.
    let port = listener.get_listen_port();
    let mut stdout = std::io::stdout().lock();
    writeln!(stdout, "LISTEN_PORT={port}")?;
    writeln!(stdout, "READY")?;
    stdout.flush()?;
    drop(stdout);

    listener.handle_forever().await.context("server exited")?;
    Ok(())
}
