# dotfiles

Shell environment for every machine on the mesh: zsh config (`zsh/`), the Homebrew
toolchain manifest (`Brewfile`), and the installer that wires a machine to them.

## Install

```sh
cd ~/claude-workspace && bash dotfiles/install.sh
```

Idempotent — run it on a fresh machine, after pulling Brewfile/mise changes, or any
time. It touches only `$HOME` (the `ZDOTDIR` line in `~/.zshenv`, the `~/.config/zsh`
symlink, mise trust) plus a missing-only `brew bundle`. Open a new shell afterwards.

## Machine-private config

Per-machine secrets and host-specific env live in `~/.config/zsh.local` — OUTSIDE the
repo (a sibling of the symlink), so no git operation can ever capture it. `.zprofile`
sources it last if present. Start from the template:

```sh
cp dotfiles/zsh/zsh.local.example ~/.config/zsh.local
```

## Completions

`zsh/completions/` is two-track: the universal files are tracked; per-machine
snapshots written by `<tool> install-completions` are gitignored. See
[`zsh/completions/README.md`](zsh/completions/README.md).
