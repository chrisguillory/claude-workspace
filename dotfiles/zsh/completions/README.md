# zsh completions

Files here are picked up by `compinit` via `fpath` (see `dotfiles/zsh/.zshrc`).

## Two tracks

**Tracked** (universal — same on every machine):

| File   | Tool       |
|--------|------------|
| `_cdk` | AWS CDK    |
| `_gh`  | GitHub CLI |

**Gitignored** (per-machine — varies by what's installed locally):

- `_claude-*`, `_document-search`, `_selenium-browser`, `_crb`, ... — written by `<tool> install-completions`. These are version-pinned snapshots of the installed CLI on each machine; syncing them would break tab-completion when CLIs are upgraded or renamed.
- Vendor-installed shell state (iTerm2, Docker, `.zcompdump`, etc.) is gitignored in `dotfiles/zsh/.gitignore`.

## Adding a universal completion

1. Drop the file here (`dotfiles/zsh/completions/_<name>`).
2. Allowlist it in `dotfiles/zsh/.gitignore`:
   ```
   !completions/_<name>
   ```
3. Commit both.
