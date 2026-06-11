# Tab completion: register our completions dir, then initialize compinit.
# Order matters — fpath must be set before compinit runs.
fpath=($HOME/.config/zsh/completions $fpath)
autoload -U compinit
compinit

# $HOME/.local/bin is where user-scope installers drop binaries (uv tool, pip --user, etc.).
if [[ ":$PATH:" != *":$HOME/.local/bin:"* ]]; then
  export PATH="$PATH:$HOME/.local/bin"
fi

# direnv hook — per-directory env switching. precmd, so interactive-only.
command -v direnv >/dev/null 2>&1 && eval "$(direnv hook zsh)"

# mise hook — per-directory env (.mise.toml venv, tool versions) on cd; revert-on-leave.
# After direnv so mise takes precedence where both apply.
command -v mise >/dev/null 2>&1 && eval "$(mise activate zsh)"
