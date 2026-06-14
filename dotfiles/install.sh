#!/bin/sh
# install.sh — point this machine's zsh at the repo's dotfiles.
# Idempotent: safe to re-run. Backs up anything it would overwrite; never deletes.
set -eu

REPO_ZSH="$(CDPATH= cd -- "$(dirname -- "$0")/zsh" && pwd)"
ZSHENV="$HOME/.zshenv"
LINK="$HOME/.config/zsh"
TS="$(date +%Y%m%d-%H%M%S)"

# 1) Ensure ~/.zshenv sets ZDOTDIR (a real file in $HOME — never symlinked).
LINE='export ZDOTDIR="$HOME/.config/zsh"'
if [ -f "$ZSHENV" ] && grep -qxF "$LINE" "$ZSHENV"; then
  echo "ZDOTDIR already set in $ZSHENV"
else
  # Guard: a file lacking a trailing newline would concatenate-corrupt its last line.
  if [ -s "$ZSHENV" ] && [ -n "$(tail -c 1 "$ZSHENV")" ]; then printf '\n' >> "$ZSHENV"; fi
  printf '%s\n' "$LINE" >> "$ZSHENV"
  echo "Appended ZDOTDIR line to $ZSHENV"
fi

# 2) Point ~/.config/zsh -> the repo's dotfiles/zsh.
mkdir -p "$(dirname "$LINK")"
if [ -L "$LINK" ]; then
  CUR="$(readlink "$LINK")"          # safe: only reached when -L is true
  if [ "$CUR" = "$REPO_ZSH" ]; then
    echo "Symlink already correct: $LINK -> $REPO_ZSH"
  else
    ln -sfn "$REPO_ZSH" "$LINK"      # -n: replace the LINK itself, never create inside its target
    echo "Re-pointed $LINK (was -> $CUR)"
  fi
elif [ -e "$LINK" ]; then
  mv "$LINK" "$LINK.bak-$TS"         # mv, never rm: a real dir may hold live state (history, completions)
  ln -s "$REPO_ZSH" "$LINK"
  echo "Backed up real $LINK -> $LINK.bak-$TS and linked"
else
  ln -s "$REPO_ZSH" "$LINK"
  echo "Linked $LINK -> $REPO_ZSH"
fi

# 3) Trust the repo's mise config (per-machine, content-hash trust DB under $HOME —
#    fresh boxes start untrusted, and any .mise.toml edit re-triggers; re-running
#    this script re-trusts).
REPO_ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
if [ -f "$REPO_ROOT/.mise.toml" ] && command -v mise >/dev/null 2>&1; then
  mise trust "$REPO_ROOT/.mise.toml"
fi

# 4) Converge to the Brewfile toolchain (missing-only).
#    check is much faster than install when everything is present (the common re-run);
#    on the install path: no metadata auto-update, no upgrades — upgrading stays manual.
BREWFILE="$(dirname -- "$0")/Brewfile"
if command -v brew >/dev/null 2>&1; then
  if brew bundle check --file="$BREWFILE" >/dev/null 2>&1; then
    echo "Brewfile satisfied"
  else
    HOMEBREW_NO_AUTO_UPDATE=1 brew bundle install --no-upgrade --file="$BREWFILE"
  fi
else
  echo "brew not found — skipped Brewfile (install Homebrew, then re-run)"
fi

echo "Done. Open a new shell (or: exec zsh)."
