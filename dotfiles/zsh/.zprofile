
# Setting PATH for Python 3.12
# The original version is saved in .zprofile.pysave
if [ -d "/Library/Frameworks/Python.framework/Versions/3.12/bin" ]; then
  PATH="/Library/Frameworks/Python.framework/Versions/3.12/bin:${PATH}"
  export PATH
fi

# Setting PATH for Python 3.10
# The original version is saved in .zprofile.pysave
if [ -d "/Library/Frameworks/Python.framework/Versions/3.10/bin" ]; then
  PATH="/Library/Frameworks/Python.framework/Versions/3.10/bin:${PATH}"
  export PATH
fi

# Select homebrew installation based on the current shell architecture. On an M1
# machine, either run the shell (e.g. Terminal app or a copy) with "Open using
# Rosetta", or run your using `arch -x86_64 $SHELL`.
case "$(uname -sm)" in
"Darwin arm64")
  eval "$(/opt/homebrew/bin/brew shellenv)"
  ;;
"Darwin x86_64")
  eval "$(/usr/local/Homebrew/bin/brew shellenv)"
  ;;
*)
  echo "Homebrew shellenv not configured. Unknown: $(uname -sm)"
esac

# GoLand launcher command for running "goland" from the terminal.
if [ -d "/Applications/GoLand.app/Contents/MacOS" ]; then
  export PATH="/Applications/GoLand.app/Contents/MacOS:$PATH"
fi

# IntelliJ IDEA launcher command for running "idea" from the terminal.
if [ -d "/Applications/IntelliJ IDEA.app/Contents/MacOS" ]; then
  export PATH="/Applications/IntelliJ IDEA.app/Contents/MacOS:$PATH"
fi

# PyCharm launcher commands for running "pycharm" and "charm" from the terminal.
if [ -d "/Applications/PyCharm.app/Contents/MacOS" ]; then
  export PATH="/Applications/PyCharm.app/Contents/MacOS:$PATH"
  alias charm="pycharm"
fi

# RubyMine launcher commands for running "rubymine" and "mine" from the terminal.
if [ -d "/Applications/RubyMine.app/Contents/MacOS" ]; then
  export PATH="/Applications/RubyMine.app/Contents/MacOS:$PATH"
  alias mine="rubymine"
fi

# WebStorm launcher commands for running "webstorm" and "storm" from the terminal.
if [ -d "/Applications/WebStorm.app/Contents/MacOS" ]; then
  export PATH="/Applications/WebStorm.app/Contents/MacOS:$PATH"
  alias storm="webstorm"
fi

# Increase the number of commands stored in history
HISTSIZE=1000000  # Number of commands to store in memory
SAVEHIST=1000000  # Number of commands to store on disk

# Optional Enhancements
setopt HIST_IGNORE_DUPS        # Avoid duplicate consecutive entries
setopt HIST_EXPIRE_DUPS_FIRST  # Expire duplicate entries first when trimming history
setopt HIST_IGNORE_ALL_DUPS    # Don't store duplicates in history (file dedupe occurs on shell exit)
setopt HIST_IGNORE_SPACE       # Ignore commands starting with a space
setopt SHARE_HISTORY           # Share history across all sessions
setopt APPEND_HISTORY          # Append history instead of overwriting
setopt INC_APPEND_HISTORY      # Immediately append to history after each command

# Additional Optional Settings for Enhanced History Management
setopt HIST_FIND_NO_DUPS      # Prevent duplicates when searching history
setopt EXTENDED_HISTORY       # Store timestamps in history

# Set the config path for ripgrep.
export RIPGREP_CONFIG_PATH="$HOME/.ripgreprc"

# --- Dock bounce control (macOS) --------------------------------------------
# Ensure Dock icon bouncing is disabled; restart Dock only if we changed it.
#
# Notes:
# - Placement: this lives in .zprofile which is sourced by login shells. If your
#   terminal does not start login shells, this block may not run. For a terminal-
#   independent, once-per-login setup, prefer a LaunchAgent at
#   ~/Library/LaunchAgents (runs at user login without a terminal).
# - Performance: this executes one 'defaults read' per login; overhead is tiny.
#   Dock is only restarted when the value changes to avoid disruption.
# - Scope: this disables all Dock icon bouncing globally. Revert with:
#     defaults delete com.apple.dock no-bouncing; killall Dock
if [[ "$OSTYPE" == darwin* ]]; then
  if ! defaults read com.apple.dock no-bouncing 2>/dev/null | grep -q '^1$'; then
    defaults write com.apple.dock no-bouncing -bool true
    /usr/bin/killall Dock >/dev/null 2>&1 || true
  fi
fi

# --- Power chime control (macOS) --------------------------------------------
# Disable the charging sound when connecting power cable
if [[ "$OSTYPE" == darwin* ]]; then
  if ! defaults read com.apple.PowerChime ChimeOnNoHardware 2>/dev/null | grep -q '^1$'; then
    defaults write com.apple.PowerChime ChimeOnNoHardware -bool true
    /usr/bin/killall PowerChime >/dev/null 2>&1 || true
  fi
fi

# Set the config path for ripgrep.
export RIPGREP_CONFIG_PATH="$HOME/.ripgreprc"

# some commented stuff to figure out later

if [[ ":$PATH:" != *":$HOME/.local/bin:"* ]]; then
  export PATH="$PATH:$HOME/.local/bin"
fi

if [ -d "$HOME/.bun/bin" ]; then
  export BUN_INSTALL="$HOME/.bun"
  export PATH="$BUN_INSTALL/bin:$PATH"
  # bun completions
  [ -s "$HOME/.bun/_bun" ] && source "$HOME/.bun/_bun"
fi


# Set up completions
# fpath+=~/.zfunc
# autoload -Uz compinit && compinit

# Set up completions for aws cli
# if command -v aws_completer > /dev/null; then
#     autoload -U +X bashcompinit && bashcompinit
#     complete -C $(which aws_completer) aws
# fi

# Enable atlas completions
# if command -v atlas > /dev/null; then
#     source <(atlas completion zsh)
# fi

# Enable docker autocompletions
# if command -v docker > /dev/null; then
#     source <(docker completion zsh)
# fi

# Enable kubectl autocompletions
# if command -v kubectl > /dev/null; then
#     source <(kubectl completion zsh)
# fi

# Ryi
# source "$HOME/.rye/env"
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!

# __conda_setup="$('/Users/chris/miniconda3/bin/conda' 'shell.zsh' 'hook' 2> /dev/null)"
# if [ $? -eq 0 ]; then
#     eval "$__conda_setup"
# else
#     if [ -f "/Users/chris/miniconda3/etc/profile.d/conda.sh" ]; then
#         . "/Users/chris/miniconda3/etc/profile.d/conda.sh"
#     else
#         export PATH="/Users/chris/miniconda3/bin:$PATH"
#     fi
# fi
# unset __conda_setup
# <<< conda initialize <<<

# Ensure history is written and reloaded frequently to preserve all history
# autoload -Uz add-zsh-hook
# function sync_history() {
#     fc -A &> /dev/null  # Save current session’s history silently
#     history -n &> /dev/null  # Reload the latest history silently
# }
# add-zsh-hook precmd sync_history  # Sync before showing a prompt

# Machine-private config (work tokens, host-specific env) — lives OUTSIDE the repo,
# beside the zsh symlink, so no repo operation can ever capture it. Sourced last,
# so it can override anything the shared config set.
[ -f "$HOME/.config/zsh.local" ] && source "$HOME/.config/zsh.local"
