# Complete Zsh Startup Order

## System parameters (Terminal: interactive login)

| What               | Command                                  | Output                             |
|--------------------|------------------------------------------|------------------------------------|
| Zsh version        | `zsh --version`                          | `zsh 5.9 (arm64-apple-darwin24.0)` |
| RCS and GLOBAL_RCS | `echo $options[rcs] $options[globalrcs]` | `on on`                            |
| ZDOTDIR            | `echo ${ZDOTDIR:-$HOME}`                 | `/Users/chris`                     |

## What counts as login vs interactive
- **login shell**: e.g., first shell on tty/SSH, or when a terminal starts a “login shell”.
- **interactive shell**: reads from a terminal (has a prompt); subshells or scripts are usually non-interactive.

## Canonical order (startup)
- **All invocations**
  - `/etc/zshenv` — always read; cannot be overridden
  - `$ZDOTDIR/.zshenv` (if `RCS` is set; see below)
- **If login shell**
  - `/etc/zprofile` (if `GLOBAL_RCS` and `RCS` are set)
  - `$ZDOTDIR/.zprofile` (if `RCS` is set)
- **If interactive shell**
  - `/etc/zshrc` (if `GLOBAL_RCS` and `RCS` are set)
  - `$ZDOTDIR/.zshrc` (if `RCS` is set)
- **If login shell (final stage)**
  - `/etc/zlogin` (if `GLOBAL_RCS` and `RCS` are set)
  - `$ZDOTDIR/.zlogin` (if `RCS` is set)

## Canonical order (shutdown)
When a login shell exits:
- `$ZDOTDIR/.zlogout` (if `RCS` is set)
- `/etc/zlogout` (if `GLOBAL_RCS` and `RCS` are set)

Notes:
- Logout files are not read if the shell terminates by `exec`-ing another process.
- If `RCS` is unset at exit time, no history file will be saved.

## Scenario matrix
- **Login + interactive**: `/etc/zshenv` → `$ /.zshenv` → `/etc/zprofile` → `$ZDOTDIR/.zprofile` → `/etc/zshrc` → `$ZDOTDIR/.zshrc` → `/etc/zlogin` → `$ZDOTDIR/.zlogin`. On exit: `$ZDOTDIR/.zlogout` → `/etc/zlogout`.
- **Login + non-interactive**: `/etc/zshenv` → `$ZDOTDIR/.zshenv` → `/etc/zprofile` → `$ZDOTDIR/.zprofile` → `/etc/zlogin` → `$ZDOTDIR/.zlogin`.
- **Non-login + interactive**: `/etc/zshenv` → `$ZDOTDIR/.zshenv` → `/etc/zshrc` → `$ZDOTDIR/.zshrc`.
- **Non-login + non-interactive**: `/etc/zshenv` → `$ZDOTDIR/.zshenv`.

## Options that affect sourcing
- **RCS**: Controls reading of all startup/shutdown files except `/etc/zshenv`.
  - `zsh -f` unsets `RCS`. Effect: only `/etc/zshenv` runs; all subsequent files (including `$ZDOTDIR/.zshenv`) are skipped.
  - If `RCS` is unset at any point, subsequent startup files are not read.
  - In `/etc/zshenv`, guard optional work with `[[ -o rcs ]]` so it doesn’t run under `zsh -f`.
- **GLOBAL_RCS**: Controls reading of system-wide files (the ones under `/etc`).
  - Can be unset to skip global files; a file in `$ZDOTDIR` may re-enable it.
  - Both `RCS` and `GLOBAL_RCS` are set by default.

## ZDOTDIR and paths
- `$ZDOTDIR` defaults to `$HOME` if unset.
- Files shown under `/etc` may reside in an installation-specific directory.

## Emulation as sh/ksh (invoked as `sh`, `ksh`, or via `su` name)
- The usual zsh startup/shutdown scripts are not executed.
- For login shells: `source /etc/profile` then `$HOME/.profile`.
- If `ENV` is set on invocation, `$ENV` is sourced after the profile scripts (after expansion/substitution).
- The `PRIVILEGED` option also affects execution of startup files.

## Compilation
- Any of these files may be precompiled with `zcompile` to produce a `.zwc`; if the `.zwc` exists and is newer, the compiled file is used.

## Practical placement guidance
- `~/.zshenv`: minimal; environment needed everywhere (avoid output). Use `[[ -o rcs ]]` to gate optional code.
- `~/.zprofile`: login-only environment/session setup (e.g., PATH, `ulimit`, starting desktop session).
- `~/.zshrc`: interactive-only config (prompt, keybindings, aliases, completion, widgets, interactive tools).
- `~/.zlogin`: commands that should run once per login after `~/.zshrc`.
- `~/.zlogout`: cleanup for login shells.

## Quick reference (files)
- `$ZDOTDIR/.zshenv`, `$ZDOTDIR/.zprofile`, `$ZDOTDIR/.zshrc`, `$ZDOTDIR/.zlogin`, `$ZDOTDIR/.zlogout`
- `/etc/zshenv`, `/etc/zprofile`, `/etc/zshrc`, `/etc/zlogin`, `/etc/zlogout`
