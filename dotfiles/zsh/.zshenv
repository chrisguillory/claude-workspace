# Rust toolchain — rustup's env file puts $HOME/.cargo/bin on PATH.
# Conditional: rustup may not be installed.
[ -f "$HOME/.cargo/env" ] && . "$HOME/.cargo/env"
