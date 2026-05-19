#!/usr/bin/env bash
#
# Audio-mesh bootstrap — idempotent setup for any Mac participating in the
# claude-remote-audio mesh (hub or peer; the bootstrap is the same).
#
# Installs, when missing:
#   - switchaudio-osx          (brew)    default audio device control
#   - blueutil                 (brew)    Bluetooth CLI
#   - roc-toolkit build deps   (brew)    scons + libs needed to compile roc-toolkit
#   - roc-toolkit              (source)  roc-send + roc-recv installed to /usr/local/bin
#   - claude-coreaudio-volume  (source)  Swift CLI for per-device Core Audio volume,
#                                        compiled from inlined source (see env var below)
#
# Excluded (deliberately — needs interactive setup or sudo+reboot):
#   - roc-vad kernel driver    one-time `curl ... | sudo bash` + sudo killall coreaudiod
#   - claude-remote-bash-daemon  PSK + alias + launchd registration are per-host secrets
#   - macOS Bluetooth TCC grant  must be a GUI click in the system permission dialog
#
# Re-running is a no-op when everything is already installed.
#
# Env vars consumed (set by _install_prereqs_on_host before dispatch):
#   - SUDO_PASSWORD                          (optional) piped into sudo -S for non-TTY installs
#   - CRA_SWIFT_CLAUDE_COREAUDIO_VOLUME_B64  (required for the Swift compile) base64-encoded
#                                            source of swift/claude-coreaudio-volume.swift

set -euo pipefail

err() {
    echo "ERROR: $*" >&2
    exit 1
}
ok() { echo "  ✓ $*"; }
info() { echo "  ▷ $*"; }

[[ "$(uname)" = "Darwin" ]] || err "macOS only (got $(uname))"
command -v brew >/dev/null || err "Homebrew not installed — https://brew.sh"

brew_install_if_missing() {
    local pkg="$1"
    if brew list "$pkg" >/dev/null 2>&1; then
        ok "$pkg"
    else
        info "installing $pkg..."
        brew install "$pkg"
    fi
}

echo "=== audio-mesh tooling ==="
brew_install_if_missing switchaudio-osx
brew_install_if_missing blueutil

echo
echo "=== roc-toolkit ==="
if command -v roc-send >/dev/null && command -v roc-recv >/dev/null; then
    ok "roc-toolkit (already installed: $(roc-send --version 2>&1 | head -1))"
else
    info "installing roc-toolkit build dependencies..."
    for pkg in scons libuv speexdsp sox libsndfile openssl@3 pkg-config ragel gengetopt cmake autoconf automake libtool; do
        brew_install_if_missing "$pkg"
    done

    info "cloning roc-toolkit source to ~/src/roc-toolkit (if missing)..."
    mkdir -p ~/src
    if [[ ! -d ~/src/roc-toolkit ]]; then
        git clone https://github.com/roc-streaming/roc-toolkit.git ~/src/roc-toolkit
    fi

    info "building roc-toolkit (~5 min compile)..."
    cd ~/src/roc-toolkit
    # --build-3rdparty=all is load-bearing on BOTH the build and the install;
    # without it on install, scons re-runs config without bundled deps and
    # errors with "openfec not found".
    scons -Q --build-3rdparty=all

    info "installing to /usr/local..."
    # Modern Apple Silicon: brew lives at /opt/homebrew, so /usr/local is the
    # conventional spot for manual artifacts like this and is normally
    # user-writable. Falls back to sudo when not — interactive prompt for local
    # runs, $SUDO_PASSWORD-piped for remote dispatch via
    # `claude-remote-audio apply --install-prereqs` (no TTY, no hang).
    if [[ -w /usr/local/bin && -w /usr/local/lib && -w /usr/local/include ]]; then
        scons -Q --build-3rdparty=all install
    elif [[ -n "${SUDO_PASSWORD:-}" ]]; then
        info "(sudo via piped password — /usr/local subdirs aren't user-writable)"
        echo "$SUDO_PASSWORD" | sudo -S -p "" scons -Q --build-3rdparty=all install
    else
        info "(sudo needed — /usr/local subdirs aren't user-writable)"
        sudo scons -Q --build-3rdparty=all install
    fi

    ok "roc-toolkit installed: $(roc-send --version 2>&1 | head -1)"
fi

echo
echo "=== claude-coreaudio-volume ==="
# Compile + install if source is provided via env var; otherwise verify the
# binary is already in PATH. The orchestrator's `--install-prereqs` flow always
# sets CRA_SWIFT_CLAUDE_COREAUDIO_VOLUME_B64 to the current source, so this
# always installs the latest version during dispatched bootstraps.
if [[ -n "${CRA_SWIFT_CLAUDE_COREAUDIO_VOLUME_B64:-}" ]]; then
    info "compiling claude-coreaudio-volume from inlined Swift source..."
    swift_src=/tmp/claude-coreaudio-volume.swift
    swift_bin=/tmp/claude-coreaudio-volume
    echo "$CRA_SWIFT_CLAUDE_COREAUDIO_VOLUME_B64" | base64 -d > "$swift_src"
    swiftc -O "$swift_src" -o "$swift_bin" -framework CoreAudio -framework Foundation

    info "installing to /usr/local/bin..."
    if [[ -w /usr/local/bin ]]; then
        mv "$swift_bin" /usr/local/bin/
    elif [[ -n "${SUDO_PASSWORD:-}" ]]; then
        echo "$SUDO_PASSWORD" | sudo -S -p "" mv "$swift_bin" /usr/local/bin/
    else
        sudo mv "$swift_bin" /usr/local/bin/
    fi
    rm -f "$swift_src"
    ok "claude-coreaudio-volume installed"
elif command -v claude-coreaudio-volume >/dev/null; then
    ok "claude-coreaudio-volume (already installed)"
else
    err "claude-coreaudio-volume not installed and no source provided via CRA_SWIFT_CLAUDE_COREAUDIO_VOLUME_B64"
fi

echo
echo "✓ bootstrap complete on $(scutil --get LocalHostName 2>/dev/null || hostname)"
echo
echo "Manual one-time steps (not handled by this script):"
echo "  • roc-vad driver:"
echo "      sudo /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/roc-streaming/roc-vad/HEAD/install.sh)\""
echo "      sudo killall coreaudiod"
echo "  • Bluetooth TCC:"
echo "      Run \`blueutil --power\` once locally. Click Allow on the system dialog."
echo "  • claude-remote-bash-daemon:"
echo "      See the claude-remote-bash README for PSK + launchd setup."
