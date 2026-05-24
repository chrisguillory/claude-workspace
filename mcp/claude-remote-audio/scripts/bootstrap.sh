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
#                                        compiled from inlined source (see env vars below)
#   - claude-tcc-probe         (source)  Swift CLI that prints AVCaptureDevice mic auth
#                                        status — disambiguates TCC denial from HAL
#                                        wedge in the roc-send-start diagnostic
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
#   - CRA_SWIFT_CLAUDE_COREAUDIO_VOLUME_B64  (required) base64-encoded source of
#                                            swift/claude-coreaudio-volume.swift
#   - CRA_SWIFT_CLAUDE_TCC_PROBE_B64         (required) base64-encoded source of
#                                            swift/claude-tcc-probe.swift

set -euo pipefail

# Resolve script's absolute directory at the top — needed because the body
# `cd`s into the roc-toolkit checkout before referencing repo-local files,
# and `$(dirname "$0")` would then resolve relative to the new cwd.
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

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
# Pinned to roc-toolkit master HEAD that includes "fix #829: panicks with sox
# 14.6.1-1" — required for our sox_ng 14.8.0 patch to drive enough internal
# driver slots. Bump this SHA when adopting a newer master; verify patches
# still apply with `git apply --check`.
ROC_TOOLKIT_PIN_SHA=f37b5b90c3414c9a6eccf0cbcfbf00e0f5be7c38
if command -v roc-send >/dev/null && command -v roc-recv >/dev/null && [[ -z "${CRA_FORCE_REBUILD:-}" ]]; then
    ok "roc-toolkit (already installed: $(roc-send --version 2>&1 | head -1))"
    info "(set CRA_FORCE_REBUILD=1 to force rebuild from pinned SHA + apply patch series)"
else
    info "installing roc-toolkit build dependencies..."
    for pkg in scons libuv speexdsp sox libsndfile openssl@3 pkg-config ragel gengetopt cmake autoconf automake libtool; do
        brew_install_if_missing "$pkg"
    done

    # Build tree lives under our managed-state dir — same convention as apply logs
    # at ~/.claude-workspace/mcp/claude-remote-audio/logs/. Avoids squatting on
    # the user's ~/src/ namespace and gives the orchestrator a single place
    # to enumerate everything claude-remote-audio owns on this host.
    ROC_TOOLKIT_DIR=~/.claude-workspace/mcp/claude-remote-audio/build/roc-toolkit
    mkdir -p "$(dirname "$ROC_TOOLKIT_DIR")"

    # One-time migration: move legacy ~/src/roc-toolkit/ checkout forward so we
    # don't lose its incremental scons build state on the next reset+rebuild.
    if [[ -d ~/src/roc-toolkit && ! -d "$ROC_TOOLKIT_DIR" ]]; then
        info "migrating ~/src/roc-toolkit/ → $ROC_TOOLKIT_DIR/"
        mv ~/src/roc-toolkit "$ROC_TOOLKIT_DIR"
        rmdir ~/src 2>/dev/null || true   # only succeeds if ~/src was ours-only
    fi

    info "cloning roc-toolkit source to $ROC_TOOLKIT_DIR (if missing)..."
    if [[ ! -d "$ROC_TOOLKIT_DIR" ]]; then
        git clone https://github.com/roc-streaming/roc-toolkit.git "$ROC_TOOLKIT_DIR"
    fi

    info "resetting $ROC_TOOLKIT_DIR to pinned SHA ${ROC_TOOLKIT_PIN_SHA:0:10}..."
    cd "$ROC_TOOLKIT_DIR"
    git fetch --quiet origin
    git checkout --quiet "$ROC_TOOLKIT_PIN_SHA"
    git reset --hard --quiet
    # Don't clean build/ — scons is incremental and clean tree forces a 10-min
    # 3rdparty rebuild every time. Just clean uncommitted source changes that
    # would conflict with patch application.
    git clean -fd --quiet -- src/ scripts/ 3rdparty/SConscript 2>/dev/null || true

    # Apply our patch series. Patches come from one of:
    #   1. CRA_PATCHES_TARBALL_B64 env var (orchestrator dispatch via
    #      --install-prereqs base64-tars the patches/ dir before sending)
    #   2. Repo-local patches/ dir adjacent to this script (when bootstrap.sh
    #      runs directly from a checkout)
    info "locating claude-remote-audio patch series..."
    if [[ -n "${CRA_PATCHES_TARBALL_B64:-}" ]]; then
        PATCH_DIR=$(mktemp -d)
        printf '%s' "$CRA_PATCHES_TARBALL_B64" | base64 -d | tar -xz -C "$PATCH_DIR"
        info "  patches loaded from env var → $PATCH_DIR"
    elif [[ -d "$SCRIPT_DIR/../patches" ]]; then
        PATCH_DIR="$SCRIPT_DIR/../patches"
        info "  patches loaded from repo → $PATCH_DIR"
    else
        err "no patches found (set CRA_PATCHES_TARBALL_B64 or run from repo with patches/ dir)"
    fi

    for patch in "$PATCH_DIR"/*.patch; do
        [[ -e "$patch" ]] || continue
        info "  applying $(basename "$patch")"
        if ! git apply --check "$patch" 2>/dev/null; then
            err "patch $(basename "$patch") does not apply cleanly against pinned SHA"
        fi
        git apply "$patch"
    done

    info "building roc-toolkit (~5 min compile, incremental after first)..."
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

    # Ad-hoc re-sign the installed binaries. macOS attaches a
    # `com.apple.provenance` xattr when sudo copies a binary from a
    # user-built location to a system path; Gatekeeper then refuses to
    # launch the binary (the symptom is roc-send/roc-recv hanging silently,
    # exit 137 under timeout, no stderr). The provenance xattr is
    # SIP-protected and cannot be stripped even by root; the workable
    # bypass is to re-sign with an ad-hoc signature, which Gatekeeper
    # accepts. Same shape as any custom-built binary going into /usr/local
    # via sudo on macOS Sequoia+.
    info "ad-hoc codesigning installed binaries (bypass Gatekeeper provenance block)..."
    for bin in /usr/local/bin/roc-send /usr/local/bin/roc-recv /usr/local/bin/roc-copy; do
        if [[ -n "${SUDO_PASSWORD:-}" ]]; then
            echo "$SUDO_PASSWORD" | sudo -S -p "" codesign --force --sign - "$bin" 2>&1
        else
            sudo codesign --force --sign - "$bin"
        fi
    done

    ok "roc-toolkit installed: $(roc-send --version 2>&1 | head -1)"
    if /usr/local/bin/roc-send --help 2>&1 | grep -q -- '--channels'; then
        ok "  patched: --channels flag present"
    else
        err "post-install verification failed: --channels flag missing from installed roc-send"
    fi
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
echo "=== claude-tcc-probe ==="
if [[ -n "${CRA_SWIFT_CLAUDE_TCC_PROBE_B64:-}" ]]; then
    info "compiling claude-tcc-probe from inlined Swift source..."
    swift_src=/tmp/claude-tcc-probe.swift
    swift_bin=/tmp/claude-tcc-probe
    echo "$CRA_SWIFT_CLAUDE_TCC_PROBE_B64" | base64 -d > "$swift_src"
    swiftc -O "$swift_src" -o "$swift_bin" -framework AVFoundation -framework Foundation

    info "installing to /usr/local/bin..."
    if [[ -w /usr/local/bin ]]; then
        mv "$swift_bin" /usr/local/bin/
    elif [[ -n "${SUDO_PASSWORD:-}" ]]; then
        echo "$SUDO_PASSWORD" | sudo -S -p "" mv "$swift_bin" /usr/local/bin/
    else
        sudo mv "$swift_bin" /usr/local/bin/
    fi
    rm -f "$swift_src"
    ok "claude-tcc-probe installed"
elif command -v claude-tcc-probe >/dev/null; then
    ok "claude-tcc-probe (already installed)"
else
    err "claude-tcc-probe not installed and no source provided via CRA_SWIFT_CLAUDE_TCC_PROBE_B64"
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
