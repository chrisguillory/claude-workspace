from __future__ import annotations

import asyncio
import subprocess
import sys
import tempfile
import textwrap
import time
import uuid
from pathlib import Path

from claude_remote_bash.diagnose.system import codesign_identifier
from claude_remote_bash.discovery import browse_hosts

__all__ = [
    'request_local_network_grant',
]


# Fallback used only when no peer is discoverable. Any RFC1918 address fires
# the LN prompt — the kernel gates on the connect attempt before routing —
# so reachability of the fallback does not matter. Link-local 169.254/16
# is unlikely to overlap a user's home subnet.
_FALLBACK_HOST = '169.254.255.1'
_FALLBACK_PORT = 22
_POLL_TIMEOUT_SECONDS = 30.0
_POLL_INTERVAL_SECONDS = 0.5


def request_local_network_grant() -> None:
    """Generate a throwaway .app bundle, fire the macOS LN prompt, poll for outcome, clean up."""
    interpreter = Path(sys.executable)
    target_host, target_port = _pick_target()

    print(f'Target interpreter: {interpreter}')
    print(f'Current Identifier: {codesign_identifier(interpreter) or "<not signed>"}')
    print(f'Trigger target:     {target_host}:{target_port}')

    with tempfile.TemporaryDirectory(prefix='CRBGrant-') as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        result_file = tmpdir / 'result.txt'
        bundle = _build_bundle(tmpdir, interpreter, result_file, target_host, target_port)

        _codesign_bundle(bundle)
        _open_bundle(bundle)

        print(f'\nBundle opened: {bundle}')
        print('macOS should show: "Allow Python to find devices on local networks?"')
        print(f'Polling for outcome (timeout: {int(_POLL_TIMEOUT_SECONDS)}s)...\n')

        outcome = _poll_for_result(result_file)
        print(f'Probe outcome: {outcome}')


def _pick_target() -> tuple[str, int]:
    """Discover a real peer to probe; fall back to a link-local address if none responds."""
    peers = asyncio.run(browse_hosts(timeout=3.0))
    for peer in peers:
        if peer.ips:
            return peer.ips[0], peer.port
    return _FALLBACK_HOST, _FALLBACK_PORT


def _build_bundle(parent: Path, interpreter: Path, result_file: Path, target_host: str, target_port: int) -> Path:
    """Lay out CRBGrant-<id>.app/Contents/{Info.plist, MacOS/probe}; return the bundle path."""
    bundle_id = uuid.uuid4().hex[:8]
    bundle = parent / f'CRBGrant-{bundle_id}.app'
    contents = bundle / 'Contents'
    macos = contents / 'MacOS'
    macos.mkdir(parents=True)

    info_plist = textwrap.dedent(f"""\
        <?xml version="1.0" encoding="UTF-8"?>
        <!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
        <plist version="1.0">
        <dict>
          <key>CFBundleExecutable</key><string>probe</string>
          <key>CFBundleIdentifier</key><string>com.chrisguillory.crb-grant.{bundle_id}</string>
          <key>CFBundleName</key><string>CRBGrant</string>
          <key>CFBundlePackageType</key><string>APPL</string>
          <key>CFBundleShortVersionString</key><string>0.0.1</string>
          <key>NSLocalNetworkUsageDescription</key>
          <string>claude-remote-bash grants this Python interpreter access to your local network mesh.</string>
          <key>LSBackgroundOnly</key><true/>
        </dict>
        </plist>
    """)
    (contents / 'Info.plist').write_text(info_plist)

    probe = textwrap.dedent(f"""\
        #!/bin/bash
        # Written by `claude-remote-bash diagnose --request-local-network`.
        {{
          echo "CRBGrant probe $(date -u +%FT%TZ)"
          "{interpreter}" - <<'PY'
        import socket
        s = socket.socket(); s.settimeout(3)
        try:
            s.connect(("{target_host}", {target_port}))
            print("RESULT: CONNECTED")
        except Exception as e:
            print(f"RESULT: {{type(e).__name__}}: {{e}}")
        PY
        }} > "{result_file}" 2>&1
    """)
    probe_path = macos / 'probe'
    probe_path.write_text(probe)
    probe_path.chmod(0o755)

    return bundle


def _codesign_bundle(bundle: Path) -> None:
    """Ad-hoc sign the bundle so codesign computes a stable CDHash for NECP attribution."""
    subprocess.run(  # noqa: S603, S607 — `codesign` is a fixed system utility
        ['/usr/bin/codesign', '--force', '--deep', '--sign', '-', str(bundle)],
        capture_output=True,
        text=True,
        check=True,
    )


def _open_bundle(bundle: Path) -> None:
    """Hand the bundle to LaunchServices, which routes through nehelper to fire the prompt."""
    subprocess.run(  # noqa: S603, S607 — `open` is a fixed system utility
        ['/usr/bin/open', str(bundle)],
        capture_output=True,
        text=True,
        check=True,
    )


def _poll_for_result(result_file: Path) -> str:
    """Wait for the probe to write its outcome; return the contents or a timeout marker."""
    deadline = time.monotonic() + _POLL_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if result_file.exists():
            return result_file.read_text().strip()
        time.sleep(_POLL_INTERVAL_SECONDS)
    return f'<timeout: no result after {int(_POLL_TIMEOUT_SECONDS)}s — user may not have clicked Allow>'
