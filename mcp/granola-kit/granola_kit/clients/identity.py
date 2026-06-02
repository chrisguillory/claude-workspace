from __future__ import annotations

__all__ = [
    'GranolaIdentity',
]

import hashlib
import plistlib
import subprocess
from collections.abc import Mapping
from pathlib import Path

from cc_lib.schemas.base import ClosedModel
from cc_lib.types import PydanticVersion


class GranolaIdentity(ClosedModel):
    """The desktop-client identity granola-kit presents to api.granola.ai.

    NON-gating: a valid token alone returns 200 (verified empirically). The headers
    exist for parity with desktop traffic, and because ``X-Client-Version`` feeds
    server-side per-flag ``min_client_version`` checks. ``detect()`` resolves the
    identity from this machine.
    """

    app_version: PydanticVersion
    device_id: str
    os_version: str

    @classmethod
    def detect(cls) -> GranolaIdentity:
        """Resolve the identity from the local system (Info.plist, ioreg, sw_vers)."""
        return cls(
            app_version=_detect_app_version(),
            device_id=_detect_device_id(),
            os_version=_detect_os_version(),
        )

    def headers(self) -> Mapping[str, str]:
        """Identity headers, sans Authorization."""
        return {
            'X-Client-Version': str(self.app_version),
            'X-Granola-Platform': 'macOS',
            'X-Granola-Os-Version': self.os_version,
            'X-Granola-Device-Id': self.device_id,
        }


def _detect_app_version() -> PydanticVersion:
    """CFBundleShortVersionString from the installed Granola.app."""
    plist = Path('/Applications/Granola.app/Contents/Info.plist')
    with plist.open('rb') as f:
        version = plistlib.load(f)['CFBundleShortVersionString']
    if not isinstance(version, str):
        raise TypeError(f'CFBundleShortVersionString is {type(version).__name__}, not str')
    return PydanticVersion(version)


def _detect_device_id() -> str:
    """SHA-256 of the machine's IOPlatformUUID, read from ioreg."""
    out = subprocess.run(
        ['ioreg', '-d2', '-c', 'IOPlatformExpertDevice'],
        capture_output=True,
        text=True,
        check=True,
        timeout=5,
    ).stdout
    for line in out.splitlines():
        if 'IOPlatformUUID' in line:
            parts = line.split('"')
            for i, part in enumerate(parts):
                if part == 'IOPlatformUUID' and i + 2 < len(parts):
                    return hashlib.sha256(parts[i + 2].encode('utf-8')).hexdigest()
    raise ValueError('IOPlatformUUID not found in ioreg output')


def _detect_os_version() -> str:
    """Product version of macOS (e.g. '15.0'), from sw_vers."""
    return subprocess.run(
        ['sw_vers', '-productVersion'],
        capture_output=True,
        text=True,
        check=True,
        timeout=5,
    ).stdout.strip()
