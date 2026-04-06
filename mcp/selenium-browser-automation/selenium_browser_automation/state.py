from __future__ import annotations

__all__ = [
    'BrowserState',
    'OriginTracker',
]

import logging
import subprocess
import tempfile
import typing
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from selenium import webdriver

from .models import Browser, ProfileState

logger = logging.getLogger(__name__)


class OriginTracker:
    """Tracks origins visited during browser session for multi-origin storage capture.

    CDP storage APIs require explicit origin specification — they have no enumeration
    API (security by design). This tracker maintains a set of all origins visited
    via navigate() so save_profile_state() knows which origins to query.

    Origin format: scheme://host:port (e.g., "https://example.com", "http://localhost:8080")
    Port is included only if non-default (not 80 for http, not 443 for https).
    """

    def __init__(self) -> None:
        self._origins: set[str] = set()

    def add_origin(self, url: str) -> str:
        """Extract and track origin from URL. Returns the origin.

        Args:
            url: Full URL (e.g., "https://example.com/path?query=1")

        Returns:
            The extracted origin (e.g., "https://example.com")
        """
        parsed = urlparse(url)
        origin = f'{parsed.scheme}://{parsed.netloc}'
        self._origins.add(origin)
        return origin

    def get_origins(self) -> Sequence[str]:
        """Get sorted list of all tracked origins."""
        return sorted(self._origins)

    def clear(self) -> None:
        """Clear all tracked origins. Called on fresh_browser=True."""
        self._origins.clear()

    def __len__(self) -> int:
        return len(self._origins)


class BrowserState:
    """Container for all browser state — initialized once at startup, never Optional."""

    @classmethod
    async def create(cls) -> typing.Self:
        """Factory method to create and initialize browser state."""
        temp_dir = tempfile.TemporaryDirectory()
        screenshot_dir = Path(temp_dir.name)

        capture_temp_dir = tempfile.TemporaryDirectory()
        capture_dir = Path(capture_temp_dir.name)

        logger.info('Temp directories initialized (screenshots: %s, captures: %s)', screenshot_dir, capture_dir)

        return cls(
            driver=None,
            temp_dir=temp_dir,
            screenshot_dir=screenshot_dir,
            capture_temp_dir=capture_temp_dir,
            capture_dir=capture_dir,
            capture_counter=0,
        )

    def __init__(
        self,
        driver: webdriver.Chrome | None,  # None = lazy initialization (created on first use)
        temp_dir: tempfile.TemporaryDirectory[str],
        screenshot_dir: Path,
        capture_temp_dir: tempfile.TemporaryDirectory[str],
        capture_dir: Path,
        capture_counter: int,
    ) -> None:
        # fmt: off
        self.driver = driver
        self.current_browser: Browser | None = None
        self.temp_dir = temp_dir
        self.screenshot_dir = screenshot_dir
        self.capture_temp_dir = capture_temp_dir
        self.capture_dir = capture_dir
        self.capture_counter = capture_counter
        self.proxy_config: dict[str, str] | None = None
        self.mitmproxy_process: subprocess.Popen[bytes] | None = None
        self.origin_tracker = OriginTracker()
        self.local_storage_cache: dict[str, list[dict[str, str]]] = {}
        self.session_storage_cache: dict[str, list[dict[str, str]]] = {}
        self.indexed_db_cache: dict[str, list[dict[str, Any]]] = {}
        self.pending_profile_state: ProfileState | None = None
        self.restored_origins: set[str] = set()
        self.response_body_capture_enabled: bool = False
        # fmt: on
