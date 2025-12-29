"""JavaScript script loader for Selenium Browser Automation.

Scripts are loaded at import time from .js files in this directory.
All scripts are validated to exist at startup (fail-fast pattern).

Usage:
    from src.scripts import TEXT_EXTRACTION_SCRIPT, ARIA_SNAPSHOT_SCRIPT
"""

from __future__ import annotations

from pathlib import Path

__all__ = [
    "TEXT_EXTRACTION_SCRIPT",
    "ARIA_SNAPSHOT_SCRIPT",
    "NETWORK_MONITOR_SETUP_SCRIPT",
    "NETWORK_MONITOR_CHECK_SCRIPT",
    "WEB_VITALS_SCRIPT",
    "RESOURCE_TIMING_SCRIPT",
    "SAFE_SERIALIZE_SCRIPT",
    "INDEXEDDB_CAPTURE_SCRIPT",
    "INDEXEDDB_RESTORE_SCRIPT",
    "build_execute_javascript_async_script",
]

_SCRIPTS_DIR = Path(__file__).parent


def _load_script(filename: str) -> str:
    """Load JavaScript from file. Raises FileNotFoundError if missing."""
    path = _SCRIPTS_DIR / filename
    return path.read_text()


# Load all scripts at import time (fail-fast validation)
TEXT_EXTRACTION_SCRIPT: str = _load_script("text_extraction.js")
ARIA_SNAPSHOT_SCRIPT: str = _load_script("aria_snapshot.js")
NETWORK_MONITOR_SETUP_SCRIPT: str = _load_script("network_monitor_setup.js")
NETWORK_MONITOR_CHECK_SCRIPT: str = _load_script("network_monitor_check.js")
WEB_VITALS_SCRIPT: str = _load_script("web_vitals.js")
RESOURCE_TIMING_SCRIPT: str = _load_script("resource_timing.js")
SAFE_SERIALIZE_SCRIPT: str = _load_script("safe_serialize.js")
INDEXEDDB_CAPTURE_SCRIPT: str = _load_script("indexeddb_capture.js")
INDEXEDDB_RESTORE_SCRIPT: str = _load_script("indexeddb_restore.js")


def build_execute_javascript_async_script(escaped_code: str) -> str:
    """Build the async wrapper script for execute_javascript.

    Args:
        escaped_code: JSON-escaped user code (from json.dumps())

    Returns:
        Complete async script with callback handling for execute_async_script
    """
    return f'''
const callback = arguments[arguments.length - 1];
const userCode = {escaped_code};

{SAFE_SERIALIZE_SCRIPT}(userCode)
    .then(function(result) {{
        callback(result);
    }})
    .catch(function(e) {{
        callback({{
            success: false,
            error: e.message || String(e),
            error_type: 'execution'
        }});
    }});
'''