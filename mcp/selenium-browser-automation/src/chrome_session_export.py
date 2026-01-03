"""
Chrome Session State Export

Exports session state (cookies, localStorage, sessionStorage, IndexedDB) from
Chrome profile files for use in Selenium automation.

This enables the workflow:
1. User logs into websites in normal Chrome (handles CAPTCHA, MFA manually)
2. Export session state from Chrome profile files
3. Import into Selenium via navigate_with_session(storage_state_file=...)
4. Continue automation with authenticated session

Storage Types Captured (matching save_storage_state):
- Cookies: Full attributes including sameSite via DIY Keychain decryption
- localStorage: All origins from LevelDB
- sessionStorage: All origins from LevelDB (Chrome persists to disk)
- IndexedDB: Records only - schema (version, keyPath, indexes) not available

Dependencies:
- pycryptodome: AES decryption for Chrome cookies
- ccl_chromium_reader: Pure Python LevelDB parser (no C compilation)

Limitations:
- macOS only (Windows/Linux require different key retrieval)
- First run prompts for Keychain access - click "Always Allow"
- IndexedDB exports records without schema; full restoration requires save_storage_state

Cookie Decryption (DIY implementation):
- Keychain: "Chrome Safe Storage" password (SHARED across all profiles)
- Key derivation: PBKDF2-HMAC-SHA1, salt='saltysalt', iterations=1003, key_len=16
- Encryption: AES-128-CBC, IV=16 spaces, v10 prefix stripped
- Chrome 130+ (DB version >= 24): SHA256(domain) prepended to value, stripped after decryption
"""

from __future__ import annotations

import re
import hashlib
import json
import os
import sqlite3
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from Crypto.Cipher import AES
from Crypto.Util.Padding import unpad
from ccl_chromium_reader import (
    ccl_chromium_indexeddb,
    ccl_chromium_localstorage,
    ccl_chromium_sessionstorage,
)


# =============================================================================
# Type Definitions
# =============================================================================

# Typed dictionaries for export data structures
# These match the existing Pydantic models in models.py for compatibility


@dataclass(frozen=True)
class CookieData:
    """Cookie with full Playwright-compatible attributes."""

    name: str
    value: str
    domain: str
    path: str
    expires: float  # Unix timestamp, -1 for session cookies
    httpOnly: bool
    secure: bool
    sameSite: str  # "Strict", "Lax", or "None"

    def to_dict(self) -> Mapping[str, Any]:
        """Convert to dict for JSON serialization."""
        return {
            "name": self.name,
            "value": self.value,
            "domain": self.domain,
            "path": self.path,
            "expires": self.expires,
            "httpOnly": self.httpOnly,
            "secure": self.secure,
            "sameSite": self.sameSite,
        }


@dataclass(frozen=True)
class StorageItem:
    """localStorage or sessionStorage key-value pair."""

    name: str
    value: str

    def to_dict(self) -> Mapping[str, str]:
        """Convert to dict for JSON serialization."""
        return {"name": self.name, "value": self.value}


@dataclass
class ExportResult:
    """Result of Chrome session export."""

    path: str
    cookie_count: int
    origin_count: int
    local_storage_keys: int
    session_storage_keys: int
    indexeddb_origins: int
    warnings: Sequence[str] = field(default_factory=list)


# =============================================================================
# Chrome Profile Path Resolution
# =============================================================================


def get_chrome_base_path() -> Path:
    """Get Chrome base directory for macOS."""
    return Path.home() / "Library/Application Support/Google/Chrome"


def get_chrome_profile_path(profile_name: str = "Default") -> Path:
    """Get Chrome profile path for macOS.

    Args:
        profile_name: Profile directory name ("Default", "Profile 1", etc.)

    Returns:
        Path to the profile directory

    Raises:
        FileNotFoundError: If profile doesn't exist, lists available profiles
    """
    base = get_chrome_base_path()
    profile_path = base / profile_name

    if not profile_path.exists():
        available = sorted(
            p.name
            for p in base.iterdir()
            if p.is_dir()
            and not p.name.startswith(".")
            and (p / "Cookies").exists()  # Only real profiles have Cookies
        )
        raise FileNotFoundError(
            f"Chrome profile not found: {profile_path}\n"
            f"Available profiles: {available}"
        )

    return profile_path


# =============================================================================
# Cookie Export
# =============================================================================

# Chrome SQLite sameSite values → Playwright format
_SAMESITE_MAP: Mapping[int, str] = {
    -1: "Lax",  # Unspecified → default to Lax
    0: "None",
    1: "Lax",
    2: "Strict",
}

# Regex for IPv4 addresses (validates 0-255 octets)
_IPV4_PATTERN = re.compile(
    r"^(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)$"
)


def _validate_domain_pattern(pattern: str) -> None:
    """Validate that a domain pattern is well-formed.

    Args:
        pattern: User-provided domain filter (e.g., "amazon.com")

    Raises:
        ValueError: If pattern is empty or missing TLD

    Examples:
        _validate_domain_pattern("amazon.com")  # OK
        _validate_domain_pattern(".amazon.com")  # OK (leading dot stripped)
        _validate_domain_pattern("localhost")  # OK (special case)
        _validate_domain_pattern("192.168.1.1")  # OK (IP address)
        _validate_domain_pattern("amazon")  # Raises ValueError
    """
    # Strip leading/trailing dots and port numbers
    cleaned = pattern.lower().strip(".")
    if ":" in cleaned:
        cleaned = cleaned.split(":")[0]  # Strip port if present

    if not cleaned:
        raise ValueError(f"Empty domain pattern: {pattern!r}")

    # Allow localhost as special case
    if cleaned == "localhost":
        return

    # Allow IP addresses (IPv4)
    if _IPV4_PATTERN.match(cleaned):
        return

    # For regular domains, require at least one dot (TLD)
    if "." not in cleaned:
        raise ValueError(
            f"Invalid domain pattern: {pattern!r} - "
            f"must include TLD (e.g., 'amazon.com' not 'amazon')"
        )


def _extract_domain_from_origin(origin: str) -> str:
    """Extract domain from various origin formats.

    Handles:
    - Full URLs: https://www.amazon.com → www.amazon.com
    - URLs with port: https://localhost:3000 → localhost
    - URLs with path: https://www.amazon.com/ → www.amazon.com
    - IndexedDB format: https_www.amazon.com_0 → www.amazon.com
    - Bare domains: amazon.com → amazon.com

    Args:
        origin: Origin string in any format

    Returns:
        Bare domain without scheme, port, or path
    """
    # Handle full URLs (https://example.com or https://example.com/)
    if "://" in origin:
        # Remove scheme
        origin = origin.split("://", 1)[1]
        # Remove path
        origin = origin.split("/", 1)[0]
        # Remove port
        origin = origin.split(":", 1)[0]
        return origin

    # Handle IndexedDB format (https_www.amazon.com_0)
    if origin.startswith(("https_", "http_")):
        # Remove scheme prefix
        origin = origin.split("_", 1)[1]
        # Remove trailing _0, _1, etc.
        if "_" in origin:
            parts = origin.rsplit("_", 1)
            if parts[-1].isdigit():
                origin = parts[0]
        return origin

    # Already a bare domain
    return origin


def _domain_matches(host: str, pattern: str) -> bool:
    """RFC 6265 domain matching - suffix match with dot boundary.

    This is the correct way to match cookie domains, NOT substring matching.
    Substring matching would be insecure (e.g., "amazon" matching "amazon.evil.com").

    Args:
        host: The cookie's host_key from SQLite (e.g., ".amazon.com")
        pattern: The filter pattern (e.g., "amazon.com")

    Returns:
        True if host matches pattern per RFC 6265 rules

    Examples:
        _domain_matches(".amazon.com", "amazon.com") → True
        _domain_matches("www.amazon.com", "amazon.com") → True
        _domain_matches("amazon.com", "amazon.com") → True
        _domain_matches("amazon.evil.com", "amazon.com") → False
        _domain_matches("notamazon.com", "amazon.com") → False
    """
    # Canonicalize: lowercase, strip leading/trailing dots, strip port
    host = host.lower().strip(".")
    pattern = pattern.lower().strip(".")
    if ":" in pattern:
        pattern = pattern.split(":")[0]

    # Exact match (handles IPs and localhost)
    if host == pattern:
        return True

    # Suffix match with dot boundary: host ends with ".pattern"
    if host.endswith("." + pattern):
        return True

    return False


# =============================================================================
# DIY Cookie Decryption (replaces browser-cookie3)
# =============================================================================
#
# Why DIY instead of browser-cookie3?
# browser-cookie3.chrome() hardcodes the Default profile path. When you call
# it with profile_name="Profile 1", it silently ignores the parameter and
# reads from Default. We need to decrypt cookies from ANY profile.
#
# The encryption key in macOS Keychain is SHARED across all profiles.
# We just need to read from the correct profile's SQLite.
#
# Verified against browser-cookie3 (MIT), pycookiecheat (MIT), and
# HackBrowserData (MIT) source code.
# =============================================================================


def _get_chrome_encryption_key() -> bytes:
    """Retrieve Chrome's encryption key from macOS Keychain.

    The key is stored under service "Chrome Safe Storage" and is
    SHARED across all Chrome profiles on the machine.

    Returns:
        Raw password bytes for PBKDF2 derivation

    Raises:
        RuntimeError: If Keychain access fails (user denied, not found)
    """
    result = subprocess.run(
        [
            "/usr/bin/security",
            "-q",
            "find-generic-password",
            "-w",
            "-a",
            "Chrome",
            "-s",
            "Chrome Safe Storage",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to retrieve Chrome Safe Storage from Keychain: {result.stderr}\n"
            "First run may require clicking 'Always Allow' when prompted."
        )
    return result.stdout.strip().encode("utf-8")


def _derive_aes_key(keychain_password: bytes) -> bytes:
    """Derive AES-128 key using Chrome's PBKDF2 parameters.

    Chrome uses fixed, well-known parameters:
    - Salt: 'saltysalt' (literal)
    - Iterations: 1003
    - Hash: SHA1
    - Key length: 16 bytes (AES-128)

    Args:
        keychain_password: Raw password from macOS Keychain

    Returns:
        16-byte AES key for cookie decryption
    """
    return hashlib.pbkdf2_hmac(
        "sha1",
        keychain_password,
        b"saltysalt",
        1003,
        dklen=16,
    )


def _get_chrome_cookies_version(cookies_db: Path) -> int:
    """Get Chrome cookie format version from database.

    Used to detect Chrome 130+ which adds domain hash to encrypted values.
    Version 24+ means SHA256(domain) is prepended to decrypted values.

    Args:
        cookies_db: Path to Cookies SQLite file

    Returns:
        Version number (24+ means domain hash present)
    """
    conn = sqlite3.connect(f"file:{cookies_db}?mode=ro", uri=True)
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT value FROM meta WHERE key = 'version'")
        row = cursor.fetchone()
        # Handle both missing row and NULL value
        if row and row[0] is not None:
            return int(row[0])
        return 0
    except sqlite3.OperationalError:
        return 0  # Assume old format if meta table missing
    finally:
        conn.close()


def _decrypt_cookie_value(
    encrypted_value: bytes,
    aes_key: bytes,
    chrome_version: int,
) -> str:
    """Decrypt a Chrome cookie value using AES-128-CBC.

    Args:
        encrypted_value: Raw bytes from SQLite encrypted_value column
        aes_key: 16-byte AES key from _derive_aes_key()
        chrome_version: From Cookies DB meta table (version key)

    Returns:
        Decrypted cookie value as string

    Raises:
        ValueError: If decryption fails (wrong key, corrupted data)
    """
    # Check for v10 prefix (macOS/Linux encrypted format)
    if not encrypted_value.startswith(b"v10"):
        # Unencrypted value (rare but possible)
        return encrypted_value.decode("utf-8", errors="replace")

    # Strip v10 prefix (3 bytes)
    ciphertext = encrypted_value[3:]

    # Decrypt with AES-128-CBC, IV = 16 spaces
    iv = b" " * 16
    cipher = AES.new(aes_key, AES.MODE_CBC, iv)
    decrypted = cipher.decrypt(ciphertext)

    # Remove PKCS7 padding
    try:
        unpadded = unpad(decrypted, AES.block_size)
    except ValueError as e:
        raise ValueError(f"PKCS7 unpadding failed: {e}") from e

    # Chrome 130+ (version >= 24): SHA256(domain) prepended to decrypted value
    # The hash is INSIDE the encrypted blob, so we strip it after decryption
    # Use >= 32 to handle empty cookie values (exactly 32 bytes = just the hash)
    if chrome_version >= 24 and len(unpadded) >= 32:
        unpadded = unpadded[32:]

    return unpadded.decode("utf-8", errors="replace")


def export_cookies(
    profile_path: Path,
    origins_filter: Sequence[str] | None = None,
) -> Sequence[CookieData]:
    """Export cookies from any Chrome profile with full decryption.

    Unlike browser-cookie3, this supports ANY Chrome profile, not just Default.
    Uses direct Keychain access + AES decryption.

    Args:
        profile_path: Path to Chrome profile directory
        origins_filter: Optional domain patterns to include

    Returns:
        Sequence of CookieData

    Raises:
        FileNotFoundError: If Cookies database doesn't exist
        RuntimeError: If Keychain access fails
        sqlite3.Error: If database read fails
        ValueError: If cookie decryption fails
    """
    cookies: list[CookieData] = []

    cookies_db = profile_path / "Cookies"
    if not cookies_db.exists():
        raise FileNotFoundError(f"Cookies database not found: {cookies_db}")

    # Get encryption key from Keychain (shared across all profiles)
    keychain_password = _get_chrome_encryption_key()
    aes_key = _derive_aes_key(keychain_password)

    # Detect Chrome version for hash handling
    chrome_version = _get_chrome_cookies_version(cookies_db)

    # Read and decrypt all cookies from this profile's SQLite
    conn = sqlite3.connect(f"file:{cookies_db}?mode=ro", uri=True)
    try:
        cursor = conn.cursor()

        cursor.execute("""
            SELECT host_key, name, path, expires_utc, is_secure,
                   is_httponly, samesite, value, encrypted_value
            FROM cookies
        """)

        for row in cursor.fetchall():
            (
                host,
                name,
                path,
                expires_utc,
                secure,
                httponly,
                samesite,
                value,
                encrypted_value,
            ) = row

            # Apply origin filter (RFC 6265 domain matching)
            if origins_filter:
                if not any(_domain_matches(host, pattern) for pattern in origins_filter):
                    continue

            # Decrypt if needed
            if encrypted_value:
                cookie_value = _decrypt_cookie_value(
                    encrypted_value, aes_key, chrome_version
                )
            else:
                cookie_value = value or ""

            # Convert Chrome epoch to Unix timestamp
            # Chrome epoch: microseconds since 1601-01-01
            # Unix epoch: seconds since 1970-01-01
            # Offset: 11644473600 seconds
            if expires_utc and expires_utc > 0:
                expires = float(expires_utc // 1_000_000 - 11644473600)
            else:
                expires = -1.0  # Session cookie

            cookies.append(
                CookieData(
                    name=name,
                    value=cookie_value,
                    domain=host,
                    path=path,
                    expires=expires,
                    httpOnly=bool(httponly),
                    secure=bool(secure),
                    sameSite=_SAMESITE_MAP.get(samesite, "Lax"),
                )
            )
    finally:
        conn.close()

    return cookies


# =============================================================================
# localStorage Export
# =============================================================================


def export_local_storage(
    profile_path: Path,
    origins_filter: Sequence[str] | None = None,
) -> Mapping[str, Sequence[StorageItem]]:
    """Export localStorage from Chrome's LevelDB.

    Args:
        profile_path: Path to Chrome profile directory
        origins_filter: Optional origin patterns to include

    Returns:
        Mapping of {origin: [StorageItem, ...]}

    Raises:
        FileNotFoundError: If localStorage path doesn't exist
    """
    storage: dict[str, list[StorageItem]] = {}

    # CRITICAL: localStorage is in "Local Storage/leveldb/", not "Local Storage/"
    leveldb_path = profile_path / "Local Storage" / "leveldb"

    if not leveldb_path.exists():
        raise FileNotFoundError(f"localStorage path not found: {leveldb_path}")

    with ccl_chromium_localstorage.LocalStoreDb(leveldb_path) as ls:
        for storage_key in ls.iter_storage_keys():
            # Filter origins (extract domain from URL, then RFC 6265 matching)
            if origins_filter:
                domain = _extract_domain_from_origin(storage_key)
                if not any(_domain_matches(domain, pattern) for pattern in origins_filter):
                    continue

            origin_storage: list[StorageItem] = []
            for record in ls.iter_records_for_storage_key(storage_key):
                value = str(record.value) if record.value else ""
                origin_storage.append(
                    StorageItem(name=record.script_key, value=value)
                )

            if origin_storage:
                storage[storage_key] = origin_storage

    return storage


# =============================================================================
# sessionStorage Export
# =============================================================================


def export_session_storage(
    profile_path: Path,
    origins_filter: Sequence[str] | None = None,
) -> Mapping[str, Sequence[StorageItem]]:
    """Export sessionStorage from Chrome's LevelDB.

    Chrome persists sessionStorage to disk for session recovery, enabling
    export from profile files. This contradicts browser specs but enables
    useful automation workflows.

    Args:
        profile_path: Path to Chrome profile directory
        origins_filter: Optional origin patterns to include

    Returns:
        Mapping of {origin: [StorageItem, ...]}

    Raises:
        FileNotFoundError: If sessionStorage path doesn't exist
    """
    storage: dict[str, list[StorageItem]] = {}

    session_storage_path = profile_path / "Session Storage"

    if not session_storage_path.exists():
        raise FileNotFoundError(f"sessionStorage path not found: {session_storage_path}")

    with ccl_chromium_sessionstorage.SessionStoreDb(session_storage_path) as ss:
        for host in ss.iter_hosts():
            # Filter origins (extract domain from URL, then RFC 6265 matching)
            if origins_filter:
                domain = _extract_domain_from_origin(host)
                if not any(_domain_matches(domain, pattern) for pattern in origins_filter):
                    continue

            origin_storage: list[StorageItem] = []
            for record in ss.iter_records_for_host(host):
                value = str(record.value) if record.value else ""
                origin_storage.append(StorageItem(name=record.key, value=value))

            if origin_storage:
                storage[host] = origin_storage

    return storage


# =============================================================================
# IndexedDB Export (Best-Effort, Records Only)
# =============================================================================


def export_indexeddb(
    profile_path: Path,
    origins_filter: Sequence[str] | None = None,
) -> Mapping[str, Any]:
    """Export IndexedDB records from Chrome's LevelDB.

    LIMITATION: ccl_chromium_reader cannot extract schema metadata (version,
    keyPath, autoIncrement, indexes). Only database/store names and records
    are available. Full restoration requires save_storage_state() from a
    Selenium session.

    Args:
        profile_path: Path to Chrome profile directory
        origins_filter: Optional origin patterns to include

    Returns:
        Mapping of {origin: {db_name: {"objectStores": {store_name: [records]}}}}

    Raises:
        FileNotFoundError: If IndexedDB path doesn't exist
    """
    databases: dict[str, Any] = {}

    indexeddb_path = profile_path / "IndexedDB"

    if not indexeddb_path.exists():
        raise FileNotFoundError(f"IndexedDB path not found: {indexeddb_path}")

    for item in indexeddb_path.iterdir():
        if not (item.is_dir() and item.name.endswith(".indexeddb.leveldb")):
            continue

        # Origin is encoded in directory name (e.g., "https_example.com_0")
        origin = item.name.replace(".indexeddb.leveldb", "")

        # Filter origins (extract domain from IndexedDB format, then RFC 6265 matching)
        if origins_filter:
            domain = _extract_domain_from_origin(origin)
            if not any(_domain_matches(domain, pattern) for pattern in origins_filter):
                continue

        wrapper = ccl_chromium_indexeddb.WrappedIndexDB(item)
        origin_dbs: dict[str, Any] = {}

        for db_id in wrapper.database_ids:
            db = wrapper[db_id]
            db_data: dict[str, Any] = {"objectStores": {}}

            for obj_store_name in db.object_store_names:
                store = db[obj_store_name]
                records: list[dict[str, Any]] = []

                for record in store.iterate_records():
                    key = str(record.key)
                    value = record.value
                    # Ensure JSON-serializable
                    if not isinstance(
                        value, (dict, list, str, int, float, bool, type(None))
                    ):
                        value = str(value)
                    records.append({"key": key, "value": value})

                if records:
                    db_data["objectStores"][obj_store_name] = records

            if db_data["objectStores"]:
                origin_dbs[db.name] = db_data

        if origin_dbs:
            databases[origin] = origin_dbs

    return databases


# =============================================================================
# Main Export Function
# =============================================================================


def export_chrome_session(
    output_file: str,
    profile_name: str = "Default",
    include_session_storage: bool = True,
    include_indexeddb: bool = False,
    origins_filter: Sequence[str] | None = None,
) -> ExportResult:
    """Export Chrome session state to Playwright-compatible JSON.

    Produces JSON compatible with navigate_with_session(storage_state_file=...) for
    restoring authenticated sessions in Selenium automation.

    Args:
        output_file: Path to save JSON output
        profile_name: Chrome profile name ("Default", "Profile 1", etc.)
        include_session_storage: Include sessionStorage (default True)
        include_indexeddb: Include IndexedDB records (default False)
        origins_filter: Only export origins matching these patterns

    Returns:
        ExportResult with statistics and warnings

    Example:
        # After logging into sites in Chrome...
        result = export_chrome_session("auth.json", origins_filter=["github.com"])

        # Then in Selenium:
        navigate_with_session("https://github.com", storage_state_file="auth.json")
    """
    # Fail if output file exists - delete first to replace
    if Path(output_file).expanduser().exists():
        raise FileExistsError(f"Output file already exists: {output_file}")

    # Resolve profile path
    profile_path = get_chrome_profile_path(profile_name)

    # Validate filter patterns
    if origins_filter:
        for pattern in origins_filter:
            _validate_domain_pattern(pattern)

    # Convert to list for filtering if provided
    filter_list = list(origins_filter) if origins_filter else None

    # Export cookies
    cookies = export_cookies(profile_path, filter_list)

    # Export localStorage
    local_storage = export_local_storage(profile_path, filter_list)

    # Export sessionStorage (optional, default on)
    session_storage: Mapping[str, Sequence[StorageItem]] = {}
    if include_session_storage:
        session_storage = export_session_storage(profile_path, filter_list)

    # Export IndexedDB (optional, default off - can be huge)
    indexeddb: Mapping[str, Any] = {}
    if include_indexeddb:
        indexeddb = export_indexeddb(profile_path, filter_list)

    # Build origins array (Playwright-compatible StorageStateOrigin format)
    all_origins = set(local_storage.keys()) | set(session_storage.keys())
    origins_list: list[dict[str, Any]] = []

    for origin in sorted(all_origins):
        origin_data: dict[str, Any] = {"origin": origin}

        # localStorage is REQUIRED by StorageStateOrigin model - always include it
        if origin in local_storage:
            origin_data["localStorage"] = [item.to_dict() for item in local_storage[origin]]
        else:
            origin_data["localStorage"] = []

        # sessionStorage is optional - only include if present
        if origin in session_storage:
            origin_data["sessionStorage"] = [item.to_dict() for item in session_storage[origin]]

        origins_list.append(origin_data)

    # Build final output (Playwright storageState format)
    output: dict[str, Any] = {
        "cookies": [cookie.to_dict() for cookie in cookies],
        "origins": origins_list,
    }

    # IndexedDB uses simplified format (records only, no schema)
    if indexeddb:
        output["indexedDB"] = dict(indexeddb)

    # Write to file with secure permissions from the start (avoid TOCTOU race)
    # SECURITY: Output contains sensitive auth tokens - create with 0o600, not umask
    output_path = Path(output_file).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fd = os.open(output_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(output, f, indent=2, default=str)

    # Calculate statistics
    ls_keys = sum(len(items) for items in local_storage.values())
    ss_keys = sum(len(items) for items in session_storage.values())

    return ExportResult(
        path=str(output_path),
        cookie_count=len(cookies),
        origin_count=len(origins_list),
        local_storage_keys=ls_keys,
        session_storage_keys=ss_keys,
        indexeddb_origins=len(indexeddb),
    )
