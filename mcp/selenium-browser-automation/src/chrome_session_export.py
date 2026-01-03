"""
Chrome Session State Export

Exports session state (cookies, localStorage, sessionStorage, IndexedDB) from
Chrome profile files for use in Selenium automation.

This enables the workflow:
1. User logs into websites in normal Chrome (handles CAPTCHA, MFA manually)
2. Export session state from Chrome profile files
3. Import into Selenium via navigate(storage_state_file=...)
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
        return int(row[0]) if row else 0
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
    if chrome_version >= 24 and len(unpadded) > 32:
        unpadded = unpadded[32:]

    return unpadded.decode("utf-8", errors="replace")


def export_cookies(
    profile_path: Path,
    origins_filter: Sequence[str] | None = None,
) -> tuple[Sequence[CookieData], Sequence[str]]:
    """Export cookies from any Chrome profile with full decryption.

    Unlike browser-cookie3, this supports ANY Chrome profile, not just Default.
    Uses direct Keychain access + AES decryption.

    Args:
        profile_path: Path to Chrome profile directory
        origins_filter: Optional domain patterns to include

    Returns:
        Tuple of (cookies, warnings)
    """
    warnings: list[str] = []
    cookies: list[CookieData] = []

    cookies_db = profile_path / "Cookies"
    if not cookies_db.exists():
        warnings.append(f"Cookies database not found: {cookies_db}")
        return cookies, warnings

    # Get encryption key from Keychain (shared across all profiles)
    try:
        keychain_password = _get_chrome_encryption_key()
        aes_key = _derive_aes_key(keychain_password)
    except RuntimeError as e:
        warnings.append(str(e))
        return cookies, warnings

    # Detect Chrome version for hash handling
    chrome_version = _get_chrome_cookies_version(cookies_db)

    # Read and decrypt all cookies from this profile's SQLite
    try:
        conn = sqlite3.connect(f"file:{cookies_db}?mode=ro", uri=True)
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

            # Apply origin filter
            if origins_filter:
                if not any(pattern in host for pattern in origins_filter):
                    continue

            # Decrypt if needed
            if encrypted_value and len(encrypted_value) > 0:
                try:
                    cookie_value = _decrypt_cookie_value(
                        encrypted_value, aes_key, chrome_version
                    )
                except ValueError as e:
                    warnings.append(f"Failed to decrypt cookie {name}@{host}: {e}")
                    continue
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

        conn.close()

    except sqlite3.Error as e:
        warnings.append(f"SQLite error reading cookies: {e}")

    return cookies, warnings


# =============================================================================
# localStorage Export
# =============================================================================


def export_local_storage(
    profile_path: Path,
    origins_filter: Sequence[str] | None = None,
) -> tuple[Mapping[str, Sequence[StorageItem]], Sequence[str]]:
    """Export localStorage from Chrome's LevelDB.

    Args:
        profile_path: Path to Chrome profile directory
        origins_filter: Optional origin patterns to include

    Returns:
        Tuple of ({origin: [StorageItem, ...]}, warnings)
    """
    warnings: list[str] = []
    storage: dict[str, list[StorageItem]] = {}

    # CRITICAL: localStorage is in "Local Storage/leveldb/", not "Local Storage/"
    leveldb_path = profile_path / "Local Storage" / "leveldb"

    if not leveldb_path.exists():
        warnings.append(f"localStorage path not found: {leveldb_path}")
        return storage, warnings

    try:
        with ccl_chromium_localstorage.LocalStoreDb(leveldb_path) as ls:
            for storage_key in ls.iter_storage_keys():
                # Filter origins
                if origins_filter:
                    if not any(pattern in storage_key for pattern in origins_filter):
                        continue

                origin_storage: list[StorageItem] = []
                for record in ls.iter_records_for_storage_key(storage_key):
                    try:
                        value = str(record.value) if record.value else ""
                        origin_storage.append(
                            StorageItem(name=record.script_key, value=value)
                        )
                    except Exception as e:
                        warnings.append(f"localStorage decode error for {storage_key}: {e}")

                if origin_storage:
                    storage[storage_key] = origin_storage

    except Exception as e:
        warnings.append(f"localStorage export failed: {e}")

    return storage, warnings


# =============================================================================
# sessionStorage Export
# =============================================================================


def export_session_storage(
    profile_path: Path,
    origins_filter: Sequence[str] | None = None,
) -> tuple[Mapping[str, Sequence[StorageItem]], Sequence[str]]:
    """Export sessionStorage from Chrome's LevelDB.

    Chrome persists sessionStorage to disk for session recovery, enabling
    export from profile files. This contradicts browser specs but enables
    useful automation workflows.

    Args:
        profile_path: Path to Chrome profile directory
        origins_filter: Optional origin patterns to include

    Returns:
        Tuple of ({origin: [StorageItem, ...]}, warnings)
    """
    warnings: list[str] = []
    storage: dict[str, list[StorageItem]] = {}

    session_storage_path = profile_path / "Session Storage"

    if not session_storage_path.exists():
        warnings.append(f"sessionStorage path not found: {session_storage_path}")
        return storage, warnings

    try:
        with ccl_chromium_sessionstorage.SessionStoreDb(session_storage_path) as ss:
            for host in ss.iter_hosts():
                # Filter origins
                if origins_filter:
                    if not any(pattern in host for pattern in origins_filter):
                        continue

                origin_storage: list[StorageItem] = []
                for record in ss.iter_records_for_host(host):
                    try:
                        value = str(record.value) if record.value else ""
                        origin_storage.append(StorageItem(name=record.key, value=value))
                    except Exception as e:
                        warnings.append(f"sessionStorage decode error for {host}: {e}")

                if origin_storage:
                    storage[host] = origin_storage

    except Exception as e:
        warnings.append(f"sessionStorage export failed: {e}")

    return storage, warnings


# =============================================================================
# IndexedDB Export (Best-Effort, Records Only)
# =============================================================================


def export_indexeddb(
    profile_path: Path,
    origins_filter: Sequence[str] | None = None,
) -> tuple[Mapping[str, Any], Sequence[str]]:
    """Export IndexedDB records from Chrome's LevelDB.

    LIMITATION: ccl_chromium_reader cannot extract schema metadata (version,
    keyPath, autoIncrement, indexes). Only database/store names and records
    are available. Full restoration requires save_storage_state() from a
    Selenium session.

    Args:
        profile_path: Path to Chrome profile directory
        origins_filter: Optional origin patterns to include

    Returns:
        Tuple of (databases_by_origin, warnings)
        Format: {origin: {db_name: {"objectStores": {store_name: [records]}}}}
    """
    warnings: list[str] = []
    databases: dict[str, Any] = {}

    indexeddb_path = profile_path / "IndexedDB"

    if not indexeddb_path.exists():
        warnings.append(f"IndexedDB path not found: {indexeddb_path}")
        return databases, warnings

    for item in indexeddb_path.iterdir():
        if not (item.is_dir() and item.name.endswith(".indexeddb.leveldb")):
            continue

        # Origin is encoded in directory name (e.g., "https_example.com_0")
        origin = item.name.replace(".indexeddb.leveldb", "")

        if origins_filter:
            if not any(pattern in origin for pattern in origins_filter):
                continue

        try:
            wrapper = ccl_chromium_indexeddb.WrappedIndexDB(item)
            origin_dbs: dict[str, Any] = {}

            for db_id in wrapper.database_ids:
                db = wrapper[db_id]
                db_data: dict[str, Any] = {"objectStores": {}}

                for obj_store_name in db.object_store_names:
                    store = db[obj_store_name]
                    records: list[dict[str, Any]] = []

                    try:
                        for record in store.iterate_records():
                            try:
                                key = str(record.key)
                                value = record.value
                                # Ensure JSON-serializable
                                if not isinstance(value, (dict, list, str, int, float, bool, type(None))):
                                    value = str(value)
                                records.append({"key": key, "value": value})
                            except Exception:
                                records.append({"key": str(record.key), "value": "<binary>"})
                    except Exception as e:
                        warnings.append(f"IndexedDB store '{obj_store_name}' read error: {e}")

                    if records:
                        db_data["objectStores"][obj_store_name] = records

                if db_data["objectStores"]:
                    origin_dbs[db.name] = db_data

            if origin_dbs:
                databases[origin] = origin_dbs

        except Exception as e:
            warnings.append(f"IndexedDB origin '{origin}' export failed: {e}")

    return databases, warnings


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

    Produces JSON compatible with navigate(storage_state_file=...) for
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
        navigate("https://github.com", storage_state_file="auth.json")
    """
    warnings: list[str] = []

    # Fail if output file exists - delete first to replace
    if Path(output_file).expanduser().exists():
        raise FileExistsError(f"Output file already exists: {output_file}")

    # Resolve profile path
    profile_path = get_chrome_profile_path(profile_name)

    # Convert to list for filtering if provided
    filter_list = list(origins_filter) if origins_filter else None

    # Export cookies
    cookies, cookie_warnings = export_cookies(profile_path, filter_list)
    warnings.extend(cookie_warnings)

    # Export localStorage
    local_storage, ls_warnings = export_local_storage(profile_path, filter_list)
    warnings.extend(ls_warnings)

    # Export sessionStorage (optional, default on)
    session_storage: Mapping[str, Sequence[StorageItem]] = {}
    if include_session_storage:
        session_storage, ss_warnings = export_session_storage(profile_path, filter_list)
        warnings.extend(ss_warnings)

    # Export IndexedDB (optional, default off - can be huge)
    indexeddb: Mapping[str, Any] = {}
    if include_indexeddb:
        indexeddb, idb_warnings = export_indexeddb(profile_path, filter_list)
        warnings.extend(idb_warnings)
        if indexeddb:
            warnings.append(
                "IndexedDB export contains records only (no schema). "
                "Full restoration with keyPath/indexes requires save_storage_state() "
                "from a Selenium session."
            )

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

    # Write to file
    output_path = Path(output_file).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    # SECURITY: Restrict file permissions (owner read/write only)
    # Output contains sensitive auth tokens
    os.chmod(output_path, 0o600)

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
        warnings=list(warnings),
    )
