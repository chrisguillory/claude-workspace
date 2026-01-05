"""
Chrome Profile State Export

Exports browser state (cookies, localStorage, sessionStorage, IndexedDB) from
Chrome profile files for use in Selenium automation.

This enables the workflow:
1. User logs into websites in normal Chrome (handles CAPTCHA, MFA manually)
2. Export profile state from Chrome profile files
3. Import into Selenium via navigate_with_profile_state(profile_state_file=...)
4. Continue automation with authenticated state

Storage Types Captured (matching save_profile_state):
- Cookies: Full attributes including sameSite via DIY Keychain decryption
- localStorage: All origins from LevelDB
- sessionStorage: All origins from LevelDB (Chrome persists to disk)
- IndexedDB: Full schema (version, keyPath, indexes) + records via dfindexeddb

Dependencies:
- pycryptodome: AES decryption for Chrome cookies
- ccl_chromium_reader: Pure Python LevelDB parser for localStorage/sessionStorage
- dfindexeddb: Google's IndexedDB parser with complete schema support

Limitations:
- macOS only (Windows/Linux require different key retrieval)
- First run prompts for Keychain access - click "Always Allow"

Cookie Decryption (DIY implementation):
- Keychain: "Chrome Safe Storage" password (SHARED across all profiles)
- Key derivation: PBKDF2-HMAC-SHA1, salt='saltysalt', iterations=1003, key_len=16
- Encryption: AES-128-CBC, IV=16 spaces, v10 prefix stripped
- Chrome 130+ (DB version >= 24): SHA256(domain) prepended to value, stripped after decryption
"""

from __future__ import annotations

import hashlib
import os
import re
import sqlite3
import subprocess
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from ccl_chromium_reader import (
    ccl_chromium_localstorage,
    ccl_chromium_sessionstorage,
)
from Crypto.Cipher import AES
from Crypto.Util.Padding import unpad

from .indexeddb_dfindexeddb import export_indexeddb_with_schema
from .models import (
    ChromeProfileStateExportResult,
    ProfileState,
    ProfileStateCookie,
    ProfileStateIndexedDB,
    ProfileStateIndexedDBIndex,
    ProfileStateIndexedDBObjectStore,
    ProfileStateIndexedDBRecord,
    ProfileStateOriginStorage,
)

# =============================================================================
# Chrome Profile Path Resolution
# =============================================================================


def get_chrome_base_path() -> Path:
    """Get Chrome base directory for macOS."""
    return Path.home() / 'Library/Application Support/Google/Chrome'


def get_chrome_profile_path(profile_name: str = 'Default') -> Path:
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
            if p.is_dir() and not p.name.startswith('.') and (p / 'Cookies').exists()  # Only real profiles have Cookies
        )
        raise FileNotFoundError(f'Chrome profile not found: {profile_path}\nAvailable profiles: {available}')

    return profile_path


# =============================================================================
# Cookie Export
# =============================================================================

# Type alias for sameSite values
type SameSiteValue = Literal['Strict', 'Lax', 'None']

# Chrome SQLite sameSite values → Playwright format
_SAMESITE_MAP: Mapping[int, SameSiteValue] = {
    -1: 'Lax',  # Unspecified → default to Lax
    0: 'None',
    1: 'Lax',
    2: 'Strict',
}

# Regex for IPv4 addresses (validates 0-255 octets)
_IPV4_PATTERN = re.compile(r'^(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)$')


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
    cleaned = pattern.lower().strip('.')
    if ':' in cleaned:
        cleaned = cleaned.split(':')[0]  # Strip port if present

    if not cleaned:
        raise ValueError(f'Empty domain pattern: {pattern!r}')

    # Allow localhost as special case
    if cleaned == 'localhost':
        return

    # Allow IP addresses (IPv4)
    if _IPV4_PATTERN.match(cleaned):
        return

    # For regular domains, require at least one dot (TLD)
    if '.' not in cleaned:
        raise ValueError(f"Invalid domain pattern: {pattern!r} - must include TLD (e.g., 'amazon.com' not 'amazon')")


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
    if '://' in origin:
        # Remove scheme
        origin = origin.split('://', 1)[1]
        # Remove path
        origin = origin.split('/', 1)[0]
        # Remove port
        origin = origin.split(':', 1)[0]
        return origin

    # Handle IndexedDB format (https_www.amazon.com_0)
    if origin.startswith(('https_', 'http_')):
        # Remove scheme prefix
        origin = origin.split('_', 1)[1]
        # Remove trailing _0, _1, etc.
        if '_' in origin:
            parts = origin.rsplit('_', 1)
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
    host = host.lower().strip('.')
    pattern = pattern.lower().strip('.')
    if ':' in pattern:
        pattern = pattern.split(':')[0]

    # Exact match (handles IPs and localhost)
    if host == pattern:
        return True

    # Suffix match with dot boundary: host ends with ".pattern"
    return host.endswith('.' + pattern)


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
            '/usr/bin/security',
            '-q',
            'find-generic-password',
            '-w',
            '-a',
            'Chrome',
            '-s',
            'Chrome Safe Storage',
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f'Failed to retrieve Chrome Safe Storage from Keychain: {result.stderr}\n'
            "First run may require clicking 'Always Allow' when prompted."
        )
    return result.stdout.strip().encode('utf-8')


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
        'sha1',
        keychain_password,
        b'saltysalt',
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
    conn = sqlite3.connect(f'file:{cookies_db}?mode=ro', uri=True)
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
    if not encrypted_value.startswith(b'v10'):
        # Unencrypted value (rare but possible)
        return encrypted_value.decode('utf-8', errors='replace')

    # Strip v10 prefix (3 bytes)
    ciphertext = encrypted_value[3:]

    # Decrypt with AES-128-CBC, IV = 16 spaces
    iv = b' ' * 16
    cipher = AES.new(aes_key, AES.MODE_CBC, iv)
    decrypted = cipher.decrypt(ciphertext)

    # Remove PKCS7 padding
    try:
        unpadded = unpad(decrypted, AES.block_size)
    except ValueError as e:
        raise ValueError(f'PKCS7 unpadding failed: {e}') from e

    # Chrome 130+ (version >= 24): SHA256(domain) prepended to decrypted value
    # The hash is INSIDE the encrypted blob, so we strip it after decryption
    # Use >= 32 to handle empty cookie values (exactly 32 bytes = just the hash)
    if chrome_version >= 24 and len(unpadded) >= 32:
        unpadded = unpadded[32:]

    return unpadded.decode('utf-8', errors='replace')


def export_cookies(
    profile_path: Path,
    origins_filter: Sequence[str] | None = None,
) -> Sequence[ProfileStateCookie]:
    """Export cookies from any Chrome profile with full decryption.

    Unlike browser-cookie3, this supports ANY Chrome profile, not just Default.
    Uses direct Keychain access + AES decryption.

    Args:
        profile_path: Path to Chrome profile directory
        origins_filter: Optional domain patterns to include

    Returns:
        Sequence of ProfileStateCookie models

    Raises:
        FileNotFoundError: If Cookies database doesn't exist
        RuntimeError: If Keychain access fails
        sqlite3.Error: If database read fails
        ValueError: If cookie decryption fails
    """
    cookies: list[ProfileStateCookie] = []

    cookies_db = profile_path / 'Cookies'
    if not cookies_db.exists():
        raise FileNotFoundError(f'Cookies database not found: {cookies_db}')

    # Get encryption key from Keychain (shared across all profiles)
    keychain_password = _get_chrome_encryption_key()
    aes_key = _derive_aes_key(keychain_password)

    # Detect Chrome version for hash handling
    chrome_version = _get_chrome_cookies_version(cookies_db)

    # Read and decrypt all cookies from this profile's SQLite
    conn = sqlite3.connect(f'file:{cookies_db}?mode=ro', uri=True)
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
                cookie_value = _decrypt_cookie_value(encrypted_value, aes_key, chrome_version)
            else:
                cookie_value = value or ''

            # Convert Chrome epoch to Unix timestamp
            # Chrome epoch: microseconds since 1601-01-01
            # Unix epoch: seconds since 1970-01-01
            # Offset: 11644473600 seconds
            if expires_utc and expires_utc > 0:
                expires = float(expires_utc // 1_000_000 - 11644473600)
            else:
                expires = -1.0  # Session cookie

            cookies.append(
                ProfileStateCookie(
                    name=name,
                    value=cookie_value,
                    domain=host,
                    path=path,
                    expires=expires,
                    http_only=bool(httponly),
                    secure=bool(secure),
                    same_site=_SAMESITE_MAP.get(samesite, 'Lax'),
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
) -> Mapping[str, Mapping[str, str]]:
    """Export localStorage from Chrome's LevelDB.

    Args:
        profile_path: Path to Chrome profile directory
        origins_filter: Optional origin patterns to include

    Returns:
        Mapping of {origin: {key: value, ...}}

    Raises:
        FileNotFoundError: If localStorage path doesn't exist
    """
    storage: dict[str, dict[str, str]] = {}

    # CRITICAL: localStorage is in "Local Storage/leveldb/", not "Local Storage/"
    leveldb_path = profile_path / 'Local Storage' / 'leveldb'

    if not leveldb_path.exists():
        raise FileNotFoundError(f'localStorage path not found: {leveldb_path}')

    with ccl_chromium_localstorage.LocalStoreDb(leveldb_path) as ls:
        for storage_key in ls.iter_storage_keys():
            # Filter origins (extract domain from URL, then RFC 6265 matching)
            if origins_filter:
                domain = _extract_domain_from_origin(storage_key)
                if not any(_domain_matches(domain, pattern) for pattern in origins_filter):
                    continue

            origin_storage: dict[str, str] = {}
            for record in ls.iter_records_for_storage_key(storage_key):
                value = str(record.value) if record.value else ''
                origin_storage[record.script_key] = value

            if origin_storage:
                storage[storage_key] = origin_storage

    return storage


# =============================================================================
# sessionStorage Export
# =============================================================================


def export_session_storage(
    profile_path: Path,
    origins_filter: Sequence[str] | None = None,
) -> Mapping[str, Mapping[str, str]]:
    """Export sessionStorage from Chrome's LevelDB.

    Chrome persists sessionStorage to disk for session recovery, enabling
    export from profile files. This contradicts browser specs but enables
    useful automation workflows.

    Args:
        profile_path: Path to Chrome profile directory
        origins_filter: Optional origin patterns to include

    Returns:
        Mapping of {origin: {key: value, ...}}

    Raises:
        FileNotFoundError: If sessionStorage path doesn't exist
    """
    storage: dict[str, dict[str, str]] = {}

    session_storage_path = profile_path / 'Session Storage'

    if not session_storage_path.exists():
        raise FileNotFoundError(f'sessionStorage path not found: {session_storage_path}')

    with ccl_chromium_sessionstorage.SessionStoreDb(session_storage_path) as ss:
        for host in ss.iter_hosts():
            # Filter origins (extract domain from URL, then RFC 6265 matching)
            if origins_filter:
                domain = _extract_domain_from_origin(host)
                if not any(_domain_matches(domain, pattern) for pattern in origins_filter):
                    continue

            origin_storage: dict[str, str] = {}
            for record in ss.iter_records_for_host(host):
                value = str(record.value) if record.value else ''
                origin_storage[record.key] = value

            if origin_storage:
                storage[host] = origin_storage

    return storage


# =============================================================================
# Main Export Function
# =============================================================================


def export_chrome_profile_state(
    output_file: str,
    chrome_profile: str = 'Default',
    include_session_storage: bool = True,
    include_indexeddb: bool = False,
    origins_filter: Sequence[str] | None = None,
) -> ChromeProfileStateExportResult:
    """Export Chrome profile state to ProfileState JSON format.

    Produces JSON compatible with navigate_with_profile_state(profile_state_file=...)
    for restoring authenticated state in Selenium automation.

    Args:
        output_file: Path to save JSON output
        chrome_profile: Chrome profile name ("Default", "Profile 1", etc.)
        include_session_storage: Include sessionStorage (default True)
        include_indexeddb: Include IndexedDB records (default False)
        origins_filter: Only export origins matching these patterns

    Returns:
        ChromeProfileStateExportResult with statistics

    Example:
        # After logging into sites in Chrome...
        result = export_chrome_profile_state("auth.json", origins_filter=["github.com"])

        # Then in Selenium:
        navigate_with_profile_state("https://github.com", profile_state_file="auth.json")
    """
    # Fail if output file exists - delete first to replace
    if Path(output_file).expanduser().exists():
        raise FileExistsError(f'Output file already exists: {output_file}')

    # Resolve profile path
    profile_path = get_chrome_profile_path(chrome_profile)

    # Validate filter patterns
    if origins_filter:
        for pattern in origins_filter:
            _validate_domain_pattern(pattern)

    # Convert to list for filtering if provided
    filter_list = list(origins_filter) if origins_filter else None

    # Export cookies (already returns Sequence[ProfileStateCookie])
    cookies = export_cookies(profile_path, filter_list)

    # Export localStorage (returns Mapping[origin, Mapping[key, value]])
    local_storage = export_local_storage(profile_path, filter_list)

    # Export sessionStorage (optional, default on)
    session_storage: Mapping[str, Mapping[str, str]] = {}
    if include_session_storage:
        session_storage = export_session_storage(profile_path, filter_list)

    # Export IndexedDB with full schema (optional, default off - can be huge)
    indexeddb: dict[str, list[dict[str, Any]]] = {}
    if include_indexeddb:
        indexeddb = export_indexeddb_with_schema(profile_path, filter_list)

    # Build origins dict with ProfileStateOriginStorage models
    all_origins = set(local_storage.keys()) | set(session_storage.keys()) | set(indexeddb.keys())
    origins_data: dict[str, ProfileStateOriginStorage] = {}

    for origin in all_origins:
        # Get localStorage for this origin (empty dict if not present)
        ls_data = dict(local_storage.get(origin, {}))

        # Get sessionStorage for this origin (None if not present)
        ss_data = dict(session_storage[origin]) if origin in session_storage else None

        # Convert IndexedDB dicts to Pydantic models
        idb_models: list[ProfileStateIndexedDB] | None = None
        if origin in indexeddb:
            idb_models = []
            for db_dict in indexeddb[origin]:
                # Convert object stores
                object_stores: list[ProfileStateIndexedDBObjectStore] = []
                for store_dict in db_dict.get('object_stores', []):
                    # Convert indexes
                    indexes: list[ProfileStateIndexedDBIndex] = [
                        ProfileStateIndexedDBIndex(
                            name=idx['name'],
                            key_path=idx['key_path'],
                            unique=idx['unique'],
                            multi_entry=idx['multi_entry'],
                        )
                        for idx in store_dict.get('indexes', [])
                    ]
                    # Convert records
                    records: list[ProfileStateIndexedDBRecord] = [
                        ProfileStateIndexedDBRecord(key=rec['key'], value=rec['value'])
                        for rec in store_dict.get('records', [])
                    ]
                    object_stores.append(
                        ProfileStateIndexedDBObjectStore(
                            name=store_dict['name'],
                            key_path=store_dict['key_path'],
                            auto_increment=store_dict['auto_increment'],
                            indexes=indexes,
                            records=records,
                        )
                    )
                idb_models.append(
                    ProfileStateIndexedDB(
                        database_name=db_dict['database_name'],
                        version=db_dict['version'],
                        object_stores=object_stores,
                    )
                )

        origins_data[origin] = ProfileStateOriginStorage(
            local_storage=ls_data,
            session_storage=ss_data,
            indexed_db=idb_models,
        )

    # Build ProfileState with typed models
    profile_state = ProfileState(
        schema_version='1.0',
        captured_at=datetime.now(UTC).isoformat(),
        cookies=list(cookies),
        origins=origins_data,
    )

    # Write to file with secure permissions (avoid TOCTOU race)
    # SECURITY: Output contains sensitive auth tokens - create with 0o600
    output_path = Path(output_file).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fd = os.open(output_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, 'w') as f:
        # Use by_alias=True to serialize IndexedDB fields as camelCase for JS compatibility
        f.write(profile_state.model_dump_json(indent=2, by_alias=True))

    # Calculate statistics
    ls_keys = sum(len(storage) for storage in local_storage.values())
    ss_keys = sum(len(storage) for storage in session_storage.values())

    return ChromeProfileStateExportResult(
        path=str(output_path),
        cookie_count=len(cookies),
        origin_count=len(origins_data),
        local_storage_keys=ls_keys,
        session_storage_keys=ss_keys,
        indexeddb_origins=len(indexeddb),
    )
