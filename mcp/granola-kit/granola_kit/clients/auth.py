from __future__ import annotations

__all__ = [
    'WorkosTokens',
    'get_access_token',
]

import base64
import hashlib
import json
import subprocess
import time
from pathlib import Path

import httpx
from cc_lib.schemas.base import SubsetModel
from cc_lib.utils.atomic_write import atomic_write
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from granola_kit.clients.identity import GranolaIdentity
from granola_kit.exceptions import GranolaAuthError

REFRESH_BUFFER_SECONDS = 60


class WorkosTokens(SubsetModel):
    """The WorkOS access/refresh token pair from Granola's local auth store."""

    access_token: str
    refresh_token: str | None = None


async def get_access_token() -> str:
    """A currently-valid WorkOS access token.

    Resolves in order — the on-disk cache, then Granola's live encrypted store
    (the legacy plaintext file only when the encrypted store is absent) — and
    refreshes via /v1/refresh-access-token when the resolved token is at or near
    its ``exp``. The cache spares a Keychain read on every invocation.
    """
    now = int(time.time())

    cached = _read_cache()
    if cached is not None and _expires_at(cached.access_token) - now > REFRESH_BUFFER_SECONDS:
        return cached.access_token

    tokens = _read_store_tokens()
    if _expires_at(tokens.access_token) - now > REFRESH_BUFFER_SECONDS:
        _write_cache(tokens)
        return tokens.access_token

    if tokens.refresh_token is None:
        raise GranolaAuthError('access token is expired and the store holds no refresh_token')
    refreshed = await _refresh(tokens.refresh_token)
    _write_cache(refreshed)
    return refreshed.access_token


def _read_store_tokens() -> WorkosTokens:
    """Read tokens from Granola's local store: the live encrypted store, else the legacy plaintext file."""
    store = _granola_store_dir()
    encrypted, dek = store / 'supabase.json.enc', store / 'storage.dek'
    if encrypted.exists() and dek.exists():
        return _read_encrypted_tokens(encrypted, dek)
    plaintext = store / 'supabase.json'
    if plaintext.exists():
        return _parse_workos_tokens(json.loads(plaintext.read_text()))
    raise GranolaAuthError(f'no Granola auth store under {store} — is Granola installed and signed in?')


def _read_encrypted_tokens(encrypted: Path, dek: Path) -> WorkosTokens:
    """Decrypt ``workos_tokens`` from Granola's post-2026 encrypted store.

    Chain (macOS): the Keychain 'Granola Safe Storage' password decrypts
    ``storage.dek`` (safeStorage v10) to a 32-byte data-encryption key; that key
    decrypts ``supabase.json.enc`` via AES-256-GCM (nonce = first 12 bytes). The
    store is present, so any failure here is fatal — we never fall back to the
    frozen plaintext file behind the user's back.
    """
    key = base64.b64decode(_safe_storage_decrypt_v10(dek.read_bytes(), _keychain_safe_storage_password()))
    blob = encrypted.read_bytes()
    plaintext = AESGCM(key).decrypt(blob[:12], blob[12:], None)
    return _parse_workos_tokens(json.loads(plaintext))


def _keychain_safe_storage_password() -> str:
    """Granola's Electron safeStorage key from the login Keychain (prompts 'Always Allow' on first use)."""
    try:
        result = subprocess.run(
            ['security', 'find-generic-password', '-w', '-s', 'Granola Safe Storage'],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise GranolaAuthError(
            "could not read 'Granola Safe Storage' from the login Keychain — approve the access prompt "
            '(Always Allow) and make sure the keychain is unlocked'
        ) from exc
    return result.stdout.strip()


def _safe_storage_decrypt_v10(blob: bytes, password: str) -> bytes:
    """Decrypt a Chromium/Electron OSCrypt 'v10' blob (macOS variant).

    AES-128-CBC; key = PBKDF2-HMAC-SHA1(password, 'saltysalt', 1003 iters, 16 bytes);
    IV = 16 spaces; PKCS7 padding.
    """
    if blob[:3] != b'v10':
        raise GranolaAuthError(f'not a v10 safeStorage blob: {blob[:3]!r}')
    key = hashlib.pbkdf2_hmac('sha1', password.encode('utf-8'), b'saltysalt', 1003, dklen=16)
    decryptor = Cipher(algorithms.AES(key), modes.CBC(b' ' * 16)).decryptor()
    plaintext = decryptor.update(blob[3:]) + decryptor.finalize()
    pad = plaintext[-1] if plaintext else 0
    return plaintext[:-pad] if 1 <= pad <= 16 else plaintext


def _parse_workos_tokens(store: object) -> WorkosTokens:
    """Extract ``workos_tokens`` (a JSON string nested inside the store JSON) into the token model."""
    if not isinstance(store, dict):
        raise GranolaAuthError(f'Granola auth store is not a JSON object: {type(store).__name__}')
    raw = store.get('workos_tokens')
    if raw is None:
        raise GranolaAuthError("Granola auth store has no 'workos_tokens'")
    return WorkosTokens.model_validate(json.loads(raw) if isinstance(raw, str) else raw)


def _expires_at(access_token: str) -> int:
    """The ``exp`` claim (unix seconds) decoded from a JWT access token."""
    parts = access_token.split('.')
    if len(parts) < 2:
        raise GranolaAuthError('access token is not a JWT')
    payload = base64.urlsafe_b64decode(parts[1] + '=' * (-len(parts[1]) % 4))
    exp = json.loads(payload).get('exp')
    if not isinstance(exp, (int, float)):
        raise GranolaAuthError("access token has no numeric 'exp' claim")
    return int(exp)


async def _refresh(refresh_token: str) -> WorkosTokens:
    """Exchange a refresh token for a fresh access token via /v1/refresh-access-token."""
    headers = {**GranolaIdentity.detect().headers(), 'Content-Type': 'application/json', 'Accept': 'application/json'}
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            'https://api.granola.ai/v1/refresh-access-token',
            json={'refresh_token': refresh_token},
            headers=headers,
        )
    response.raise_for_status()
    refreshed = _parse_refresh_response(response.json())
    # The refresh response often omits refresh_token — carry the one we just used.
    return (
        refreshed
        if refreshed.refresh_token is not None
        else refreshed.model_copy(update={'refresh_token': refresh_token})
    )


def _parse_refresh_response(payload: object) -> WorkosTokens:
    """Validate the /v1/refresh-access-token response into the token model."""
    if not isinstance(payload, dict) or not payload.get('access_token'):
        raise GranolaAuthError(f'refresh response did not return an access_token: {payload!r}')
    return WorkosTokens.model_validate(payload)


def _read_cache() -> WorkosTokens | None:
    """The cached token pair, or None when no cache exists yet."""
    path = _token_cache_path()
    if not path.exists():
        return None
    return WorkosTokens.model_validate_json(path.read_text())


def _write_cache(tokens: WorkosTokens) -> None:
    """Persist the token pair to the cache (atomic) so the next run skips the Keychain read."""
    path = _token_cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(path, tokens.model_dump_json().encode())


def _granola_store_dir() -> Path:
    """Granola's macOS application-support directory, where the auth store lives."""
    return Path.home() / 'Library' / 'Application Support' / 'Granola'


def _token_cache_path() -> Path:
    """granola-kit's token cache file."""
    return Path.home() / '.claude-workspace' / 'granola-kit' / 'token-cache.json'
