"""Chrome profile metadata extraction module."""

from __future__ import annotations

import json
import os
import platform
import sys
from pathlib import Path

from .models import (
    ChromeInfoCacheEntry,
    ChromeLocalState,
    ChromeProfileEssential,
    ChromeProfileFull,
    ChromeProfilesResult,
)


class ChromeProfileError(Exception):
    """Base exception for Chrome profile operations."""

    pass


class ChromeNotFoundError(ChromeProfileError):
    """Chrome installation not found."""

    pass


class ProfileNotFoundError(ChromeProfileError):
    """Specified profile directory not found."""

    pass


class MetadataParseError(ChromeProfileError):
    """Error parsing Chrome metadata JSON."""

    pass


def get_chrome_base_path() -> Path:
    r"""Get Chrome user data directory (platform-aware).

    Platform paths:
    - macOS:   ~/Library/Application Support/Google/Chrome/
    - Linux:   ~/.config/google-chrome/
    - Windows: %LOCALAPPDATA%\Google\Chrome\User Data\

    Returns:
        Path to Chrome base directory

    Raises:
        ChromeNotFoundError: If Chrome installation not found
        KeyError: If required environment variable not set (Windows)
    """
    system = platform.system()

    if system == "Darwin":
        # macOS
        path = Path("~/Library/Application Support/Google/Chrome").expanduser()

    elif system == "Linux":
        # Linux (respects XDG Base Directory Specification)
        path = Path("~/.config/google-chrome").expanduser()

    elif system == "Windows":
        # Windows (requires LOCALAPPDATA environment variable)
        localappdata = os.environ["LOCALAPPDATA"]  # Raises KeyError if not set
        path = Path(localappdata) / "Google/Chrome/User Data"

    else:
        raise ChromeNotFoundError(f"Unsupported platform: {system}")

    if not path.exists():
        raise ChromeNotFoundError(f"Chrome installation not found at: {path}")

    return path


def list_profile_directories(base_path: Path) -> list[str]:
    """Find all profile directories (Default, Profile 1, etc.).

    Args:
        base_path: Chrome base directory

    Returns:
        List of profile directory names
    """
    profiles = []

    # Check for Default profile
    if (base_path / "Default").is_dir():
        profiles.append("Default")

    # Check for Profile 1, Profile 2, etc.
    for item in base_path.iterdir():
        if item.is_dir() and item.name.startswith("Profile "):
            profiles.append(item.name)

    return sorted(profiles)


def load_local_state(base_path: Path) -> ChromeLocalState:
    """Load and parse Local State JSON with strict Pydantic validation.

    Args:
        base_path: Chrome base directory

    Returns:
        Parsed and validated ChromeLocalState model

    Raises:
        MetadataParseError: If Local State cannot be loaded, parsed, or validated
    """
    local_state_path = base_path / "Local State"

    if not local_state_path.exists():
        raise MetadataParseError(f"Local State file not found: {local_state_path}")

    try:
        with open(local_state_path) as f:
            data = json.load(f)
        return ChromeLocalState(**data)
    except json.JSONDecodeError as e:
        raise MetadataParseError(f"Failed to parse Local State JSON: {e}") from e
    except Exception as e:
        raise MetadataParseError(f"Failed to validate Local State: {e}") from e


def load_profile_preferences(profile_path: Path) -> dict | None:
    """Load and parse profile's Preferences JSON (may not exist).

    Args:
        profile_path: Path to profile directory

    Returns:
        Parsed Preferences dictionary, or None if file doesn't exist
    """
    prefs_path = profile_path / "Preferences"

    if not prefs_path.exists():
        return None

    try:
        with open(prefs_path) as f:
            return json.load(f)
    except Exception:
        # Non-fatal: return None if Preferences can't be read
        return None


def get_profile_metadata(
    profile_dir: str, base_path: Path, verbose: bool = False
) -> ChromeProfileEssential | ChromeProfileFull:
    """Extract metadata for a single profile.

    Data sources:
    1. Local State -> profile.info_cache[profile_dir]
    2. Preferences (optional) -> profile section

    Args:
        profile_dir: "Default", "Profile 1", etc.
        base_path: Chrome base directory
        verbose: If True, return ChromeProfileFull with all fields

    Returns:
        ChromeProfileEssential (default) or ChromeProfileFull (verbose)

    Raises:
        ProfileNotFoundError: If profile directory doesn't exist
        MetadataParseError: If metadata cannot be parsed
    """
    profile_path = base_path / profile_dir

    if not profile_path.is_dir():
        raise ProfileNotFoundError(f"Profile directory not found: {profile_path}")

    # Load Local State with strict Pydantic validation
    local_state = load_local_state(base_path)
    info_cache = local_state.profile.info_cache

    if profile_dir not in info_cache:
        raise MetadataParseError(
            f"Profile {profile_dir} not found in Local State info_cache"
        )

    profile_data: ChromeInfoCacheEntry = info_cache[profile_dir]

    # Extract essential fields from validated Pydantic model
    essential_data = {
        "profile_dir": profile_dir,
        "name": profile_data.name,
        "user_name": profile_data.user_name,
        "gaia_name": profile_data.gaia_name,
        "gaia_id": profile_data.gaia_id,
        "avatar_icon": profile_data.avatar_icon,
        "is_managed": profile_data.is_managed != 0,
        "is_ephemeral": profile_data.is_ephemeral,
        "active_time": profile_data.active_time,
    }

    if not verbose:
        return ChromeProfileEssential(**essential_data)

    # Extract additional fields for verbose view from validated Pydantic model
    full_data = {
        **essential_data,
        "background_apps": profile_data.background_apps,
        "hosted_domain": profile_data.hosted_domain,
        "is_using_default_name": profile_data.is_using_default_name,
        "is_using_default_avatar": profile_data.is_using_default_avatar,
        "is_consented_primary_account": profile_data.is_consented_primary_account,
        "gaia_given_name": profile_data.gaia_given_name,
        "gaia_picture_file_name": profile_data.gaia_picture_file_name,
        "last_downloaded_gaia_picture_url_with_size": profile_data.last_downloaded_gaia_picture_url_with_size,
        "profile_color_seed": profile_data.profile_color_seed,
        "profile_highlight_color": profile_data.profile_highlight_color,
        "metrics_bucket_index": profile_data.metrics_bucket_index,
        "first_account_name_hash": profile_data.first_account_name_hash,
        "force_signin_profile_locked": profile_data.force_signin_profile_locked,
        "is_glic_eligible": profile_data.is_glic_eligible,
        "is_using_new_placeholder_avatar_icon": profile_data.is_using_new_placeholder_avatar_icon,
        "user_accepted_account_management": profile_data.user_accepted_account_management,
        "default_avatar_fill_color": profile_data.default_avatar_fill_color,
        "default_avatar_stroke_color": profile_data.default_avatar_stroke_color,
    }

    return ChromeProfileFull(**full_data)


def list_all_profiles(verbose: bool = False) -> ChromeProfilesResult:
    """List all Chrome profiles with metadata.

    Args:
        verbose: If True, return full metadata for each profile

    Returns:
        ChromeProfilesResult with all profiles

    Raises:
        ChromeNotFoundError: If Chrome installation not found
        MetadataParseError: If metadata cannot be parsed
    """
    base_path = get_chrome_base_path()
    profile_dirs = list_profile_directories(base_path)

    profiles = []
    for profile_dir in profile_dirs:
        try:
            profile = get_profile_metadata(profile_dir, base_path, verbose=verbose)
            profiles.append(profile)
        except (ProfileNotFoundError, MetadataParseError) as e:
            # Log warning but continue with other profiles
            print(
                f"[chrome_profiles] Warning: Failed to load {profile_dir}: {e}",
                file=sys.stderr,
            )
            continue

    # Determine default profile (first one, typically "Default")
    default_profile = profile_dirs[0] if profile_dirs else None

    return ChromeProfilesResult(
        profiles=profiles,
        total_count=len(profiles),
        default_profile=default_profile,
        chrome_base_path=str(base_path),
    )
