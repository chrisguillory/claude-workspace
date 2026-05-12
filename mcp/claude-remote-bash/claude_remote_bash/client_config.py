"""Client-side preferences for claude-remote-bash (groups, future settings).

The file at ``~/.claude-workspace/mcp/claude-remote-bash/client_config.json``
holds personal, machine-local preferences for the CLI client. Today it
carries just the user's named host groups; the top-level wrapper leaves
room for future keys (default timeout, output format, etc.) without
breaking the loader.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from cc_lib.utils import get_claude_workspace_config_home_dir

__all__ = [
    'CLIENT_CONFIG_FILE',
    'GroupOfGroupsError',
    'MalformedClientConfigError',
    'load_groups',
]


CLIENT_CONFIG_FILE: Path = get_claude_workspace_config_home_dir() / 'mcp' / 'claude-remote-bash' / 'client_config.json'


class MalformedClientConfigError(ValueError):
    """``groups`` key in client_config.json is not a JSON object."""


class GroupOfGroupsError(ValueError):
    """A group's value list contains a name that is also a group (Phase 1 disallows this)."""


def load_groups() -> Mapping[str, Sequence[str]]:
    """Return the user-defined groups map, or an empty dict if there's nothing to load.

    Behavior across degenerate inputs (matched to the plan's locked spec):

    * file absent             → empty map
    * file present, 0 bytes   → empty map (json.loads raises; caught)
    * ``{}`` or missing key   → empty map
    * ``{"groups": {}}``      → empty map
    * ``{"groups": null}``    → empty map
    * ``{"groups": <not-a-dict>}`` → MalformedClientConfigError
    * group value list contains a name that's also a group → GroupOfGroupsError

    Group keys are lowercased on load so they compare consistently with
    the selector atom (which is also lowercased).
    """
    if not CLIENT_CONFIG_FILE.exists():
        return {}

    raw = CLIENT_CONFIG_FILE.read_text()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        if raw.strip() == '':
            return {}
        raise

    groups_raw = data.get('groups') if isinstance(data, dict) else None
    if groups_raw is None:
        return {}
    if not isinstance(groups_raw, dict):
        raise MalformedClientConfigError(
            f"client_config.json: 'groups' must be a JSON object, got {type(groups_raw).__name__}"
        )

    groups: dict[str, Sequence[str]] = {name.lower(): list(members) for name, members in groups_raw.items()}

    group_names = set(groups.keys())
    for name, members in groups.items():
        for member in members:
            if member.lower() in group_names:
                raise GroupOfGroupsError(
                    f'group {name!r} references group {member!r} — '
                    'Phase 1 does not support group-of-groups (or self-reference)'
                )

    return groups
