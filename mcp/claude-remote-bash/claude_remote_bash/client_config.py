"""Client-side configuration loaded from ``client_config.json``."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from cc_lib.schemas import ClosedModel
from pydantic import ValidationError, field_validator, model_validator

from claude_remote_bash.exceptions import ConfigError
from claude_remote_bash.paths import CLIENT_CONFIG

__all__ = [
    'ClientConfig',
]


class ClientConfig(ClosedModel):
    """User's named host groups."""

    groups: Mapping[str, Sequence[str]] = {}
    """Group name → ordered list of host aliases. Keys lowercased on load."""

    @classmethod
    def load(cls) -> ClientConfig:
        """Load the client config from disk, or return an empty default if the file is missing."""
        if not CLIENT_CONFIG.exists():
            return cls()
        try:
            return cls.model_validate_json(CLIENT_CONFIG.read_text())
        except ValidationError as exc:
            msg = exc.errors()[0]['msg'].removeprefix('Value error, ')
            raise ConfigError(f'{CLIENT_CONFIG}: {msg}') from exc

    @field_validator('groups', mode='before')
    @classmethod
    def _lowercase_keys(cls, v: Any) -> Any:
        """Lowercase group names so the selector grammar can match them case-insensitively."""
        if isinstance(v, dict):
            return {str(name).lower(): list(members) for name, members in v.items()}
        return v

    @model_validator(mode='after')
    def _reject_group_of_groups(self) -> ClientConfig:
        """Reject any group whose value list names another group."""
        names = set(self.groups.keys())
        for owner, members in self.groups.items():
            for member in members:
                if member.lower() in names:
                    raise ValueError(
                        f'group {owner!r} references group {member!r} — '
                        'group-of-groups and self-reference are not supported'
                    )
        return self

    @model_validator(mode='after')
    def _reject_empty_groups(self) -> ClientConfig:
        """Reject any group with no members — an empty group cannot be targeted."""
        for name, members in self.groups.items():
            if not members:
                raise ValueError(f'group {name!r} has no members — list at least one host or remove the group')
        return self
