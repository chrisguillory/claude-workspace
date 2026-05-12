"""Parse the ``-t, --target`` selector grammar into a resolution-ready atom list.

A selector is a comma-separated list. Each atom is one of:

* A literal ``ip:port`` (contains ``:``) — direct address, bypasses discovery.
* A discovered-host alias (e.g. ``M2``) — matched case-insensitively.
* A group name from ``client_config.json`` — replaced inline by its host list.

Grammar rules (locked by the plan):

* ASCII whitespace stripped from each atom before lookup — ``"M2 , M3"`` is
  equivalent to ``"M2,M3"``.
* Empty atom inside the list (``"M2,,M3"``) → ``SelectorError`` naming the position.
* Trailing comma (``"M2,M3,"``) → ``SelectorError``.
* Empty selector (``""`` or whitespace-only) → ``SelectorError``.
* Pre-expansion duplicate (``"M2,M2"``) → ``SelectorError``.
* Group name that's also a discovered alias → ``SelectorError`` (collision).
* Unknown atom (not an alias, not a group, not ``ip:port``) → ``SelectorError``.

Returns the resolution-ready atom list with post-expansion overlap
deduped (first-seen order preserved).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence, Set

__all__ = [
    'SelectorError',
    'parse',
]


class SelectorError(ValueError):
    """The selector string violates one of the locked grammar rules."""


def parse(
    selector: str,
    *,
    groups: Mapping[str, Sequence[str]],
    discovered_aliases: Set[str],
) -> Sequence[str]:
    """Parse ``selector`` into an ordered, deduped list of atoms ready for resolution.

    Each returned atom is either a literal ``ip:port`` string or a
    lowercased host alias. Callers feed each one to ``_lookup_alias``.

    Args:
        selector: Raw ``-t, --target`` value from the CLI.
        groups: ``client_config.json`` groups map (keys already lowercased).
        discovered_aliases: Lowercased aliases from the mDNS discovery
            result, used to validate alias atoms and detect collisions
            with group names.

    Raises:
        SelectorError: Any grammar violation, unknown atom, or collision.
    """
    if not selector.strip():
        raise SelectorError('Empty selector. Specify at least one host alias, group, or ip:port.')

    raw_parts = selector.split(',')
    atoms: list[str] = []
    for position, raw in enumerate(raw_parts, start=1):
        atom = raw.strip()
        if not atom:
            if position == len(raw_parts):
                raise SelectorError(f'Trailing comma in selector: {selector!r}')
            raise SelectorError(f'Empty atom at position {position} in selector: {selector!r}')
        atoms.append(atom)

    seen_lower: dict[str, int] = {}
    for position, atom in enumerate(atoms, start=1):
        key = atom.lower()
        prior = seen_lower.get(key)
        if prior is not None:
            raise SelectorError(
                f'Duplicate atom {atom!r} at positions {prior} and {position}. '
                'Remove the duplicate (post-expansion overlaps from groups are dedup'
                'd silently; pre-expansion duplicates signal user confusion).'
            )
        seen_lower[key] = position

    for name in groups:
        if name in discovered_aliases:
            raise SelectorError(
                f'Group {name!r} conflicts with discovered host alias {name!r}. '
                'Rename the group in ~/.claude-workspace/mcp/claude-remote-bash/client_config.json.'
            )

    resolved: list[str] = []
    seen_resolved: set[str] = set()

    def _append(atom: str) -> None:
        key = atom.lower() if ':' not in atom else atom
        if key not in seen_resolved:
            seen_resolved.add(key)
            resolved.append(atom if ':' in atom else key)

    for atom in atoms:
        if ':' in atom:
            _append(atom)
            continue
        atom_lower = atom.lower()
        if atom_lower in groups:
            for member in groups[atom_lower]:
                member_stripped = member.strip()
                if ':' not in member_stripped and member_stripped.lower() not in discovered_aliases:
                    raise SelectorError(
                        f'Group {atom!r} references unknown host {member!r}. '
                        'Check the alias matches a daemon advertised on the LAN '
                        '(run `claude-remote-bash discover`).'
                    )
                _append(member_stripped)
            continue
        if atom_lower in discovered_aliases:
            _append(atom)
            continue
        raise SelectorError(
            f'Unknown atom {atom!r}. Not a discovered host alias and not a group. '
            'Run `claude-remote-bash discover` to see available hosts; '
            'check ~/.claude-workspace/mcp/claude-remote-bash/client_config.json for groups.'
        )

    return resolved
