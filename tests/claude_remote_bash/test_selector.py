"""Tests for `claude_remote_bash.selector.parse`."""

from __future__ import annotations

import pytest
from claude_remote_bash.selector import SelectorError, parse

DISCOVERED = frozenset({'m2', 'm3', 'm4', 'm5'})
GROUPS = {
    'fleet': ['M2', 'M3', 'M4'],
    'workers': ['M3', 'M4'],
}


class TestHappyPath:
    def test_single_alias(self) -> None:
        assert parse('M2', groups=GROUPS, discovered_aliases=DISCOVERED) == ['m2']

    def test_comma_list(self) -> None:
        assert parse('M2,M3', groups=GROUPS, discovered_aliases=DISCOVERED) == ['m2', 'm3']

    def test_group_expansion(self) -> None:
        assert parse('fleet', groups=GROUPS, discovered_aliases=DISCOVERED) == ['m2', 'm3', 'm4']

    def test_group_plus_ad_hoc_with_dedupe(self) -> None:
        # M2 already in fleet — silent post-expansion dedupe
        assert parse('fleet,M2', groups=GROUPS, discovered_aliases=DISCOVERED) == ['m2', 'm3', 'm4']

    def test_literal_ip_port(self) -> None:
        # ip:port bypasses discovery validation
        assert parse(
            '192.168.4.22:51648',
            groups={},
            discovered_aliases=frozenset(),
        ) == ['192.168.4.22:51648']

    def test_mixed_ip_port_and_alias(self) -> None:
        assert parse(
            'M2,192.168.4.22:51648',
            groups=GROUPS,
            discovered_aliases=DISCOVERED,
        ) == ['m2', '192.168.4.22:51648']

    def test_case_insensitive_alias(self) -> None:
        assert parse('m2,M3', groups=GROUPS, discovered_aliases=DISCOVERED) == ['m2', 'm3']

    def test_case_insensitive_group(self) -> None:
        # Group keys come in lowercased from client_config (loader normalizes)
        assert parse('Fleet', groups=GROUPS, discovered_aliases=DISCOVERED) == ['m2', 'm3', 'm4']


class TestWhitespace:
    def test_whitespace_around_atoms_stripped(self) -> None:
        assert parse('M2 , M3', groups=GROUPS, discovered_aliases=DISCOVERED) == ['m2', 'm3']

    def test_leading_trailing_whitespace_in_selector(self) -> None:
        assert parse('  M2,M3  ', groups=GROUPS, discovered_aliases=DISCOVERED) == ['m2', 'm3']


class TestGrammarErrors:
    def test_empty_selector_rejected(self) -> None:
        with pytest.raises(SelectorError, match='Empty selector'):
            parse('', groups=GROUPS, discovered_aliases=DISCOVERED)

    def test_whitespace_only_selector_rejected(self) -> None:
        with pytest.raises(SelectorError, match='Empty selector'):
            parse('   ', groups=GROUPS, discovered_aliases=DISCOVERED)

    def test_empty_atom_in_middle_rejected(self) -> None:
        with pytest.raises(SelectorError, match='Empty atom at position 2'):
            parse('M2,,M3', groups=GROUPS, discovered_aliases=DISCOVERED)

    def test_trailing_comma_rejected(self) -> None:
        with pytest.raises(SelectorError, match='Trailing comma'):
            parse('M2,M3,', groups=GROUPS, discovered_aliases=DISCOVERED)

    def test_pre_expansion_duplicate_rejected(self) -> None:
        with pytest.raises(SelectorError, match=r'Duplicate atom .* positions 1 and 2'):
            parse('M2,M2', groups=GROUPS, discovered_aliases=DISCOVERED)

    def test_pre_expansion_duplicate_case_insensitive(self) -> None:
        with pytest.raises(SelectorError, match=r'Duplicate atom'):
            parse('M2,m2', groups=GROUPS, discovered_aliases=DISCOVERED)

    def test_unknown_atom_rejected(self) -> None:
        with pytest.raises(SelectorError, match=r"Unknown atom 'bogus'"):
            parse('M2,bogus', groups=GROUPS, discovered_aliases=DISCOVERED)

    def test_group_naming_discovered_alias_collision_rejected(self) -> None:
        # User defined group 'M2' but M2 is also a discovered alias → collision
        bad_groups = {'m2': ['M3', 'M4']}
        with pytest.raises(SelectorError, match=r"Group 'm2' conflicts with discovered host alias"):
            parse('M2', groups=bad_groups, discovered_aliases=DISCOVERED)

    def test_group_referencing_unknown_host_rejected(self) -> None:
        bad_groups = {'fleet': ['M2', 'phantom']}
        with pytest.raises(SelectorError, match=r"Group 'fleet' references unknown host 'phantom'"):
            parse('fleet', groups=bad_groups, discovered_aliases=DISCOVERED)


class TestEdgeCases:
    def test_post_expansion_overlap_silently_deduped(self) -> None:
        # fleet contains M2, M3; workers contains M3, M4. Overlap on M3.
        # Pre-expansion atoms (fleet, workers) are distinct so no dup error.
        assert parse('fleet,workers', groups=GROUPS, discovered_aliases=DISCOVERED) == ['m2', 'm3', 'm4']

    def test_empty_groups_map_still_works_for_aliases(self) -> None:
        assert parse('M2,M3', groups={}, discovered_aliases=DISCOVERED) == ['m2', 'm3']

    def test_no_discovered_hosts_still_accepts_ip_port(self) -> None:
        assert parse(
            '10.0.0.1:1234',
            groups={},
            discovered_aliases=frozenset(),
        ) == ['10.0.0.1:1234']

    def test_order_preserved_across_groups(self) -> None:
        # workers is intentionally first so we can verify order
        assert parse('workers,fleet', groups=GROUPS, discovered_aliases=DISCOVERED) == ['m3', 'm4', 'm2']
