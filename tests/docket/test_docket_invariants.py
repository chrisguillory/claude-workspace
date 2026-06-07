from __future__ import annotations

import re
from pathlib import Path


class TestDocketInvariants:
    """Structural invariants for the `docket/` store.

    The collision gate plus shape checks the add-to-docket skill can't enforce alone. The
    load-bearing one is per-directory NN uniqueness: two PRs claiming the same number merge
    cleanly on disk (different slugs → different paths), so this check fails the merged state
    and forces a renumber.
    """

    DOCKET = Path(__file__).parents[2] / 'docket'
    TYPES = ('tech-debt', 'feature', 'follow-up', 'idea')
    ENTRY_RE = re.compile(r'^\d{2,}-[a-z0-9]+(?:-[a-z0-9]+)*\.md$')

    def test_only_valid_type_dirs(self) -> None:
        """docket/ holds only the four type directories — a typo'd dir silently orphans entries."""
        if not self.DOCKET.is_dir():
            return
        subdirs = {p.name for p in self.DOCKET.iterdir() if p.is_dir()}
        assert subdirs <= set(self.TYPES), f'unexpected docket type dirs: {sorted(subdirs - set(self.TYPES))}'

    def test_entry_filenames_well_formed(self) -> None:
        """Every entry is NN-slug.md — 2+ digit zero-padded number, kebab slug."""
        for type_name in self.TYPES:
            for entry in self._entries(type_name):
                assert self.ENTRY_RE.match(entry.name), f'malformed docket entry: {type_name}/{entry.name}'

    def test_no_duplicate_numbers_per_type(self) -> None:
        """No two entries in a type dir share a number — the collision gate."""
        for type_name in self.TYPES:
            nums = [e.name.split('-', 1)[0] for e in self._entries(type_name)]
            dupes = sorted({n for n in nums if nums.count(n) > 1})
            assert not dupes, f'duplicate docket numbers in {type_name}/: {dupes}'

    def test_required_frontmatter(self) -> None:
        """Every entry carries `area` + `title` in YAML frontmatter (the load-bearing fields)."""
        for type_name in self.TYPES:
            for entry in self._entries(type_name):
                text = entry.read_text()
                assert text.startswith('---\n'), f'{type_name}/{entry.name}: missing frontmatter'
                frontmatter = text.split('---\n', 2)[1]
                for field in ('area:', 'title:'):
                    assert field in frontmatter, f'{type_name}/{entry.name}: missing required `{field}`'

    def _entries(self, type_name: str) -> list[Path]:
        type_dir = self.DOCKET / type_name
        return sorted(type_dir.glob('*.md')) if type_dir.is_dir() else []
