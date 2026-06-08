from __future__ import annotations

import re
from pathlib import Path


class TestDocketInvariants:
    """Structural invariants for the `docket/` store.

    The collision gate plus shape checks the add-to-docket skill can't enforce alone. The
    load-bearing one is per-directory NN uniqueness: two PRs claiming the same number merge
    cleanly on disk (different slugs → different paths), so this check fails the merged state and
    forces a renumber. Each type's own README defines its entry shape; that's not enforced here —
    only the cross-type structure is.
    """

    DOCKET = Path(__file__).parents[2] / 'docket'
    ENTRY_RE = re.compile(r'^\d{2,}-[a-z0-9]+(?:-[a-z0-9]+)*\.md$')

    def test_type_dirs_are_self_documenting(self) -> None:
        """Every type dir carries a README — the store declares its own types and how to file them."""
        for type_dir in self._type_dirs():
            assert (type_dir / 'README.md').is_file(), f'docket/{type_dir.name}/ has no README.md'

    def test_entry_filenames_well_formed(self) -> None:
        """Every entry is NN-slug.md — 2+ digit zero-padded number, kebab slug."""
        for type_dir in self._type_dirs():
            for entry in self._entries(type_dir):
                assert self.ENTRY_RE.match(entry.name), f'malformed docket entry: {type_dir.name}/{entry.name}'

    def test_no_duplicate_numbers_per_type(self) -> None:
        """No two entries in a type dir share a number — the collision gate."""
        for type_dir in self._type_dirs():
            nums = [e.name.split('-', 1)[0] for e in self._entries(type_dir)]
            dupes = sorted({n for n in nums if nums.count(n) > 1})
            assert not dupes, f'duplicate docket numbers in {type_dir.name}/: {dupes}'

    def _type_dirs(self) -> list[Path]:
        return sorted(p for p in self.DOCKET.iterdir() if p.is_dir()) if self.DOCKET.is_dir() else []

    def _entries(self, type_dir: Path) -> list[Path]:
        return sorted(p for p in type_dir.glob('*.md') if p.name != 'README.md')
