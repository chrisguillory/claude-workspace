"""Discover AddressBook sync sources and map them to account providers."""

from __future__ import annotations

__all__ = [
    'SourceRegistry',
]

import logging
import pathlib
import sqlite3
from collections.abc import Mapping, Sequence
from typing import cast

from imessage_kit.types import ContactSource, ContactSourceKind

logger = logging.getLogger(__name__)

ACCOUNTS_DB = pathlib.Path.home() / 'Library' / 'Accounts' / 'Accounts4.sqlite'
ADDRESSBOOK_BASE = pathlib.Path.home() / 'Library' / 'Application Support' / 'AddressBook'
MAIN_DB = ADDRESSBOOK_BASE / 'AddressBook-v22.abcddb'
SOURCES_DIR = ADDRESSBOOK_BASE / 'Sources'

# Accounts4 parent account type → ContactSourceKind
_KIND_BY_PARENT_TYPE: Mapping[str, ContactSourceKind] = {
    'Gmail': 'Google',
    'iCloud': 'iCloud',
}


class SourceRegistry:
    """Discovers AddressBook sources and maps them to sync account providers.

    Primary mapping via ``~/Library/Accounts/Accounts4.sqlite`` JOIN. Any
    source directory not present in the CardDAV join is attributed to iCloud
    CloudKit by process of elimination. Fails fast when the heuristic is
    ambiguous (multiple unmapped sources) or when the main AddressBook DB
    contains person records (which it should not — it is container metadata).
    """

    def __init__(self, sources: Sequence[ContactSource]) -> None:
        self._sources = sources

    @classmethod
    def discover(cls) -> SourceRegistry:
        """Scan disk and build a SourceRegistry. Fails fast on ambiguity."""
        _assert_main_db_has_no_persons()

        source_dirs = _list_source_dirs()
        if not source_dirs:
            msg = (
                f'No AddressBook sources found in {SOURCES_DIR}. '
                f'Messages.app and Contacts.app may not be configured on this Mac.'
            )
            raise RuntimeError(msg)

        mapped = _query_carddav_accounts()

        # Split sources into mapped (have Accounts4 entry) and unmapped (CloudKit candidates)
        mapped_dirs: list[pathlib.Path] = []
        unmapped_dirs: list[pathlib.Path] = []
        for d in source_dirs:
            (mapped_dirs if d.name in mapped else unmapped_dirs).append(d)

        if len(unmapped_dirs) > 1:
            uuids = ', '.join(d.name for d in unmapped_dirs)
            msg = (
                f'Ambiguous CloudKit attribution: {len(unmapped_dirs)} AddressBook sources '
                f'are not in Accounts4.sqlite. Cannot determine which is iCloud CloudKit. '
                f'Unmapped source UUIDs: {uuids}'
            )
            raise RuntimeError(msg)

        sources: list[ContactSource] = []
        for d in mapped_dirs:
            info = mapped[d.name]
            parent_type = info['parent_type']
            if parent_type is None:
                msg = f'Accounts4.sqlite has NULL parent_type for source {d.name}.'
                raise RuntimeError(msg)
            kind = _KIND_BY_PARENT_TYPE.get(parent_type)
            if kind is None:
                msg = (
                    f'Unknown Accounts4 parent type {parent_type!r} for source {d.name}. '
                    f'Extend ContactSourceKind and _KIND_BY_PARENT_TYPE to handle it.'
                )
                raise RuntimeError(msg)
            sources.append(
                ContactSource(
                    source_uuid=d.name,
                    kind=kind,
                    display_name=info['account_name'] or kind,
                    email=info['email'],
                    contact_count=_count_records(d / 'AddressBook-v22.abcddb'),
                    db_path=d / 'AddressBook-v22.abcddb',
                )
            )

        # Exactly zero or one unmapped dir by the check above
        sources.extend(
            ContactSource(
                source_uuid=d.name,
                kind='iCloud',
                display_name='iCloud (CloudKit)',
                email=None,
                contact_count=_count_records(d / 'AddressBook-v22.abcddb'),
                db_path=d / 'AddressBook-v22.abcddb',
            )
            for d in unmapped_dirs
        )

        logger.info(
            'Discovered %d AddressBook source(s): %s',
            len(sources),
            ', '.join(f'{s.display_name}={s.contact_count}' for s in sources),
        )
        return cls(sources)

    @property
    def sources(self) -> Sequence[ContactSource]:
        """All discovered sources."""
        return self._sources

    def get_by_display_name(self, name: str) -> ContactSource | None:
        """Match by display name, case-insensitive. Returns None if not found."""
        name_lower = name.lower()
        for source in self._sources:
            if source.display_name.lower() == name_lower:
                return source
        return None


def _list_source_dirs() -> Sequence[pathlib.Path]:
    """Subdirectories of ~/Library/Application Support/AddressBook/Sources/."""
    if not SOURCES_DIR.exists():
        return []
    return sorted(d for d in SOURCES_DIR.iterdir() if d.is_dir() and not d.name.startswith('.'))


def _query_carddav_accounts() -> Mapping[str, Mapping[str, str | None]]:
    """Map source UUIDs to account info via Accounts4.sqlite CardDAV JOIN.

    Returns a mapping keyed by source UUID with ``email``, ``account_name``,
    ``parent_type`` values. Sources not in the result are candidates for
    CloudKit attribution by elimination.
    """
    conn = sqlite3.connect(f'file:{ACCOUNTS_DB}?mode=ro', uri=True)
    try:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT child.ZIDENTIFIER        AS uuid,
                   child.ZUSERNAME          AS email,
                   parent.ZACCOUNTDESCRIPTION AS account_name,
                   pt.ZACCOUNTTYPEDESCRIPTION AS parent_type
            FROM ZACCOUNT child
            JOIN ZACCOUNT parent ON child.ZPARENTACCOUNT = parent.Z_PK
            JOIN ZACCOUNTTYPE pt ON parent.ZACCOUNTTYPE = pt.Z_PK
            JOIN ZACCOUNTTYPE ct ON child.ZACCOUNTTYPE  = ct.Z_PK
            WHERE ct.ZIDENTIFIER = 'com.apple.account.CardDAV'
            """
        ).fetchall()
    finally:
        conn.close()

    return {
        cast(str, r['uuid']): {
            'email': r['email'],
            'account_name': r['account_name'],
            'parent_type': r['parent_type'],
        }
        for r in rows
    }


def _count_records(db_path: pathlib.Path) -> int:
    """Count ZABCDRECORD rows in a source database."""
    if not db_path.exists():
        return 0
    conn = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True)
    try:
        return cast(int, conn.execute('SELECT COUNT(*) FROM ZABCDRECORD').fetchone()[0])
    finally:
        conn.close()


def _assert_main_db_has_no_persons() -> None:
    """Fail fast if the main AddressBook DB contains person records.

    On observed systems the main DB holds only container metadata
    (ZUNIQUEID ending in ':ABContainer'); all contacts live in per-source
    DBs under Sources/. If this assumption is ever wrong, we want a loud
    failure rather than silent double-counting.
    """
    if not MAIN_DB.exists():
        return
    conn = sqlite3.connect(f'file:{MAIN_DB}?mode=ro', uri=True)
    try:
        count = conn.execute(
            """
            SELECT COUNT(*) FROM ZABCDRECORD
            WHERE ZFIRSTNAME IS NOT NULL
               OR ZLASTNAME IS NOT NULL
               OR ZORGANIZATION IS NOT NULL
               OR ZNICKNAME IS NOT NULL
            """
        ).fetchone()[0]
    finally:
        conn.close()

    if count > 0:
        msg = (
            f'Main AddressBook DB {MAIN_DB} unexpectedly contains {count} person record(s). '
            f'imessage-kit assumes the main DB holds only container metadata and all persons '
            f'live in per-source DBs. Update SourceRegistry to handle the main DB as a source.'
        )
        raise RuntimeError(msg)
