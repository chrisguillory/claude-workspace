"""Resolve iMessage handles to contact names via macOS AddressBook.

Reads per-source AddressBook databases directly (Google, iCloud CardDAV,
iCloud CloudKit). Deduplicates contacts that appear in multiple sources
and tags each merged contact with its contributing source display names.
"""

from __future__ import annotations

__all__ = [
    'ContactResolver',
]

import logging
import re
import sqlite3
import time
from collections.abc import Mapping, Sequence

from thefuzz import fuzz  # type: ignore[import-untyped]  # no py.typed marker, stubs unpublished

from imessage_kit.types import Contact, ContactSource

logger = logging.getLogger(__name__)


class ContactResolver:
    """Resolve phone numbers and emails to contact display names.

    Loads contacts from the explicit source registry passed at construction.
    Deduplicates across sources by (first_name, last_name) with phone OR
    email overlap as the merge predicate. Each resulting Contact carries
    the list of source display names it was merged from.
    """

    CACHE_TTL_S = 300  # 5 minutes

    # Relevance scoring tiers for _match_score. Ordered descending.
    # 1.0 (used inline) is a perfect match; 0.0 is no match.
    SCORE_EXACT_NAME_PART = 0.95  # first OR last name exactly equals the query
    SCORE_SUBSTRING_IN_NAME = 0.85  # query appears as a substring in the full name
    FUZZY_MATCH_MIN_RATIO = 70  # fuzz.token_sort_ratio (0-100) floor below which we return no match

    def __init__(self, sources: Sequence[ContactSource]) -> None:
        self._sources = sources
        self._contacts: Sequence[Contact] | None = None
        self._handle_to_name: dict[str, str] = {}
        self._cache_time: float = 0.0

    def resolve(self, handle: str) -> str:
        """Resolve a handle (phone/email) to a display name, or return handle unchanged."""
        self._ensure_cache()
        return self._handle_to_name.get(_normalize_phone(handle), handle)

    def resolve_handle_map(self, handle_map: Mapping[int, str]) -> Mapping[int, str]:
        """Resolve all handles in a map to display names."""
        self._ensure_cache()
        return {
            rowid: self._handle_to_name.get(_normalize_phone(identifier), identifier)
            for rowid, identifier in handle_map.items()
        }

    def lookup(self, query: str, *, source: str | None = None) -> Sequence[Contact]:
        """Search contacts by name, phone, or email.

        Args:
            query: Name (fuzzy), phone number, or email to search for.
            source: Display name of a source to filter by (e.g., 'Google',
                'iCloud'). Matches case-insensitively against any of the
                contact's merged sources. Raises ValueError if the source
                does not exist in the registry.
        """
        self._ensure_cache()
        if self._contacts is None:
            return []

        if source is not None:
            source_lower = source.lower()
            known = {s.display_name.lower() for s in self._sources}
            if source_lower not in known:
                available = ', '.join(s.display_name for s in self._sources)
                msg = f'Unknown source {source!r}. Available: {available}'
                raise ValueError(msg)

        query_lower = query.lower()
        query_digits = _normalize_phone(query)
        results: list[tuple[float, Contact]] = []

        for contact in self._contacts:
            if source is not None:
                contact_sources_lower = {s.lower() for s in contact.sources}
                if source.lower() not in contact_sources_lower:
                    continue
            score = self._match_score(contact, query_lower, query_digits)
            if score > 0.0:
                results.append((score, contact))

        results.sort(key=lambda x: x[0], reverse=True)
        return [contact for _, contact in results]

    @property
    def is_accessible(self) -> bool:
        """True if the registry has at least one source with a readable DB."""
        return any(s.db_path.exists() for s in self._sources)

    @property
    def contact_count(self) -> int:
        """Total deduplicated contact count across all sources."""
        self._ensure_cache()
        return len(self._contacts) if self._contacts else 0

    def _ensure_cache(self) -> None:
        """Refresh contact cache if stale."""
        if self._contacts is not None and (time.monotonic() - self._cache_time) < self.CACHE_TTL_S:
            return
        self._load_all()

    def _load_all(self) -> None:
        """Load contacts from every source, dedup, and build the handle→name index."""
        per_source: list[tuple[ContactSource, Sequence[Contact]]] = [
            (source, _load_from_db(source)) for source in self._sources
        ]

        contacts = _dedup_across_sources(per_source)

        handle_to_name: dict[str, str] = {}
        for contact in contacts:
            name = contact.display_name
            if not name:
                continue
            for phone in contact.phone_numbers:
                normalized = _normalize_phone(phone)
                if normalized:
                    handle_to_name[normalized] = name
            for email in contact.emails:
                handle_to_name[email.lower()] = name

        self._contacts = contacts
        self._handle_to_name = handle_to_name
        self._cache_time = time.monotonic()
        logger.info(
            'Loaded %d unique contacts from %d source(s)',
            len(contacts),
            len(self._sources),
        )

    def _match_score(self, contact: Contact, query_lower: str, query_digits: str) -> float:
        """Score a contact against a search query. Returns 0.0 for no match."""
        # Phone match
        if query_digits and len(query_digits) >= 4:
            for phone in contact.phone_numbers:
                if query_digits in _normalize_phone(phone):
                    return 1.0

        # Email match
        for email in contact.emails:
            if query_lower in email.lower():
                return 1.0

        # Name matching
        name = contact.display_name
        if not name:
            return 0.0

        name_lower = name.lower()

        # Exact name match
        if query_lower == name_lower:
            return 1.0

        # First or last name exact match
        first = (contact.first_name or '').lower()
        last = (contact.last_name or '').lower()
        if query_lower in (first, last):
            return self.SCORE_EXACT_NAME_PART

        # Substring match
        if query_lower in name_lower:
            return self.SCORE_SUBSTRING_IN_NAME

        # Fuzzy match (token_sort_ratio avoids substring false positives)
        ratio: int = fuzz.token_sort_ratio(query_lower, name_lower)
        if ratio >= self.FUZZY_MATCH_MIN_RATIO:
            return ratio / 100.0

        return 0.0


def _load_from_db(source: ContactSource) -> Sequence[Contact]:
    """Load raw contacts from a single AddressBook database, tagged with the source name."""
    conn = sqlite3.connect(f'file:{source.db_path}?mode=ro', uri=True)
    conn.row_factory = sqlite3.Row

    try:
        records = conn.execute("""
            SELECT Z_PK, ZFIRSTNAME, ZLASTNAME, ZORGANIZATION, ZNICKNAME
            FROM ZABCDRECORD
        """).fetchall()

        phones_by_owner: dict[int, list[str]] = {}
        for row in conn.execute('SELECT ZOWNER, ZFULLNUMBER FROM ZABCDPHONENUMBER'):
            pk = row['ZOWNER']
            number = row['ZFULLNUMBER']
            if number:
                phones_by_owner.setdefault(pk, []).append(number)

        emails_by_owner: dict[int, list[str]] = {}
        for row in conn.execute('SELECT ZOWNER, ZADDRESS FROM ZABCDEMAILADDRESS'):
            pk = row['ZOWNER']
            addr = row['ZADDRESS']
            if addr:
                emails_by_owner.setdefault(pk, []).append(addr)

        contacts: list[Contact] = []
        for rec in records:
            pk = rec['Z_PK']
            first = rec['ZFIRSTNAME']
            last = rec['ZLASTNAME']

            if first and last:
                display = f'{first} {last}'
            elif first:
                display = first
            elif last:
                display = last
            elif rec['ZORGANIZATION']:
                display = rec['ZORGANIZATION']
            elif rec['ZNICKNAME']:
                display = rec['ZNICKNAME']
            else:
                display = None

            contacts.append(
                Contact(
                    display_name=display,
                    first_name=first,
                    last_name=last,
                    phone_numbers=phones_by_owner.get(pk, []),
                    emails=emails_by_owner.get(pk, []),
                    sources=[source.display_name],
                )
            )

        return contacts
    finally:
        conn.close()


def _dedup_across_sources(
    per_source: Sequence[tuple[ContactSource, Sequence[Contact]]],
) -> Sequence[Contact]:
    """Merge contacts appearing in multiple sources.

    Groups by (first_name, last_name) case-insensitively. Within each group,
    contacts merge when they share at least one normalized phone number
    OR one lowercased email. Merged contacts get the union of phones,
    emails, and source labels.

    Contacts without both first and last names are passed through unchanged
    (not merged), preserving organization-only and nickname-only records
    that may legitimately collide on name.
    """
    all_contacts: list[Contact] = []
    for _source, contacts in per_source:
        all_contacts.extend(contacts)

    groups: dict[tuple[str, str], list[Contact]] = {}
    passthrough: list[Contact] = []
    for contact in all_contacts:
        if contact.first_name and contact.last_name:
            key = (contact.first_name.lower(), contact.last_name.lower())
            groups.setdefault(key, []).append(contact)
        else:
            passthrough.append(contact)

    merged: list[Contact] = []
    for group in groups.values():
        merged.extend(_merge_by_handle_overlap(group))
    merged.extend(passthrough)
    return merged


def _merge_by_handle_overlap(group: Sequence[Contact]) -> Sequence[Contact]:
    """Merge contacts in a name-group when they share a phone or email.

    Union-find: two contacts are in the same component if they share any
    normalized phone or lowercased email. All components collapse into
    one merged Contact each.
    """
    if len(group) == 1:
        return group

    # Union-find
    parent = list(range(len(group)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        parent[find(i)] = find(j)

    # Build normalized handle sets per contact, then union those with overlap
    handles: list[set[str]] = []
    for c in group:
        phones = {_normalize_phone(p) for p in c.phone_numbers if _normalize_phone(p)}
        emails = {e.lower() for e in c.emails}
        handles.append(phones | emails)

    for i, hi in enumerate(handles):
        for j in range(i + 1, len(group)):
            if hi & handles[j]:
                union(i, j)

    # Collect components
    components: dict[int, list[int]] = {}
    for i in range(len(group)):
        components.setdefault(find(i), []).append(i)

    merged: list[Contact] = []
    for indices in components.values():
        representatives = [group[i] for i in indices]
        merged.append(_merge_contacts(representatives))
    return merged


def _merge_contacts(contacts: Sequence[Contact]) -> Contact:
    """Merge a list of Contact instances into one, unioning handles and sources."""
    if len(contacts) == 1:
        return contacts[0]

    # Deduplicate phones by normalized form; keep the first seen original formatting
    seen_phone_norms: set[str] = set()
    merged_phones: list[str] = []
    for c in contacts:
        for phone in c.phone_numbers:
            norm = _normalize_phone(phone)
            if norm and norm not in seen_phone_norms:
                seen_phone_norms.add(norm)
                merged_phones.append(phone)

    seen_emails: set[str] = set()
    merged_emails: list[str] = []
    for c in contacts:
        for email in c.emails:
            lower = email.lower()
            if lower not in seen_emails:
                seen_emails.add(lower)
                merged_emails.append(email)

    seen_sources: set[str] = set()
    merged_sources: list[str] = []
    for c in contacts:
        for src in c.sources:
            if src not in seen_sources:
                seen_sources.add(src)
                merged_sources.append(src)

    primary = contacts[0]
    return Contact(
        display_name=primary.display_name,
        first_name=primary.first_name,
        last_name=primary.last_name,
        phone_numbers=merged_phones,
        emails=merged_emails,
        sources=merged_sources,
    )


def _normalize_phone(phone: str) -> str:
    """Strip non-digits, remove leading +1 country code."""
    digits = re.sub(r'\D', '', phone)
    if len(digits) == 11 and digits.startswith('1'):
        digits = digits[1:]
    return digits
