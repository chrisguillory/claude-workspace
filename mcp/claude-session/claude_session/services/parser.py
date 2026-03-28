"""
Session parser service - JSONL file parsing and validation.

Framework-agnostic service for loading and validating Claude Code session files.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path

from claude_session.schemas.session import SessionRecord
from claude_session.schemas.session.models import validate_session_record

__all__ = [
    'SessionParserService',
    'logger',
]


logger = logging.getLogger(__name__)

# -- Session Parser Service ----------------------------------------------------


class SessionParserService:
    """
    Service for parsing Claude Code session JSONL files.

    Pure domain logic - loads and validates JSONL files into typed SessionRecord objects.
    """

    async def load_session_files(self, session_files: Sequence[Path]) -> Mapping[str, Sequence[SessionRecord]]:
        """
        Load and parse session JSONL files.

        Args:
            session_files: Sequence of JSONL file paths
        Returns:
            Dict mapping filename to list of validated records
        """
        files_data: dict[str, Sequence[SessionRecord]] = {}

        for file_path in session_files:
            logger.info('Loading %s', file_path.name)

            records = await self._parse_jsonl_file(file_path)
            files_data[file_path.name] = records

            logger.info('Loaded %d records from %s', len(records), file_path.name)

        return files_data

    async def _parse_jsonl_file(
        self, file_path: Path
    ) -> Sequence[SessionRecord]:
        """
        Parse a single JSONL file into validated records.

        Args:
            file_path: Path to JSONL file
        Returns:
            List of validated SessionRecord objects
        """
        records = []

        with open(file_path, encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                # Parse JSON (fail fast on error)
                raw_data = json.loads(line)

                # Validate with type-dispatch (fast path, better error messages)
                record = validate_session_record(raw_data)
                records.append(record)

        return records
