"""Access iMessage attachments with HEIC conversion and base64 encoding."""

from __future__ import annotations

__all__ = [
    'AttachmentHandler',
]

import base64
import logging
import pathlib
import shutil
import subprocess
import tempfile
from collections.abc import Set

from imessage_kit.types import AttachmentSave, AttachmentView

logger = logging.getLogger(__name__)


class AttachmentHandler:
    """Read attachment files, convert HEIC, encode for delivery."""

    HEIC_TYPES: Set[str] = {'image/heic', 'image/heif'}
    MAX_VIEW_BYTES = 10 * 1024 * 1024  # 10 MB cap for base64 inline delivery

    def __init__(self, temp_dir: pathlib.Path) -> None:
        self._temp_dir = temp_dir

    def resolve_path(self, db_filename: str | None) -> pathlib.Path | None:
        """Expand ~/Library/... paths from chat.db to absolute paths."""
        if not db_filename:
            return None
        if db_filename.startswith('~'):
            return pathlib.Path(db_filename).expanduser()
        return pathlib.Path(db_filename)

    def view(
        self,
        path: pathlib.Path,
        mime_type: str | None,
        attachment_id: int,
    ) -> AttachmentView:
        """Read an image and return base64 for Claude vision. HEIC auto-converted to JPEG."""
        if not path.exists():
            msg = f'File not on disk — may be iCloud-offloaded: {path}'
            raise FileNotFoundError(msg)

        file_size = path.stat().st_size
        if file_size > self.MAX_VIEW_BYTES:
            msg = f'File too large for inline viewing ({file_size:,} bytes, max {self.MAX_VIEW_BYTES:,}). Use mode="save" instead.'
            raise ValueError(msg)

        original_mime = mime_type
        if mime_type in self.HEIC_TYPES:
            data = self._convert_heic_to_jpeg(path)
            delivered_mime = 'image/jpeg'
        else:
            data = path.read_bytes()
            delivered_mime = mime_type or 'application/octet-stream'

        return AttachmentView(
            attachment_id=attachment_id,
            filename=path.name,
            original_mime_type=original_mime,
            delivered_mime_type=delivered_mime,
            total_bytes=len(data),
            base64_data=base64.b64encode(data).decode('ascii'),
        )

    def save(
        self,
        path: pathlib.Path,
        mime_type: str | None,
        attachment_id: int,
    ) -> AttachmentSave:
        """Copy native file to temp dir. No conversion."""
        if not path.exists():
            msg = f'File not on disk — may be iCloud-offloaded: {path}'
            raise FileNotFoundError(msg)

        dest = self._temp_dir / path.name
        # Handle name collisions
        counter = 1
        while dest.exists():
            dest = self._temp_dir / f'{path.stem}_{counter}{path.suffix}'
            counter += 1

        shutil.copy2(path, dest)
        logger.info('Saved attachment to %s', dest)

        return AttachmentSave(
            attachment_id=attachment_id,
            filename=path.name,
            mime_type=mime_type,
            total_bytes=dest.stat().st_size,
            saved_path=str(dest),
        )

    def _convert_heic_to_jpeg(self, heic_path: pathlib.Path) -> bytes:
        """Convert HEIC/HEIF to JPEG using macOS sips."""
        with tempfile.NamedTemporaryFile(suffix='.jpeg', delete=False, dir=self._temp_dir) as tmp:
            tmp_path = pathlib.Path(tmp.name)

        try:
            subprocess.run(
                ['sips', '-s', 'format', 'jpeg', str(heic_path), '--out', str(tmp_path)],
                check=True,
                capture_output=True,
                timeout=30,
            )
            return tmp_path.read_bytes()
        finally:
            tmp_path.unlink(missing_ok=True)
