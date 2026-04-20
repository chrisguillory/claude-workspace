"""Extract plain text from iMessage attributedBody blobs.

Since macOS Ventura, 93%+ of messages store text in ``attributedBody``
(a binary NSMutableAttributedString serialized as typedstream) rather
than the ``text`` column. Edited messages always have NULL ``text``.

Algorithm combines three MIT-licensed implementations:
    - tszaks: dynamic 0x2b marker scan (most robust)
    - carterlasalle: little-endian variable-length integer (correct byte order)
    - anipotts: NSMutableString fallback + U+FFFC filtering
"""

from __future__ import annotations

__all__ = [
    'extract_text',
]

import struct


def extract_text(blob: bytes) -> str | None:
    """Extract plain text from an attributedBody typedstream blob.

    Returns None if the blob cannot be parsed or contains no text.
    """
    if not blob:
        return None

    # Step 1: Find NSString marker, fallback to NSMutableString
    marker = b'NSString'
    pos = blob.find(marker)
    if pos == -1:
        marker = b'NSMutableString'
        pos = blob.find(marker)
    if pos == -1:
        return None

    # Step 2: Dynamic 0x2b scan from marker end (not fixed 5-byte preamble)
    search_start = pos + len(marker)
    tag_pos = blob.find(b'\x2b', search_start, search_start + 20)
    if tag_pos == -1 or tag_pos >= len(blob) - 1:
        return None

    # Step 3: Read variable-length little-endian integer
    offset = tag_pos + 1
    if offset >= len(blob):
        return None

    length_byte = blob[offset]
    offset += 1

    if length_byte < 0x80:
        text_length = length_byte
    elif length_byte == 0x81:
        if offset + 2 > len(blob):
            return None
        text_length = struct.unpack_from('<H', blob, offset)[0]
        offset += 2
    elif length_byte == 0x82:
        if offset + 3 > len(blob):
            return None
        text_length = int.from_bytes(blob[offset : offset + 3], 'little')
        offset += 3
    elif length_byte == 0x83:
        if offset + 4 > len(blob):
            return None
        text_length = struct.unpack_from('<I', blob, offset)[0]
        offset += 4
    else:
        return None

    # Step 4: Extract UTF-8 text
    if offset + text_length > len(blob):
        return None
    try:
        text = blob[offset : offset + text_length].decode('utf-8')
    except UnicodeDecodeError:
        return None

    # Step 5: Filter U+FFFC (object replacement character for attachments)
    text = text.replace('\ufffc', '').strip()

    return text if text else None
