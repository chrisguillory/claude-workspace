"""Mach-O binary inspection — code signature fields without the codesign subprocess."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    'CodeDirectory',
    'MachOSignature',
]


@dataclass(frozen=True)
class CodeDirectory:
    """Parsed CodeDirectory blob fields."""

    flags: int
    identifier: str


class MachOSignature:
    """Read Mach-O code-signature fields from disk.

    ``None`` return = not a 64-bit Mach-O or no embedded CodeDirectory blob.
    """

    MH_MAGIC_64 = 0xFEEDFACF
    LC_CODE_SIGNATURE = 0x1D
    CS_ADHOC = 0x2
    CSMAGIC_EMBEDDED_SIGNATURE = 0xFADE0CC0
    CSMAGIC_CODEDIRECTORY = 0xFADE0C02

    @classmethod
    def is_adhoc(cls, path: Path) -> bool | None:
        """True if the binary is ad-hoc signed (patched); False if Anthropic-signed."""
        cd = cls.read(path)
        return None if cd is None else bool(cd.flags & cls.CS_ADHOC)

    @classmethod
    def identifier(cls, path: Path) -> str | None:
        """Codesign Identifier (e.g., ``com.anthropic.claude-code``)."""
        cd = cls.read(path)
        return None if cd is None else cd.identifier

    @classmethod
    def read(cls, path: Path) -> CodeDirectory | None:
        """Parse the CodeDirectory blob. ``None`` if the file lacks one."""
        with path.open('rb') as f:
            magic = struct.unpack('<I', f.read(4))[0]
            if magic != cls.MH_MAGIC_64:
                return None
            f.seek(16)
            ncmds = struct.unpack('<I', f.read(4))[0]
            f.seek(32)
            for _ in range(ncmds):
                pos = f.tell()
                cmd, cmdsize = struct.unpack('<II', f.read(8))
                if cmd == cls.LC_CODE_SIGNATURE:
                    sig_offset = struct.unpack('<I', f.read(4))[0]
                    f.seek(sig_offset)
                    if struct.unpack('>I', f.read(4))[0] != cls.CSMAGIC_EMBEDDED_SIGNATURE:
                        return None
                    _length, count = struct.unpack('>II', f.read(8))
                    for _ in range(count):
                        blob_type, blob_offset = struct.unpack('>II', f.read(8))
                        if blob_type == 0:  # CSSLOT_CODEDIRECTORY
                            cd_start = sig_offset + blob_offset
                            f.seek(cd_start)
                            if struct.unpack('>I', f.read(4))[0] != cls.CSMAGIC_CODEDIRECTORY:
                                return None
                            _cd_length, _cd_version, cd_flags = struct.unpack('>III', f.read(12))
                            _hash_offset, ident_offset = struct.unpack('>II', f.read(8))
                            f.seek(cd_start + ident_offset)
                            ident = bytearray()
                            while (b := f.read(1)) and b != b'\x00':
                                ident.extend(b)
                            return CodeDirectory(flags=cd_flags, identifier=ident.decode('utf-8'))
                    return None
                f.seek(pos + cmdsize)
        return None
