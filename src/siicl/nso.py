"""Reader for NSO executables (magic `NSO0`), the format of exefs modules."""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass

from siicl import lz4

MAGIC = b"NSO0"
HEADER_SIZE = 0x100
SEGMENT_NAMES = ("text", "rodata", "data")


class NsoError(ValueError):
    """The data is not an NSO executable, or it is damaged."""


@dataclass(frozen=True)
class Segment:
    name: str
    file_offset: int
    memory_offset: int
    size: int
    stored_size: int
    compressed: bool
    hashed: bool
    sha256: bytes


@dataclass
class Nso:
    flags: int
    module_id: bytes
    bss_size: int
    segments: dict[str, Segment]
    raw: bytes

    def segment_data(self, name: str) -> bytes:
        """Return a segment's bytes, decompressed and checked against its hash."""
        seg = self.segments[name]
        end = seg.file_offset + seg.stored_size
        if end > len(self.raw):
            raise NsoError(f"{name} segment runs past the end of the file")
        stored = self.raw[seg.file_offset : end]
        data = lz4.decompress_block(stored, seg.size) if seg.compressed else stored
        if len(data) != seg.size:
            raise NsoError(f"{name} segment is {len(data)} bytes, expected {seg.size}")
        if seg.hashed and hashlib.sha256(data).digest() != seg.sha256:
            raise NsoError(f"{name} segment does not match its SHA-256 hash")
        return data


def parse(raw: bytes) -> Nso:
    if len(raw) < HEADER_SIZE or raw[:4] != MAGIC:
        raise NsoError("not an NSO executable (magic is not 'NSO0')")
    flags = struct.unpack_from("<I", raw, 0x0C)[0]
    stored_sizes = struct.unpack_from("<III", raw, 0x60)
    segments = {}
    for i, name in enumerate(SEGMENT_NAMES):
        file_off, mem_off, size = struct.unpack_from("<III", raw, 0x10 + i * 0x10)
        compressed = bool(flags & (1 << i))
        segments[name] = Segment(
            name=name,
            file_offset=file_off,
            memory_offset=mem_off,
            size=size,
            stored_size=stored_sizes[i] if compressed else size,
            compressed=compressed,
            hashed=bool(flags & (1 << (i + 3))),
            sha256=raw[0xA0 + i * 0x20 : 0xC0 + i * 0x20],
        )
    return Nso(
        flags=flags,
        module_id=raw[0x40:0x60],
        bss_size=struct.unpack_from("<I", raw, 0x3C)[0],
        segments=segments,
        raw=raw,
    )
