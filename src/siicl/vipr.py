"""The parts of VIPR resource files that SIICL needs.

A VIPR file holds one image in the engine's binary format. Its GPU data (shader
binaries, texture pixels) is not inside it: each piece is a separate archive
entry, stored just before the image in the archive. The image lists these
external buffers with their sizes. docs/vzip-format.md describes this.
"""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass, field

from siicl.vzip import VzipEntry, VzipReader

MAGIC = b"VIPR"
BACKENDS = {5: "NVN", 12: "NVN gen2"}
KINDS = {1: "buffer", 2: "texture", 4: "shader"}

# A buffer record: buffer count (tag 0xBFFF), buffer index (tag 0xCFFF), 8 zero bytes, size.
_BUFFER_RECORD = re.compile(rb"(.{4})\x00\x00\xff\xbf(.{4})\x00\x00\xff\xcf\x00{8}(.{8})", re.S)


@dataclass(frozen=True)
class GpuBufferType:
    backend: int
    kind: int

    @property
    def label(self) -> str:
        backend = BACKENDS.get(self.backend, f"backend {self.backend}")
        return f"{backend} {KINDS.get(self.kind, f'kind {self.kind}')}"


@dataclass(frozen=True)
class BufferRef:
    count: int
    index: int
    size: int


def gpu_buffer_type(data: bytes) -> GpuBufferType | None:
    """The GPU buffer type that the image header declares, if any."""
    if data[:4] != MAGIC or len(data) < 0x1C:
        return None
    (word,) = struct.unpack_from("<I", data, 0x18)
    if word == 0:
        return None
    return GpuBufferType((word >> 26) & 0xF, (word >> 22) & 7)


def buffer_refs(data: bytes) -> list[BufferRef]:
    """The image's external buffer records, in the order they appear."""
    refs = []
    for count, index, size in _BUFFER_RECORD.findall(data):
        refs.append(
            BufferRef(struct.unpack("<I", count)[0], struct.unpack("<I", index)[0], struct.unpack("<Q", size)[0])
        )
    return refs


@dataclass(frozen=True)
class Owner:
    resource: VzipEntry
    ref: BufferRef


@dataclass
class BufferMatch:
    owners: dict[int, Owner] = field(default_factory=dict)  # data offset of a buffer -> its image
    resources: int = 0
    refs: int = 0
    unmatched: list[tuple[VzipEntry, BufferRef]] = field(default_factory=list)


def match_buffers(reader: VzipReader) -> BufferMatch:
    """Match each image's buffer records to the entries stored just before the image.

    For every VIPR image, the candidates are the entries between the previous
    VIPR image and this one. A record matches the first unclaimed candidate of
    the same size, taking records in index order. Records with no match usually
    point at data that the packer stored only once, for an earlier image.
    """
    result = BufferMatch()
    run: list[VzipEntry] = []
    for e in reader.unique_data():
        if reader.head(e, 4) != MAGIC:
            run.append(e)
            continue
        refs = buffer_refs(reader.read(e))
        if refs:
            result.resources += 1
            result.refs += len(refs)
            free = list(run)
            for ref in sorted(refs, key=lambda r: r.index):
                hit = next((c for c in free if c.size == ref.size), None)
                if hit is None:
                    result.unmatched.append((e, ref))
                    continue
                free.remove(hit)
                result.owners[hit.offset] = Owner(e, ref)
        run = []
    return result
