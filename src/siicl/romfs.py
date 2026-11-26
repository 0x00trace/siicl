"""Build a RomFS image from a list of files, in the layout the game's file system library reads.

A NUP container stores the romfs as separate files, but a game reads its romfs as one image (from
OpenDataStorageByCurrentProcess) and finds files in it itself. siicl builds the image's header and
tables in memory and reads file data from the container only when the game asks for it, since a
romfs file can be many gigabytes.

The layout, as switchbrew's RomFS page describes it:

- a 0x50-byte header: its own size, then the offset and size of the directory hash table, the
  directory entries, the file hash table and the file entries, then the offset of the file data
- directory entries: parent, next sibling, first child directory, first file, next entry in the
  same hash bucket, name length, name (padded to 4 bytes); offsets are into the entry table, and
  0xFFFFFFFF means none. The root directory is the entry at offset 0 and has no name.
- file entries: parent, next sibling, data offset (from the start of the file data), size, next
  entry in the same hash bucket, name length, name
- hash tables: one entry offset per bucket, for the first entry in it
"""

from __future__ import annotations

import struct
from collections.abc import Callable
from dataclasses import dataclass, field

HEADER_SIZE = 0x50
NONE = 0xFFFF_FFFF
DATA_ALIGN = 0x10


@dataclass
class RomFile:
    path: str  # relative to the romfs root, "dir/name"
    size: int
    read: Callable[[int, int], bytes]  # (offset, size) -> data


def path_hash(parent: int, name: bytes) -> int:
    h = parent ^ 123456789
    for c in name:
        h = ((h >> 5) | (h << 27)) & 0xFFFF_FFFF
        h ^= c
    return h


def bucket_count(n: int) -> int:
    if n < 3:
        return 3
    if n < 19:
        return n | 1
    while any(n % p == 0 for p in (2, 3, 5, 7, 11, 13, 17)):
        n += 1
    return n


@dataclass
class _Dir:
    name: bytes
    parent: _Dir | None
    dirs: dict[bytes, _Dir] = field(default_factory=dict)
    files: list[tuple[bytes, RomFile]] = field(default_factory=list)
    offset: int = 0


def _pad4(b: bytes) -> bytes:
    return b + bytes(-len(b) % 4)


class RomFsImage:
    def __init__(self, files: list[RomFile]) -> None:
        root = _Dir(b"", None)
        for f in files:
            parts = [p.encode() for p in f.path.strip("/").split("/")]
            d = root
            for part in parts[:-1]:
                d = d.dirs.setdefault(part, _Dir(part, d))
            d.files.append((parts[-1], f))
        dirs: list[_Dir] = []
        todo = [root]
        while todo:  # breadth first, children sorted by name
            d = todo.pop(0)
            dirs.append(d)
            todo += [d.dirs[k] for k in sorted(d.dirs)]
        offset = 0
        for d in dirs:
            d.offset = offset
            offset += 0x18 + len(_pad4(d.name))
        entries = [(d, name, f) for d in dirs for name, f in sorted(d.files, key=lambda x: x[0])]
        file_offsets, offset, data_offset = [], 0, 0
        self._extents: list[tuple[int, RomFile]] = []
        for _, name, f in entries:
            file_offsets.append(offset)
            offset += 0x20 + len(_pad4(name))
            self._extents.append((data_offset, f))
            data_offset = (data_offset + f.size + DATA_ALIGN - 1) // DATA_ALIGN * DATA_ALIGN

        dir_buckets = [NONE] * bucket_count(len(dirs))
        dir_next: dict[int, int] = {}
        for d in dirs:
            parent = d.parent.offset if d.parent else 0
            b = path_hash(parent, d.name) % len(dir_buckets)
            dir_next[d.offset] = dir_buckets[b]
            dir_buckets[b] = d.offset
        file_buckets = [NONE] * bucket_count(len(entries))
        file_next: dict[int, int] = {}
        for (d, name, _), off in zip(entries, file_offsets, strict=True):
            b = path_hash(d.offset, name) % len(file_buckets)
            file_next[off] = file_buckets[b]
            file_buckets[b] = off

        first_file = {}
        for (d, _, _), off in zip(entries, file_offsets, strict=True):
            first_file.setdefault(d.offset, off)
        dir_table = bytearray()
        for d in dirs:
            siblings = [d.parent.dirs[k] for k in sorted(d.parent.dirs)] if d.parent else [d]
            i = siblings.index(d)
            sibling = siblings[i + 1].offset if i + 1 < len(siblings) else NONE
            child = d.dirs[min(d.dirs)].offset if d.dirs else NONE
            parent = d.parent.offset if d.parent else 0
            dir_table += struct.pack(
                "<IIIIII", parent, sibling, child, first_file.get(d.offset, NONE), dir_next[d.offset], len(d.name)
            )
            dir_table += _pad4(d.name)
        file_table = bytearray()
        for i, ((d, name, f), off) in enumerate(zip(entries, file_offsets, strict=True)):
            sibling = file_offsets[i + 1] if i + 1 < len(entries) and entries[i + 1][0] is d else NONE
            file_table += struct.pack(
                "<IIQQII", d.offset, sibling, self._extents[i][0], f.size, file_next[off], len(name)
            )
            file_table += _pad4(name)

        tables = [
            struct.pack(f"<{len(dir_buckets)}I", *dir_buckets),
            bytes(dir_table),
            struct.pack(f"<{len(file_buckets)}I", *file_buckets),
            bytes(file_table),
        ]
        pos, fields = HEADER_SIZE, []
        for t in tables:
            fields += [pos, len(t)]
            pos += len(t)
        self.data_offset = (pos + 0x1FF) // 0x200 * 0x200
        header = struct.pack("<10Q", HEADER_SIZE, *fields, self.data_offset)
        self._meta = (header + b"".join(tables)).ljust(self.data_offset, b"\0")
        self.size = self.data_offset + data_offset

    def read(self, offset: int, size: int) -> bytes:
        """size bytes from offset; past the end of the image, zeros."""
        out = bytearray()
        end = offset + size
        if offset < self.data_offset:
            out += self._meta[offset : min(end, self.data_offset)]
            offset = min(end, self.data_offset)
        while offset < end:
            rel = offset - self.data_offset
            extent = next(((start, f) for start, f in self._extents if start <= rel < start + f.size), None)
            if extent is None:
                nxt = min((s for s, _ in self._extents if s > rel), default=None)
                gap = (end - offset) if nxt is None else min(end - offset, nxt - rel)
                out += bytes(gap)
                offset += gap
                continue
            start, f = extent
            n = min(end - offset, start + f.size - rel)
            out += f.read(rel - start, n)
            offset += n
        return bytes(out)
