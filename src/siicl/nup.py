"""Reader for NUP dump containers.

The layout was worked out from one sample and is described in
docs/nup-format.md. Values this module does not understand are kept in
`Nup.unknown` so they can be compared across samples.
"""

from __future__ import annotations

import hashlib
import struct
from compression import zstd
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, Self

MAGIC = b"NUP"
TOC_MAGIC = b"TOC"
HEADER_SIZE = 0x80
TOC_DIRS_OFFSET = 0x70
DIR_RECORD_SIZE = 0x18
FILE_ENTRY_SIZE = 0x50

COMPRESSION_NONE = 0
COMPRESSION_ZSTD = 1
COMPRESSION_NAMES = {COMPRESSION_NONE: "none", COMPRESSION_ZSTD: "zstd"}

METADATA_KINDS = {1: "NPDM copy", 3: "JPEG image", 5: "text", 8: "text"}

DEFAULT_MAX_READ = 2 << 30


class NupError(ValueError):
    """The file is not a NUP container, or it is damaged."""


@dataclass(frozen=True)
class NupFile:
    path: str
    size: int
    offset: int
    stored_size: int
    sha256: bytes
    compression: int

    @property
    def name(self) -> str:
        return self.path.rsplit("/", 1)[-1]


@dataclass(frozen=True)
class MetadataRecord:
    kind: int
    data: bytes

    @property
    def description(self) -> str:
        return METADATA_KINDS.get(self.kind, "unknown")


@dataclass
class Nup:
    version: int
    title_id: int
    toc_sha256: bytes
    data_offset: int
    file_size: int
    metadata_offset: int
    metadata_size: int
    files: list[NupFile]
    unknown: dict[str, int] = field(default_factory=dict)


def _u32(buf: bytes, off: int) -> int:
    return struct.unpack_from("<I", buf, off)[0]


def _u64(buf: bytes, off: int) -> int:
    return struct.unpack_from("<Q", buf, off)[0]


def parse(header: bytes, toc: bytes) -> Nup:
    """Parse the header and the table of contents."""
    try:
        return _parse(header, toc)
    except struct.error as e:
        raise NupError(f"truncated NUP structure: {e}") from None


def _parse(header: bytes, toc: bytes) -> Nup:
    if len(header) < HEADER_SIZE or header[:3] != MAGIC:
        raise NupError("not a NUP container (magic is not 'NUP')")
    toc_sha = header[0x18:0x38]
    if hashlib.sha256(toc).digest() != toc_sha:
        raise NupError("table of contents does not match its SHA-256 hash")
    if toc[:3] != TOC_MAGIC:
        raise NupError("table of contents magic is not 'TOC'")

    dir_count = _u32(toc, 0x08)
    file_count = _u64(toc, 0x10)
    str_off, str_size = _u64(toc, 0x18), _u64(toc, 0x20)
    if dir_count < 1 or str_off + str_size > len(toc):
        raise NupError("table of contents has bad counts or offsets")
    files_off = TOC_DIRS_OFFSET + dir_count * DIR_RECORD_SIZE
    if files_off + file_count * FILE_ENTRY_SIZE > str_off:
        raise NupError("file entries overlap the string table")
    strings = toc[str_off : str_off + str_size]

    def name_at(off: int, length: int) -> str:
        if off + length > len(strings):
            raise NupError("name is outside the string table")
        return strings[off : off + length].decode("utf-8", "replace")

    dirs = []
    for i in range(dir_count):
        o = TOC_DIRS_OFFSET + i * DIR_RECORD_SIZE
        noff, nlen, first, count = struct.unpack_from("<IIII", toc, o)
        dirs.append((name_at(noff, nlen), first, count))

    raw_files = []
    for i in range(file_count):
        o = files_off + i * FILE_ENTRY_SIZE
        noff, nlen, size, offset, stored = struct.unpack_from("<IIQQQ", toc, o)
        digest = toc[o + 0x20 : o + 0x40]
        compression = _u32(toc, o + 0x40)
        raw_files.append((name_at(noff, nlen), size, offset, stored, digest, compression))

    # Build full paths by walking the directory tree from the root.
    paths: dict[int, str] = {}
    sub_dirs = dir_count - 1
    seen: set[int] = set()

    def walk(dir_index: int, prefix: str) -> None:
        if dir_index in seen:
            raise NupError("directory tree has a cycle")
        seen.add(dir_index)
        _, first, count = dirs[dir_index]
        for node in range(first, first + count):
            if node < sub_dirs:
                child = node + 1
                walk(child, f"{prefix}{dirs[child][0]}/")
            elif node - sub_dirs < file_count:
                paths[node - sub_dirs] = prefix + raw_files[node - sub_dirs][0]
            else:
                raise NupError("directory refers to a node that does not exist")

    walk(0, "")
    files = [
        NupFile(paths.get(i, name), size, offset, stored, digest, comp)
        for i, (name, size, offset, stored, digest, comp) in enumerate(raw_files)
    ]

    return Nup(
        version=header[3],
        title_id=_u64(toc, 0x30),
        toc_sha256=toc_sha,
        data_offset=_u64(header, 0x38),
        file_size=_u64(header, 0x40),
        metadata_offset=_u64(toc, 0x50),
        metadata_size=_u64(toc, 0x58),
        files=files,
        unknown={
            "header_0x06": struct.unpack_from("<H", header, 0x06)[0],
            "header_0x48": _u32(header, 0x48),
            "toc_0x04": _u32(toc, 0x04),
            "toc_0x3c": _u32(toc, 0x3C),
            "toc_0x40": _u32(toc, 0x40),
            "toc_0x44": _u32(toc, 0x44),
            "toc_0x48": _u32(toc, 0x48),
        },
    )


def parse_metadata(block: bytes) -> list[MetadataRecord]:
    records = []
    off = 0
    while off + 8 <= len(block):
        kind, size = struct.unpack_from("<II", block, off)
        if off + 8 + size > len(block):
            raise NupError("metadata record runs past the end of the block")
        records.append(MetadataRecord(kind, block[off + 8 : off + 8 + size]))
        off += 8 + size
    return records


class NupReader:
    """Open a NUP container and read its files."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._f: BinaryIO = open(self.path, "rb")
        try:
            header = self._f.read(HEADER_SIZE)
            if len(header) < HEADER_SIZE or header[:3] != MAGIC:
                raise NupError(f"{self.path} is not a NUP container")
            toc_off, toc_size = _u64(header, 0x08), _u64(header, 0x10)
            if toc_size > 64 << 20:
                raise NupError("table of contents is implausibly large")
            self._f.seek(toc_off)
            self.info = parse(header, self._f.read(toc_size))
        except BaseException:
            self._f.close()
            raise

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._f.close()

    @property
    def files(self) -> list[NupFile]:
        return self.info.files

    def find(self, path: str) -> NupFile:
        for entry in self.info.files:
            if entry.path == path:
                return entry
        raise KeyError(path)

    def read(self, entry: NupFile, *, max_size: int = DEFAULT_MAX_READ) -> bytes:
        """Return the file's data, decompressed and checked against its hash."""
        if entry.size > max_size or entry.stored_size > max_size:
            raise NupError(f"{entry.path} is {entry.size} bytes, over the {max_size}-byte read limit")
        self._f.seek(entry.offset)
        raw = self._f.read(entry.stored_size)
        if len(raw) != entry.stored_size:
            raise NupError(f"{entry.path} runs past the end of the container")
        if entry.compression == COMPRESSION_ZSTD:
            try:
                data = zstd.decompress(raw)
            except zstd.ZstdError as e:
                raise NupError(f"{entry.path}: zstd data is damaged: {e}") from None
        elif entry.compression == COMPRESSION_NONE:
            data = raw
        else:
            raise NupError(f"{entry.path} uses unknown compression {entry.compression}")
        if len(data) != entry.size:
            raise NupError(f"{entry.path} decompressed to {len(data)} bytes, expected {entry.size}")
        if hashlib.sha256(data).digest() != entry.sha256:
            raise NupError(f"{entry.path} does not match its SHA-256 hash")
        return data

    def peek(self, entry: NupFile, n: int) -> bytes:
        """Return the first `n` stored bytes of an entry, without decompressing or checking it."""
        self._f.seek(entry.offset)
        return self._f.read(min(n, entry.stored_size))

    def metadata(self) -> list[MetadataRecord]:
        if not self.info.metadata_size:
            return []
        self._f.seek(self.info.metadata_offset)
        return parse_metadata(self._f.read(self.info.metadata_size))
