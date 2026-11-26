"""Reader for VZIP archives.

A VZIP archive is a ZIP64 file with an index in front of its first entry. The
index maps each name to its data, and several names can share one copy of the
data. Compressed data is a sequence of linked LZ4 blocks. The layout was worked
out from one sample and is described in docs/vzip-format.md.
"""

from __future__ import annotations

import bisect
import re
import struct
import zlib
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import BinaryIO, Self

from siicl import lz4, nup

MAGIC = b"VZIP"
HEADER_SIZE = 12
RECORD_SIZE = 40
BLOCK_SIZE = 0x10000
COMPRESSION_NONE = 0
COMPRESSION_LZ4_LINKED = 1
COMPRESSION_NAMES = {COMPRESSION_NONE: "none", COMPRESSION_LZ4_LINKED: "lz4"}
DEFAULT_MAX_READ = 1 << 30

ZIP_LOCAL = b"PK\x03\x04"
ZIP_CENTRAL = b"PK\x01\x02"
ZIP_EOCD = b"PK\x05\x06"
ZIP64_LOCATOR = b"PK\x06\x07"
ZIP64_EOCD = b"PK\x06\x06"
SAFE_NAME = re.compile(r"[0-9A-Za-z_][0-9A-Za-z_.-]{0,127}")
PATH_STRING = re.compile(rb"/?data/[\x21-\x7e][\x20-\x7e]{1,300}?(?=\x00)")

_FNV64_OFFSET, _FNV64_PRIME = 0xCBF29CE484222325, 0x100000001B3
_FNV32_OFFSET, _FNV32_PRIME = 0x811C9DC5, 0x01000193


def fnv1a64(data: bytes) -> int:
    h = _FNV64_OFFSET
    for c in data:
        h = ((h ^ c) * _FNV64_PRIME) & 0xFFFFFFFFFFFFFFFF
    return h


def fnv1a32(data: bytes) -> int:
    h = _FNV32_OFFSET
    for c in data:
        h = ((h ^ c) * _FNV32_PRIME) & 0xFFFFFFFF
    return h


def name_for_path(path: str) -> str:
    """The archive's name for an asset path.

    This is FNV-1a 64 of the path without its leading '/', with the letters A-Z
    lowercased (as the game's lookup does), written in hex.
    """
    p = path[1:] if path.startswith("/") else path
    return format(fnv1a64(bytes(c + 0x20 if 0x41 <= c <= 0x5A else c for c in p.encode())), "x")


def key_for_name(name: str) -> int:
    """The index key for a name: FNV-1a 32 of the name's text."""
    return fnv1a32(name.encode())


class VzipError(ValueError):
    """The data is not a VZIP archive, or it is damaged."""


@dataclass(frozen=True)
class VzipEntry:
    index: int
    name: str
    key: int
    offset: int
    stored_size: int
    size: int
    compression: int
    flags: int


@dataclass(frozen=True)
class ZipEntry:
    name: str
    crc32: int
    stored_size: int
    size: int
    header_offset: int


def file_type(head: bytes) -> str:
    """A short label for a file's type, taken from its first bytes."""
    if head[:4] == b"RIFF" and head[8:12] == b"WAVE":
        return "RIFF/WAVE"
    if len(head) >= 4 and all(32 <= c < 127 for c in head[:4]):
        return head[:4].decode("ascii")
    sample = head[:64]
    if sample and all(32 <= c < 127 or c in (9, 10, 13) for c in sample):
        first = sample.lstrip()[:1].decode("ascii")
        return f"text ({first})" if first in "{[<" else "text"
    return "0x" + head[:4].hex() if head else "empty"


class VzipReader:
    """Read a VZIP archive stored at `base` in an open file."""

    def __init__(self, f: BinaryIO, base: int = 0, size: int | None = None, *, owns_file: bool = False) -> None:
        self._f = f
        self.base = base
        if size is None:
            size = f.seek(0, 2) - base
        self.size = size
        self._owns_file = owns_file
        try:
            self._read_index()
        except BaseException:
            self.close()
            raise

    @classmethod
    def open(cls, path: str | Path) -> Self:
        """Open a VZIP file, or the VZIP archive inside a NUP container's romfs."""
        path = Path(path)
        with open(path, "rb") as f:
            magic = f.read(4)
        if magic == MAGIC:
            return cls(open(path, "rb"), owns_file=True)
        if magic[:3] != nup.MAGIC:
            raise VzipError(f"{path} is neither a VZIP archive nor a NUP container")
        with nup.NupReader(path) as r:
            found = [
                e
                for e in r.files
                if e.compression == nup.COMPRESSION_NONE and e.path.startswith("romfs/") and r.peek(e, 4) == MAGIC
            ]
        if not found:
            raise VzipError(f"{path} has no VZIP archive in its romfs")
        return cls(open(path, "rb"), found[0].offset, found[0].size, owns_file=True)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        if self._owns_file:
            self._f.close()

    # -- low-level reads -------------------------------------------------------

    def read_at(self, offset: int, n: int) -> bytes:
        if offset < 0 or n < 0 or offset + n > self.size:
            raise VzipError(f"read of {n} bytes at {offset:#x} is outside the archive")
        self._f.seek(self.base + offset)
        data = self._f.read(n)
        if len(data) != n:
            raise VzipError(f"archive is truncated at {offset:#x}")
        return data

    # -- index -----------------------------------------------------------------

    def _read_index(self) -> None:
        magic, count, strings_size = struct.unpack("<4sII", self.read_at(0, HEADER_SIZE))
        if magic != MAGIC:
            raise VzipError("not a VZIP archive (magic is not 'VZIP')")
        records = self.read_at(HEADER_SIZE, count * RECORD_SIZE)
        strings = self.read_at(HEADER_SIZE + count * RECORD_SIZE, strings_size)
        self.index_size = HEADER_SIZE + count * RECORD_SIZE + strings_size
        entries = []
        for i, (name_off, flags, offset, stored, size, comp, key) in enumerate(struct.iter_unpack("<IIQQQII", records)):
            end = strings.find(b"\0", name_off)
            if name_off >= len(strings) or end < 0:
                raise VzipError(f"index record {i} has a name outside the string table")
            if offset + stored > self.size:
                raise VzipError(f"index record {i} points outside the archive")
            name = strings[name_off:end].decode("ascii", "replace")
            entries.append(VzipEntry(i, name, key, offset, stored, size, comp, flags))
        self.entries: list[VzipEntry] = entries
        self.by_name: dict[str, VzipEntry] = {e.name: e for e in entries}
        self._keys = [e.key for e in entries]

    @property
    def keys_sorted(self) -> bool:
        return all(a <= b for a, b in zip(self._keys, self._keys[1:], strict=False))

    def find(self, name: str) -> VzipEntry:
        try:
            return self.by_name[name]
        except KeyError:
            raise KeyError(name) from None

    def find_path(self, path: str) -> VzipEntry:
        """The entry for an asset path such as 'data/textures/a.png.import'."""
        return self.find(name_for_path(path))

    def lookup(self, key: int) -> list[VzipEntry]:
        """Entries with this 32-bit key (the index is sorted by key)."""
        lo = bisect.bisect_left(self._keys, key)
        hi = bisect.bisect_right(self._keys, key)
        return self.entries[lo:hi]

    def unique_data(self) -> list[VzipEntry]:
        """One entry per stored copy of data, in archive order."""
        seen: dict[int, VzipEntry] = {}
        for e in self.entries:
            seen.setdefault(e.offset, e)
        return sorted(seen.values(), key=lambda e: e.offset)

    def aliases(self, entry: VzipEntry) -> list[VzipEntry]:
        return [e for e in self.entries if e.offset == entry.offset]

    # -- ZIP part ----------------------------------------------------------------

    @cached_property
    def zip_entries(self) -> dict[str, ZipEntry]:
        """The ZIP central directory, by name."""
        tail_len = min(self.size, 0xFFFF + 22 + 20)
        tail = self.read_at(self.size - tail_len, tail_len)
        i = tail.rfind(ZIP_EOCD)
        if i < 0:
            raise VzipError("no ZIP end-of-central-directory record")
        _, _, _, _, count, cd_size, cd_off, _ = struct.unpack_from("<IHHHHIIH", tail, i)
        if count == 0xFFFF or cd_size == 0xFFFFFFFF or cd_off == 0xFFFFFFFF:
            j = i - 20
            if j < 0 or tail[j : j + 4] != ZIP64_LOCATOR:
                raise VzipError("ZIP64 end-of-central-directory locator is missing")
            (z64_off,) = struct.unpack_from("<Q", tail, j + 8)
            rec = self.read_at(z64_off, 56)
            if rec[:4] != ZIP64_EOCD:
                raise VzipError("ZIP64 end-of-central-directory record is missing")
            count, cd_size, cd_off = struct.unpack_from("<QQQ", rec, 32)
        cd = self.read_at(cd_off, cd_size)
        out: dict[str, ZipEntry] = {}
        o = 0
        for _ in range(count):
            if cd[o : o + 4] != ZIP_CENTRAL:
                raise VzipError(f"bad central directory entry at {cd_off + o:#x}")
            crc, csz, usz, nlen, xlen, clen = struct.unpack_from("<IIIHHH", cd, o + 16)
            (lho,) = struct.unpack_from("<I", cd, o + 42)
            name = cd[o + 46 : o + 46 + nlen].decode("ascii", "replace")
            extra = cd[o + 46 + nlen : o + 46 + nlen + xlen]
            x = 0
            while x + 4 <= len(extra):
                tag, sz = struct.unpack_from("<HH", extra, x)
                if tag == 0x0001:
                    vals = iter(struct.unpack_from(f"<{sz // 8}Q", extra, x + 4))
                    if usz == 0xFFFFFFFF:
                        usz = next(vals)
                    if csz == 0xFFFFFFFF:
                        csz = next(vals)
                    if lho == 0xFFFFFFFF:
                        lho = next(vals)
                x += 4 + sz
            out[name] = ZipEntry(name, crc, csz, usz, lho)
            o += 46 + nlen + xlen + clen
        return out

    @cached_property
    def _crc_by_offset(self) -> dict[int, int]:
        zips = self.zip_entries
        return {e.offset: zips[e.name].crc32 for e in self.entries if e.name in zips}

    def crc32(self, entry: VzipEntry) -> int | None:
        """The CRC-32 the ZIP central directory gives for this entry's data, if any."""
        return self._crc_by_offset.get(entry.offset)

    # -- data --------------------------------------------------------------------

    def _blocks(self, raw: bytes):
        p = 0
        while p < len(raw):
            if p + 4 > len(raw):
                raise VzipError("LZ4 block header runs past the end of the data")
            (n,) = struct.unpack_from("<I", raw, p)
            if p + 4 + n > len(raw):
                raise VzipError("LZ4 block runs past the end of the data")
            yield raw[p + 4 : p + 4 + n]
            p += 4 + n

    def read(self, entry: VzipEntry, *, verify: bool = True, max_size: int = DEFAULT_MAX_READ) -> bytes:
        """Return an entry's data, decompressed and (by default) checked against its CRC-32."""
        if entry.size > max_size:
            raise VzipError(f"{entry.name} is {entry.size} bytes, over the {max_size}-byte read limit")
        raw = self.read_at(entry.offset, entry.stored_size)
        if entry.compression == COMPRESSION_NONE:
            data = raw
        elif entry.compression == COMPRESSION_LZ4_LINKED:
            try:
                data = lz4.decompress_linked(self._blocks(raw), entry.size, BLOCK_SIZE)
            except ValueError as e:
                raise VzipError(f"{entry.name}: {e}") from None
        else:
            raise VzipError(f"{entry.name} uses unknown compression {entry.compression}")
        if len(data) != entry.size:
            raise VzipError(f"{entry.name} is {len(data)} bytes, expected {entry.size}")
        if verify:
            crc = self.crc32(entry)
            if crc is not None and zlib.crc32(data) != crc:
                raise VzipError(f"{entry.name} does not match its CRC-32")
        return data

    def head(self, entry: VzipEntry, n: int = 64) -> bytes:
        """The first `n` bytes of an entry's data, decoding as little as possible."""
        n = min(n, entry.size)
        if entry.compression != COMPRESSION_LZ4_LINKED:
            return self.read_at(entry.offset, min(n, entry.stored_size))
        (first,) = struct.unpack("<I", self.read_at(entry.offset, 4))
        try:
            return lz4.decompress_head(self.read_at(entry.offset + 4, first), n)
        except ValueError as e:
            raise VzipError(f"{entry.name}: {e}") from None


def recover_paths(reader: VzipReader, scan_bytes: int = 0x10000) -> dict[str, str]:
    """Find asset paths for index names.

    Resource (VIPR) and text files name other assets by path. This collects the
    path strings in the first `scan_bytes` of each such file and keeps the ones
    whose hash is a name in the index. Returns {name: path}.
    """
    candidates: set[str] = set()
    for e in reader.unique_data():
        head = reader.head(e, 4)
        if head[:4] != b"VIPR" and head[:1] not in (b"{", b"["):
            continue
        for m in PATH_STRING.findall(reader.head(e, scan_bytes)):
            s = m.decode("latin-1").lstrip("/")
            candidates.update((s, s.removesuffix(".import"), s + ".import"))
    found = {}
    for s in candidates:
        n = name_for_path(s)
        if n in reader.by_name:
            found[n] = s
    return found


def output_name(entry: VzipEntry) -> str:
    """A safe file name for an entry: its own name if harmless, else its index number."""
    return entry.name if SAFE_NAME.fullmatch(entry.name) else f"entry{entry.index:06d}"


# -- whole-archive passes ------------------------------------------------------


@dataclass
class PassResult:
    checked: int = 0
    bytes: int = 0
    failures: list[tuple[str, str]] = field(default_factory=list)
    matches: list[tuple[str, int]] = field(default_factory=list)

    def merge(self, other: PassResult) -> None:
        self.checked += other.checked
        self.bytes += other.bytes
        self.failures += other.failures
        self.matches += other.matches


def check_entries(reader: VzipReader, indices: list[int], pattern: bytes | None = None) -> PassResult:
    """Decompress and CRC-check entries by index; optionally look for `pattern` in each."""
    result = PassResult()
    for i in indices:
        e = reader.entries[i]
        try:
            data = reader.read(e, max_size=1 << 62)
        except VzipError as err:
            result.failures.append((e.name, str(err)))
            continue
        result.checked += 1
        result.bytes += len(data)
        if pattern is not None:
            pos = data.find(pattern)
            if pos >= 0:
                result.matches.append((e.name, pos))
    return result


_worker: VzipReader | None = None


def _init_worker(path: str) -> None:
    global _worker
    _worker = VzipReader.open(path)


def _check_group(indices: list[int], pattern: bytes | None) -> PassResult:
    assert _worker is not None
    return check_entries(_worker, indices, pattern)


def check_archive(
    path: str | Path,
    *,
    jobs: int = 1,
    pattern: bytes | None = None,
    progress: Callable[[PassResult, int], None] | None = None,
) -> PassResult:
    """Run check_entries over every stored copy of data, in `jobs` processes."""
    with VzipReader.open(path) as r:
        todo = r.unique_data()
        total = len(todo)
        if jobs <= 1:
            result = check_entries(r, [e.index for e in todo], pattern)
            if progress:
                progress(result, total)
            return result
    # Balance the groups by size, then read each group in archive order.
    groups: list[list[VzipEntry]] = [[] for _ in range(jobs * 8)]
    loads = [0] * len(groups)
    for e in sorted(todo, key=lambda e: -e.size):
        k = loads.index(min(loads))
        groups[k].append(e)
        loads[k] += e.size + 4096
    result = PassResult()
    with ProcessPoolExecutor(max_workers=jobs, initializer=_init_worker, initargs=(str(path),)) as pool:
        futures = [
            pool.submit(_check_group, [e.index for e in sorted(g, key=lambda e: e.offset)], pattern)
            for g in groups
            if g
        ]
        for fut in as_completed(futures):
            result.merge(fut.result())
            if progress:
                progress(result, total)
    return result
