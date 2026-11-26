"""Names for code addresses, from the modules' dynamic symbol tables.

Each module's MOD0 header (its offset is the word after the module's first instruction) points to
its .dynamic section, whose DT_SYMTAB, DT_STRTAB and DT_HASH or DT_GNU_HASH entries give the
symbol table. Values in a module's .dynamic are offsets from the module's base. Only defined
function and object symbols are kept.
"""

from __future__ import annotations

import bisect
import shutil
import struct
import subprocess
from dataclasses import dataclass

MOD0 = b"MOD0"
DT_NULL, DT_HASH, DT_STRTAB, DT_SYMTAB, DT_STRSZ = 0, 4, 5, 6, 10
DT_GNU_HASH = 0x6FFFFEF5
STT_OBJECT, STT_FUNC = 1, 2
CXXFILT = shutil.which("aarch64-linux-gnu-c++filt") or shutil.which("c++filt")


@dataclass(frozen=True)
class Symbol:
    address: int
    size: int
    name: str


def _symbol_count(read, base: int, tags: dict[int, int]) -> int:
    if DT_HASH in tags:
        return struct.unpack("<II", read(base + tags[DT_HASH], 8))[1]
    if DT_GNU_HASH not in tags:
        return 0
    at = base + tags[DT_GNU_HASH]
    nbuckets, symoffset, bloom_size, _shift = struct.unpack("<IIII", read(at, 16))
    buckets_at = at + 16 + 8 * bloom_size
    buckets = struct.unpack(f"<{nbuckets}I", read(buckets_at, 4 * nbuckets))
    last = max(buckets, default=0)
    if last < symoffset:
        return symoffset
    chain_at = buckets_at + 4 * nbuckets
    while not struct.unpack("<I", read(chain_at + 4 * (last - symoffset), 4))[0] & 1:
        last += 1
    return last + 1


def module_symbols(read, base: int, text: bytes) -> list[Symbol]:
    """The symbols of the module at base. read(address, size) reads guest memory; text starts the module."""
    mod0 = base + struct.unpack_from("<I", text, 4)[0]
    if read(mod0, 4) != MOD0:
        return []
    dynamic = mod0 + struct.unpack("<i", read(mod0 + 4, 4))[0]
    tags: dict[int, int] = {}
    for i in range(512):
        tag, value = struct.unpack("<qQ", read(dynamic + 16 * i, 16))
        if tag == DT_NULL:
            break
        tags.setdefault(tag, value)
    if DT_SYMTAB not in tags or DT_STRTAB not in tags:
        return []
    count = _symbol_count(read, base, tags)
    strtab = read(base + tags[DT_STRTAB], tags.get(DT_STRSZ, 0))
    table = read(base + tags[DT_SYMTAB], 24 * count)
    out = []
    for i in range(count):
        name_off, info, _other, shndx, value, size = struct.unpack_from("<IBBHQQ", table, 24 * i)
        if value and shndx and info & 0xF in (STT_FUNC, STT_OBJECT):
            end = strtab.find(b"\0", name_off)
            out.append(Symbol(base + value, size, strtab[name_off:end].decode("ascii", "replace")))
    out.sort(key=lambda s: s.address)
    return out


def demangle(names: list[str]) -> list[str]:
    if not CXXFILT or not names:
        return names
    done = subprocess.run([CXXFILT], input="\n".join(names), capture_output=True, text=True)
    lines = done.stdout.splitlines()
    return lines if len(lines) == len(names) else names


class SymbolTable:
    def __init__(self, symbols: list[Symbol]) -> None:
        self.symbols = sorted(symbols, key=lambda s: s.address)
        self._starts = [s.address for s in self.symbols]

    def lookup(self, address: int) -> tuple[Symbol, int] | None:
        """The symbol that holds address, and the offset into it."""
        i = bisect.bisect_right(self._starts, address) - 1
        if i < 0:
            return None
        s = self.symbols[i]
        if s.size and address >= s.address + s.size:
            return None
        return s, address - s.address
