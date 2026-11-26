"""The guest address space: where things are mapped, with what permissions and in which state.

The layout follows the system's 39-bit address space, the type games use, with fixed addresses
instead of random ones, so every run places everything at the same address. The kernel reports
these regions to the game through GetInfo, so the game finds them itself.

Every mapping is backed by host memory that siicl allocates (`HostMemory`). Unicorn maps the
guest pages straight onto it, so two guest ranges can share the same memory, as MapMemory and
shared memory need.
"""

from __future__ import annotations

import bisect
import ctypes
import mmap
from dataclasses import dataclass, field, replace
from enum import IntEnum, IntFlag

from siicl.cpu import EXECUTE, PAGE, READ, WRITE, Cpu

GIB = 1 << 30
ADDRESS_END = 1 << 64


class MemoryState(IntEnum):
    """What a range of guest memory holds, as QueryMemory reports it (svc::MemoryState)."""

    FREE = 0x00
    IO = 0x01
    STATIC = 0x02
    CODE = 0x03
    CODE_DATA = 0x04
    NORMAL = 0x05
    SHARED = 0x06
    ALIAS = 0x07
    ALIAS_CODE = 0x08
    ALIAS_CODE_DATA = 0x09
    IPC = 0x0A
    STACK = 0x0B
    THREAD_LOCAL = 0x0C
    TRANSFERRED = 0x0D
    SHARED_TRANSFERRED = 0x0E
    SHARED_CODE = 0x0F
    INACCESSIBLE = 0x10


class MemoryAttribute(IntFlag):
    NONE = 0
    LOCKED = 0x01
    IPC_LOCKED = 0x02
    DEVICE_SHARED = 0x04
    UNCACHED = 0x08
    PERMISSION_LOCKED = 0x10


@dataclass(frozen=True)
class Range:
    start: int
    size: int

    @property
    def end(self) -> int:
        return self.start + self.size

    def contains(self, address: int, size: int = 1) -> bool:
        return self.start <= address and address + size <= self.end


@dataclass(frozen=True)
class Layout:
    """Where each region of the address space is.

    space: every address the process may map. code: where the loader places the modules. alias,
    heap and stack: the regions of those names that GetInfo reports. kernel_map: where the kernel
    maps its own things for the process, such as thread-local storage. remote: the Switch 2's
    remote region, which GetInfo also reports.
    """

    space: Range
    code: Range
    alias: Range
    heap: Range
    stack: Range
    kernel_map: Range
    remote: Range


LAYOUT_39BIT = Layout(
    space=Range(0x0800_0000, (1 << 39) - 0x0800_0000),
    code=Range(0x0800_0000, 16 * GIB - 0x0800_0000),
    alias=Range(64 * GIB, 64 * GIB),
    heap=Range(128 * GIB, 32 * GIB),
    stack=Range(192 * GIB, 2 * GIB),
    kernel_map=Range(256 * GIB, 64 * GIB),
    remote=Range(384 * GIB, 32 * GIB),
)


class HostMemory:
    """Anonymous host memory that backs guest pages. The pages are allocated when first touched."""

    def __init__(self, size: int) -> None:
        self.size = size
        self._map = mmap.mmap(-1, size)
        self._anchor = ctypes.c_char.from_buffer(self._map)  # keeps the mapping exported, so it never moves
        self.address = ctypes.addressof(self._anchor)


@dataclass(frozen=True)
class Block:
    """A mapped range. name says what it is in logs, such as "main .text"."""

    start: int
    size: int
    perm: int
    state: MemoryState
    name: str = ""
    attribute: int = 0
    memory: HostMemory | None = field(default=None, compare=False, repr=False)
    offset: int = 0  # where the block starts in memory

    @property
    def end(self) -> int:
        return self.start + self.size

    @property
    def host_address(self) -> int:
        return self.memory.address + self.offset


class AddressSpaceError(ValueError):
    pass


class AddressSpace:
    def __init__(self, cpu: Cpu, layout: Layout = LAYOUT_39BIT) -> None:
        self.cpu = cpu
        self.layout = layout
        self._blocks: list[Block] = []  # sorted by start, never overlapping

    @property
    def blocks(self) -> list[Block]:
        return list(self._blocks)

    def mapped_size(self) -> int:
        return sum(b.size for b in self._blocks)

    def map(
        self,
        start: int,
        size: int,
        perm: int,
        state: MemoryState,
        name: str = "",
        *,
        memory: HostMemory | None = None,
        offset: int = 0,
        attribute: int = 0,
    ) -> Block:
        """Map size bytes at start, on new zeroed memory or on memory from offset."""
        _check_pages(start, size)
        if not self.layout.space.contains(start, size):
            raise AddressSpaceError(f"{start:#x}+{size:#x} is outside the address space")
        if not self.is_free(start, size):
            raise AddressSpaceError(f"{start:#x}+{size:#x} overlaps memory that is already mapped")
        if memory is None:
            memory, offset = HostMemory(size), 0
        block = Block(start, size, perm, state, name, attribute, memory, offset)
        self.cpu.map_host(start, size, perm, block.host_address)
        bisect.insort(self._blocks, block, key=lambda b: b.start)
        return block

    def unmap(self, start: int, size: int) -> list[Block]:
        """Unmap a range that is mapped all the way through, and return the blocks it held."""
        blocks = self.range(start, size)
        for b in blocks:
            self.cpu.unmap(b.start, b.size)
            self._blocks.remove(b)
        return blocks

    def update(self, start: int, size: int, *, perm=None, state=None, attribute=None, name=None) -> None:
        """Change the permissions, state, attribute or name of a range that is mapped all the way through."""
        for b in self.range(start, size):
            changes = {k: v for k, v in (("perm", perm), ("state", state), ("attribute", attribute), ("name", name))}
            new = replace(b, **{k: v for k, v in changes.items() if v is not None})
            if new.perm != b.perm:
                self.cpu.protect(b.start, b.size, new.perm)
            self._blocks[self._blocks.index(b)] = new

    def range(self, start: int, size: int) -> list[Block]:
        """The blocks that cover start..start+size, split at its ends. Raises if part of it is unmapped."""
        _check_pages(start, size)
        self._split(start)
        self._split(start + size)
        blocks = [b for b in self._blocks if start <= b.start and b.end <= start + size]
        covered = start
        for b in blocks:
            if b.start != covered:
                break
            covered = b.end
        if covered != start + size:
            raise AddressSpaceError(f"{start:#x}+{size:#x} is not mapped all the way through")
        return blocks

    def is_free(self, start: int, size: int) -> bool:
        i = bisect.bisect_left(self._blocks, start, key=lambda b: b.start)
        if i > 0 and self._blocks[i - 1].end > start:
            return False
        return not (i < len(self._blocks) and self._blocks[i].start < start + size)

    def free_ranges(self, start: int, size: int) -> list[Range]:
        """The unmapped parts of start..start+size."""
        out, address, end = [], start, start + size
        for b in self._blocks:
            if b.end <= address or b.start >= end:
                continue
            if b.start > address:
                out.append(Range(address, b.start - address))
            address = max(address, b.end)
        if address < end:
            out.append(Range(address, end - address))
        return out

    def find_free(self, region: Range, size: int) -> int:
        """The lowest page-aligned address in region with size free bytes."""
        address = region.start
        for b in self._blocks:
            if b.end <= address:
                continue
            if b.start >= address + size:
                break
            address = (b.end + PAGE - 1) // PAGE * PAGE
        if not region.contains(address, size):
            raise AddressSpaceError(f"no {size:#x} bytes free in {region.start:#x}+{region.size:#x}")
        return address

    def map_free(self, region: Range, size: int, perm: int, state: MemoryState, name: str = "") -> Block:
        return self.map(self.find_free(region, size), size, perm, state, name)

    def query(self, address: int) -> Block:
        """What QueryMemory reports for address: the block that holds it, merged with neighbours that
        have the same state, permissions and attribute; for unmapped memory, the free gap around it.
        Past the end of the address space, everything is inaccessible."""
        space_end = self.layout.space.end
        if address >= space_end:
            return Block(space_end, ADDRESS_END - space_end, 0, MemoryState.INACCESSIBLE)
        i = bisect.bisect_right(self._blocks, address, key=lambda b: b.start) - 1
        if i < 0 or address >= self._blocks[i].end:
            start = self._blocks[i].end if i >= 0 else 0
            end = self._blocks[i + 1].start if i + 1 < len(self._blocks) else space_end
            return Block(start, end - start, 0, MemoryState.FREE)
        first = last = i
        same = lambda a, b: a.end == b.start and (a.perm, a.state, a.attribute) == (b.perm, b.state, b.attribute)  # noqa: E731
        while first > 0 and same(self._blocks[first - 1], self._blocks[first]):
            first -= 1
        while last + 1 < len(self._blocks) and same(self._blocks[last], self._blocks[last + 1]):
            last += 1
        b = self._blocks[i]
        start, end = self._blocks[first].start, self._blocks[last].end
        return replace(b, start=start, size=end - start, offset=b.offset - (b.start - start))

    def _split(self, address: int) -> None:
        i = bisect.bisect_right(self._blocks, address, key=lambda b: b.start) - 1
        if i < 0:
            return
        b = self._blocks[i]
        if b.start < address < b.end:
            head = replace(b, size=address - b.start)
            tail = replace(b, start=address, size=b.end - address, offset=b.offset + address - b.start)
            self._blocks[i : i + 1] = [head, tail]


def _check_pages(start: int, size: int) -> None:
    if start % PAGE or size % PAGE or size <= 0:
        raise AddressSpaceError(f"{start:#x}+{size:#x} is not a whole number of pages")


def perm_text(perm: int) -> str:
    return "".join(c if perm & bit else "-" for c, bit in (("r", READ), ("w", WRITE), ("x", EXECUTE)))
