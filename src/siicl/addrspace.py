"""The guest address space: where things are mapped, with what permissions and in which state.

The layout follows the system's 39-bit address space, the type games use, with fixed addresses
instead of random ones, so every run places everything at the same address. The kernel reports
these regions to the game through GetInfo, so the game finds them itself.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from enum import IntEnum

from siicl.cpu import EXECUTE, PAGE, READ, WRITE, Cpu

GIB = 1 << 30


class MemoryState(IntEnum):
    """What a range of guest memory holds, as QueryMemory reports it (svc::MemoryState)."""

    FREE = 0x00
    CODE = 0x03
    CODE_DATA = 0x04
    NORMAL = 0x05
    STACK = 0x0B
    THREAD_LOCAL = 0x0C


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

    space: every address the process may use. code: where the loader places the modules.
    alias, heap and stack: the regions of those names that GetInfo reports. kernel_map: where the
    kernel maps its own things for the process, such as thread-local storage.
    """

    space: Range
    code: Range
    alias: Range
    heap: Range
    stack: Range
    kernel_map: Range


LAYOUT_39BIT = Layout(
    space=Range(0x0800_0000, (1 << 39) - 0x0800_0000),
    code=Range(0x0800_0000, 16 * GIB - 0x0800_0000),
    alias=Range(64 * GIB, 64 * GIB),
    heap=Range(128 * GIB, 32 * GIB),
    stack=Range(192 * GIB, 2 * GIB),
    kernel_map=Range(256 * GIB, 64 * GIB),
)


@dataclass(frozen=True)
class Block:
    """A mapped range. name says what it is in logs, such as "main .text"."""

    start: int
    size: int
    perm: int
    state: MemoryState
    name: str = ""

    @property
    def end(self) -> int:
        return self.start + self.size


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

    def map(self, start: int, size: int, perm: int, state: MemoryState, name: str = "") -> Block:
        if start % PAGE or size % PAGE or size <= 0:
            raise AddressSpaceError(f"{start:#x}+{size:#x} is not a whole number of pages")
        if not self.layout.space.contains(start, size):
            raise AddressSpaceError(f"{start:#x}+{size:#x} is outside the address space")
        i = bisect.bisect_left([b.start for b in self._blocks], start)
        if (i > 0 and self._blocks[i - 1].end > start) or (
            i < len(self._blocks) and self._blocks[i].start < start + size
        ):
            raise AddressSpaceError(f"{start:#x}+{size:#x} overlaps memory that is already mapped")
        self.cpu.map(start, size, perm)
        block = Block(start, size, perm, state, name)
        self._blocks.insert(i, block)
        return block

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
        """The block that holds address; for unmapped memory, the whole free gap around it."""
        i = bisect.bisect_right([b.start for b in self._blocks], address) - 1
        if i >= 0 and address < self._blocks[i].end:
            return self._blocks[i]
        start = self._blocks[i].end if i >= 0 else 0
        end = self._blocks[i + 1].start if i + 1 < len(self._blocks) else 1 << 64
        return Block(start, end - start, 0, MemoryState.FREE)


def perm_text(perm: int) -> str:
    return "".join(c if perm & bit else "-" for c, bit in (("r", READ), ("w", WRITE), ("x", EXECUTE)))
