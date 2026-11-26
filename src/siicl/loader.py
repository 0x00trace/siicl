"""Place a game's executable modules in the guest address space, as the system's loader does.

The modules are NSO files from the exefs. They go one after another, page-aligned, from the start
of the code region, in this order: rtld, main, subsdk0 to subsdk9, sdk. In each module, .text is
mapped read and execute, .rodata read-only, and .data with .bss after it read and write. The
first module's first instruction is the process's entry point. rtld runs there first and
relocates the other modules itself.
"""

from __future__ import annotations

from dataclasses import dataclass

from siicl import nso
from siicl.addrspace import AddressSpace, MemoryState, Range
from siicl.cpu import EXECUTE, PAGE, READ, WRITE

MODULE_ORDER = ("rtld", "main", *(f"subsdk{i}" for i in range(10)), "sdk")


class LoaderError(ValueError):
    pass


@dataclass(frozen=True)
class Module:
    name: str
    base: int
    size: int
    text: Range
    rodata: Range
    data: Range  # .data and .bss
    module_id: bytes

    @property
    def end(self) -> int:
        return self.base + self.size

    def contains(self, address: int) -> bool:
        return self.base <= address < self.end


def _page_up(n: int) -> int:
    return (n + PAGE - 1) // PAGE * PAGE


def load(space: AddressSpace, modules: dict[str, bytes], base: int | None = None) -> list[Module]:
    """Map and write the modules, and return them in load order. Names outside MODULE_ORDER are an error."""
    unknown = sorted(set(modules) - set(MODULE_ORDER))
    if unknown:
        raise LoaderError(f"not an exefs module name: {', '.join(unknown)}")
    if "rtld" not in modules and "main" not in modules:
        raise LoaderError("no rtld or main module to start")
    address = space.layout.code.start if base is None else base
    loaded = []
    for name in (n for n in MODULE_ORDER if n in modules):
        exe = nso.parse(modules[name])
        loaded.append(_load_one(space, name, exe, address))
        address = loaded[-1].end
    return loaded


def _load_one(space: AddressSpace, name: str, exe: nso.Nso, base: int) -> Module:
    segs = exe.segments
    for seg in segs.values():
        if seg.memory_offset % PAGE:
            raise LoaderError(f"{name}: {seg.name} starts at {seg.memory_offset:#x}, not on a page boundary")
    text = Range(base + segs["text"].memory_offset, _page_up(segs["text"].size))
    rodata = Range(base + segs["rodata"].memory_offset, _page_up(segs["rodata"].size))
    data = Range(base + segs["data"].memory_offset, _page_up(segs["data"].size + exe.bss_size))
    if text.start != base or not text.end <= rodata.start or not rodata.end <= data.start:
        raise LoaderError(f"{name}: segments are not in the order .text, .rodata, .data")
    for rng, perm, state, label in (
        (text, READ | EXECUTE, MemoryState.CODE, ".text"),
        (rodata, READ, MemoryState.CODE, ".rodata"),
        (data, READ | WRITE, MemoryState.CODE_DATA, ".data"),
    ):
        if rng.size:
            space.map(rng.start, rng.size, perm, state, f"{name} {label}")
    for seg_name, rng in (("text", text), ("rodata", rodata), ("data", data)):
        space.cpu.write(rng.start, exe.segment_data(seg_name))
    return Module(name, base, data.end - base, text, rodata, data, exe.module_id)
