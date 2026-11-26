"""Tests for the guest address space and the module loader, on synthetic NSO files."""

import struct

import pytest

from siicl import addrspace, cpu, loader
from siicl.addrspace import MemoryState
from siicl.cpu import EXECUTE, READ, WRITE
from tests.builders import build_nso

BASE = addrspace.LAYOUT_39BIT.code.start


def new_space():
    return addrspace.AddressSpace(cpu.Cpu())


def test_places_modules_one_after_another_in_load_order():
    mods = {
        "sdk": build_nso(b"\x01" * 0x10, b"r", b"d", bss=0x2000),
        "main": build_nso(b"\x02" * 0x1800, b"r" * 0x1001, b"main data", bss=0x10),
        "rtld": build_nso(b"\x03" * 4, bss=0),
    }
    space = new_space()
    loaded = loader.load(space, mods)
    assert [(m.name, m.base - BASE, m.size) for m in loaded] == [
        ("rtld", 0, 0x3000),
        ("main", 0x3000, 0x5000),
        ("sdk", 0x8000, 0x5000),
    ]
    main = loaded[1]
    assert (main.text.start, main.rodata.start, main.data.start, main.data.size) == (
        BASE + 0x3000,
        BASE + 0x5000,
        BASE + 0x7000,
        0x1000,
    )
    assert [(b.start - BASE, b.size, b.perm, b.state, b.name) for b in space.blocks][3:6] == [
        (0x3000, 0x2000, READ | EXECUTE, MemoryState.CODE, "main .text"),
        (0x5000, 0x2000, READ, MemoryState.CODE, "main .rodata"),
        (0x7000, 0x1000, READ | WRITE, MemoryState.CODE_DATA, "main .data"),
    ]
    assert space.cpu.read(main.text.start, 0x1800) == b"\x02" * 0x1800
    assert space.cpu.read(main.data.start, 0x1000) == b"main data" + bytes(0x1000 - 9)
    sdk = loaded[2]
    assert space.cpu.read(sdk.data.start, sdk.data.size) == b"d" + bytes(0x2FFF)


def test_rejects_names_that_are_not_exefs_modules():
    with pytest.raises(loader.LoaderError, match="extra"):
        loader.load(new_space(), {"main": build_nso(b"\0" * 4), "extra": build_nso(b"\0" * 4)})
    with pytest.raises(loader.LoaderError, match="no rtld or main"):
        loader.load(new_space(), {"sdk": build_nso(b"\0" * 4)})


def test_rejects_a_segment_that_does_not_start_on_a_page():
    raw = bytearray(build_nso(b"\0" * 4, compress=False))
    struct.pack_into("<I", raw, 0x24, 0x1010)  # .rodata memory offset
    with pytest.raises(loader.LoaderError, match="rodata starts at 0x1010"):
        loader.load(new_space(), {"main": bytes(raw)})


def test_address_space_finds_free_ranges_and_refuses_overlaps():
    space = new_space()
    region = addrspace.Range(BASE, 0x10000)
    a = space.map_free(region, 0x2000, READ, MemoryState.NORMAL, "a")
    space.map(BASE + 0x3000, 0x1000, READ, MemoryState.NORMAL, "b")
    assert a.start == BASE
    assert space.find_free(region, 0x1000) == BASE + 0x2000
    assert space.find_free(region, 0x2000) == BASE + 0x4000
    with pytest.raises(addrspace.AddressSpaceError, match="overlaps"):
        space.map(BASE + 0x1000, 0x1000, READ, MemoryState.NORMAL)
    with pytest.raises(addrspace.AddressSpaceError, match="outside"):
        space.map(0x1000, 0x1000, READ, MemoryState.NORMAL)
    with pytest.raises(addrspace.AddressSpaceError, match="no 0x20000 bytes free"):
        space.find_free(region, 0x20000)


def test_query_returns_the_block_or_the_free_gap_around_an_address():
    space = new_space()
    space.map(BASE, 0x1000, READ, MemoryState.NORMAL, "a")
    space.map(BASE + 0x3000, 0x1000, READ | WRITE, MemoryState.STACK, "b")
    assert space.query(BASE + 0x10).name == "a"
    gap = space.query(BASE + 0x2000)
    assert (gap.start, gap.size, gap.state, gap.perm) == (BASE + 0x1000, 0x2000, MemoryState.FREE, 0)
    assert space.query(0).start == 0 and space.query(0).end == BASE
    assert space.query(BASE + 0x5000).end == 1 << 64
