"""Tests for reading names from a module's dynamic symbol table, on a synthetic module."""

import struct

from siicl import symbols


def module():
    """A module image with MOD0 at 0x100, .dynamic at 0x200, then DT_HASH, the symbols and the strings."""
    image = bytearray(0x1000)
    struct.pack_into("<II", image, 0, 0x14000000, 0x100)  # b . ; MOD0 offset
    struct.pack_into("<4si", image, 0x100, b"MOD0", 0x100)  # .dynamic is 0x100 after MOD0
    dynamic = [(4, 0x300), (6, 0x400), (5, 0x500), (10, 0x40), (0, 0)]
    for i, (tag, value) in enumerate(dynamic):
        struct.pack_into("<qQ", image, 0x200 + 16 * i, tag, value)
    struct.pack_into("<II", image, 0x300, 1, 3)  # one bucket, three symbols
    strings = b"\0_Z3fooi\0bar\0"
    image[0x500 : 0x500 + len(strings)] = strings
    struct.pack_into("<IBBHQQ", image, 0x418, 1, 0x12, 0, 1, 0x800, 0x20)  # _Z3fooi, a function
    struct.pack_into("<IBBHQQ", image, 0x430, 9, 0x11, 0, 0, 0x900, 0x8)  # bar, undefined: skipped
    return bytes(image)


def test_reads_defined_symbols_and_finds_the_one_holding_an_address():
    image = module()
    base = 0x8000000
    found = symbols.module_symbols(lambda a, n: image[a - base : a - base + n], base, image[:8])
    assert found == [symbols.Symbol(base + 0x800, 0x20, "_Z3fooi")]
    table = symbols.SymbolTable(found)
    assert table.lookup(base + 0x810) == (found[0], 0x10)
    assert table.lookup(base + 0x820) is None


def test_demangles_names_when_cxxfilt_is_installed():
    out = symbols.demangle(["_Z3fooi"])
    assert out == (["foo(int)"] if symbols.CXXFILT else ["_Z3fooi"])
