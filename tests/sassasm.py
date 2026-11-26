"""Build synthetic instruction words from named bit fields, for tests.

Each line of a program is (text, opcode, fields) or (text, opcode, fields, guard).
program() checks that siicl.sass prints exactly the given text for each
instruction at its address, so a test reads as assembly and a wrong field fails
loudly.
"""

import struct

from siicl import sass

CTRL = 0x7E0  # scheduling bits for an independent instruction
M64 = (1 << 64) - 1
SREG = {"SR_LANEID": 0x00, "SR_TID.X": 0x21, "SR_TID.Y": 0x22, "SR_TID.Z": 0x23, "SR_CTAID.X": 0x25}


def word(opcode: int, fields: dict[int, int], guard: int = 7) -> tuple[int, int]:
    w = opcode | (guard << 12) | (CTRL << 105)
    for bit, value in fields.items():
        w |= value << bit
    return w & M64, w >> 64


def program(*lines) -> bytes:
    out = bytearray()
    for i, (text, opcode, fields, *guard) in enumerate(lines):
        lo, hi = word(opcode, fields, *guard)
        got = sass.decode(lo, hi, i * 16).text()
        assert got == text, f"line {i}: the fields give {got!r}, not {text!r}"
        out += struct.pack("<QQ", lo, hi)
    return bytes(out)


def rel(line: int, target: int) -> int:
    """Branch offset field from instruction line to instruction target."""
    return ((target - line - 1) * 16) & ((1 << 50) - 1)


def s2r(d: int, sr: str):
    return (f"S2R R{d}, {sr}", 0x919, {16: d, 72: SREG[sr]})


def mov_imm(d: int, value: int):
    return (f"MOV R{d}, {value:#x}", 0x802, {16: d, 32: value})


def store(coord: int, value: int, index: int = 1):
    """SUST.P.1D.R: write one word to the 1D surface with handle (0, index)."""
    return (
        f"SUST.P.1D.R.IGN [R{coord}], R{value}, 0x0, {index:#x}",
        0xB9B,
        {24: coord, 32: value, 40: index, 72: 1},
    )


def exit_(guard: int = 7, guard_text: str = ""):
    return (f"{guard_text}EXIT", 0x94D, {87: 7}, guard)


def bra_self(line: int):
    return (f"BRA {line * 16:#x}", 0x947, {32: rel(line, line), 87: 7})
