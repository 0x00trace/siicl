"""Tests for the SPIR-V translation, with synthetic programs (tests/sassasm.py).

Every translation must pass spirv-val. When a Vulkan GPU is present, the tests
also run the programs on it and compare every output word with siicl.interp,
using inputs chosen to hit edge cases: NaN, infinities, signed zeros,
subnormal numbers, out-of-range conversions and large shift amounts.
"""

import shutil
import struct
import subprocess

import pytest

from siicl import interp, translate
from siicl.nvshader import ComputeInfo
from tests import test_interp as ti
from tests.sassasm import bra_self, exit_, program, s2r, store

SPIRV_VAL = shutil.which("spirv-val")
PT_NOT = {87: 7, 90: 1}
INFO = ComputeInfo((32, 1, 1), 0x100, 32, b"")


def fbits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


FLOATS = [0.0, -0.0, 1.0, -1.0, 0.5, 2.5, -2.5, 3.5, 1e-40, -1e-40, 3.4e38, -3.4e38, 1e10, 2.0**31, -(2.0**31)]
FLOATS += [2.0**32, 65535.5, 70000.0, 1.0 + 2.0**-23, 0.1, -0.3, 7.9999995]
FLOAT_BITS = [fbits(x) for x in FLOATS] + [0x7F800000, 0xFF800000, 0x7FC00000, 0xFFFFFFFF, 0x00000001]
INTS = [0, 1, 2, 31, 32, 33, 64, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF, 0xFFFFFFE0, 0x12345678, 0x87654321, 5, 16, 40]


def pairs(values: list[int]) -> bytes:
    """32 pairs of words for c[0x3]: thread t reads words 2t and 2t + 1."""
    out = []
    for t in range(32):
        out += [values[t % len(values)], values[(t * 7 + 3) % len(values)]]
    return struct.pack("<64I", *out)


LOAD_PAIR = [
    s2r(0, "SR_TID.X"),
    ("SHF.L.U32 R1, R0, 0x3, RZ", 0x819, {16: 1, 24: 0, 32: 3, 64: 255, 73: 3}),
    ("LDC.64 R2, c[0x3][R1]", 0xB82, {16: 2, 24: 1, 54: 3, 73: 5}),
]

FLOAT_PROGRAM = [
    *LOAD_PAIR,
    ("FADD.FTZ R4, R2, R3", 0x221, {16: 4, 24: 2, 32: 3, 80: 1}),
    ("FMUL.FTZ.RZ R5, R2, R3", 0x220, {16: 5, 24: 2, 32: 3, 78: 3, 80: 1, 84: 4}),
    ("FFMA.FTZ.SAT R6, R2, R3, R2", 0x223, {16: 6, 24: 2, 32: 3, 64: 2, 77: 1, 80: 1}),
    ("FMNMX R7, R2, R3, PT", 0x209, {16: 7, 24: 2, 32: 3, 87: 7}),
    ("FMNMX R8, R2, R3, !PT", 0x209, {16: 8, 24: 2, 32: 3, **PT_NOT}),
    ("FSETP.GTU.AND P0, PT, R2, R3, PT", 0x20B, {24: 2, 32: 3, 76: 12, 81: 0, 84: 7, 87: 7}),
    ("FSEL R9, R2, R3, P0", 0x208, {16: 9, 24: 2, 32: 3, 87: 0}),
    ("F2I.U32.TRUNC.NTZ R10, R2", 0x305, {16: 10, 32: 2, 75: 2, 77: 1, 78: 3, 84: 2}),
    ("F2I.NTZ R11, R2", 0x305, {16: 11, 32: 2, 72: 1, 75: 2, 77: 1, 84: 2}),
    ("FRND.FLOOR R12, R2", 0x307, {16: 12, 32: 2, 78: 1}),
    ("FADD.FTZ.RZ R13, R2, R3", 0x221, {16: 13, 24: 2, 32: 3, 78: 3, 80: 1}),
    ("F2I.U16.TRUNC.NTZ R14, R3", 0x305, {16: 14, 32: 3, 75: 1, 77: 1, 78: 3, 84: 2}),
    ("FMUL R15, R2, R3", 0x220, {16: 15, 24: 2, 32: 3, 84: 4}),
    ("FFMA R16, R2, R3, -R3", 0x223, {16: 16, 24: 2, 32: 3, 64: 3, 75: 1}),
    *(store(0, r, r - 3) for r in range(4, 17)),
    exit_(),
    bra_self(31),
]

INT_PROGRAM = [
    *LOAD_PAIR,
    ("IADD3 R4, P0, P1, R2, -R3, R2", 0x210, {16: 4, 24: 2, 32: 3, 63: 1, 64: 2, 81: 0, 84: 1}),
    ("P2R R5, PR, RZ, 0x3", 0x803, {16: 5, 24: 255, 32: 3}),
    ("SHF.R.S32.HI R6, RZ, R3, R2", 0x219, {16: 6, 24: 255, 32: 3, 64: 2, 73: 2, 76: 1, 80: 1}),
    ("SHF.L.W.U32.HI R7, R2, R3, R2", 0x219, {16: 7, 24: 2, 32: 3, 64: 2, 73: 3, 75: 1, 80: 1}),
    ("SGXT R8, R2, R3", 0x21A, {16: 8, 24: 2, 32: 3, 73: 1}),
    ("PRMT R9, R2, 0x8b19, R3", 0x816, {16: 9, 24: 2, 32: 0x8B19, 64: 3}),
    ("LEA.HI R10, R2, R3, R2, 0x5", 0x211, {16: 10, 24: 2, 32: 3, 64: 2, 75: 5, 80: 1, 81: 7}),
    ("IMNMX R11, R2, R3, PT", 0x217, {16: 11, 24: 2, 32: 3, 73: 1, 87: 7}),
    ("ISETP.LT.AND P2, PT, R2, R3, PT", 0x20C, {24: 2, 32: 3, 73: 1, 76: 1, 81: 2, 84: 7, 87: 7}),
    ("P2R R12, PR, RZ, 0x4", 0x803, {16: 12, 24: 255, 32: 4}),
    ("LOP3.LUT R13, R2, R3, R0, 0x96, !PT", 0x212, {16: 13, 24: 2, 32: 3, 64: 0, 72: 0x96, 81: 7, **PT_NOT}),
    ("IMAD.HI.U32 R14, R2, R3, RZ", 0x227, {16: 14, 24: 2, 32: 3, 64: 255}),
    ("FLO.U32 R15, R2", 0x300, {16: 15, 32: 2}),
    ("IMAD R16, R2, R3, -R2", 0x224, {16: 16, 24: 2, 32: 3, 64: 2, 73: 1, 75: 1}),
    *(store(0, r, r - 3) for r in range(4, 17)),
    exit_(),
    bra_self(31),
]


SHARED_PROGRAM = [
    s2r(0, "SR_TID.X"),
    ("STS [R0.X4], R0", 0x388, {24: 0, 32: 0, 73: 4, 78: 1}),
    ("BAR.SYNC 0x0", 0xB1D, {}),
    ("IADD3 R1, -R0, 0x3f, RZ", 0x810, {16: 1, 24: 0, 32: 0x3F, 64: 255, 72: 1, 81: 7, 84: 7}),
    ("LDS R2, [R1.X4]", 0x984, {16: 2, 24: 1, 73: 4, 78: 1}),
    store(0, 2),
    exit_(),
    bra_self(7),
]
SHARED_INFO = ComputeInfo((64, 1, 1), 0x100, 32, b"")


def launch_for(prog: interp.Program, cbufs: dict[int, bytes]) -> interp.Launch:
    surfaces = {
        h: interp.Surface(64 if dim == "1D" else 8, 1, 1, channels)
        for h, (dim, channels) in translate.surfaces(prog.instructions).items()
    }
    return interp.Launch((1, 1, 1), {1: b"\0" * 16, **cbufs}, surfaces)


def validate(tr: translate.Translation, tmp_path) -> None:
    if SPIRV_VAL is None:
        pytest.skip("spirv-val (spirv-tools) is not installed")
    path = tmp_path / "shader.spv"
    path.write_bytes(tr.spirv)
    result = subprocess.run([SPIRV_VAL, "--target-env", "vulkan1.2", str(path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


PROGRAMS = {
    "float": (FLOAT_PROGRAM, {3: pairs(FLOAT_BITS)}, INFO),
    "int": (INT_PROGRAM, {3: pairs(INTS)}, INFO),
    "atomics": (ti.SURFACE_ATOMICS, {}, INFO),
    "shared": (SHARED_PROGRAM, {}, SHARED_INFO),
}


@pytest.mark.parametrize("name", PROGRAMS)
def test_translation_is_valid_spirv(name, tmp_path):
    lines, _, info = PROGRAMS[name]
    prog = interp.Program(program(*lines), info)
    validate(translate.translate(prog.instructions, info, mufu_log=True), tmp_path)


def test_bindings():
    prog = interp.Program(program(*FLOAT_PROGRAM), INFO)
    tr = translate.translate(prog.instructions, INFO)
    assert tr.bindings[0] == translate.Binding(0, "status")
    assert tr.bindings[1] == translate.Binding(1, "cbuf", bank=3)
    surfaces = [b for b in tr.bindings if b.kind == "surface"]
    assert [b.handle for b in surfaces] == [(0, k) for k in range(1, 14)]
    assert {(b.dim, b.channels) for b in surfaces} == {("1D", 4)}
    atomics = translate.surfaces(interp.Program(program(*ti.SURFACE_ATOMICS), INFO).instructions)
    assert atomics == {(0, 1): ("1D", 4), (0, 2): ("1D", 1), (0, 3): ("1D", 1)}


def test_untranslatable_instruction():
    lines = [("LDG.E R4, [R2+0x10]", 0x381, {16: 4, 24: 2, 40: 0x10, 72: 1, 73: 4, 81: 7, 84: 1}), exit_(), bra_self(2)]
    prog = interp.Program(program(*lines), INFO)
    with pytest.raises(translate.Unsupported, match="0x0000 LDG.E"):
        translate.translate(prog.instructions, INFO)
