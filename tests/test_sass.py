"""Tests for the instruction decoder, with synthetic instruction words.

Each case builds an instruction from named bit fields (no game data) and gives
the text that nvdisasm --binary SM86 prints for it. When nvdisasm is installed,
a further test checks that it still prints exactly that text.
"""

import struct
import subprocess
from dataclasses import dataclass, field

import pytest

from siicl import nvshader, sass

CTRL = 0x7E0  # scheduling bits for an independent instruction, as NVIDIA's compiler writes them
CTRL_REUSE = 0x00FF1  # scheduling bits that nvdisasm accepts together with an operand-reuse flag


@dataclass
class Case:
    name: str
    opcode: int
    fields: dict[int, int]
    text: str
    guard: int = 7
    addr: int = 0
    ctrl: int = CTRL
    words: tuple[int, int] = field(init=False)

    def __post_init__(self) -> None:
        w = self.opcode | (self.guard << 12) | (self.ctrl << 105)
        for bit, value in self.fields.items():
            w |= value << bit
        self.words = (w & ((1 << 64) - 1), w >> 64)


CASES = [
    Case("FFMA registers", 0x223, {16: 1, 24: 2, 32: 3, 64: 4}, "FFMA R1, R2, R3, R4"),
    Case(
        "FFMA modifiers",
        0x223,
        {16: 1, 24: 2, 32: 3, 64: 4, 72: 1, 73: 1, 75: 1, 77: 1, 80: 1},
        "FFMA.FTZ.SAT R1, -|R2|, R3, -R4",
    ),
    Case(
        "FFMA reuse",
        0x223,
        {16: 1, 24: 2, 32: 3, 64: 4, 80: 1, 122: 1},
        "FFMA.FTZ R1, R2.reuse, R3, R4",
        ctrl=CTRL_REUSE,
    ),
    Case("FFMA immediate C", 0x423, {16: 5, 24: 6, 32: 0x3F000000, 64: 7, 78: 3}, "FFMA.RZ R5, R6, R7, 0.5"),
    Case("FFMA cbuf C", 0x623, {16: 5, 24: 6, 38: 0x40, 54: 3, 63: 1, 64: 7}, "FFMA R5, R6, R7, -c[0x3][0x40]"),
    Case("FFMA immediate B", 0x823, {16: 1, 24: 2, 32: 0xC0200000, 64: 4, 74: 1, 80: 1}, "FFMA.FTZ R1, R2, -2.5, |R4|"),
    Case(
        "FFMA reuse C",
        0x823,
        {16: 1, 24: 2, 32: 0x40000000, 64: 4, 80: 1, 124: 1},
        "FFMA.FTZ R1, R2, 2, R4.reuse",
        ctrl=CTRL_REUSE,
    ),
    Case("FFMA cbuf B", 0xA23, {16: 1, 24: 2, 38: 0x6C0, 54: 7, 64: 0xC, 75: 1}, "FFMA R1, R2, c[0x7][0x6c0], -R12"),
    Case("FFMA ureg B", 0xC23, {16: 1, 24: 2, 32: 5, 64: 3, 91: 1}, "FFMA R1, R2, UR5, R3"),
    Case("FMUL scale", 0x220, {16: 1, 24: 2, 32: 3, 80: 1, 84: 5}, "FMUL.FTZ.M2 R1, R2, R3"),
    Case(
        "FMUL large immediate", 0x820, {16: 1, 24: 2, 32: 0x501502F9, 84: 4}, "FMUL R1, R2, 1.00000000000000000000e+10"
    ),
    Case(
        "FMUL small immediate", 0x820, {16: 1, 24: 2, 32: 0x358637BD, 84: 4}, "FMUL R1, R2, 9.9999999747524270788e-07"
    ),
    Case("FMUL infinity", 0x820, {16: 1, 24: 2, 32: 0x7F800000, 84: 4}, "FMUL R1, R2, +INF"),
    Case("FADD register", 0x221, {16: 1, 24: 2, 32: 3, 63: 1, 73: 1}, "FADD R1, |R2|, -R3"),
    Case("FADD immediate", 0x421, {16: 1, 24: 2, 32: 0xC0C00000, 80: 1}, "FADD.FTZ R1, R2, -6"),
    Case("FADD cbuf", 0x621, {16: 1, 24: 2, 38: 0x604, 54: 7, 77: 1}, "FADD.SAT R1, R2, c[0x7][0x604]"),
    Case("FMNMX", 0x209, {16: 1, 24: 2, 32: 3, 80: 1, 87: 7, 90: 1}, "FMNMX.FTZ R1, R2, R3, !PT"),
    Case("FSEL", 0x208, {16: 1, 24: 2, 32: 0xFF, 87: 2}, "FSEL R1, R2, RZ, P2"),
    Case(
        "FSETP",
        0x80B,
        {24: 2, 32: 0x3F000000, 74: 1, 76: 0xE, 80: 1, 81: 1, 84: 7, 87: 3, 90: 1},
        "FSETP.GEU.FTZ.OR P1, PT, R2, 0.5, !P3",
    ),
    Case("MUFU", 0x308, {16: 1, 32: 2, 62: 1, 74: 5}, "MUFU.RSQ R1, |R2|"),
    Case("FRND", 0x307, {16: 1, 32: 2, 75: 2, 78: 2, 84: 2}, "FRND.CEIL R1, R2"),
    Case(
        "F2I unsigned", 0x305, {16: 1, 32: 2, 72: 0, 75: 2, 77: 1, 78: 3, 80: 1, 84: 2}, "F2I.FTZ.U32.TRUNC.NTZ R1, R2"
    ),
    Case("F2I byte", 0x305, {16: 1, 32: 2, 72: 0, 78: 3, 80: 1, 84: 2}, "F2I.FTZ.U8.TRUNC R1, R2"),
    Case("F2I signed", 0x305, {16: 1, 32: 2, 72: 1, 84: 2}, "F2I.S8 R1, R2"),
    Case("I2FP", 0x245, {16: 1, 32: 2, 74: 1, 76: 1, 84: 2}, "I2FP.F32.S32 R1, R2"),
    Case("MOV register", 0x202, {16: 1, 32: 2, 72: 0xF}, "MOV R1, R2"),
    Case("MOV immediate", 0x802, {16: 1, 32: 0x3F800000, 72: 0xF}, "MOV R1, 0x3f800000"),
    Case("MOV cbuf", 0xA02, {16: 1, 38: 0x208, 54: 2, 72: 0xF}, "MOV R1, c[0x2][0x208]"),
    Case("LDC indexed", 0xB82, {16: 2, 24: 4, 38: 0x30, 54: 1, 73: 5}, "LDC.64 R2, c[0x1][R4+0x30]"),
    Case("ULDC", 0xAB9, {16: 4, 24: 0x3F, 38: 0x14C, 54: 6, 73: 4}, "ULDC UR4, c[0x6][0x14c]"),
    Case("IPA", 0x326, {16: 1, 32: 0xFF, 64: 0x1F, 81: 7, 87: 7}, "IPA.PASS R1, a[0x7c]"),
    Case("ALD", 0x321, {16: 4, 24: 0xFF, 32: 0xFF, 40: 0x80, 74: 1}, "ALD.64 R4, a[0x80]"),
    Case("AST", 0x322, {24: 0xFF, 32: 0x10, 40: 0xC8, 64: 0xFF}, "AST a[0xc8], R16"),
    Case("AST vertex", 0x322, {24: 0xFF, 32: 0x10, 40: 0xC8, 64: 3}, "AST a[0xc8], R16, R3"),
    Case("BRA", 0x947, {32: 0x20, 87: 7}, "BRA 0x40", addr=0x10),
    Case("BRA predicate", 0x947, {32: 0x40, 87: 2, 90: 1}, "@P1 BRA !P2, 0x150", guard=1, addr=0x100),
    Case("BSSY", 0x945, {16: 0, 32: 0x60, 87: 7}, "BSSY B0, 0x90", addr=0x20),
    Case("BSYNC", 0x941, {16: 1, 87: 7}, "BSYNC B1"),
    Case("EXIT", 0x94D, {87: 7}, "EXIT"),
    Case("NOP", 0x918, {}, "NOP"),
    Case("KILL", 0x95B, {87: 7}, "@P1 KILL", guard=1),
    Case(
        "IADD3",
        0x210,
        {16: 1, 24: 2, 32: 3, 64: 4, 72: 1, 75: 1, 77: 7, 80: 1, 81: 7, 84: 7, 87: 7, 90: 1},
        "IADD3 R1, -R2, R3, -R4",
    ),
    Case(
        "IADD3 carry",
        0xA10,
        {16: 1, 24: 2, 38: 0x60, 54: 0, 64: 0xFF, 77: 7, 80: 1, 81: 1, 84: 7, 87: 7, 90: 1},
        "IADD3 R1, P1, R2, c[0x0][0x60], RZ",
    ),
    Case(
        "IADD3.X",
        0xA10,
        {16: 1, 24: 0xFF, 38: 0x64, 54: 0, 64: 0xFF, 74: 1, 77: 7, 80: 1, 81: 7, 84: 7, 87: 1},
        "IADD3.X R1, RZ, c[0x0][0x64], RZ, P1, !PT",
    ),
    Case("IMAD", 0x224, {16: 1, 24: 2, 32: 3, 64: 4, 73: 1, 81: 7, 87: 7, 90: 1}, "IMAD R1, R2, R3, R4"),
    Case("IMAD.MOV", 0x224, {16: 1, 24: 0xFF, 32: 0xFF, 64: 5, 81: 7, 87: 7, 90: 1}, "IMAD.MOV.U32 R1, RZ, RZ, R5"),
    Case("IMAD.SHL", 0x824, {16: 1, 24: 2, 32: 0x10, 64: 0xFF, 81: 7, 87: 7, 90: 1}, "IMAD.SHL.U32 R1, R2, 0x10, RZ"),
    Case("IMAD.IADD", 0x824, {16: 1, 24: 2, 32: 1, 64: 3, 73: 1, 81: 7, 87: 7, 90: 1}, "IMAD.IADD R1, R2, 0x1, R3"),
    Case(
        "IMAD signed immediate",
        0x824,
        {16: 1, 24: 2, 32: 0xFFFFFFEF, 64: 3, 73: 1, 81: 7, 87: 7, 90: 1},
        "IMAD R1, R2, -0x11, R3",
    ),
    Case(
        "IMAD.HI",
        0x827,
        {16: 1, 24: 2, 32: 0xCCCCCCCD, 64: 0xFF, 81: 7, 87: 7, 90: 1},
        "IMAD.HI.U32 R1, R2, -0x33333333, RZ",
    ),
    Case("LEA", 0x211, {16: 1, 24: 2, 32: 3, 64: 0xFF, 75: 3, 81: 7, 87: 7, 90: 1}, "LEA R1, R2, R3, 0x3"),
    Case(
        "LEA.HI",
        0x811,
        {16: 1, 24: 2, 32: 4, 64: 0xFF, 75: 4, 80: 1, 81: 7, 87: 7, 90: 1},
        "LEA.HI R1, R2, 0x4, RZ, 0x4",
    ),
    Case(
        "LOP3",
        0x812,
        {16: 1, 24: 2, 32: 0x10000, 64: 3, 72: 0xE2, 81: 7, 87: 7, 90: 1},
        "LOP3.LUT R1, R2, 0x10000, R3, 0xe2, !PT",
    ),
    Case(
        "LOP3 predicate out",
        0x212,
        {16: 0xFF, 24: 2, 32: 3, 64: 0xFF, 72: 0xC0, 81: 1, 87: 7, 90: 1},
        "LOP3.LUT P1, RZ, R2, R3, RZ, 0xc0, !PT",
    ),
    Case(
        "PLOP3", 0x81C, {64: 0, 68: 7, 72: 0x15, 77: 1, 81: 0, 84: 7, 87: 0}, "PLOP3.LUT P0, PT, P0, P1, PT, 0xa8, 0x0"
    ),
    Case(
        "SHF.R", 0x819, {16: 1, 24: 0xFF, 32: 0x10, 64: 2, 73: 1, 74: 1, 76: 1, 80: 1}, "SHF.R.U32.HI R1, RZ, 0x10, R2"
    ),
    Case("SHF.L.W", 0x219, {16: 1, 24: 2, 32: 3, 64: 0xFF, 73: 1, 74: 1, 75: 1}, "SHF.L.W.U32 R1, R2, R3, RZ"),
    Case("SGXT", 0x81A, {16: 1, 24: 2, 32: 3}, "SGXT.U32 R1, R2, 0x3"),
    Case("SEL", 0x807, {16: 1, 24: 0xFF, 32: 1, 87: 1}, "SEL R1, RZ, 0x1, P1"),
    Case("IMNMX", 0x817, {16: 1, 24: 2, 32: 3, 87: 7}, "IMNMX.U32 R1, R2, 0x3, PT"),
    Case("PRMT", 0x816, {16: 1, 24: 2, 32: 0x7610, 64: 0xFF}, "PRMT R1, R2, 0x7610, RZ"),
    Case("ISETP unsigned", 0x20C, {24: 2, 32: 0xFF, 76: 5, 81: 0, 84: 7, 87: 7}, "ISETP.NE.U32.AND P0, PT, R2, RZ, PT"),
    Case(
        "ISETP signed immediate",
        0x80C,
        {24: 5, 32: 0xFFFFFFFE, 73: 1, 76: 6, 81: 0, 84: 7, 87: 7},
        "ISETP.GE.AND P0, PT, R5, -0x2, PT",
    ),
    Case(
        "UISETP",
        0x88C,
        {24: 4, 32: 7, 73: 1, 76: 6, 81: 0, 84: 7, 87: 7, 91: 1},
        "UISETP.GE.AND UP0, UPT, UR4, 0x7, UPT",
    ),
    Case("IABS", 0x213, {16: 1, 32: 2}, "IABS R1, R2"),
    Case("POPC", 0x309, {16: 1, 32: 2}, "POPC R1, R2"),
    Case("FLO", 0x300, {16: 1, 32: 2, 81: 7}, "FLO.U32 R1, R2"),
    Case("BREV", 0x301, {16: 1, 32: 2}, "BREV R1, R2"),
    Case("UMOV", 0x882, {16: 4, 32: 1}, "UMOV UR4, 0x1"),
    Case(
        "UIADD3",
        0x890,
        {16: 4, 24: 4, 32: 0xFFFFFFFD, 64: 0x3F, 77: 7, 80: 1, 81: 7, 84: 7, 87: 7, 90: 1, 91: 1},
        "UIADD3 UR4, UR4, -0x3, URZ",
    ),
    Case("UIMAD", 0x8A4, {16: 4, 24: 6, 32: 3, 64: 7, 73: 1, 91: 1}, "UIMAD UR4, UR6, 0x3, UR7"),
    Case("S2R", 0x919, {16: 3, 72: 0x21}, "S2R R3, SR_TID.X"),
    Case("CS2R", 0x805, {16: 4, 72: 0xFF, 80: 1}, "CS2R R4, SRZ"),
    Case("P2R", 0x803, {16: 1, 24: 0xFF, 32: 0x40}, "P2R R1, PR, RZ, 0x40"),
    Case("R2P", 0x804, {24: 0x11, 32: 3}, "R2P PR, R17, 0x3"),
    Case("DEPBAR", 0x91A, {38: 3, 44: 5, 47: 1}, "DEPBAR.LE SB5, 0x3"),
    Case("BAR", 0xB1D, {}, "BAR.SYNC 0x0"),
    Case("BRX", 0x949, {24: 0, 32: 0xFFFFFA00, 64: 0x3FFFF, 87: 7}, "BRX R0 -0x600"),
    Case("BRX predicate", 0x949, {24: 0, 32: 0xFFFFFA00, 64: 0x3FFFF, 87: 0}, "BRX P0, R0 -0x600"),
    Case("CALL", 0x944, {32: 0x10, 86: 1, 87: 7}, "CALL.REL.NOINC 0x1f50", addr=0x1F30),
    Case("RET", 0x950, {24: 0xC, 32: 0xFFFFF230, 64: 0x3FFFF, 86: 1, 87: 7}, "RET.REL.NODEC R12 0x0", addr=0xDC0),
    Case("BREAK", 0x942, {16: 5, 87: 7}, "@P0 BREAK B5", guard=0),
    Case("WARPSYNC", 0x948, {32: 0xFFFFFFFF, 87: 7}, "WARPSYNC 0xffffffff"),
    Case("HADD2 to F32", 0x230, {16: 1, 24: 0xFF, 32: 2, 60: 3, 72: 1, 78: 1}, "HADD2.F32 R1, -RZ, R2.H1_H1"),
    Case("HMUL2", 0x232, {16: 1, 24: 2, 32: 3, 74: 2}, "HMUL2 R1, R2.H0_H0, R3"),
    Case("HFMA2", 0x231, {16: 1, 24: 2, 32: 3, 60: 2, 64: 4, 72: 1}, "HFMA2 R1, -R2, R3.H0_H0, R4"),
    Case("F2FP", 0x23E, {16: 1, 24: 0xFF, 32: 3}, "F2FP.PACK_AB R1, RZ, R3"),
    Case("I2F byte", 0x306, {16: 1, 32: 2, 60: 2, 76: 1}, "I2F.U8 R1, R2.B2"),
    Case("SHFL immediates", 0xF89, {16: 1, 24: 2, 40: 0x1F, 53: 1, 58: 3, 81: 7}, "SHFL.BFLY PT, R1, R2, 0x1, 0x1f"),
    Case("SHFL register lane", 0x589, {16: 1, 24: 2, 32: 3, 40: 0x1F, 81: 7}, "SHFL.IDX PT, R1, R2, R3, 0x1f"),
    Case("FSWZADD", 0x822, {16: 1, 24: 2, 32: 0x66, 64: 3}, "FSWZADD R1, R2, R3, PNNPPNNP"),
    Case(
        "TEX",
        0xB60,
        {16: 6, 24: 4, 32: 5, 40: 8, 54: 2, 60: 1, 61: 1, 64: 4, 72: 0xF, 81: 7, 84: 1},
        "TEX.SCR R4, R6, R4, R5, 0x2, 0x8, 2D",
    ),
    Case(
        "TEX level zero",
        0xB60,
        {16: 2, 24: 0xB, 32: 0xC, 40: 8, 54: 2, 60: 1, 61: 1, 64: 0xFF, 72: 1, 81: 7, 84: 1, 87: 1, 90: 1},
        "TEX.SCR.LZ.NODEP RZ, R2, R11, R12, 0x2, 0x8, 2D, 0x1",
    ),
    Case(
        "TLD",
        0xB66,
        {16: 7, 24: 8, 32: 0xF, 40: 0xA, 54: 0, 60: 1, 61: 5, 64: 0xFF, 72: 1, 81: 7, 84: 1, 87: 1},
        "TLD.SCR.LZ RZ, R7, R8, R15, 0x0, 0xa, ARRAY_2D, 0x1",
    ),
    Case(
        "TLD4",
        0xB63,
        {16: 0x24, 24: 0, 32: 1, 40: 8, 54: 0, 60: 1, 61: 1, 64: 0x22, 72: 0xF, 76: 1, 81: 7, 84: 1, 87: 1},
        "TLD4.SCR.G.AOFFI R34, R36, R0, R1, 0x0, 0x8, 2D",
    ),
    Case(
        "TXQ",
        0xB6F,
        {16: 0, 24: 4, 40: 0x1C, 54: 2, 64: 0xFF, 72: 3, 81: 7, 84: 1},
        "TXQ RZ, R0, R4, TEX_HEADER_DIMENSION, 0x2, 0x1c, 0x3",
    ),
    Case("LDS", 0x984, {16: 8, 24: 0, 40: 0x1400, 73: 6, 78: 3}, "LDS.128 R8, [R0.X16+0x1400]"),
    Case("STS", 0x388, {24: 1, 32: 2, 40: 4, 73: 4}, "STS [R1+0x4], R2"),
    Case("LDG", 0x381, {16: 4, 24: 6, 40: 0x10, 72: 1, 73: 4, 76: 8, 81: 7, 84: 1}, "LDG.E.CONSTANT R4, [R6+0x10]"),
    Case("STG", 0x386, {24: 2, 32: 3, 72: 1, 73: 4, 76: 0xA, 84: 1}, "STG.E.STRONG.SM [R2], R3"),
    Case("LDL", 0x983, {16: 1, 24: 0xFF, 40: 0xC, 73: 4, 84: 3}, "LDL.LU R1, [0xc]"),
    Case("STL", 0x387, {24: 0xFF, 32: 5, 73: 4, 84: 1}, "STL [RZ], R5"),
    Case("ATOMS", 0x38C, {16: 1, 24: 2, 32: 3}, "ATOMS.ADD R1, [R2], R3"),
    Case("RED", 0x98E, {24: 2, 32: 1, 64: 0x3F, 72: 1, 73: 0, 76: 0xE, 84: 1, 87: 0}, "RED.E.ADD.STRONG.GPU [R2], R1"),
    Case(
        "RED F16x2",
        0x98E,
        {24: 2, 32: 1, 64: 0x3F, 72: 1, 73: 4, 76: 0xE, 84: 1, 87: 0},
        "RED.E.ADD.F16x2.RN.STRONG.GPU [R2], R1",
    ),
]


@pytest.mark.parametrize("case", CASES, ids=[c.name for c in CASES])
def test_decodes_like_nvdisasm(case):
    lo, hi = case.words
    assert sass.decode(lo, hi, case.addr).text() == case.text


@pytest.mark.skipif(nvshader.find_nvdisasm() is None, reason="nvdisasm (cuda) is not installed")
@pytest.mark.parametrize("case", CASES, ids=[c.name for c in CASES])
def test_expected_text_is_what_nvdisasm_prints(case, tmp_path):
    nop = Case("NOP", 0x918, {}, "NOP").words
    code = struct.pack("<QQ", *nop) * (case.addr // 16) + struct.pack("<QQ", *case.words)
    path = tmp_path / "code.bin"
    path.write_bytes(code)
    out = subprocess.run([nvshader.find_nvdisasm(), "--binary", "SM86", str(path)], capture_output=True, text=True)
    line = next(ln for ln in out.stdout.splitlines() if f"/*{case.addr:04x}*/" in ln)
    assert line.split("*/", 1)[1].strip().rstrip(";").rstrip() == case.text


def test_unknown_opcode_is_an_error():
    with pytest.raises(sass.UnknownInstruction):
        sass.decode(0xFFF, 0, 0)


def test_operands_are_structured():
    lo, hi = Case("x", 0x223, {16: 1, 24: 2, 32: 3, 64: 4, 72: 1, 80: 1}, "").words
    ins = sass.decode(lo, hi)
    assert ins.name == "FFMA" and "FTZ" in ins.mods
    dst, a, b, c = ins.operands
    assert (dst.kind, dst.value) == ("reg", 1)
    assert a.neg and a.value == 2
    assert (b.value, c.value) == (3, 4)


@pytest.mark.parametrize(
    "bits, text",
    [
        (0x3F000000, "0.5"),
        (0x501502F9, "1.00000000000000000000e+10"),
        (0x358637BD, "9.9999999747524270788e-07"),
        (0x7F800000, "+INF "),
        (0xFF800000, "-INF "),
    ],
)
def test_float_immediates(bits, text):
    assert sass._f32(bits) == text
