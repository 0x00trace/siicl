"""Decoder for NVIDIA's 128-bit GPU instruction encoding, as used by the Switch 2's Ampere GPU.

`decode(lo, hi, address)` turns one instruction (two little-endian 64-bit
halves) into an `Instruction`: its mnemonic, modifiers, predicate guard and
operands. `Instruction.text()` prints it the way `nvdisasm --binary SM86`
does, so the decoder can be checked against NVIDIA's disassembler.

The field layouts were worked out by comparing encodings with nvdisasm output
(docs/sass-decoder.md). Opcodes the decoder does not know raise
`UnknownInstruction`.
"""

from __future__ import annotations

import math
import struct
from collections.abc import Callable
from dataclasses import dataclass, field


class UnknownInstruction(ValueError):
    """The decoder has no layout for this opcode."""


# ---------------------------------------------------------------------------
# Operands


@dataclass
class Operand:
    kind: str  # reg ureg pred upred breg cbuf imm fimm attr label sreg text
    value: int | float | str = 0
    bank: int = 0  # cbuf: bank number
    index: Operand | None = None  # cbuf and attr: index register
    neg: bool = False
    abs: bool = False
    not_: bool = False  # predicates
    reuse: bool = False
    suffix: str = ""  # such as ".H0_H0" or ".X4"
    extra: Operand | None = None  # addr: uniform register added to the address
    scale: int = 1  # addr: the base register is multiplied by this
    hidden: bool = False  # kept for its meaning, but nvdisasm doesn't print it (such as a PT predicate)

    def core(self) -> str:
        k, v = self.kind, self.value
        if k == "reg":
            return "RZ" if v == 255 else f"R{v}"
        if k == "ureg":
            return "URZ" if v == 63 else f"UR{v}"
        if k == "pred":
            return ("!" if self.not_ else "") + ("PT" if v == 7 else f"P{v}")
        if k == "upred":
            return ("!" if self.not_ else "") + ("UPT" if v == 7 else f"UP{v}")
        if k == "breg":
            return f"B{v}"
        if k == "cbuf":
            inner = f"{v:#x}"
            if self.index is not None and not _is_zero_reg(self.index):
                inner = self.index.core() + (f"+{v:#x}" if v else "")
            return f"c[{self.bank:#x}][{inner}]"
        if k == "attr":
            return f"a[{v:#x}]"
        if k == "addr":
            return _addr_text(self)
        if k == "himm":
            return _f16(v)
        if k == "brx":
            return f"{self.index.core()} {_hex(v)}"
        if k == "ret":
            return f"{self.index.core()} {v:#x}"
        if k == "imm":
            return _hex(v)
        if k == "fimm":
            return _f32(v)
        if k == "label":
            return f"{v:#x}"
        return str(v)

    def text(self) -> str:
        s = self.core()
        if self.abs:
            s = f"|{s}|"
        if self.neg:
            s = "-" + s
        if self.reuse and not _is_zero_reg(self):
            s += ".reuse"
        return s + self.suffix


def _is_zero_reg(op: Operand) -> bool:
    return (op.kind == "reg" and op.value == 255) or (op.kind == "ureg" and op.value == 63)


def _hidden_if(op: Operand, hide: bool) -> Operand:
    op.hidden = hide
    return op


def _hex(v: int) -> str:
    return f"-{-v:#x}" if v < 0 else f"{v:#x}"


def _f32(bits: int) -> str:
    (f,) = struct.unpack("<f", struct.pack("<I", bits & 0xFFFFFFFF))
    if math.isinf(f):
        return "+INF " if f > 0 else "-INF "
    if math.isnan(f):
        return ("-" if bits >> 31 else "+") + ("QNAN " if bits & 0x400000 else "SNAN ")
    # nvdisasm switches to %.20e (keeping trailing zeros) from 1e10 up; %.20g covers the rest,
    # including exponent form for small values. The corpus has no values between 1e8 and 1e10.
    return f"{f:.20e}" if abs(f) >= 1e10 else f"{f:.20g}"


# ---------------------------------------------------------------------------
# Instructions


@dataclass
class Instruction:
    address: int
    opcode: int
    name: str
    mods: list[str] = field(default_factory=list)
    guard: Operand | None = None
    operands: list[Operand] = field(default_factory=list)
    separator: str = ", "
    uniform_guard: bool = False
    cbuf_space: bool = False  # nvdisasm prints "c[0x6] [0x398]" for some packed-half operands

    def text(self) -> str:
        head = self.name + "".join("." + m for m in self.mods if m)
        if self.guard is not None:
            head = f"@{self.guard.core()} {head}"
        shown = [op for op in self.operands if not op.hidden]
        if not shown:
            return head
        body = self.separator.join(op.text() for op in shown)
        if self.cbuf_space:
            body = body.replace("][", "] [")
        return (head + " " + body).rstrip()


class Bits:
    """A 128-bit instruction word with field accessors."""

    __slots__ = ("w", "address")

    def __init__(self, lo: int, hi: int, address: int) -> None:
        self.w = lo | (hi << 64)
        self.address = address

    def f(self, start: int, width: int) -> int:
        return (self.w >> start) & ((1 << width) - 1)

    def b(self, bit: int) -> bool:
        return bool((self.w >> bit) & 1)

    def s(self, start: int, width: int) -> int:
        v = self.f(start, width)
        return v - (1 << width) if v >> (width - 1) else v

    # common operand fields
    def reg(self, start: int, *, neg: int | None = None, abs_: int | None = None, reuse: int | None = None) -> Operand:
        return Operand("reg", self.f(start, 8), neg=self._b(neg), abs=self._b(abs_), reuse=self._b(reuse))

    def ureg(self, start: int, *, neg: int | None = None, abs_: int | None = None) -> Operand:
        return Operand("ureg", self.f(start, 6), neg=self._b(neg), abs=self._b(abs_))

    def pred(self, start: int, not_bit: int | None = None) -> Operand:
        return Operand("pred", self.f(start, 3), not_=self._b(not_bit))

    def upred(self, start: int, not_bit: int | None = None) -> Operand:
        return Operand("upred", self.f(start, 3), not_=self._b(not_bit))

    def cbuf(self, *, neg: int | None = None, abs_: int | None = None) -> Operand:
        return Operand("cbuf", self.f(38, 16), bank=self.f(54, 5), neg=self._b(neg), abs=self._b(abs_))

    def imm(self, signed: bool = False) -> Operand:
        return Operand("imm", self.s(32, 32) if signed else self.f(32, 32))

    def fimm(self) -> Operand:
        return Operand("fimm", self.f(32, 32))

    def _b(self, bit: int | None) -> bool:
        return bit is not None and self.b(bit)

    def guard(self) -> Operand | None:
        idx, neg = self.f(12, 3), self.b(15)
        if idx == 7 and not neg:
            return None
        return Operand("pred", idx, not_=neg)


Decoder = Callable[[Bits], Instruction]
_TABLE: dict[int, Decoder] = {}


def _op(*opcodes: int) -> Callable[[Decoder], Decoder]:
    def reg(fn: Decoder) -> Decoder:
        for o in opcodes:
            _TABLE[o] = fn
        return fn

    return reg


def known_opcodes() -> set[int]:
    return set(_TABLE)


def decode(lo: int, hi: int, address: int = 0) -> Instruction:
    """Decode one instruction. Raises UnknownInstruction for opcodes without a layout."""
    bits = Bits(lo, hi, address)
    opcode = bits.f(0, 12)
    fn = _TABLE.get(opcode)
    if fn is None:
        raise UnknownInstruction(f"opcode {opcode:#05x} at {address:#x}")
    ins = fn(bits)
    ins.address, ins.opcode = address, opcode
    ins.guard = bits.guard()
    if ins.guard is not None and ins.uniform_guard:
        ins.guard.kind = "upred"
    return ins


_UNIFORM = {"UIADD3", "UIMAD", "ULEA", "ULOP3", "USHF", "UISETP", "UMOV", "UFLO", "UPOPC", "UBREV", "USEL", "UPRMT"}


def _ins(name: str, mods: list[str], operands: list[Operand], **kw) -> Instruction:
    kw.setdefault("uniform_guard", name in _UNIFORM)
    return Instruction(0, 0, name, mods, None, operands, **kw)


# ---------------------------------------------------------------------------
# Shared operand layouts

_ROUND = {0: "", 1: "RM", 2: "RP", 3: "RZ"}
_MEM_SIZE = {0: "U8", 1: "S8", 2: "U16", 3: "S16", 4: "", 5: "64", 6: "128", 7: "U.128"}
_FCMP = ["F", "LT", "EQ", "LE", "GT", "NE", "GE", "NUM", "NAN", "LTU", "EQU", "LEU", "GTU", "NEU", "GEU", "T"]
_ICMP = ["F", "LT", "EQ", "LE", "GT", "NE", "GE", "T"]
_BOOL = ["AND", "OR", "XOR", "?"]


def _form(b: Bits) -> int:
    return b.f(9, 3)


def _src_a(b: Bits) -> Operand:
    return b.reg(24, neg=72, abs_=73, reuse=122)


def _src_b2(b: Bits, *, is_float: bool, reuse: int = 123) -> Operand:
    """Second source of a two-source ALU operation in forms 1, 4, 5 and 6."""
    f = _form(b)
    if f == 1:
        return b.reg(32, neg=63, abs_=62, reuse=reuse)
    if f == 4:
        return b.fimm() if is_float else b.imm()
    if f == 5:
        return b.cbuf(neg=63, abs_=62)
    if f == 6:
        return b.ureg(32, neg=63, abs_=62)
    raise UnknownInstruction(f"form {f}")


def _src_bc(b: Bits, *, is_float: bool) -> tuple[Operand, Operand]:
    """Second and third sources of a three-source ALU operation."""
    f = _form(b)
    if f == 1:
        return b.reg(32, neg=63, abs_=62, reuse=123), b.reg(64, neg=75, abs_=74, reuse=124)
    if f == 2:
        return b.reg(64, neg=75, abs_=74, reuse=123), (b.fimm() if is_float else b.imm())
    if f == 3:
        return b.reg(64, neg=75, abs_=74, reuse=123), b.cbuf(neg=63, abs_=62)
    if f == 4:
        return (b.fimm() if is_float else b.imm()), b.reg(64, neg=75, abs_=74, reuse=124)
    if f == 5:
        return b.cbuf(neg=63, abs_=62), b.reg(64, neg=75, abs_=74, reuse=124)
    if f == 6:
        return b.ureg(32, neg=63, abs_=62), b.reg(64, neg=75, abs_=74, reuse=124)
    if f == 7:
        return b.reg(64, neg=75, abs_=74, reuse=123), b.ureg(32, neg=63, abs_=62)
    raise UnknownInstruction(f"form {f}")


def _dst(b: Bits) -> Operand:
    return b.reg(16)


# ---------------------------------------------------------------------------
# Floating-point arithmetic


@_op(0x223, 0x423, 0x623, 0x823, 0xA23, 0xC23, 0xE23)
def _ffma(b: Bits) -> Instruction:
    src_b, src_c = _src_bc(b, is_float=True)
    mods = ["FTZ" if b.b(80) else "", _ROUND[b.f(78, 2)], "SAT" if b.b(77) else ""]
    return _ins("FFMA", mods, [_dst(b), _src_a(b), src_b, src_c])


_FMUL_SCALE = {0: "?", 1: "D8", 2: "D4", 3: "D2", 4: "", 5: "M2", 6: "M4", 7: "M8"}


@_op(0x220, 0x820, 0xA20, 0xC20)
def _fmul(b: Bits) -> Instruction:
    mods = ["FTZ" if b.b(80) else "", _FMUL_SCALE[b.f(84, 3)], _ROUND[b.f(78, 2)], "SAT" if b.b(77) else ""]
    return _ins("FMUL", mods, [_dst(b), _src_a(b), _src_b2(b, is_float=True)])


@_op(0x221, 0x421, 0x621, 0xE21)
def _fadd(b: Bits) -> Instruction:
    f = _form(b)
    if f == 1:
        src = b.reg(32, neg=63, abs_=62, reuse=124)
    elif f == 2:
        src = b.fimm()
    elif f == 3:
        src = b.cbuf(neg=63, abs_=62)
    else:
        src = b.ureg(32, neg=63, abs_=62)
    mods = ["FTZ" if b.b(80) else "", _ROUND[b.f(78, 2)], "SAT" if b.b(77) else ""]
    return _ins("FADD", mods, [_dst(b), _src_a(b), src])


@_op(0x209, 0x809, 0xA09, 0xC09)
def _fmnmx(b: Bits) -> Instruction:
    return _ins("FMNMX", ["FTZ" if b.b(80) else ""], [_dst(b), _src_a(b), _src_b2(b, is_float=True), b.pred(87, 90)])


@_op(0x208, 0x808, 0xA08, 0xC08)
def _fsel(b: Bits) -> Instruction:
    return _ins("FSEL", [], [_dst(b), _src_a(b), _src_b2(b, is_float=True), b.pred(87, 90)])


@_op(0x20B, 0x80B, 0xA0B, 0xC0B)
def _fsetp(b: Bits) -> Instruction:
    mods = [_FCMP[b.f(76, 4)], "FTZ" if b.b(80) else "", _BOOL[b.f(74, 2)]]
    ops = [b.pred(81), b.pred(84), _src_a(b), _src_b2(b, is_float=True), b.pred(87, 90)]
    return _ins("FSETP", mods, ops)


_MUFU = ["COS", "SIN", "EX2", "LG2", "RCP", "RSQ", "RCP64H", "RSQ64H", "SQRT", "TANH", "?", "?", "?", "?", "?", "?"]


@_op(0x308, 0x908, 0xB08, 0xD08)
def _mufu(b: Bits) -> Instruction:
    src = _src_unary(b, is_float=True)
    if b.b(73) and b.b(60):
        src.suffix = ".H1"
    return _ins("MUFU", [_MUFU[b.f(74, 4)], "F16" if b.b(73) else ""], [_dst(b), src])


def _src_unary(b: Bits, *, is_float: bool) -> Operand:
    f = _form(b)
    if f == 1:
        return b.reg(32, neg=63, abs_=62, reuse=123)
    if f == 4:
        return b.fimm() if is_float else b.imm()
    if f == 5:
        return b.cbuf(neg=63, abs_=62)
    if f == 6:
        return b.ureg(32, neg=63, abs_=62)
    raise UnknownInstruction(f"form {f}")


_FRND = {0: "", 1: "FLOOR", 2: "CEIL", 3: "TRUNC"}


@_op(0x307, 0x907, 0xB07, 0xD07)
def _frnd(b: Bits) -> Instruction:
    return _ins("FRND", [_FRND[b.f(78, 2)], "FTZ" if b.b(80) else ""], [_dst(b), _src_unary(b, is_float=True)])


_INT_SIZE = {0: "8", 1: "16", 2: "32", 3: "64"}
_FLOAT_SIZE = {1: "F16", 2: "F32", 3: "F64"}


def _int_type(signed: bool, size: int) -> str:
    t = ("S" if signed else "U") + _INT_SIZE[size]
    return "" if t == "S32" else t


@_op(0x305, 0x905, 0xB05, 0xD05)
def _f2i(b: Bits) -> Instruction:
    src_type = {1: "F16", 2: "", 3: "F64"}.get(b.f(84, 2), "?")
    mods = [
        src_type,
        "FTZ" if b.b(80) else "",
        _int_type(b.b(72), b.f(75, 2)),
        _FRND[b.f(78, 2)],
        "NTZ" if b.b(77) else "",
    ]
    return _ins("F2I", mods, [_dst(b), _src_unary(b, is_float=True)])


@_op(0x245, 0x845, 0xA45, 0xC45)
def _i2fp(b: Bits) -> Instruction:
    dst = _FLOAT_SIZE.get(b.f(75, 2), f"???{b.f(75, 2)}")
    mods = [dst, ("S" if b.b(74) else "U") + _INT_SIZE[b.f(84, 2)], _ROUND[b.f(78, 2)]]
    return _ins("I2FP", mods, [_dst(b), _src_unary(b, is_float=False)])


# ---------------------------------------------------------------------------
# Moves and constants


@_op(0x202, 0x802, 0xA02, 0xC02)
def _mov(b: Bits) -> Instruction:
    return _ins("MOV", [], [_dst(b), _src_unary(b, is_float=False)])


@_op(0xB82)
def _ldc(b: Bits) -> Instruction:
    c = Operand("cbuf", b.f(38, 16), bank=b.f(54, 5), index=b.reg(24))
    return _ins("LDC", [_MEM_SIZE[b.f(73, 3)]], [_dst(b), c])


@_op(0xAB9, 0xABB)
def _uldc(b: Bits) -> Instruction:
    c = Operand("cbuf", b.f(38, 16), bank=b.f(54, 5), index=b.ureg(24) if b.f(0, 12) == 0xABB else None)
    return _ins("ULDC", [_MEM_SIZE[b.f(73, 3)]], [b.ureg(16), c])


# ---------------------------------------------------------------------------
# Attributes


_IPA_MODE = {0: "PASS", 1: "CONSTANT", 2: "?", 3: "?"}


@_op(0x326)
def _ipa(b: Bits) -> Instruction:
    return _ins("IPA", [_IPA_MODE[b.f(78, 2)]], [_dst(b), Operand("attr", b.f(62, 11))])


_ATTR_SIZE = {0: "", 1: "64", 2: "96", 3: "128"}


@_op(0x321)
def _ald(b: Bits) -> Instruction:
    return _ins("ALD", [_ATTR_SIZE[b.f(74, 2)]], [_dst(b), Operand("attr", b.f(40, 11))])


@_op(0x322)
def _ast(b: Bits) -> Instruction:
    ops = [Operand("attr", b.f(40, 11)), b.reg(32), _hidden_if(b.reg(64), b.f(64, 8) == 255)]
    return _ins("AST", [_ATTR_SIZE[b.f(74, 2)]], ops)


# ---------------------------------------------------------------------------
# Control flow


def _target(b: Bits) -> Operand:
    return Operand("label", b.address + 16 + b.s(32, 50))


@_op(0x947)
def _bra(b: Bits) -> Instruction:
    return _ins("BRA", [], _with_pred(b, [_target(b)]))


@_op(0x945)
def _bssy(b: Bits) -> Instruction:
    return _ins("BSSY", [], _with_pred(b, [Operand("breg", b.f(16, 4)), _target(b)]))


@_op(0x941)
def _bsync(b: Bits) -> Instruction:
    return _ins("BSYNC", [], _with_pred(b, [Operand("breg", b.f(16, 4))]))


def _with_pred(b: Bits, ops: list[Operand]) -> list[Operand]:
    """Prefix the source predicate at bit 87, hidden when it is PT."""
    p = b.pred(87, 90)
    return [_hidden_if(p, _is_pt(p)), *ops]


@_op(0x94D)
def _exit(b: Bits) -> Instruction:
    return _ins("EXIT", [], _with_pred(b, []))


@_op(0x918)
def _nop(b: Bits) -> Instruction:
    return _ins("NOP", [], [])


@_op(0x95B)
def _kill(b: Bits) -> Instruction:
    return _ins("KILL", [], [])


# ---------------------------------------------------------------------------
# Integer arithmetic and logic


def _src_b2i(b: Bits, *, signed: bool = False, uniform: bool = False) -> Operand:
    """Second source of a two-source integer operation (forms 1, 4, 5, 6)."""
    f = _form(b)
    if f == 1:
        return b.ureg(32) if uniform else b.reg(32, neg=63, reuse=123)
    if f == 4:
        return b.imm(signed)
    if f == 5:
        return b.cbuf(neg=63)
    if f == 6:
        return b.ureg(32, neg=63)
    raise UnknownInstruction(f"form {f}")


def _src_bci(b: Bits, *, signed: bool = False, uniform: bool = False) -> tuple[Operand, Operand]:
    """Second and third sources of a three-source integer operation."""
    f = _form(b)
    r = b.ureg if uniform else None

    def reg(start: int, neg: int, reuse: int) -> Operand:
        return b.ureg(start, neg=neg) if uniform else b.reg(start, neg=neg, reuse=reuse)

    if f == 1:
        return reg(32, 63, 123), reg(64, 75, 124)
    if f == 2:
        return reg(64, 75, 123), b.imm(signed)
    if f == 3:
        return reg(64, 75, 123), b.cbuf(neg=63)
    if f == 4:
        return b.imm(signed), reg(64, 75, 124)
    if f == 5:
        return b.cbuf(neg=63), reg(64, 75, 124)
    if f == 6:
        return b.ureg(32, neg=63), reg(64, 75, 124)
    if f == 7:
        return reg(64, 75, 123), b.ureg(32, neg=63)
    del r
    raise UnknownInstruction(f"form {f}")


def _is_pt(op: Operand) -> bool:
    return op.kind in ("pred", "upred") and op.value == 7 and not op.not_


@_op(0x210, 0x410, 0x610, 0x810, 0xA10, 0xC10, 0xE10, 0x290, 0x890)
def _iadd3(b: Bits) -> Instruction:
    uniform = b.f(0, 12) in (0x290, 0x890)
    src_b, src_c = _src_bci(b, signed=True, uniform=uniform)
    pr = b.upred if uniform else b.pred
    a = b.ureg(24, neg=72) if uniform else b.reg(24, neg=72, reuse=122)
    dst = b.ureg(16) if uniform else _dst(b)
    ops = [dst] + [_hidden_if(p, _is_pt(p)) for p in (pr(81), pr(84))] + [a, src_b, src_c]
    x = b.b(74)
    if x:
        ops += [pr(87, 90), pr(77, 80)]
    return _ins("UIADD3" if uniform else "IADD3", ["X" if x else ""], ops)


def _imad_name(b: Bits, a: Operand, src_b: Operand, src_c: Operand) -> list[str]:
    unsigned = not b.b(73)
    x = b.b(74)
    form = _form(b)
    if x:
        return ["X"]
    if _is_zero_reg(a) and _is_zero_reg(src_b) and form in (1, 2, 3):
        return ["MOV", "U32" if unsigned else ""]
    if src_b.kind == "imm" and src_b.value == 1 and form == 4:
        return ["IADD", "U32" if unsigned else ""]
    # nvdisasm prints SHL for powers of two, except 0x10000 (and 0x80000000, which is negative).
    v = src_b.value if src_b.kind == "imm" else 0
    if (
        unsigned
        and src_c.kind == "reg"
        and src_c.value == 255
        and 0 < v < 0x80000000
        and v & (v - 1) == 0
        and v != 0x10000
    ):
        return ["SHL", "U32"]
    return ["U32" if unsigned else ""]


@_op(0x224, 0x424, 0x624, 0x824, 0xA24, 0xC24, 0xE24)
def _imad(b: Bits) -> Instruction:
    src_b, src_c = _src_bci(b, signed=True)
    a = b.reg(24, neg=72, reuse=122)
    mods = _imad_name(b, a, src_b, src_c)
    ops = [_dst(b), a, src_b, src_c]
    if b.b(74):
        ops.append(b.pred(87, 90))
    return _ins("IMAD", mods, ops)


@_op(0x227, 0x427, 0x627, 0x827, 0xA27, 0xC27, 0xE27)
def _imad_hi(b: Bits) -> Instruction:
    src_b, src_c = _src_bci(b, signed=True)
    return _ins("IMAD", ["HI", "" if b.b(73) else "U32"], [_dst(b), b.reg(24, neg=72, reuse=122), src_b, src_c])


@_op(0x2A4, 0x4A4, 0x8A4)
def _uimad(b: Bits) -> Instruction:
    src_b, src_c = _src_bci(b, signed=True, uniform=True)
    return _ins("UIMAD", ["" if b.b(73) else "U32"], [b.ureg(16), b.ureg(24), src_b, src_c])


@_op(0x211, 0x811, 0xA11, 0xC11, 0x291, 0x891)
def _lea(b: Bits) -> Instruction:
    uniform = b.f(0, 12) in (0x291, 0x891)
    hi = b.b(80)
    src_b = _src_b2i(b, uniform=uniform)
    reg = (lambda s: b.ureg(s)) if uniform else (lambda s: b.reg(s, reuse=None))
    a = b.ureg(24) if uniform else b.reg(24, neg=72, reuse=122)
    dst = b.ureg(16) if uniform else _dst(b)
    pr = b.upred if uniform else b.pred
    ops = [dst] + [_hidden_if(p, _is_pt(p)) for p in (pr(81),)] + [a, src_b]
    if hi:
        ops.append(reg(64))
    ops.append(Operand("imm", b.f(75, 5)))
    if b.b(74):
        ops.append(pr(87, 90))
    return _ins("ULEA" if uniform else "LEA", ["HI" if hi else "", "X" if b.b(74) else ""], ops)


def _lut(b: Bits) -> Operand:
    return Operand("imm", b.f(72, 8))


@_op(0x212, 0x412, 0x612, 0x812, 0xA12, 0xC12, 0xE12, 0x292, 0x892)
def _lop3(b: Bits) -> Instruction:
    uniform = b.f(0, 12) in (0x292, 0x892)
    src_b, src_c = _src_bci(b, uniform=uniform)
    for s in (src_b, src_c):
        s.neg = False
    pr = b.upred if uniform else b.pred
    dst = b.ureg(16) if uniform else _dst(b)
    a = b.ureg(24) if uniform else b.reg(24, reuse=122)
    ops = [_hidden_if(p, _is_pt(p)) for p in (pr(81),)] + [dst, a, src_b, src_c, _lut(b), pr(87, 90)]
    return _ins("ULOP3" if uniform else "LOP3", ["LUT"], ops)


@_op(0x81C)
def _plop3(b: Bits) -> Instruction:
    lut = (b.f(72, 5) << 3) | b.f(64, 3)
    pc = b.upred(68, 71) if b.b(67) else b.pred(68, 71)
    ops = [b.pred(81), b.pred(84), b.pred(87, 90), b.pred(77, 80), pc, Operand("imm", lut), Operand("imm", b.f(16, 8))]
    return _ins("PLOP3", ["LUT"], ops)


_SHF_TYPE = {0: "S64", 1: "U64", 2: "S32", 3: "U32"}


@_op(0x219, 0x619, 0x819, 0xA19, 0xC19, 0x299, 0x899)
def _shf(b: Bits) -> Instruction:
    uniform = b.f(0, 12) in (0x299, 0x899)
    src_b, src_c = _src_bci(b, uniform=uniform)
    for s in (src_b, src_c):
        s.neg = False
    mods = ["R" if b.b(76) else "L", "W" if b.b(75) else "", _SHF_TYPE[b.f(73, 2) | 2], "HI" if b.b(80) else ""]
    dst = b.ureg(16) if uniform else _dst(b)
    a = b.ureg(24) if uniform else b.reg(24, reuse=122)
    return _ins("USHF" if uniform else "SHF", mods, [dst, a, src_b, src_c])


@_op(0x21A, 0x81A, 0xA1A, 0xC1A)
def _sgxt(b: Bits) -> Instruction:
    mods = ["W" if b.b(75) else "", "" if b.b(73) else "U32"]
    return _ins("SGXT", mods, [_dst(b), b.reg(24, reuse=122), _src_b2i(b)])


@_op(0x21B, 0x81B, 0xA1B, 0xC1B)
def _bmsk(b: Bits) -> Instruction:
    return _ins("BMSK", ["W" if b.b(75) else ""], [_dst(b), b.reg(24, reuse=122), _src_b2i(b)])


@_op(0x207, 0x807, 0xA07, 0xC07)
def _sel(b: Bits) -> Instruction:
    return _ins("SEL", [], [_dst(b), b.reg(24, reuse=122), _src_b2i(b), b.pred(87, 90)])


@_op(0x217, 0x817, 0xA17, 0xC17)
def _imnmx(b: Bits) -> Instruction:
    return _ins("IMNMX", ["" if b.b(73) else "U32"], [_dst(b), b.reg(24, reuse=122), _src_b2i(b), b.pred(87, 90)])


_PRMT = {0: "", 1: "F4E", 2: "B4E", 3: "RC8", 4: "ECL", 5: "ECR", 6: "RC16", 7: "?"}


@_op(0x216, 0x416, 0x616, 0x816, 0xA16, 0xC16, 0xE16)
def _prmt(b: Bits) -> Instruction:
    src_b, src_c = _src_bci(b)
    for s in (src_b, src_c):
        s.neg = False
    return _ins("PRMT", [_PRMT[b.f(72, 3)]], [_dst(b), b.reg(24, reuse=122), src_b, src_c])


def _isetp_ops(b: Bits, uniform: bool) -> list[Operand]:
    pr = b.upred if uniform else b.pred
    a = b.ureg(24) if uniform else b.reg(24, reuse=122)
    ops = [pr(81), pr(84), a, _src_b2i(b, signed=True, uniform=uniform), pr(87, 90)]
    if b.b(72):
        ops.append(pr(68, 71))
    return ops


@_op(0x20C, 0x80C, 0xA0C, 0xC0C, 0x28C, 0x88C)
def _isetp(b: Bits) -> Instruction:
    uniform = b.f(0, 12) in (0x28C, 0x88C)
    mods = [_ICMP[b.f(76, 3)], "" if b.b(73) else "U32", _BOOL[b.f(74, 2)], "EX" if b.b(72) else ""]
    return _ins("UISETP" if uniform else "ISETP", mods, _isetp_ops(b, uniform))


@_op(0x213, 0x813, 0xA13, 0xC13)
def _iabs(b: Bits) -> Instruction:
    return _ins("IABS", [], [_dst(b), _src_unary(b, is_float=False)])


@_op(0x309, 0x909, 0xB09, 0xD09)
def _popc(b: Bits) -> Instruction:
    return _ins("POPC", [], [_dst(b), _src_unary(b, is_float=False)])


@_op(0x300, 0x900, 0xB00, 0xD00, 0x2BD)
def _flo(b: Bits) -> Instruction:
    uniform = b.f(0, 12) == 0x2BD
    mods = ["" if b.b(73) else "U32", "SH" if b.b(74) else ""]
    dst = b.ureg(16) if uniform else _dst(b)
    src = b.ureg(32) if uniform else _src_unary(b, is_float=False)
    return _ins("UFLO" if uniform else "FLO", mods, [dst, src])


@_op(0x301, 0x901, 0xB01, 0xD01)
def _brev(b: Bits) -> Instruction:
    return _ins("BREV", [], [_dst(b), _src_unary(b, is_float=False)])


@_op(0x882, 0xC82)
def _umov(b: Bits) -> Instruction:
    src = b.imm() if _form(b) == 4 else b.ureg(32)
    return _ins("UMOV", [], [b.ureg(16), src])


_VOTE = {0: "ALL", 1: "ANY", 2: "EQ", 3: "?"}


@_op(0x806)
def _vote(b: Bits) -> Instruction:
    return _ins("VOTE", [_VOTE[b.f(72, 2)]], [_dst(b), b.pred(81), b.pred(87, 90)])


@_op(0x886)
def _voteu(b: Bits) -> Instruction:
    ops = [_hidden_if(b.ureg(16), b.f(16, 6) == 63), b.upred(81), b.pred(87, 90)]
    return _ins("VOTEU", [_VOTE[b.f(72, 2)]], ops)


@_op(0x20A, 0x80A, 0xA0A, 0xC0A)
def _fset(b: Bits) -> Instruction:
    mods = ["BF", _FCMP[b.f(76, 4)], "FTZ" if b.b(80) else "", _BOOL[b.f(74, 2)]]
    return _ins("FSET", mods, [_dst(b), _src_a(b), _src_b2(b, is_float=True), b.pred(87, 90)])


# ---------------------------------------------------------------------------
# Special registers and predicates

_SR = {
    0x00: "SR_LANEID",
    0x21: "SR_TID.X",
    0x22: "SR_TID.Y",
    0x23: "SR_TID.Z",
    0x25: "SR_CTAID.X",
    0x26: "SR_CTAID.Y",
    0x27: "SR_CTAID.Z",
    0x38: "SR_EQMASK",
    0x39: "SR_LTMASK",
    0x3A: "SR_LEMASK",
    0x3B: "SR_GTMASK",
    0x3C: "SR_GEMASK",
    0x50: "SR_CLOCKLO",
    0x51: "SR_CLOCKHI",
    0xFF: "SRZ",
}


def _sreg(b: Bits) -> Operand:
    n = b.f(72, 8)
    return Operand("sreg", _SR.get(n, f"SR{n}"))


@_op(0x919)
def _s2r(b: Bits) -> Instruction:
    return _ins("S2R", [], [_dst(b), _sreg(b)])


@_op(0x805)
def _cs2r(b: Bits) -> Instruction:
    return _ins("CS2R", ["" if b.b(80) else "32"], [_dst(b), _sreg(b)])


@_op(0x803)
def _p2r(b: Bits) -> Instruction:
    return _ins("P2R", [], [_dst(b), Operand("text", "PR"), b.reg(24), b.imm()])


@_op(0x804)
def _r2p(b: Bits) -> Instruction:
    return _ins("R2P", [], [Operand("text", "PR"), b.reg(24), b.imm()])


# ---------------------------------------------------------------------------
# More control flow


@_op(0x91A)
def _depbar(b: Bits) -> Instruction:
    return _ins("DEPBAR", ["LE" if b.b(47) else ""], [Operand("text", f"SB{b.f(44, 3)}"), Operand("imm", b.f(38, 6))])


@_op(0xB1D)
def _bar(b: Bits) -> Instruction:
    return _ins("BAR", ["SYNC"], [Operand("imm", b.f(54, 4))])


@_op(0x949)
def _brx(b: Bits) -> Instruction:
    target = Operand("brx", b.s(32, 50), index=b.reg(24))
    return _ins("BRX", [], _with_pred(b, [target]))


@_op(0x950)
def _ret(b: Bits) -> Instruction:
    target = Operand("ret", _target(b).value, index=b.reg(24))
    return _ins("RET", ["REL", "NODEC" if b.b(86) else ""], _with_pred(b, [target]))


@_op(0x944)
def _call(b: Bits) -> Instruction:
    return _ins("CALL", ["REL", "NOINC" if b.b(86) else ""], _with_pred(b, [_target(b)]))


@_op(0x942)
def _break(b: Bits) -> Instruction:
    return _ins("BREAK", [], _with_pred(b, [Operand("breg", b.f(16, 4))]))


@_op(0x948)
def _warpsync(b: Bits) -> Instruction:
    return _ins("WARPSYNC", [], _with_pred(b, [b.imm()]))


# ---------------------------------------------------------------------------
# Half-precision (packed f16) arithmetic

_HSWZ = {0: "", 1: ".F32", 2: ".H0_H0", 3: ".H1_H1"}


def _hreg(b: Bits, start: int, swz: int, *, neg: int | None, abs_: int | None, reuse: int | None) -> Operand:
    op = b.reg(start, neg=neg, abs_=abs_, reuse=reuse)
    op.suffix = _HSWZ[b.f(swz, 2)]
    return op


def _f16(bits: int) -> str:
    (f,) = struct.unpack("<e", struct.pack("<H", bits & 0xFFFF))
    return _f32(struct.unpack("<I", struct.pack("<f", f))[0])


def _himm(b: Bits) -> list[Operand]:
    return [Operand("himm", b.f(48, 16)), Operand("himm", b.f(32, 16))]


def _hcbuf(b: Bits) -> Operand:
    op = b.cbuf(neg=63, abs_=62)
    op.suffix = _HSWZ[b.f(60, 2)]
    return op


@_op(0x230, 0x430, 0x630, 0x830, 0xA30, 0xC30)
def _hadd2(b: Bits) -> Instruction:
    f = _form(b)
    a = _hreg(b, 24, 74, neg=72, abs_=73, reuse=122)
    if f == 1:
        srcs = [_hreg(b, 32, 60, neg=63, abs_=62, reuse=124)]
    elif f == 2:
        srcs = _himm(b)
    elif f == 3:
        srcs = [_hcbuf(b)]
    else:
        raise UnknownInstruction(f"form {f}")
    mods = ["F32" if b.b(78) else "", "FTZ" if b.b(80) else "", "SAT" if b.b(77) else ""]
    ins = _ins("HADD2", mods, [_dst(b), a, *srcs])
    if f == 3:
        ins.cbuf_space = True
    return ins


@_op(0x232, 0x832, 0xA32)
def _hmul2(b: Bits) -> Instruction:
    f = _form(b)
    a = _hreg(b, 24, 74, neg=72, abs_=73, reuse=122)
    if f == 1:
        srcs = [_hreg(b, 32, 60, neg=63, abs_=62, reuse=123)]
    elif f == 4:
        srcs = _himm(b)
    else:
        srcs = [_hcbuf(b)]
    mods = ["F32" if b.b(78) else "", "FTZ" if b.b(80) else "", "SAT" if b.b(77) else ""]
    return _ins("HMUL2", mods, [_dst(b), a, *srcs])


@_op(0x231, 0x431, 0x631, 0x831, 0xA31)
def _hfma2(b: Bits) -> Instruction:
    f = _form(b)
    a = _hreg(b, 24, 74, neg=72, abs_=73, reuse=122)
    if f == 1:
        srcs = [_hreg(b, 32, 60, neg=63, abs_=62, reuse=123), _hreg(b, 64, 81, neg=None, abs_=None, reuse=124)]
    else:
        raise UnknownInstruction(f"form {f}")
    if b.b(86) and not srcs[0].suffix:
        srcs[0].suffix = ".H0_NH1"  # seen on two instructions only
    mods = ["F32" if b.b(78) else "", "FTZ" if b.b(80) else "", "SAT" if b.b(77) else ""]
    return _ins("HFMA2", mods, [_dst(b), a, *srcs])


@_op(0x234, 0x434, 0x634)
def _hsetp2(b: Bits) -> Instruction:
    f = _form(b)
    a = _hreg(b, 24, 74, neg=72, abs_=73, reuse=122)
    srcs = [_hreg(b, 32, 60, neg=63, abs_=62, reuse=123)] if f == 1 else _himm(b)
    mods = [_FCMP[b.f(76, 4)], "FTZ" if b.b(80) else "", _BOOL[b.f(69, 2)]]
    return _ins("HSETP2", mods, [b.pred(81), b.pred(84), a, *srcs, b.pred(87, 90)])


@_op(0x23E, 0xA3E)
def _f2fp(b: Bits) -> Instruction:
    return _ins("F2FP", ["PACK_AB"], [_dst(b), b.reg(24, reuse=122), _src_b2(b, is_float=True)])


_I2F_SRC = {0: "U8", 1: "U16", 2: "U32", 3: "U64"}


@_op(0x306, 0xD06)
def _i2f(b: Bits) -> Instruction:
    size = b.f(84, 2)
    src = b.reg(32, neg=63, abs_=62) if _form(b) == 1 else b.ureg(32)
    sel = b.f(60, 2)
    if sel:
        src.suffix = f".B{sel}" if size == 0 else f".H{sel}"
    t = _I2F_SRC[size]
    if b.b(74):
        t = "S" + t[1:]
    return _ins("I2F", [t], [_dst(b), src])


# ---------------------------------------------------------------------------
# Warp-level operations

_SHFL = ["IDX", "UP", "DOWN", "BFLY"]


@_op(0x389, 0x589, 0x989, 0xF89)
def _shfl(b: Bits) -> Instruction:
    f = _form(b)
    lane = b.reg(32) if f in (1, 2) else Operand("imm", b.f(53, 5))
    clamp = b.reg(64) if f in (1, 4) else Operand("imm", b.f(40, 13))
    return _ins("SHFL", [_SHFL[b.f(58, 2)]], [b.pred(81), _dst(b), b.reg(24), lane, clamp])


@_op(0x822)
def _fswzadd(b: Bits) -> Instruction:
    pat = "".join("N" if b.f(32, 8) >> i & 1 else "P" for i in range(8))
    mods = ["FTZ" if b.b(80) else "", "NDV" if b.b(77) else ""]
    return _ins("FSWZADD", mods, [_dst(b), b.reg(24, reuse=122), b.reg(64, reuse=124), Operand("text", pat)])


@_op(0x925)
def _pixld(b: Bits) -> Instruction:
    return _ins("PIXLD", ["COVMASK"], [_dst(b)])


@_op(0x3A1)
def _match(b: Bits) -> Instruction:
    return _ins("MATCH", ["ALL" if not b.b(79) else "ANY", "U64" if b.b(73) else ""], [b.pred(81), _dst(b), b.reg(24)])


_REDUX = {0: "AND", 1: "OR", 2: "XOR", 3: "SUM", 4: "MIN", 5: "MAX"}


@_op(0x3C4)
def _redux(b: Bits) -> Instruction:
    return _ins("REDUX", [_REDUX.get(b.f(78, 3), "?")], [b.ureg(16), b.reg(24)])


# ---------------------------------------------------------------------------
# Texture sampling

_TEX_DIM = ["1D", "2D", "3D", "CUBE", "ARRAY_1D", "ARRAY_2D", "?6", "ARRAY_CUBE"]
_LOD = {0: "", 1: "LZ", 2: "LB", 3: "LL", 4: "?4", 5: "?5", 6: "LBA", 7: "LLA"}


def _tex_ops(b: Bits, handle: str, *, with_dim: bool = True) -> list[Operand]:
    ops = [b.reg(64), _dst(b), b.reg(24), b.reg(32)]
    if handle == "cbuf":
        ops += [Operand("imm", b.f(54, 5)), Operand("imm", b.f(40, 14))]
    elif handle == "ureg":
        ops += [b.ureg(40), Operand("imm", b.f(54, 5))]
    if with_dim:
        ops.append(Operand("text", _TEX_DIM[b.f(61, 3)]))
    mask = b.f(72, 4)
    ops.append(_hidden_if(Operand("imm", mask), mask == 0xF))
    return ops


_TEX_CACHE = {1: "", 6: "INVALID6", 7: "INVALID7"}


def _tex_cache(b: Bits) -> str:
    return _TEX_CACHE.get(b.f(84, 3), f"?{b.f(84, 3)}")


def _tex_handle(b: Bits) -> str:
    return {0xB: "cbuf", 0xF: "ureg", 0x3: "bindless"}[b.f(8, 4)]


@_op(0x361, 0xB60, 0xF60)
def _tex(b: Bits) -> Instruction:
    h = _tex_handle(b)
    mods = [
        "SCR" if b.b(60) else "",
        "B" if h == "bindless" else "",
        _LOD[b.f(87, 3)],
        "AOFFI" if b.b(76) else "",
        "DC" if b.b(78) else "",
        _tex_cache(b),
        "NODEP" if b.b(90) else "",
    ]
    return _ins("TEX", mods, _tex_ops(b, h))


@_op(0x366, 0xB66, 0xF66)
def _tld(b: Bits) -> Instruction:
    h = _tex_handle(b)
    mods = [
        "SCR" if b.b(60) else "",
        "B" if h == "bindless" else "",
        _LOD[b.f(87, 3)],
        "AOFFI" if b.b(76) else "",
        _tex_cache(b),
        "NODEP" if b.b(90) else "",
    ]
    return _ins("TLD", mods, _tex_ops(b, h))


_TLD4_COMP = ["R", "G", "B", "A"]


@_op(0x364, 0xB63, 0xF63)
def _tld4(b: Bits) -> Instruction:
    h = _tex_handle(b)
    mods = [
        "SCR" if b.b(60) else "",
        _TLD4_COMP[b.f(87, 2)],
        "AOFFI" if b.b(76) else "",
        "DC" if b.b(78) else "",
        _tex_cache(b),
        "NODEP" if b.b(90) else "",
    ]
    return _ins("TLD4", mods, _tex_ops(b, h))


@_op(0x36D, 0xB6C, 0xF6C)
def _txd(b: Bits) -> Instruction:
    h = _tex_handle(b)
    mods = ["AOFFI" if b.b(76) else "", _tex_cache(b), "NODEP" if b.b(90) else ""]
    return _ins("TXD", mods, _tex_ops(b, h))


_TXQ = {0: "TEX_HEADER_DIMENSION", 1: "TEX_HEADER_TEXTURE_TYPE", 2: "TEX_HEADER_SAMPLER_POS", 3: "?"}


@_op(0x370, 0xB6F, 0xF6F)
def _txq(b: Bits) -> Instruction:
    h = _tex_handle(b)
    ops = [b.reg(64), _dst(b), b.reg(24), Operand("text", _TXQ.get(b.f(62, 2), "?"))]
    if h == "cbuf":
        ops += [Operand("imm", b.f(54, 5)), Operand("imm", b.f(40, 14))]
    elif h == "ureg":
        ops += [b.ureg(40), Operand("imm", b.f(54, 5))]
    mask = b.f(72, 4)
    ops.append(_hidden_if(Operand("imm", mask), mask == 0xF))
    return _ins("TXQ", ["NODEP" if b.b(90) else ""], ops)


# ---------------------------------------------------------------------------
# Memory

_SCALE = {0: 1, 1: 4, 2: 8, 3: 16}


def _addr(base: Operand, offset: int, *, extra: Operand | None = None, scale: int = 1) -> Operand:
    """A memory address: base register (times scale), plus an optional uniform register, plus offset."""
    return Operand("addr", offset, index=base, extra=extra, scale=scale)


def _addr_text(op: Operand) -> str:
    base, extra, offset = op.index, op.extra, op.value
    parts = []
    if not _is_zero_reg(base):
        parts.append(base.core() + (f".X{op.scale}" if op.scale > 1 else ""))
    if extra is not None and not _is_zero_reg(extra):
        if parts:
            parts[-1] += ".U32"
        parts.append(extra.core())
    s = "+".join(parts)
    if offset:
        s = (s + "+" if s else "") + _hex(offset)
    elif not s:
        s = "RZ" if base.kind == "reg" else base.core()
    return f"[{s}]"


@_op(0x984)
def _lds(b: Bits) -> Instruction:
    a = _addr(b.reg(24), b.s(40, 24), scale=_SCALE[b.f(78, 2)])
    return _ins("LDS", [_MEM_SIZE[b.f(73, 3)]], [_dst(b), a])


@_op(0x388)
def _sts(b: Bits) -> Instruction:
    a = _addr(b.reg(24), b.s(40, 24), scale=_SCALE[b.f(78, 2)])
    return _ins("STS", [_MEM_SIZE[b.f(73, 3)]], [a, b.reg(32)])


_LDL_CACHE = {0: "EF", 1: "", 2: "EL", 3: "LU", 4: "EU", 5: "NA", 6: "?", 7: "?"}


@_op(0x983)
def _ldl(b: Bits) -> Instruction:
    a = _addr(b.reg(24), b.s(40, 24))
    return _ins("LDL", [_LDL_CACHE[b.f(84, 3)], _MEM_SIZE[b.f(73, 3)]], [_dst(b), a])


@_op(0x387)
def _stl(b: Bits) -> Instruction:
    a = _addr(b.reg(24), b.s(40, 24))
    return _ins("STL", [_LDL_CACHE[b.f(84, 3)], _MEM_SIZE[b.f(73, 3)]], [a, b.reg(32)])


_SEM = {0x8: "CONSTANT", 0xA: "STRONG.SM", 0xE: "STRONG.GPU", 0xC: "STRONG.CTA?", 0x0: "", 0x2: "?2"}


@_op(0x381, 0x981)
def _ldg(b: Bits) -> Instruction:
    ur = b.ureg(32) if b.f(0, 12) == 0x981 and b.b(91) else None
    a = _addr(b.reg(24), b.s(40, 24), extra=ur)
    mods = ["E" if b.b(72) else "", _MEM_SIZE[b.f(73, 3)], _SEM.get(b.f(76, 4), "?")]
    return _ins("LDG", mods, [_dst(b), a])


@_op(0x386, 0x986)
def _stg(b: Bits) -> Instruction:
    ur = b.ureg(64) if b.f(0, 12) == 0x986 and b.b(91) else None
    a = _addr(b.reg(24), b.s(40, 24), extra=ur)
    mods = ["E" if b.b(72) else "", _MEM_SIZE[b.f(73, 3)], _SEM.get(b.f(76, 4), "?")]
    return _ins("STG", mods, [a, b.reg(32)])


_ATOM_OP = ["ADD", "MIN", "MAX", "INC", "DEC", "AND", "OR", "XOR"]
_ATOM_TYPE = {0: "", 4: "F16x2.RN"}


@_op(0x38C)
def _atoms(b: Bits) -> Instruction:
    return _ins("ATOMS", [_ATOM_OP[b.f(87, 3)]], [_dst(b), _addr(b.reg(24), b.s(40, 24)), b.reg(32)])


@_op(0x3A8)
def _atomg(b: Bits) -> Instruction:
    mods = ["E" if b.b(72) else "", _ATOM_OP[b.f(87, 3)], _SEM.get(b.f(76, 4), "?")]
    return _ins("ATOMG", mods, [b.pred(81), _dst(b), _addr(b.reg(24), b.s(40, 24)), b.reg(32)])


@_op(0x98E)
def _red(b: Bits) -> Instruction:
    a = _addr(b.reg(24), b.s(40, 24), extra=b.ureg(64) if b.b(91) else None)
    mods = ["E" if b.b(72) else "", _ATOM_OP[b.f(87, 3)], _ATOM_TYPE.get(b.f(73, 3), "?"), _SEM.get(b.f(76, 4), "?")]
    return _ins("RED", mods, [a, b.reg(32)])


# ---------------------------------------------------------------------------
# Surfaces

_SU_DIM = ["1D", "1D_BUFFER", "1D_ARRAY", "2D", "2D_ARRAY", "3D", "?6", "?7"]
_SU_MASK = {1: "R", 3: "RG", 7: "RGB", 15: ""}


def _su_addr(b: Bits) -> Operand:
    """Surface coordinates: the first of consecutive registers, one per dimension."""
    return Operand("addr", 0, index=b.reg(24))


def _su_handle(b: Bits) -> list[Operand]:
    return [Operand("imm", b.f(54, 5)), Operand("imm", b.f(40, 14))]


def _su_mods(b: Bits, *, op: bool = False) -> list[str]:
    mask = b.f(72, 4)
    typed = mask == 0
    mods = ["D" if typed else "P", _SU_DIM[b.f(61, 3)], "" if typed else _SU_MASK.get(mask, f"M{mask:x}")]
    if op:
        mods += [_ATOM_OP[b.f(87, 3)], _SEM.get(b.f(76, 4), "?")]
    return mods + ["IGN"]


@_op(0xB9B)
def _sust(b: Bits) -> Instruction:
    return _ins("SUST", _su_mods(b), [_su_addr(b), b.reg(32), *_su_handle(b)])


@_op(0xB97)
def _suld(b: Bits) -> Instruction:
    return _ins("SULD", _su_mods(b), [_dst(b), _su_addr(b), *_su_handle(b)])


@_op(0x59F)
def _sured(b: Bits) -> Instruction:
    return _ins("SURED", _su_mods(b, op=True), [_su_addr(b), b.reg(32), *_su_handle(b)])


@_op(0x593)
def _suatom(b: Bits) -> Instruction:
    ops = [_dst(b), _su_addr(b), b.reg(32), *_su_handle(b)]
    return _ins("SUATOM", _su_mods(b, op=True), ops)
