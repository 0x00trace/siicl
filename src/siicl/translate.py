"""Translate compute programs to SPIR-V for Vulkan.

`translate(instructions, info)` returns a Translation: a SPIR-V module and the
resource bindings it expects. SPIR-V requires structured control flow, so the
program's branches become a loop around a switch on a block number: each case
runs one basic block of the original program and stores the number of the
next one. That works for any branch pattern, and it gives a place to count
steps, so a program that loops on its inputs stops after a limit.

What each instruction computes follows siicl.interp (docs/interpreter.md), the
reference that translations are checked against. Instructions, modifiers and
operands without a translation raise Unsupported.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

from siicl import sass, spirv
from siicl.nvshader import ComputeInfo

EXIT_BLOCK = 0xFFFFFFFF
DEFAULT_STEP_LIMIT = 1_000_000  # basic blocks per thread
STATUS_STEP_LIMIT = 1  # status bit: a thread stopped at the step limit
STATUS_WORDS = 2  # status flags, then the number of MUFU log entries
NAN = 0x7FFFFFFF
SU_COORDS = {"1D": 1, "1D_ARRAY": 2, "2D": 2, "2D_ARRAY": 3, "3D": 3}
SU_IMAGE = {"1D": ("1D", False), "1D_ARRAY": ("1D", True), "2D": ("2D", False), "2D_ARRAY": ("2D", True)}
SU_IMAGE["3D"] = ("3D", False)
SU_CHANNELS = {"R": 1, "RG": 2, "RGB": 3, "": 4}


class Unsupported(ValueError):
    """The translator has no translation for this instruction, modifier or operand."""


@dataclass(frozen=True)
class Binding:
    """One descriptor of set 0."""

    binding: int
    kind: str  # "status", "cbuf" or "surface"
    bank: int = 0  # cbuf: constant buffer bank
    handle: tuple[int, int] = (0, 0)  # surface: the two numbers of the instruction's handle
    dim: str = ""  # surface: 1D, 1D_ARRAY, 2D, 2D_ARRAY or 3D
    channels: int = 0  # surface: 32-bit words per texel, 1 (R32ui) or 4 (Rgba32ui)


@dataclass
class Translation:
    spirv: bytes
    bindings: list[Binding]
    local_size: tuple[int, int, int]
    step_limit: int
    mufu_log: bool


def surfaces(instructions: list[sass.Instruction]) -> dict[tuple[int, int], tuple[str, int]]:
    """The dimension and channel count of each surface handle the program uses.

    A surface with atomic operations has one channel (R32ui); others have four (Rgba32ui).
    """
    found: dict[tuple[int, int], tuple[str, int]] = {}
    for ins in instructions:
        if ins.name not in ("SUST", "SULD", "SURED", "SUATOM"):
            continue
        handle = (ins.operands[-2].value, ins.operands[-1].value)
        dim = ins.mods[1]
        atomic = ins.name in ("SURED", "SUATOM")
        if atomic and ins.mods[0] != "D" or not atomic and ins.mods[0] != "P":
            raise Unsupported(f"{ins.name}.{ins.mods[0]}")
        mask = 1 if atomic else SU_CHANNELS.get(ins.mods[2])
        if mask is None:
            raise Unsupported(f"surface component mask {ins.mods[2]}")
        old_dim, old_atomic, old_mask = found.get(handle, (dim, atomic, mask))
        if old_dim != dim:
            raise Unsupported(f"surface {handle} is used as {old_dim} and {dim}")
        found[handle] = (dim, old_atomic or atomic, max(old_mask, mask))
    out = {}
    for handle, (dim, atomic, mask) in found.items():
        if atomic and mask > 1:
            raise Unsupported(f"surface {handle} has atomics and stores of {mask} components")
        out[handle] = (dim, 1 if atomic else 4)
    return out


def translate(
    instructions: list[sass.Instruction],
    info: ComputeInfo,
    *,
    step_limit: int = DEFAULT_STEP_LIMIT,
    mufu_log: bool = False,
) -> Translation:
    """Translate a compute program. With mufu_log, the shader records each MUFU input and result."""
    return _Translator(instructions, info, step_limit, mufu_log).run()


class _Translator:
    def __init__(self, instructions: list[sass.Instruction], info: ComputeInfo, step_limit: int, log: bool) -> None:
        self.ins = instructions
        self.info = info
        self.step_limit = step_limit
        self.log = log
        self.m = m = spirv.Module()
        for cap in ("Shader", "Int64", "ImageQuery", "SignedZeroInfNanPreserve"):
            m.capability(cap)
        self.u32, self.u64, self.i32, self.f32, self.bool = m.uint(), m.int(64), m.int(32, True), m.float(), m.bool()
        self.uvec3, self.uvec4 = m.vector(self.u32, 3), m.vector(self.u32, 4)
        self.vars: list[int] = []  # OpVariable words for the entry block
        self.regs: dict[tuple[str, int], int] = {}
        self.interface: list[int] = []
        self.bindings: list[Binding] = []
        self.cbuf_vars: dict[int, int] = {}
        self.images: dict[tuple[int, int], tuple[int, int, str]] = {}  # handle -> (variable, type, dim)
        self.builtins: dict[str, int] = {}

    # -- resources ------------------------------------------------------------------

    def _ssbo(self, binding: Binding) -> int:
        m = self.m
        block = m.struct(m.runtime_array(self.u32, 4), key="words")
        if block not in getattr(self, "_blocks", set()):
            m.decorate(block, "Block")
            m.member_decorate(block, 0, "Offset", 0)
            self._blocks = getattr(self, "_blocks", set()) | {block}
        var = m.variable(m.pointer("StorageBuffer", block), "StorageBuffer")
        m.decorate(var, "DescriptorSet", 0)
        m.decorate(var, "Binding", binding.binding)
        self.interface.append(var)
        self.bindings.append(binding)
        return var

    def _setup_resources(self) -> None:
        m = self.m
        self.status = self._ssbo(Binding(0, "status"))
        m.name(self.status, "status")
        banks = sorted({op.bank for ins in self.ins for op in _all_operands(ins) if op.kind == "cbuf"})
        for bank in banks:
            self.cbuf_vars[bank] = var = self._ssbo(Binding(len(self.bindings), "cbuf", bank=bank))
            m.name(var, f"cbuf{bank}")
        for handle, (dim, channels) in sorted(surfaces(self.ins).items()):
            image_dim, arrayed = SU_IMAGE[dim]
            if image_dim == "1D":
                m.capability("Image1D")
            ty = m.image(self.u32, image_dim, arrayed=arrayed, fmt="R32ui" if channels == 1 else "Rgba32ui")
            var = m.variable(m.pointer("UniformConstant", ty), "UniformConstant")
            b = Binding(len(self.bindings), "surface", handle=handle, dim=dim, channels=channels)
            m.decorate(var, "DescriptorSet", 0)
            m.decorate(var, "Binding", b.binding)
            m.name(var, f"surface_{handle[0]:x}_{handle[1]:x}")
            self.interface.append(var)
            self.bindings.append(b)
            self.images[handle] = (var, ty, dim)
        self.shared_words = self.info.shared_size // 4
        if self.shared_words:
            arr = m.array(self.u32, self.shared_words)
            self.shared = m.variable(m.pointer("Workgroup", arr), "Workgroup")
            m.name(self.shared, "shared")
            self.interface.append(self.shared)

    def builtin(self, name: str) -> int:
        if name not in self.builtins:
            m = self.m
            var = m.variable(m.pointer("Input", self.uvec3), "Input")
            m.decorate(var, "BuiltIn", spirv.BUILTIN[name])
            self.interface.append(var)
            self.builtins[name] = var
        return self.builtins[name]

    # -- variables ------------------------------------------------------------------------

    def var(self, kind: str, n: int) -> int:
        """The Function variable for register n of a kind: R, P, UR, UP, or an internal name."""
        key = (kind, n)
        if key not in self.regs:
            m = self.m
            ty = self.bool if kind in ("P", "UP") else self.u32
            init = m.boolean(False) if ty == self.bool else m.u32(0)
            vid = m.id()
            self.vars += spirv._encode(spirv.OPS["OpVariable"], [m.pointer("Function", ty), vid, 7, init])
            m.name(vid, f"{kind}{n}")
            self.regs[key] = vid
        return self.regs[key]

    def load(self, kind: str, n: int) -> int:
        ty = self.bool if kind in ("P", "UP") else self.u32
        return self.m.val("OpLoad", ty, self.var(kind, n))

    def store(self, kind: str, n: int, value: int) -> None:
        self.m.op("OpStore", self.var(kind, n), value)

    # -- small helpers ----------------------------------------------------------------------

    def v(self, opname: str, ty: int, *args: int) -> int:
        return self.m.val(opname, ty, *args)

    def u(self, value: int) -> int:
        return self.m.u32(value)

    def b(self, opname: str, *args: int) -> int:
        return self.m.val(opname, self.bool, *args)

    def sel(self, cond: int, a: int, b: int, ty: int | None = None) -> int:
        return self.m.val("OpSelect", ty or self.u32, cond, a, b)

    def as_f(self, x: int) -> int:
        return self.v("OpBitcast", self.f32, x)

    def as_u(self, x: int) -> int:
        return self.v("OpBitcast", self.u32, x)

    def fop(self, opname: str, *args: int) -> int:
        r = self.v(opname, self.f32, *args)
        self.m.decorate(r, "NoContraction")
        return r

    def if_(self, cond: int):
        """Start a selection: code until end_if() runs only when cond is true."""
        m = self.m
        then, after = m.id(), m.id()
        m.op("OpSelectionMerge", after, 0)
        m.op("OpBranchConditional", cond, then, after)
        m.label(then)
        return after

    def end_if(self, after: int) -> None:
        self.m.op("OpBranch", after)
        self.m.label(after)

    def ftz(self, x: int) -> int:
        exp = self.v("OpBitwiseAnd", self.u32, x, self.u(0x7F800000))
        sub = self.b("OpIEqual", exp, self.u(0))
        return self.sel(sub, self.v("OpBitwiseAnd", self.u32, x, self.u(0x80000000)), x)

    def is_nan(self, x: int) -> int:
        return self.b("OpUGreaterThan", self.v("OpBitwiseAnd", self.u32, x, self.u(0x7FFFFFFF)), self.u(0x7F800000))

    def sat(self, x: int) -> int:
        neg = self.b("OpINotEqual", self.v("OpBitwiseAnd", self.u32, x, self.u(0x80000000)), self.u(0))
        zero = self.b("OpLogicalOr", self.is_nan(x), neg)
        return self.sel(zero, self.u(0), self.m.ext(self.u32, "UMin", x, self.u(0x3F800000)))

    def to64(self, x: int) -> int:
        return self.v("OpUConvert", self.u64, x)

    def lo32(self, x: int) -> int:
        return self.v("OpUConvert", self.u32, x)

    def bit(self, p: int) -> int:
        return self.sel(p, self.u(1), self.u(0))

    # -- operands ----------------------------------------------------------------------------

    def src(self, op: sass.Operand, *, fp: bool = False, not_neg: bool = False) -> int:
        k, n = op.kind, op.value
        if op.suffix:
            raise Unsupported(f"operand suffix {op.suffix}")
        if k == "reg":
            x = self.u(0) if n == 255 else self.load("R", n)
        elif k == "ureg":
            x = self.u(0) if n == 63 else self.load("UR", n)
        elif k in ("imm", "fimm"):
            x = self.u(n)
        elif k == "cbuf":
            if op.index is None or sass._is_zero_reg(op.index):
                x = self.cbuf_word(op.bank, self.u(n >> 2)) if n % 4 == 0 else self.cbuf_bytes(op.bank, self.u(n), 4)[0]
            else:
                off = self.v("OpIAdd", self.u32, self.src(op.index), self.u(n))
                x = self.cbuf_bytes(op.bank, off, 4)[0]
        else:
            raise Unsupported(f"{k} operand")
        if fp:
            if op.abs:
                x = self.v("OpBitwiseAnd", self.u32, x, self.u(0x7FFFFFFF))
            if op.neg:
                x = self.v("OpBitwiseXor", self.u32, x, self.u(0x80000000))
            return x
        if op.abs:
            raise Unsupported("absolute value of an integer operand")
        if op.neg:
            x = self.v("OpNot" if not_neg else "OpSNegate", self.u32, x)
        return x

    def psrc(self, op: sass.Operand) -> int:
        if op.kind not in ("pred", "upred"):
            raise Unsupported(f"{op.kind} operand as a predicate")
        x = self.m.boolean(True) if op.value == 7 else self.load("P" if op.kind == "pred" else "UP", op.value)
        return self.b("OpLogicalNot", x) if op.not_ else x

    def dst(self, op: sass.Operand, value: int, index: int = 0) -> None:
        n = op.value + index
        if op.kind == "reg":
            if op.value != 255 and n < 255:
                self.store("R", n, value)
        elif op.kind == "ureg":
            if op.value != 63 and n < 63:
                self.store("UR", n, value)
        else:
            raise Unsupported(f"{op.kind} operand as a destination")

    def pdst(self, op: sass.Operand, value: int) -> None:
        if op.kind not in ("pred", "upred"):
            raise Unsupported(f"{op.kind} operand as a predicate destination")
        if op.value != 7:
            self.store("P" if op.kind == "pred" else "UP", op.value, value)

    # -- constant buffers -------------------------------------------------------------------

    def cbuf_word(self, bank: int, index: int) -> int:
        """Word index of a bank, or 0 outside it."""
        m = self.m
        buf = self.cbuf_vars[bank]
        length = self.v("OpArrayLength", self.u32, buf, 0)
        inside = self.b("OpULessThan", index, length)
        safe = self.sel(inside, index, self.u(0))
        ptr = self.v("OpAccessChain", m.pointer("StorageBuffer", self.u32), buf, self.u(0), safe)
        return self.sel(inside, self.v("OpLoad", self.u32, ptr), self.u(0))

    def cbuf_bytes(self, bank: int, offset: int, size: int) -> list[int]:
        """size bytes at a byte offset, as 32-bit words; all zero unless every byte is inside the bank."""
        buf = self.cbuf_vars[bank]
        length = self.to64(self.v("OpArrayLength", self.u32, buf, 0))
        end = self.v("OpIAdd", self.u64, self.to64(offset), self.m.u64(size))
        inside = self.b("OpULessThanEqual", end, self.v("OpShiftLeftLogical", self.u64, length, self.m.u64(2)))
        shift = self.to64(
            self.v("OpShiftLeftLogical", self.u32, self.v("OpBitwiseAnd", self.u32, offset, self.u(3)), self.u(3))
        )
        first = self.v("OpShiftRightLogical", self.u32, offset, self.u(2))
        words = []
        for k in range((size + 3) // 4):
            i = self.v("OpIAdd", self.u32, first, self.u(k))
            lo, hi = self.cbuf_word(bank, i), self.cbuf_word(bank, self.v("OpIAdd", self.u32, i, self.u(1)))
            pair = self.v(
                "OpBitwiseOr",
                self.u64,
                self.v("OpShiftLeftLogical", self.u64, self.to64(hi), self.m.u64(32)),
                self.to64(lo),
            )
            words.append(self.sel(inside, self.lo32(self.v("OpShiftRightLogical", self.u64, pair, shift)), self.u(0)))
        if size < 4:
            words[0] = self.v("OpBitwiseAnd", self.u32, words[0], self.u((1 << (8 * size)) - 1))
        return words

    # -- program structure ---------------------------------------------------------------------

    def run(self) -> Translation:
        m = self.m
        self._setup_resources()
        n = len(self.ins)
        leaders = {0}
        for i, ins in enumerate(self.ins):
            if ins.name in ("BRA", "CALL"):
                target = next(op for op in ins.operands if op.kind == "label").value // 16
                if not 0 <= target < n:
                    raise Unsupported(f"branch target {target * 16:#x} is outside the program")
                leaders.add(target)
            if ins.name in ("BRA", "CALL", "EXIT", "BRX", "RET") and i + 1 < n:
                leaders.add(i + 1)
        starts = sorted(leaders)
        blocks = list(zip(starts, starts[1:] + [n], strict=True))
        body: list[int] = []
        m.code = body
        header, dispatch, smerge, cont, merge, default = (m.id() for _ in range(6))
        labels = {start: m.id() for start, _ in blocks}
        pc = self.var("pc", 0)
        steps = self.var("steps", 0)
        m.op("OpBranch", header)
        m.label(header)
        m.op("OpLoopMerge", merge, cont, 0)
        m.op("OpBranch", dispatch)
        m.label(dispatch)
        pcv = self.v("OpLoad", self.u32, pc)
        m.op("OpSelectionMerge", smerge, 0)
        cases = [w for start, _ in blocks for w in (start, labels[start])]
        m.op("OpSwitch", pcv, default, *cases)
        for start, end in blocks:
            m.label(labels[start])
            ended = False
            for i in range(start, end):
                try:
                    ended = self.emit(self.ins[i], i)
                except Unsupported as err:
                    raise Unsupported(f"{i * 16:#06x} {self.ins[i].text()}: {err}") from err
            if not ended:
                m.op("OpStore", pc, self.u(end if end < n else EXIT_BLOCK))
            m.op("OpBranch", smerge)
        m.label(default)
        m.op("OpStore", pc, self.u(EXIT_BLOCK))
        m.op("OpBranch", smerge)
        m.label(smerge)
        count = self.v("OpIAdd", self.u32, self.v("OpLoad", self.u32, steps), self.u(1))
        m.op("OpStore", steps, count)
        after = self.if_(self.b("OpUGreaterThanEqual", count, self.u(self.step_limit)))
        self.status_or(STATUS_STEP_LIMIT)
        m.op("OpStore", pc, self.u(EXIT_BLOCK))
        self.end_if(after)
        m.op("OpBranch", cont)
        m.label(cont)
        done = self.b("OpIEqual", self.v("OpLoad", self.u32, pc), self.u(EXIT_BLOCK))
        m.op("OpBranchConditional", done, merge, header)
        m.label(merge)
        m.op("OpReturn")
        m.op("OpFunctionEnd")
        fn, entry = m.id(), m.id()
        head = spirv._encode(spirv.OPS["OpFunction"], [m.void(), fn, 0, m.function_type(m.void())])
        head += spirv._encode(spirv.OPS["OpLabel"], [entry])
        m.code = head + self.vars + body
        m.name(fn, "main")
        m.entry_point(fn, "main", self.interface)
        m.execution_mode(fn, "LocalSize", *self.info.local_size)
        m.execution_mode(fn, "SignedZeroInfNanPreserve", 32)
        return Translation(m.binary(), self.bindings, self.info.local_size, self.step_limit, self.log)

    def status_word(self, index: int) -> int:
        m = self.m
        return self.v("OpAccessChain", m.pointer("StorageBuffer", self.u32), self.status, self.u(0), self.u(index))

    def status_or(self, bits: int) -> None:
        self.v("OpAtomicOr", self.u32, self.status_word(0), self.u(spirv.SCOPE["Device"]), self.u(0), self.u(bits))

    # -- instructions --------------------------------------------------------------------------

    def emit(self, ins: sass.Instruction, index: int) -> bool:
        """Translate one instruction. Returns True if it stored the next block number."""
        name = ins.name
        if name in _CONTROL:
            return _CONTROL[name](self, ins, index)
        rule = _RULES.get(name)
        if rule is None:
            raise Unsupported(f"no translation for {name}")
        after = self.if_(self.psrc(ins.guard)) if ins.guard is not None else None
        rule(self, ins)
        if after is not None:
            self.end_if(after)
        return False

    def cond(self, ins: sass.Instruction) -> int:
        """Guard and predicate operand (bit 87) together."""
        c = self.m.boolean(True)
        if ins.guard is not None:
            c = self.psrc(ins.guard)
        pred = ins.operands[0] if ins.operands and ins.operands[0].kind == "pred" else None
        if pred is not None and not sass._is_pt(pred):
            c = self.b("OpLogicalAnd", c, self.psrc(pred))
        return c


def _all_operands(ins: sass.Instruction):
    for op in ins.operands:
        yield op
        if op.index is not None:
            yield op.index


def _mods(ins: sass.Instruction, allowed: set[str] = frozenset()) -> set[str]:
    mods = {m for m in ins.mods if m}
    extra = mods - allowed
    if extra:
        raise Unsupported(f"modifier {'.'.join(sorted(extra))}")
    return mods


_RULES: dict = {}
_CONTROL: dict = {}


def _rule(*names: str):
    def reg(f):
        for n in names:
            _RULES[n] = f
        return f

    return reg


def _control(*names: str):
    def reg(f):
        for n in names:
            _CONTROL[n] = f
        return f

    return reg


# -- control flow ----------------------------------------------------------------------------------


def _label(ins: sass.Instruction) -> int:
    return next(op for op in ins.operands if op.kind == "label").value // 16


@_control("BRA")
def _bra(t: _Translator, ins: sass.Instruction, index: int) -> bool:
    _mods(ins)
    target = t.u(_label(ins))
    nxt = t.u(index + 1 if index + 1 < len(t.ins) else EXIT_BLOCK)
    t.m.op("OpStore", t.var("pc", 0), t.sel(t.cond(ins), target, nxt))
    return True


@_control("CALL")
def _call(t: _Translator, ins: sass.Instruction, index: int) -> bool:
    _mods(ins, {"REL", "NOINC"})
    if "NOINC" not in ins.mods:
        raise Unsupported("CALL that returns")
    return _bra(t, sass.Instruction(ins.address, ins.opcode, "BRA", [], ins.guard, ins.operands), index)


@_control("EXIT")
def _exit(t: _Translator, ins: sass.Instruction, index: int) -> bool:
    _mods(ins)
    nxt = t.u(index + 1 if index + 1 < len(t.ins) else EXIT_BLOCK)
    t.m.op("OpStore", t.var("pc", 0), t.sel(t.cond(ins), t.u(EXIT_BLOCK), nxt))
    return True


@_control("BSSY", "BSYNC", "BREAK", "WARPSYNC", "NOP", "DEPBAR")
def _no_effect(t: _Translator, ins: sass.Instruction, index: int) -> bool:
    # Convergence barriers only change which threads run together, which the loop doesn't model.
    return False


@_control("BAR")
def _bar(t: _Translator, ins: sass.Instruction, index: int) -> bool:
    _mods(ins, {"SYNC"})
    if ins.guard is not None or ins.operands[0].value != 0:
        raise Unsupported("BAR.SYNC with a guard or another barrier than 0")
    sem = spirv.SEMANTICS["AcquireRelease"] | spirv.SEMANTICS["WorkgroupMemory"] | spirv.SEMANTICS["ImageMemory"]
    wg = t.u(spirv.SCOPE["Workgroup"])
    t.m.op("OpControlBarrier", wg, wg, t.u(sem))
    return False


# -- floating point ---------------------------------------------------------------------------------

_ROUND = {"RM", "RP", "RZ"}


def _rounding(mods: set[str]) -> str:
    return next((m for m in mods if m in _ROUND), "")


def _finish(t: _Translator, r: int, mods: set[str]) -> int:
    if "FTZ" in mods:
        r = t.ftz(r)
    return t.sat(r) if "SAT" in mods else r


def _toward_zero(t: _Translator, r: int, err: int, a: int, b: int) -> int:
    """Round-to-nearest result r and its exact error -> the result rounded toward zero."""
    rf = t.as_f(r)
    zero = t.m.f32(0)
    pos = t.b("OpLogicalAnd", t.b("OpFOrdGreaterThan", rf, zero), t.b("OpFOrdLessThan", err, zero))
    neg = t.b("OpLogicalAnd", t.b("OpFOrdLessThan", rf, zero), t.b("OpFOrdGreaterThan", err, zero))
    r = t.sel(t.b("OpLogicalOr", pos, neg), t.v("OpISub", t.u32, r, t.u(1)), r)
    finite_in = t.b("OpLogicalNot", t.b("OpLogicalOr", t.b("OpIsInf", t.as_f(a)), t.b("OpIsInf", t.as_f(b))))
    overflow = t.b("OpLogicalAnd", t.b("OpIsInf", t.as_f(r)), finite_in)
    largest = t.v("OpBitwiseOr", t.u32, t.v("OpBitwiseAnd", t.u32, r, t.u(0x80000000)), t.u(0x7F7FFFFF))
    return t.sel(overflow, largest, r)


@_rule("FFMA")
def _ffma(t: _Translator, ins: sass.Instruction) -> None:
    mods = _mods(ins, {"FTZ", "SAT"})
    d, a, b, c = ins.operands
    xs = [t.src(op, fp=True) for op in (a, b, c)]
    if "FTZ" in mods:
        xs = [t.ftz(x) for x in xs]
    r = t.as_u(t.m.ext(t.f32, "Fma", *(t.as_f(x) for x in xs)))
    t.dst(d, _finish(t, r, mods))


@_rule("FADD")
def _fadd(t: _Translator, ins: sass.Instruction) -> None:
    mods = _mods(ins, {"FTZ", "SAT", "RZ"})
    d, a, b = ins.operands
    x, y = t.src(a, fp=True), t.src(b, fp=True)
    if "FTZ" in mods:
        x, y = t.ftz(x), t.ftz(y)
    xf, yf = t.as_f(x), t.as_f(y)
    s = t.fop("OpFAdd", xf, yf)
    r = t.as_u(s)
    if _rounding(mods) == "RZ":
        bb = t.fop("OpFSub", s, xf)
        err = t.fop("OpFAdd", t.fop("OpFSub", xf, t.fop("OpFSub", s, bb)), t.fop("OpFSub", yf, bb))
        r = _toward_zero(t, r, err, x, y)
    t.dst(d, _finish(t, r, mods))


_FMUL_SCALE = {"D8": 0x3E000000, "D4": 0x3E800000, "D2": 0x3F000000, "M2": 0x40000000, "M4": 0x40800000}
_FMUL_SCALE["M8"] = 0x41000000


@_rule("FMUL")
def _fmul(t: _Translator, ins: sass.Instruction) -> None:
    mods = _mods(ins, {"FTZ", "SAT", "RZ", *_FMUL_SCALE})
    d, a, b = ins.operands
    x, y = t.src(a, fp=True), t.src(b, fp=True)
    if "FTZ" in mods:
        x, y = t.ftz(x), t.ftz(y)
    xf, yf = t.as_f(x), t.as_f(y)
    p = t.fop("OpFMul", xf, yf)
    scale = next((_FMUL_SCALE[m] for m in mods if m in _FMUL_SCALE), None)
    if scale is not None:
        if _rounding(mods):
            raise Unsupported("FMUL with a scale and a rounding mode")
        p = t.fop("OpFMul", p, t.m.f32(scale))
    r = t.as_u(p)
    if _rounding(mods) == "RZ":
        err = t.m.ext(t.f32, "Fma", xf, yf, t.v("OpFNegate", t.f32, p))
        r = _toward_zero(t, r, err, x, y)
    t.dst(d, _finish(t, r, mods))


def _order_key(t: _Translator, x: int) -> int:
    """An unsigned integer that orders floats as their values do, with -0 just below +0.

    Comparing floats as integers keeps the GPU's handling of subnormal numbers out of the result:
    some GPUs flush them in float comparisons even without FTZ.
    """
    negative = t.b("OpINotEqual", t.v("OpBitwiseAnd", t.u32, x, t.u(0x80000000)), t.u(0))
    return t.sel(negative, t.v("OpNot", t.u32, x), t.v("OpBitwiseOr", t.u32, x, t.u(0x80000000)))


def _fminmax(t: _Translator, x: int, y: int, is_min: int) -> int:
    xn, yn = t.is_nan(x), t.is_nan(y)
    below = t.b("OpULessThan", _order_key(t, x), _order_key(t, y))
    r = t.sel(is_min, t.sel(below, x, y), t.sel(below, y, x))
    r = t.sel(yn, x, r)
    r = t.sel(xn, t.sel(yn, t.u(NAN), y), r)
    return r


@_rule("FMNMX")
def _fmnmx(t: _Translator, ins: sass.Instruction) -> None:
    mods = _mods(ins, {"FTZ"})
    d, a, b, p = ins.operands
    x, y = t.src(a, fp=True), t.src(b, fp=True)
    if "FTZ" in mods:
        x, y = t.ftz(x), t.ftz(y)
    t.dst(d, _fminmax(t, x, y, t.psrc(p)))


@_rule("FSEL", "SEL")
def _sel(t: _Translator, ins: sass.Instruction) -> None:
    _mods(ins)
    d, a, b, p = ins.operands
    fp = ins.name == "FSEL"
    t.dst(d, t.sel(t.psrc(p), t.src(a, fp=fp), t.src(b, fp=fp)))


_FCMP = {
    "LT": "OpFOrdLessThan",
    "EQ": "OpFOrdEqual",
    "LE": "OpFOrdLessThanEqual",
    "GT": "OpFOrdGreaterThan",
    "NE": "OpFOrdNotEqual",
    "GE": "OpFOrdGreaterThanEqual",
    "LTU": "OpFUnordLessThan",
    "EQU": "OpFUnordEqual",
    "LEU": "OpFUnordLessThanEqual",
    "GTU": "OpFUnordGreaterThan",
    "NEU": "OpFUnordNotEqual",
    "GEU": "OpFUnordGreaterThanEqual",
}
_BOOL = {"AND": "OpLogicalAnd", "OR": "OpLogicalOr", "XOR": "OpLogicalNotEqual"}


def _fcompare(t: _Translator, name: str, x: int, y: int) -> int:
    if name in _FCMP:
        # On the bits, as for FMNMX: +0 and -0 are equal, NaN is unordered.
        kx, ky = _order_key(t, x), _order_key(t, y)
        zeros = t.b("OpIEqual", t.v("OpBitwiseAnd", t.u32, t.v("OpBitwiseOr", t.u32, x, y), t.u(0x7FFFFFFF)), t.u(0))
        eq = t.b("OpLogicalOr", t.b("OpIEqual", x, y), zeros)
        lt = t.b("OpLogicalAnd", t.b("OpULessThan", kx, ky), t.b("OpLogicalNot", eq))
        gt = t.b("OpLogicalAnd", t.b("OpUGreaterThan", kx, ky), t.b("OpLogicalNot", eq))
        base = name[:2]
        r = {
            "LT": lt,
            "EQ": eq,
            "LE": t.b("OpLogicalOr", lt, eq),
            "GT": gt,
            "NE": t.b("OpLogicalNot", eq),
            "GE": t.b("OpLogicalOr", gt, eq),
        }[base]
        unordered = t.b("OpLogicalOr", t.is_nan(x), t.is_nan(y))
        if name.endswith("U"):
            return t.b("OpLogicalOr", unordered, r)
        return t.b("OpLogicalAnd", t.b("OpLogicalNot", unordered), r)
    if name == "NUM":
        return t.b("OpLogicalNot", t.b("OpLogicalOr", t.is_nan(x), t.is_nan(y)))
    if name == "NAN":
        return t.b("OpLogicalOr", t.is_nan(x), t.is_nan(y))
    if name in ("F", "T"):
        return t.m.boolean(name == "T")
    raise Unsupported(f"float comparison {name}")


def _combine(t: _Translator, op: str, c: int, q: int) -> int:
    if op not in _BOOL:
        raise Unsupported(f"boolean operation {op}")
    return t.b(_BOOL[op], c, q)


@_rule("FSETP")
def _fsetp(t: _Translator, ins: sass.Instruction) -> None:
    cmp_name, _, bop = ins.mods
    mods = _mods(ins, {cmp_name, bop, "FTZ"})
    p0, p1, a, b, pin = ins.operands
    x, y = t.src(a, fp=True), t.src(b, fp=True)
    if "FTZ" in mods:
        x, y = t.ftz(x), t.ftz(y)
    c, q = _fcompare(t, cmp_name, x, y), t.psrc(pin)
    t.pdst(p0, _combine(t, bop, c, q))
    t.pdst(p1, _combine(t, bop, t.b("OpLogicalNot", c), q))


@_rule("FSET")
def _fset(t: _Translator, ins: sass.Instruction) -> None:
    _, cmp_name, _, bop = ins.mods
    mods = _mods(ins, {"BF", cmp_name, bop, "FTZ"})
    d, a, b, pin = ins.operands
    x, y = t.src(a, fp=True), t.src(b, fp=True)
    if "FTZ" in mods:
        x, y = t.ftz(x), t.ftz(y)
    r = _combine(t, bop, _fcompare(t, cmp_name, x, y), t.psrc(pin))
    t.dst(d, t.sel(r, t.u(0x3F800000), t.u(0)))


_MUFU = {"RSQ": "InverseSqrt", "SQRT": "Sqrt", "EX2": "Exp2", "LG2": "Log2", "SIN": "Sin", "COS": "Cos"}
TWO_PI = 0x40C90FDB


@_rule("MUFU")
def _mufu(t: _Translator, ins: sass.Instruction) -> None:
    func = ins.mods[0]
    _mods(ins, {func})
    if func not in _MUFU and func != "RCP":
        raise Unsupported(f"MUFU.{func}")
    d, a = ins.operands
    raw = t.src(a, fp=True)
    x = t.as_f(t.ftz(raw))
    if func == "RCP":
        r = t.fop("OpFDiv", t.m.f32(0x3F800000), x)
    elif func in ("SIN", "COS"):
        # MUFU takes turns. Taking away whole turns first is exact, and keeps the angle within pi.
        turns = t.fop("OpFSub", x, t.m.ext(t.f32, "RoundEven", x))
        r = t.m.ext(t.f32, _MUFU[func], t.fop("OpFMul", turns, t.m.f32(TWO_PI)))
    else:
        r = t.m.ext(t.f32, _MUFU[func], x)
    out = t.ftz(t.as_u(r))
    if t.log:
        _log_mufu(t, ins.address, raw, out)
    t.dst(d, out)


def _log_mufu(t: _Translator, address: int, raw: int, out: int) -> None:
    m = t.m
    slot = t.v("OpAtomicIAdd", t.u32, t.status_word(1), t.u(spirv.SCOPE["Device"]), t.u(0), t.u(1))
    length = t.v("OpArrayLength", t.u32, t.status, 0)
    first = t.v("OpIAdd", t.u32, t.v("OpIMul", t.u32, slot, t.u(3)), t.u(STATUS_WORDS))
    after = t.if_(t.b("OpULessThan", t.v("OpIAdd", t.u32, first, t.u(2)), length))
    for k, value in enumerate((t.u(address), raw, out)):
        ptr = t.v(
            "OpAccessChain", m.pointer("StorageBuffer", t.u32), t.status, t.u(0), t.v("OpIAdd", t.u32, first, t.u(k))
        )
        m.op("OpStore", ptr, value)
    t.end_if(after)


_FRND = {"": "RoundEven", "FLOOR": "Floor", "CEIL": "Ceil", "TRUNC": "Trunc"}


def _round_integral(t: _Translator, x: int, mode: str) -> int:
    """x rounded to an integral value with mode, as float bits. Subnormal inputs are handled on the bits,
    because some GPUs flush them even without FTZ."""
    r = t.as_u(t.m.ext(t.f32, _FRND[mode], t.as_f(x)))
    sign = t.v("OpBitwiseAnd", t.u32, x, t.u(0x80000000))
    exp_zero = t.b("OpIEqual", t.v("OpBitwiseAnd", t.u32, x, t.u(0x7F800000)), t.u(0))
    subnormal = t.b("OpLogicalAnd", exp_zero, t.b("OpINotEqual", t.v("OpBitwiseAnd", t.u32, x, t.u(0x7FFFFF)), t.u(0)))
    negative = t.b("OpINotEqual", sign, t.u(0))
    if mode == "FLOOR":
        tiny = t.sel(negative, t.u(0xBF800000), t.u(0))
    elif mode == "CEIL":
        tiny = t.sel(negative, t.u(0x80000000), t.u(0x3F800000))
    else:
        tiny = sign
    return t.sel(subnormal, tiny, r)


@_rule("FRND")
def _frnd(t: _Translator, ins: sass.Instruction) -> None:
    mode = ins.mods[0]
    mods = _mods(ins, {mode, "FTZ"})
    d, a = ins.operands
    x = t.src(a, fp=True)
    if "FTZ" in mods:
        x = t.ftz(x)
    t.dst(d, t.sel(t.is_nan(x), t.u(NAN), _round_integral(t, x, mode)))


_INT_RANGE = {
    "U8": (0, 0xFF),
    "S8": (-0x80, 0x7F),
    "U16": (0, 0xFFFF),
    "S16": (-0x8000, 0x7FFF),
    "U32": (0, 0xFFFFFFFF),
    "": (-0x80000000, 0x7FFFFFFF),
}


def _f32_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


@_rule("F2I")
def _f2i(t: _Translator, ins: sass.Instruction) -> None:
    src_type, _, int_type, mode, _ = ins.mods
    if src_type or int_type not in _INT_RANGE:
        raise Unsupported(f"F2I from {src_type or 'F32'} to {int_type or 'S32'}")
    mods = _mods(ins, {"FTZ", int_type, mode, "NTZ"})
    if "NTZ" not in mods:
        raise Unsupported("F2I without NTZ")
    d, a = ins.operands
    x = t.src(a, fp=True)
    if "FTZ" in mods:
        x = t.ftz(x)
    r = t.as_f(_round_integral(t, x, mode))
    lo, hi = _INT_RANGE[int_type]
    is_signed = lo < 0
    # Compare against powers of two, which are exact floats, then convert values that fit.
    too_big = t.b("OpFOrdGreaterThanEqual", r, t.m.f32(_f32_bits(float(hi + 1))))
    too_small = t.b("OpFOrdLessThan", r, t.m.f32(_f32_bits(float(lo))))
    if is_signed:
        conv = t.as_u(t.v("OpConvertFToS", t.i32, t.sel(t.b("OpLogicalOr", too_big, too_small), t.m.f32(0), r, t.f32)))
    else:
        conv = t.v("OpConvertFToU", t.u32, t.sel(t.b("OpLogicalOr", too_big, too_small), t.m.f32(0), r, t.f32))
    out = t.sel(too_big, t.u(hi), t.sel(too_small, t.u(lo & 0xFFFFFFFF), conv))
    t.dst(d, t.sel(t.is_nan(x), t.u(0), out))


_INT_SRC = {"U8": (8, False), "S8": (8, True), "U16": (16, False), "S16": (16, True), "U32": (32, False)}
_INT_SRC["S32"] = (32, True)


def _int_field(t: _Translator, x: int, offset: int, size: int, is_signed: bool) -> int:
    if size == 32:
        return x
    op = "OpBitFieldSExtract" if is_signed else "OpBitFieldUExtract"
    return t.v(op, t.u32, x, t.u(offset), t.u(size))


@_rule("I2FP")
def _i2fp(t: _Translator, ins: sass.Instruction) -> None:
    dst_type, src_type, rnd = ins.mods
    if dst_type != "F32" or src_type not in _INT_SRC or rnd:
        raise Unsupported(f"I2FP.{dst_type}.{src_type}{'.' + rnd if rnd else ''}")
    size, is_signed = _INT_SRC[src_type]
    d, a = ins.operands
    x = _int_field(t, t.src(a), 0, size, is_signed)
    f = t.v("OpConvertSToF", t.f32, t.v("OpBitcast", t.i32, x)) if is_signed else t.v("OpConvertUToF", t.f32, x)
    t.dst(d, t.as_u(f))


@_rule("I2F")
def _i2f(t: _Translator, ins: sass.Instruction) -> None:
    (src_type,) = ins.mods
    if src_type not in _INT_SRC:
        raise Unsupported(f"I2F.{src_type}")
    size, is_signed = _INT_SRC[src_type]
    d, a = ins.operands
    sel = int(a.suffix[2:]) if a.suffix else 0
    plain = sass.Operand(a.kind, a.value, a.bank, a.index, a.neg, a.abs)
    x = _int_field(t, t.src(plain), sel * size, size, is_signed)
    f = t.v("OpConvertSToF", t.f32, t.v("OpBitcast", t.i32, x)) if is_signed else t.v("OpConvertUToF", t.f32, x)
    t.dst(d, t.as_u(f))


# -- moves, constants and special registers ------------------------------------------------------


@_rule("MOV", "UMOV")
def _mov(t: _Translator, ins: sass.Instruction) -> None:
    _mods(ins)
    d, a = ins.operands
    t.dst(d, t.src(a))


_LDC_SIZE = {"U8": (1, False), "S8": (1, True), "U16": (2, False), "S16": (2, True), "": (4, False), "64": (8, False)}


@_rule("LDC", "ULDC")
def _ldc(t: _Translator, ins: sass.Instruction) -> None:
    (size_name,) = ins.mods
    if size_name not in _LDC_SIZE:
        raise Unsupported(f"{ins.name}.{size_name}")
    size, is_signed = _LDC_SIZE[size_name]
    d, c = ins.operands
    off = t.u(c.value)
    if c.index is not None and not sass._is_zero_reg(c.index):
        off = t.v("OpIAdd", t.u32, t.src(c.index), off)
    words = t.cbuf_bytes(c.bank, off, size)
    if is_signed:
        words[0] = t.v("OpBitFieldSExtract", t.u32, words[0], t.u(0), t.u(8 * size))
    for k, w in enumerate(words):
        t.dst(d, w, k)


_SREG_COMPONENT = {"X": 0, "Y": 1, "Z": 2}


@_rule("S2R")
def _s2r(t: _Translator, ins: sass.Instruction) -> None:
    _mods(ins)
    d, s = ins.operands
    name = s.value
    if name.startswith(("SR_TID.", "SR_CTAID.")):
        var = t.builtin("LocalInvocationId" if name.startswith("SR_TID") else "WorkgroupId")
        vec = t.v("OpLoad", t.uvec3, var)
        t.dst(d, t.v("OpCompositeExtract", t.u32, vec, _SREG_COMPONENT[name[-1]]))
    elif name == "SRZ":
        t.dst(d, t.u(0))
    else:
        raise Unsupported(f"special register {name}")


@_rule("CS2R")
def _cs2r(t: _Translator, ins: sass.Instruction) -> None:
    mods = _mods(ins, {"32"})
    d, s = ins.operands
    if s.value != "SRZ":
        raise Unsupported(f"CS2R from {s.value}")
    for k in range(1 if "32" in mods else 2):
        t.dst(d, t.u(0), k)


@_rule("P2R")
def _p2r(t: _Translator, ins: sass.Instruction) -> None:
    _mods(ins)
    d, _, a, m = ins.operands
    mask = m.value & 0x7F
    pr = t.u(0)
    for i in range(7):
        if mask >> i & 1:
            pr = t.v("OpBitwiseOr", t.u32, pr, t.sel(t.load("P", i), t.u(1 << i), t.u(0)))
    kept = t.v("OpBitwiseAnd", t.u32, t.src(a), t.u(~mask & 0xFFFFFFFF))
    t.dst(d, t.v("OpBitwiseOr", t.u32, kept, pr))


@_rule("R2P")
def _r2p(t: _Translator, ins: sass.Instruction) -> None:
    _mods(ins)
    _, a, m = ins.operands
    x = t.src(a)
    for i in range(7):
        if m.value >> i & 1:
            t.store("P", i, t.b("OpINotEqual", t.v("OpBitwiseAnd", t.u32, x, t.u(1 << i)), t.u(0)))


# -- integer arithmetic and logic -------------------------------------------------------------------


@_rule("IADD3", "UIADD3")
def _iadd3(t: _Translator, ins: sass.Instruction) -> None:
    mods = _mods(ins, {"X"})
    x = "X" in mods
    d, p0, p1, a, b, c, *carry_in = ins.operands
    total = t.m.u64(0 if x else sum(1 for s in (a, b, c) if s.neg))
    for s in (a, b, c):
        total = t.v("OpIAdd", t.u64, total, t.to64(t.src(s, not_neg=True)))
    for p in carry_in:
        total = t.v("OpIAdd", t.u64, total, t.to64(t.bit(t.psrc(p))))
    t.dst(d, t.lo32(total))
    t.pdst(p0, t.b("OpUGreaterThanEqual", total, t.m.u64(1 << 32)))
    t.pdst(p1, t.b("OpUGreaterThanEqual", total, t.m.u64(2 << 32)))


@_rule("IMAD", "UIMAD")
def _imad(t: _Translator, ins: sass.Instruction) -> None:
    mods = _mods(ins, {"X", "MOV", "IADD", "SHL", "U32", "HI"})
    d, a, b, c, *carry_in = ins.operands
    x = "X" in mods
    if "HI" in mods:
        if not (c.kind == "reg" and c.value == 255):
            raise Unsupported("IMAD.HI with an addend")
        op = "OpUMulExtended" if "U32" in mods else "OpSMulExtended"
        pair = t.m.struct(t.u32, t.u32, key="mulext")
        t.dst(d, t.v("OpCompositeExtract", t.u32, t.v(op, pair, t.src(a), t.src(b)), 1))
        return
    r = t.v("OpIAdd", t.u32, t.v("OpIMul", t.u32, t.src(a), t.src(b)), t.src(c, not_neg=x))
    if carry_in:
        r = t.v("OpIAdd", t.u32, r, t.bit(t.psrc(carry_in[0])))
    t.dst(d, r)


@_rule("LEA", "ULEA")
def _lea(t: _Translator, ins: sass.Instruction) -> None:
    mods = _mods(ins, {"HI", "X"})
    ops = list(ins.operands)
    d, p, a, b = ops[:4]
    rest = ops[4:]
    c = rest.pop(0) if "HI" in mods else None
    shift = rest.pop(0).value
    cin = rest.pop(0) if "X" in mods else None
    if a.neg:
        raise Unsupported("LEA with a negated operand")
    if c is not None:
        wide = t.v(
            "OpBitwiseOr", t.u64, t.v("OpShiftLeftLogical", t.u64, t.to64(t.src(c)), t.m.u64(32)), t.to64(t.src(a))
        )
        v = t.v("OpShiftRightLogical", t.u64, t.v("OpShiftLeftLogical", t.u64, wide, t.m.u64(shift)), t.m.u64(32))
        v = t.to64(t.lo32(v))
    else:
        v = t.to64(t.v("OpShiftLeftLogical", t.u32, t.src(a), t.u(shift)))
    total = t.v("OpIAdd", t.u64, v, t.to64(t.src(b)))
    if cin is not None:
        total = t.v("OpIAdd", t.u64, total, t.to64(t.bit(t.psrc(cin))))
    t.dst(d, t.lo32(total))
    t.pdst(p, t.b("OpUGreaterThanEqual", total, t.m.u64(1 << 32)))


def _lut3(t: _Translator, lut: int, a: int, b: int, c: int) -> int:
    if lut == 0:
        return t.u(0)
    if lut == 0xFF:
        return t.u(0xFFFFFFFF)
    na, nb, nc = (t.v("OpNot", t.u32, x) for x in (a, b, c))
    r = None
    for i in range(8):
        if lut >> i & 1:
            term = t.v("OpBitwiseAnd", t.u32, a if i & 4 else na, b if i & 2 else nb)
            term = t.v("OpBitwiseAnd", t.u32, term, c if i & 1 else nc)
            r = term if r is None else t.v("OpBitwiseOr", t.u32, r, term)
    return r


@_rule("LOP3", "ULOP3")
def _lop3(t: _Translator, ins: sass.Instruction) -> None:
    _mods(ins, {"LUT"})
    p, d, a, b, c, lut, pin = ins.operands
    r = _lut3(t, lut.value, t.src(a), t.src(b), t.src(c))
    t.dst(d, r)
    if not sass._is_pt(p):
        t.pdst(p, t.b("OpLogicalOr", t.b("OpINotEqual", r, t.u(0)), t.psrc(pin)))


@_rule("PLOP3")
def _plop3(t: _Translator, ins: sass.Instruction) -> None:
    _mods(ins, {"LUT"})
    p0, p1, a, b, c, lut0, lut1 = ins.operands
    idx = t.v("OpBitwiseOr", t.u32, t.sel(t.psrc(a), t.u(4), t.u(0)), t.sel(t.psrc(b), t.u(2), t.u(0)))
    idx = t.v("OpBitwiseOr", t.u32, idx, t.sel(t.psrc(c), t.u(1), t.u(0)))
    for p, lut in ((p0, lut0), (p1, lut1)):
        if not sass._is_pt(p):
            bit = t.v("OpBitwiseAnd", t.u32, t.v("OpShiftRightLogical", t.u32, t.u(lut.value), idx), t.u(1))
            t.pdst(p, t.b("OpINotEqual", bit, t.u(0)))


@_rule("SHF", "USHF")
def _shf(t: _Translator, ins: sass.Instruction) -> None:
    direction, wrap, dtype, hi = ins.mods
    _mods(ins, {direction, wrap, dtype, hi})
    if dtype not in ("U32", "S32"):
        raise Unsupported(f"SHF.{dtype}")
    d, a, b, c = ins.operands
    s = t.src(b)
    s = t.v("OpBitwiseAnd", t.u32, s, t.u(31)) if wrap else t.m.ext(t.u32, "UMin", s, t.u(32))
    s = t.to64(s)
    x = t.v("OpBitwiseOr", t.u64, t.v("OpShiftLeftLogical", t.u64, t.to64(t.src(c)), t.m.u64(32)), t.to64(t.src(a)))
    if direction == "R":
        x = t.v("OpShiftRightArithmetic" if dtype == "S32" else "OpShiftRightLogical", t.u64, x, s)
    else:
        x = t.v("OpShiftLeftLogical", t.u64, x, s)
    if hi:
        x = t.v("OpShiftRightLogical", t.u64, x, t.m.u64(32))
    t.dst(d, t.lo32(x))


@_rule("SGXT")
def _sgxt(t: _Translator, ins: sass.Instruction) -> None:
    mods = _mods(ins, {"W", "U32"})
    d, a, b = ins.operands
    n = t.src(b)
    n = t.v("OpBitwiseAnd", t.u32, n, t.u(31)) if "W" in mods else t.m.ext(t.u32, "UMin", n, t.u(32))
    op = "OpBitFieldUExtract" if "U32" in mods else "OpBitFieldSExtract"
    t.dst(d, t.v(op, t.u32, t.src(a), t.u(0), n))


@_rule("IMNMX")
def _imnmx(t: _Translator, ins: sass.Instruction) -> None:
    mods = _mods(ins, {"U32"})
    d, a, b, p = ins.operands
    x, y = t.src(a), t.src(b)
    lo, hi = ("UMin", "UMax") if "U32" in mods else ("SMin", "SMax")
    t.dst(d, t.sel(t.psrc(p), t.m.ext(t.u32, lo, x, y), t.m.ext(t.u32, hi, x, y)))


@_rule("PRMT")
def _prmt(t: _Translator, ins: sass.Instruction) -> None:
    _mods(ins)
    d, a, sel, b = ins.operands
    x = t.v("OpBitwiseOr", t.u64, t.v("OpShiftLeftLogical", t.u64, t.to64(t.src(b)), t.m.u64(32)), t.to64(t.src(a)))
    s = t.src(sel)
    r = t.u(0)
    for i in range(4):
        n = t.v("OpBitFieldUExtract", t.u32, s, t.u(4 * i), t.u(4))
        pos = t.to64(t.v("OpShiftLeftLogical", t.u32, t.v("OpBitwiseAnd", t.u32, n, t.u(7)), t.u(3)))
        byte = t.v("OpBitwiseAnd", t.u32, t.lo32(t.v("OpShiftRightLogical", t.u64, x, pos)), t.u(0xFF))
        sign = t.b("OpINotEqual", t.v("OpBitwiseAnd", t.u32, byte, t.u(0x80)), t.u(0))
        replicate = t.b("OpINotEqual", t.v("OpBitwiseAnd", t.u32, n, t.u(8)), t.u(0))
        byte = t.sel(replicate, t.sel(sign, t.u(0xFF), t.u(0)), byte)
        r = t.v("OpBitwiseOr", t.u32, r, t.v("OpShiftLeftLogical", t.u32, byte, t.u(8 * i)))
    t.dst(d, r)


_ICMP_S = {"LT": "OpSLessThan", "EQ": "OpIEqual", "LE": "OpSLessThanEqual", "GT": "OpSGreaterThan"}
_ICMP_S |= {"NE": "OpINotEqual", "GE": "OpSGreaterThanEqual"}
_ICMP_U = {"LT": "OpULessThan", "EQ": "OpIEqual", "LE": "OpULessThanEqual", "GT": "OpUGreaterThan"}
_ICMP_U |= {"NE": "OpINotEqual", "GE": "OpUGreaterThanEqual"}


@_rule("ISETP", "UISETP")
def _isetp(t: _Translator, ins: sass.Instruction) -> None:
    cmp_name, _, bop, ex = ins.mods
    if ex:
        raise Unsupported("ISETP.EX")
    mods = _mods(ins, {cmp_name, bop, "U32"})
    p0, p1, a, b, pin = ins.operands
    x, y = t.src(a), t.src(b)
    if cmp_name in ("F", "T"):
        c = t.m.boolean(cmp_name == "T")
    else:
        c = t.b((_ICMP_U if "U32" in mods else _ICMP_S)[cmp_name], x, y)
    q = t.psrc(pin)
    t.pdst(p0, _combine(t, bop, c, q))
    t.pdst(p1, _combine(t, bop, t.b("OpLogicalNot", c), q))


@_rule("IABS")
def _iabs(t: _Translator, ins: sass.Instruction) -> None:
    _mods(ins)
    d, a = ins.operands
    t.dst(d, t.m.ext(t.u32, "SAbs", t.src(a)))


@_rule("POPC", "UPOPC")
def _popc(t: _Translator, ins: sass.Instruction) -> None:
    _mods(ins)
    d, a = ins.operands
    t.dst(d, t.v("OpBitCount", t.u32, t.src(a)))


@_rule("BREV", "UBREV")
def _brev(t: _Translator, ins: sass.Instruction) -> None:
    _mods(ins)
    d, a = ins.operands
    t.dst(d, t.v("OpBitReverse", t.u32, t.src(a)))


@_rule("FLO", "UFLO")
def _flo(t: _Translator, ins: sass.Instruction) -> None:
    mods = _mods(ins, {"U32", "SH"})
    d, a = ins.operands
    pos = t.m.ext(t.u32, "FindUMsb" if "U32" in mods else "FindSMsb", t.src(a))
    if "SH" in mods:
        none = t.b("OpIEqual", pos, t.u(0xFFFFFFFF))
        pos = t.sel(none, pos, t.v("OpISub", t.u32, t.u(31), pos))
    t.dst(d, pos)


# -- shared memory -----------------------------------------------------------------------------------

_MEM_SIZE = {"U8": (1, False), "S8": (1, True), "U16": (2, False), "S16": (2, True), "": (4, False)}
_MEM_SIZE |= {"64": (8, False), "128": (16, False)}


def _shared_addr(t: _Translator, op: sass.Operand) -> int:
    if op.extra is not None:
        raise Unsupported("shared address with a uniform register")
    base = t.v("OpIMul", t.u32, t.src(op.index), t.u(op.scale))
    return t.v("OpIAdd", t.u32, base, t.u(op.value))


def _shared_ptr(t: _Translator, word: int) -> int:
    return t.v("OpAccessChain", t.m.pointer("Workgroup", t.u32), t.shared, word)


def _shared_inside(t: _Translator, addr: int, size: int) -> int:
    end = t.v("OpIAdd", t.u64, t.to64(addr), t.m.u64(size))
    return t.b("OpULessThanEqual", end, t.m.u64(t.shared_words * 4))


@_rule("LDS")
def _lds(t: _Translator, ins: sass.Instruction) -> None:
    (size_name,) = ins.mods
    if size_name not in _MEM_SIZE or not t.shared_words:
        raise Unsupported(f"LDS.{size_name} without shared memory" if not t.shared_words else f"LDS.{size_name}")
    size, is_signed = _MEM_SIZE[size_name]
    d, addr = ins.operands
    a = _shared_addr(t, addr)
    inside = _shared_inside(t, a, size)
    first = t.sel(inside, t.v("OpShiftRightLogical", t.u32, a, t.u(2)), t.u(0))
    for k in range(max(1, size // 4)):
        w = t.v("OpLoad", t.u32, _shared_ptr(t, t.v("OpIAdd", t.u32, first, t.u(k))))
        if size < 4:
            shift = t.v("OpShiftLeftLogical", t.u32, t.v("OpBitwiseAnd", t.u32, a, t.u(3)), t.u(3))
            w = _int_field(t, t.v("OpShiftRightLogical", t.u32, w, shift), 0, 8 * size, is_signed)
        t.dst(d, t.sel(inside, w, t.u(0)), k)


@_rule("STS")
def _sts(t: _Translator, ins: sass.Instruction) -> None:
    (size_name,) = ins.mods
    if size_name not in _MEM_SIZE or not t.shared_words:
        raise Unsupported(f"STS.{size_name}")
    size, _ = _MEM_SIZE[size_name]
    addr, s = ins.operands
    if s.kind != "reg":
        raise Unsupported(f"store of a {s.kind} operand")
    a = _shared_addr(t, addr)
    after = t.if_(_shared_inside(t, a, size))
    first = t.v("OpShiftRightLogical", t.u32, a, t.u(2))
    if size >= 4:
        for k in range(size // 4):
            n = s.value + k
            value = t.u(0) if s.value == 255 or n >= 255 else t.load("R", n)
            t.m.op("OpStore", _shared_ptr(t, t.v("OpIAdd", t.u32, first, t.u(k))), value)
    else:
        shift = t.v("OpShiftLeftLogical", t.u32, t.v("OpBitwiseAnd", t.u32, a, t.u(3)), t.u(3))
        mask = t.v("OpShiftLeftLogical", t.u32, t.u((1 << (8 * size)) - 1), shift)
        value = t.v("OpBitwiseAnd", t.u32, t.v("OpShiftLeftLogical", t.u32, t.src(s), shift), mask)
        ptr = _shared_ptr(t, first)
        wg, relaxed = t.u(spirv.SCOPE["Workgroup"]), t.u(0)
        t.v("OpAtomicAnd", t.u32, ptr, wg, relaxed, t.v("OpNot", t.u32, mask))
        t.v("OpAtomicOr", t.u32, ptr, wg, relaxed, value)
    t.end_if(after)


_ATOMIC = {"ADD": "OpAtomicIAdd", "AND": "OpAtomicAnd", "OR": "OpAtomicOr", "XOR": "OpAtomicXor"}


@_rule("ATOMS")
def _atoms(t: _Translator, ins: sass.Instruction) -> None:
    (op_name,) = ins.mods
    if op_name not in _ATOMIC or not t.shared_words:
        raise Unsupported(f"ATOMS.{op_name}")
    d, addr, s = ins.operands
    a = _shared_addr(t, addr)
    inside = _shared_inside(t, a, 4)
    after = t.if_(inside)
    ptr = _shared_ptr(t, t.v("OpShiftRightLogical", t.u32, a, t.u(2)))
    old = t.v(_ATOMIC[op_name], t.u32, ptr, t.u(spirv.SCOPE["Workgroup"]), t.u(0), t.src(s))
    t.dst(d, old)
    t.end_if(after)


# -- surfaces ------------------------------------------------------------------------------------------


def _surface(t: _Translator, ins: sass.Instruction) -> tuple[int, int, int, int]:
    """(image, coordinates, inside, count) for a surface instruction."""
    handle = (ins.operands[-2].value, ins.operands[-1].value)
    var, ty, dim = t.images[handle]
    addr = next(op for op in ins.operands if op.kind == "addr")
    if addr.index.kind != "reg" or addr.index.value == 255:
        raise Unsupported("surface coordinates without a register")
    n = SU_COORDS[dim]
    regs = [t.load("R", addr.index.value + k) for k in range(n)]
    if dim.endswith("ARRAY"):
        regs[-1] = t.v("OpBitwiseAnd", t.u32, regs[-1], t.u(0xFFFF))
    image = t.v("OpLoad", ty, var)
    if n == 1:
        coords, size = t.v("OpBitcast", t.i32, regs[0]), t.v("OpImageQuerySize", t.i32, image)
        inside = t.b("OpULessThan", regs[0], t.as_u(size))
    else:
        ivec, uvec = t.m.vector(t.i32, n), t.m.vector(t.u32, n)
        coords = t.v("OpBitcast", ivec, t.v("OpCompositeConstruct", uvec, *regs))
        size = t.v("OpBitcast", uvec, t.v("OpImageQuerySize", ivec, image))
        inside = None
        for k in range(n):
            ok = t.b("OpULessThan", regs[k], t.v("OpCompositeExtract", t.u32, size, k))
            inside = ok if inside is None else t.b("OpLogicalAnd", inside, ok)
    return image, coords, inside, n


@_rule("SUST")
def _sust(t: _Translator, ins: sass.Instruction) -> None:
    count = SU_CHANNELS[ins.mods[2]]
    image, coords, inside, _ = _surface(t, ins)
    s = ins.operands[1].value
    parts = [t.load("R", s + k) if k < count and s + k < 255 else t.u(0) for k in range(4)]
    texel = t.v("OpCompositeConstruct", t.uvec4, *parts)
    after = t.if_(inside)
    t.m.op("OpImageWrite", image, coords, texel)
    t.end_if(after)


@_rule("SULD")
def _suld(t: _Translator, ins: sass.Instruction) -> None:
    count = SU_CHANNELS[ins.mods[2]]
    image, coords, inside, _ = _surface(t, ins)
    d = ins.operands[0]
    m = t.m
    then, other, after = m.id(), m.id(), m.id()
    m.op("OpSelectionMerge", after, 0)
    m.op("OpBranchConditional", inside, then, other)
    m.label(then)
    texel = t.v("OpImageRead", t.uvec4, image, coords)
    for k in range(count):
        t.dst(d, t.v("OpCompositeExtract", t.u32, texel, k), k)
    m.op("OpBranch", after)
    m.label(other)
    for k in range(count):
        t.dst(d, t.u(0), k)
    m.op("OpBranch", after)
    m.label(after)


def _su_atomic(t: _Translator, ins: sass.Instruction, with_result: bool) -> None:
    _, _, _, op_name, sem = ins.mods[:5]
    if op_name not in _ATOMIC:
        raise Unsupported(f"surface atomic {op_name}")
    if sem not in ("", "STRONG.GPU", "STRONG.SM"):
        raise Unsupported(f"memory semantics {sem}")
    _, coords, inside, _ = _surface(t, ins)
    handle = (ins.operands[-2].value, ins.operands[-1].value)
    var = t.images[handle][0]
    d = ins.operands[0] if with_result else None
    s = ins.operands[2] if with_result else ins.operands[1]
    value = t.src(s)
    m = t.m
    then, other, after = m.id(), m.id(), m.id()
    m.op("OpSelectionMerge", after, 0)
    m.op("OpBranchConditional", inside, then, other)
    m.label(then)
    ptr = t.v("OpImageTexelPointer", m.pointer("Image", t.u32), var, coords, t.u(0))
    old = t.v(_ATOMIC[op_name], t.u32, ptr, t.u(spirv.SCOPE["Device"]), t.u(0), value)
    if d is not None:
        t.dst(d, old)
    m.op("OpBranch", after)
    m.label(other)
    if d is not None:
        t.dst(d, t.u(0))
    m.op("OpBranch", after)
    m.label(after)


@_rule("SURED")
def _sured(t: _Translator, ins: sass.Instruction) -> None:
    _su_atomic(t, ins, with_result=False)


@_rule("SUATOM")
def _suatom(t: _Translator, ins: sass.Instruction) -> None:
    _su_atomic(t, ins, with_result=True)
