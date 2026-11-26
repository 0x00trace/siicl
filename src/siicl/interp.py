"""Reference interpreter for compute programs in NVIDIA's 128-bit GPU instruction encoding.

`run(program, launch)` executes a compute program, decoded by siicl.sass, on the
CPU: each workgroup of the grid in turn, warp by warp, thread by thread. It is
the reference that translations of a program (such as SPIR-V for another GPU)
are checked against, so it aims for exact results rather than speed.

The rules for what each instruction computes come from NVIDIA's public
documentation (the PTX ISA), the code that NVIDIA's compiler writes for these
GPUs, and the sample's own programs. docs/interpreter.md lists them, including
the ones that are guesses. An instruction, modifier or operand without a rule
raises Unsupported when it runs, instead of computing something.
"""

from __future__ import annotations

import math
import struct
from collections.abc import Callable
from dataclasses import dataclass, field

from siicl import sass
from siicl.nvshader import INSTRUCTION_SIZE, ComputeInfo

M32 = 0xFFFFFFFF
WARP_SIZE = 32
NAN = 0x7FFFFFFF  # the NaN that NVIDIA GPUs return from arithmetic
DEFAULT_STEP_LIMIT = 50_000_000


class Unsupported(ValueError):
    """The interpreter has no rule for this instruction, modifier or operand."""


class ExecutionError(RuntimeError):
    """The program can't continue: a bad memory access, a deadlock, or too many steps."""


# ---------------------------------------------------------------------------
# 32-bit floating point

_F32 = struct.Struct("<f")
_U32 = struct.Struct("<I")
_F64 = struct.Struct("<d")
_U64 = struct.Struct("<Q")
RN, RM, RP, RZ = 0, 1, 2, 3
_ROUNDING = {"": RN, "RM": RM, "RP": RP, "RZ": RZ}
F32_MAX = 3.4028234663852886e38


def f32(bits: int) -> float:
    return _F32.unpack(_U32.pack(bits))[0]


def bits32(x: float) -> int:
    """The bits of a value that is exactly representable as a 32-bit float."""
    return _U32.unpack(_F32.pack(x))[0]


def ftz(bits: int) -> int:
    """Flush a subnormal value to zero, keeping its sign."""
    return bits if bits & 0x7F800000 else bits & 0x80000000


def to_f32(x: float, rnd: int = RN) -> int:
    """Round a 64-bit float to 32 bits with the given rounding mode."""
    if x != x:
        return NAN
    if rnd == RN:
        try:
            return _U32.unpack(_F32.pack(x))[0]
        except OverflowError:
            return 0xFF800000 if x < 0 else 0x7F800000
    sign = 0x80000000 if math.copysign(1.0, x) < 0 else 0
    a = abs(x)
    if a == 0 or math.isinf(a):
        return sign | (0x7F800000 if a else 0)
    e = math.frexp(a)[1]
    q = max(e - 24, -149)  # exponent of one unit in the last place
    scaled = math.ldexp(a, -q)
    n = math.floor(scaled)
    if scaled != n and ((rnd == RP and not sign) or (rnd == RM and sign)):
        n += 1
    v = math.ldexp(n, q)
    if v > F32_MAX:
        toward_zero = rnd == RZ or (rnd == RP and sign) or (rnd == RM and not sign)
        return sign | (0x7F7FFFFF if toward_zero else 0x7F800000)
    return sign | bits32(v)


def _sum_odd(p: float, c: float) -> float:
    """p + c rounded to 64 bits with round-to-odd, so that a later rounding to 32 bits is exact."""
    s = p + c
    if not math.isfinite(s):
        return s
    bb = s - p
    err = (p - (s - bb)) + (c - bb)
    if err:
        b = _U64.unpack(_F64.pack(s))[0]
        if not b & 1:
            b += 1 if (err > 0) == (s > 0) else -1
            s = _F64.unpack(_U64.pack(b))[0]
    return s


def _sat(r: int) -> int:
    if (r & 0x7FFFFFFF) > 0x7F800000 or r & 0x80000000:
        return 0
    return min(r, 0x3F800000)


def fma32(a: int, b: int, c: int, rnd: int = RN, flush: bool = False, sat: bool = False) -> int:
    if flush:
        a, b, c = ftz(a), ftz(b), ftz(c)
    r = to_f32(_sum_odd(f32(a) * f32(b), f32(c)), rnd)  # the product is exact in 64 bits
    if flush:
        r = ftz(r)
    return _sat(r) if sat else r


def add32(a: int, b: int, rnd: int = RN, flush: bool = False, sat: bool = False) -> int:
    if flush:
        a, b = ftz(a), ftz(b)
    x, y = f32(a), f32(b)
    r = to_f32(x + y if rnd == RN else _sum_odd(x, y), rnd)
    if flush:
        r = ftz(r)
    return _sat(r) if sat else r


def mul32(a: int, b: int, rnd: int = RN, flush: bool = False, sat: bool = False, scale: float = 1.0) -> int:
    if flush:
        a, b = ftz(a), ftz(b)
    r = to_f32(f32(a) * f32(b) * scale, rnd)
    if flush:
        r = ftz(r)
    return _sat(r) if sat else r


def _is_nan(bits: int) -> bool:
    return (bits & 0x7FFFFFFF) > 0x7F800000


def fmin32(a: int, b: int) -> int:
    if _is_nan(a):
        return NAN if _is_nan(b) else b
    if _is_nan(b):
        return a
    x, y = f32(a), f32(b)
    if x != y:
        return a if x < y else b
    return a | b  # -0 is smaller than +0


def fmax32(a: int, b: int) -> int:
    if _is_nan(a):
        return NAN if _is_nan(b) else b
    if _is_nan(b):
        return a
    x, y = f32(a), f32(b)
    if x != y:
        return a if x > y else b
    return a & b


def _fcmp(name: str) -> Callable[[float, float], bool]:
    ordered = {
        "LT": lambda x, y: x < y,
        "EQ": lambda x, y: x == y,
        "LE": lambda x, y: x <= y,
        "GT": lambda x, y: x > y,
        "NE": lambda x, y: x != y,
        "GE": lambda x, y: x >= y,
    }
    if name in ordered:
        f = ordered[name]
        return lambda x, y: x == x and y == y and f(x, y)
    if name.endswith("U") and name[:-1] in ordered:
        f = ordered[name[:-1]]
        return lambda x, y: x != x or y != y or f(x, y)
    if name == "NUM":
        return lambda x, y: x == x and y == y
    if name == "NAN":
        return lambda x, y: x != x or y != y
    if name == "F":
        return lambda x, y: False
    if name == "T":
        return lambda x, y: True
    raise Unsupported(f"float comparison {name}")


def mufu(func: str, bits: int) -> int:
    """MUFU, computed exactly and rounded to 32 bits. The hardware's results are approximations."""
    x = f32(ftz(bits))
    if func == "RCP":
        r = math.copysign(math.inf, x) if x == 0 else 1.0 / x
    elif func == "RSQ":
        if x != x or x < 0:
            r = math.nan
        elif x == 0:
            r = math.copysign(math.inf, x)
        else:
            r = 1.0 / math.sqrt(x)
    elif func == "SQRT":
        r = math.nan if x < 0 else math.sqrt(x)
    elif func == "EX2":
        try:
            r = 2.0**x
        except OverflowError:
            r = math.inf
    elif func == "LG2":
        if x != x or x < 0:
            r = math.nan
        elif x == 0:
            r = -math.inf
        else:
            r = math.log2(x)
    elif func in ("SIN", "COS"):
        # The input is in turns: the compiler multiplies by 1/(2*pi) first.
        if not math.isfinite(x):
            r = math.nan
        else:
            turns = x - round(x)
            r = (math.sin if func == "SIN" else math.cos)(math.tau * turns)
    else:
        raise Unsupported(f"MUFU.{func}")
    return ftz(to_f32(r))


_FRND = {"": round, "FLOOR": math.floor, "CEIL": math.ceil, "TRUNC": math.trunc}


def frnd32(bits: int, mode: str, flush: bool) -> int:
    if flush:
        bits = ftz(bits)
    x = f32(bits)
    if not math.isfinite(x):
        return NAN if x != x else bits
    v = float(_FRND[mode](x))
    return bits32(math.copysign(v, x) if v == 0 else v)


_INT_RANGE = {
    "U8": (0, 0xFF),
    "S8": (-0x80, 0x7F),
    "U16": (0, 0xFFFF),
    "S16": (-0x8000, 0x7FFF),
    "U32": (0, M32),
    "": (-0x80000000, 0x7FFFFFFF),
}


def f2i32(bits: int, mode: str, int_type: str, flush: bool) -> int:
    if flush:
        bits = ftz(bits)
    lo, hi = _INT_RANGE[int_type]
    x = f32(bits)
    if x != x:
        return 0
    v = (hi if x > 0 else lo) if math.isinf(x) else _FRND[mode](x)
    return max(lo, min(hi, v)) & M32


def signed(v: int) -> int:
    return v - (1 << 32) if v & 0x80000000 else v


# ---------------------------------------------------------------------------
# Resources


@dataclass
class Surface:
    """A storage image of width x height x depth texels, each holding `channels` 32-bit words.

    For array surfaces, depth is the number of layers. Surface instructions
    address it by texel coordinates, and the words hold the raw register bits
    that the program stores.
    """

    width: int
    height: int = 1
    depth: int = 1
    channels: int = 4
    data: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        n = self.width * self.height * self.depth * self.channels
        if not self.data:
            self.data = [0] * n
        elif len(self.data) != n:
            raise ValueError(f"surface data has {len(self.data)} words, expected {n}")

    def index(self, x: int, y: int, z: int) -> int | None:
        if 0 <= x < self.width and 0 <= y < self.height and 0 <= z < self.depth:
            return ((z * self.height + y) * self.width + x) * self.channels
        return None


class Memory:
    """Global memory: separate regions at 64-bit addresses."""

    BASE = 0x1_0000_0000

    def __init__(self) -> None:
        self.regions: list[tuple[int, bytearray]] = []

    def add(self, data: bytes | int, *, align: int = 0x1000) -> int:
        """Add a region holding data (or that many zero bytes) and return its address."""
        buf = bytearray(data) if isinstance(data, (bytes, bytearray)) else bytearray(data)
        end = max((base + len(b) for base, b in self.regions), default=self.BASE)
        base = -(-end // align) * align
        self.regions.append((base, buf))
        return base

    def region(self, address: int, size: int) -> tuple[bytearray, int]:
        for base, buf in self.regions:
            if base <= address and address + size <= base + len(buf):
                return buf, address - base
        raise ExecutionError(f"global memory access of {size} bytes at {address:#x} is outside every region")


@dataclass
class Launch:
    """What a program runs with: the grid size and the resources its instructions name."""

    grid: tuple[int, int, int] = (1, 1, 1)
    cbufs: dict[int, bytes] = field(default_factory=dict)  # constant buffers by bank
    surfaces: dict[tuple[int, int], Surface] = field(default_factory=dict)  # by the handle's two numbers
    memory: Memory = field(default_factory=Memory)
    # MUFU results to use instead of computing them, by (instruction address, input bits). A check
    # against a GPU fills this from the GPU's own results, which are approximations.
    mufu_results: dict[tuple[int, int], int] | None = None


@dataclass
class Result:
    instructions: int  # thread-instructions executed
    workgroups: int


# ---------------------------------------------------------------------------
# Execution state

RUN, EXITED, WAIT_BSYNC, WAIT_WARPSYNC, WAIT_BAR = range(5)


class Thread:
    __slots__ = ("r", "p", "pc", "state", "wait", "tid", "lane", "warp")

    def __init__(self, tid: tuple[int, int, int], lane: int, warp: Warp) -> None:
        self.r = [0] * 256  # R255 is RZ: never written, so it reads as zero
        self.p = [False] * 7 + [True]  # P7 is PT
        self.pc = 0  # instruction index
        self.state = RUN
        self.wait = 0
        self.tid = tid
        self.lane = lane
        self.warp = warp


class Warp:
    __slots__ = ("threads", "ur", "up", "b", "cta")

    def __init__(self, cta: Cta) -> None:
        self.threads: list[Thread] = []
        self.ur = [0] * 64  # UR63 is URZ
        self.up = [False] * 7 + [True]  # UP7 is UPT
        self.b = [0] * 16  # convergence barriers: masks of lanes
        self.cta = cta

    def mask(self, threads: list[Thread]) -> int:
        m = 0
        for t in threads:
            m |= 1 << t.lane
        return m

    def release(self) -> bool:
        """Let waiting threads continue once everyone they wait for has arrived."""
        exited = self.mask([t for t in self.threads if t.state == EXITED])
        released = False
        for state in (WAIT_BSYNC, WAIT_WARPSYNC):
            groups: dict[int, list[Thread]] = {}
            for t in self.threads:
                if t.state == state:
                    groups.setdefault(t.wait, []).append(t)
            for key, ts in groups.items():
                need = (self.b[key] if state == WAIT_BSYNC else key) & ~exited
                if need & ~self.mask(ts) == 0:
                    for t in ts:
                        t.state, t.pc = RUN, t.pc + 1
                    released = True
        return released


class Cta:
    __slots__ = ("ctaid", "warps", "threads", "shared")

    def __init__(self, ctaid: tuple[int, int, int], local: tuple[int, int, int], shared_size: int) -> None:
        self.ctaid = ctaid
        self.shared = bytearray(shared_size)
        self.threads: list[Thread] = []
        self.warps: list[Warp] = []
        lx, ly, lz = local
        for i in range(lx * ly * lz):
            if i % WARP_SIZE == 0:
                self.warps.append(Warp(self))
            w = self.warps[-1]
            t = Thread((i % lx, i // lx % ly, i // (lx * ly)), i % WARP_SIZE, w)
            w.threads.append(t)
            self.threads.append(t)


Step = Callable[[list[Thread], Warp], None]
ThreadFn = Callable[[Thread], None]


# ---------------------------------------------------------------------------
# Programs


class Program:
    """A decoded compute program and its launch parameters."""

    def __init__(self, code: bytes, info: ComputeInfo) -> None:
        self.info = info
        self.instructions = [
            sass.decode(*struct.unpack_from("<QQ", code, off), off)
            for off in range(0, len(code) - INSTRUCTION_SIZE + 1, INSTRUCTION_SIZE)
        ]

    def unsupported(self) -> list[tuple[sass.Instruction, str]]:
        """Instructions the interpreter has no rule for, with the reason."""
        ctx = _Context(Launch(), self.info)
        out = []
        for ins in self.instructions:
            try:
                _compile(ins, ctx)
            except Unsupported as err:
                out.append((ins, str(err)))
        return out


def run(program: Program, launch: Launch, *, step_limit: int = DEFAULT_STEP_LIMIT, reverse: bool = False) -> Result:
    """Run every workgroup of the grid. Changes the launch's surfaces and memory in place.

    reverse runs workgroups, warps and the threads of each instruction in the opposite order. Results
    that differ between the two orders depend on the order in which threads run.
    """
    ctx = _Context(launch, program.info)
    code: list[Step] = []
    for ins in program.instructions:
        try:
            code.append(_compile(ins, ctx))
        except Unsupported as err:
            code.append(_raiser(Unsupported(str(err))))
    info = program.info
    steps = 0
    gx, gy, gz = launch.grid
    ctas = [(x, y, z) for z in range(gz) for y in range(gy) for x in range(gx)]
    for ctaid in reversed(ctas) if reverse else ctas:
        ctx.cta = cta = Cta(ctaid, info.local_size, info.shared_size)
        if reverse:
            cta.warps.reverse()
            for w in cta.warps:
                w.threads.reverse()
        steps = _run_cta(cta, code, program.instructions, steps, step_limit)
    return Result(steps, len(ctas))


def _raiser(err: Exception) -> Step:
    def step(group: list[Thread], warp: Warp) -> None:
        raise err

    return step


def _run_cta(cta: Cta, code: list[Step], ins: list[sass.Instruction], steps: int, limit: int) -> int:
    n = len(code)
    while True:
        for warp in cta.warps:
            while True:
                runnable = [t for t in warp.threads if t.state == RUN]
                if not runnable:
                    if warp.release():
                        continue
                    break
                pc = min(t.pc for t in runnable)
                if pc >= n:
                    raise ExecutionError(f"a thread ran past the end of the program at {pc * INSTRUCTION_SIZE:#x}")
                group = [t for t in runnable if t.pc == pc]
                try:
                    code[pc](group, warp)
                except (Unsupported, ExecutionError) as err:
                    raise type(err)(f"{pc * INSTRUCTION_SIZE:#06x} {ins[pc].text()}: {err}") from err
                steps += len(group)
                if steps > limit:
                    raise ExecutionError(f"stopped after {limit:,} thread-instructions (step limit)")
        live = [t for t in cta.threads if t.state != EXITED]
        if not live:
            return steps
        waiting = [t for t in live if t.state == WAIT_BAR]
        if len(waiting) == len(live) and len({t.wait for t in waiting}) == 1:
            for t in waiting:
                t.state, t.pc = RUN, t.pc + 1
            continue
        where = sorted({f"{t.pc * INSTRUCTION_SIZE:#x}" for t in live})
        raise ExecutionError(f"deadlock: {len(live)} threads wait at {', '.join(where[:8])}")


class _Context:
    """What compiling an instruction needs: the launch's resources and the current workgroup."""

    def __init__(self, launch: Launch, info: ComputeInfo) -> None:
        self.launch = launch
        self.info = info
        self.cbufs = {1: info.constants, **launch.cbufs}
        self.cta: Cta | None = None

    def cbuf(self, bank: int, offset: int, size: int = 4) -> int:
        data = self.cbufs.get(bank, b"")
        offset &= M32
        if offset + size > len(data):
            return 0
        return int.from_bytes(data[offset : offset + size], "little")


# ---------------------------------------------------------------------------
# Operands


Reader = Callable[[Thread], int]
PredReader = Callable[[Thread], bool]
Writer = Callable[[Thread, int], None]
PredWriter = Callable[[Thread, bool], None]


def _const(v: int) -> Reader:
    return lambda t: v


def _src(op: sass.Operand, ctx: _Context, *, fp: bool = False, not_neg: bool = False) -> Reader:
    """A reader for a source operand. fp: negation and absolute value act on the sign bit.

    not_neg: negation is a bitwise NOT (the carry-in forms of IADD3 and IMAD).
    """
    k, v = op.kind, op.value
    base: Reader
    value: int | None = None
    if k == "reg":
        base = (lambda t: t.r[v]) if v != 255 else _const(0)
        if v == 255:
            value = 0
    elif k == "ureg":
        base = (lambda t: t.warp.ur[v]) if v != 63 else _const(0)
        if v == 63:
            value = 0
    elif k in ("imm", "fimm"):
        value = v & M32
    elif k == "cbuf":
        if op.index is None or sass._is_zero_reg(op.index):
            value = ctx.cbuf(op.bank, v)
        else:
            idx = _src(op.index, ctx)
            bank = op.bank
            base = lambda t: ctx.cbuf(bank, idx(t) + v)  # noqa: E731
    else:
        raise Unsupported(f"{k} operand")
    if op.suffix:
        raise Unsupported(f"operand suffix {op.suffix}")
    if fp:
        mask = 0x7FFFFFFF if op.abs else M32
        flip = 0x80000000 if op.neg else 0

        def mod(x: int) -> int:
            return (x & mask) ^ flip

    else:
        if op.abs:
            raise Unsupported("absolute value of an integer operand")
        if not op.neg:
            mod = None
        elif not_neg:

            def mod(x: int) -> int:
                return ~x & M32

        else:

            def mod(x: int) -> int:
                return -x & M32

    if value is not None:
        return _const(mod(value) if mod else value)
    if mod is None:
        return base
    return lambda t: mod(base(t))


def _psrc(op: sass.Operand) -> PredReader:
    n, neg = op.value, op.not_
    if op.kind == "pred":
        return lambda t: t.p[n] != neg
    if op.kind == "upred":
        return lambda t: t.warp.up[n] != neg
    raise Unsupported(f"{op.kind} operand as a predicate")


def _dst(op: sass.Operand, count: int = 1) -> Writer:
    """A writer for count consecutive registers, given a value of 32 * count bits."""
    n = op.value
    if op.kind == "reg":
        if n == 255:
            return lambda t, v: None
        if count == 1:

            def w(t: Thread, v: int) -> None:
                t.r[n] = v

            return w

        def wn(t: Thread, v: int) -> None:
            for i in range(count):
                if n + i < 255:
                    t.r[n + i] = (v >> (32 * i)) & M32

        return wn
    if op.kind == "ureg":
        if n == 63:
            return lambda t, v: None

        def wu(t: Thread, v: int) -> None:
            for i in range(count):
                if n + i < 63:
                    t.warp.ur[n + i] = (v >> (32 * i)) & M32

        return wu
    raise Unsupported(f"{op.kind} operand as a destination")


def _pdst(op: sass.Operand) -> PredWriter:
    n = op.value
    if op.kind not in ("pred", "upred"):
        raise Unsupported(f"{op.kind} operand as a predicate destination")
    if n == 7:
        return lambda t, v: None
    if op.kind == "pred":

        def w(t: Thread, v: bool) -> None:
            t.p[n] = v

        return w

    def wu(t: Thread, v: bool) -> None:
        t.warp.up[n] = v

    return wu


def _reg_pair(op: sass.Operand) -> Reader:
    """A 64-bit value from a register and the next one."""
    n = op.value
    if op.kind == "reg":
        return _const(0) if n == 255 else (lambda t: t.r[n] | (t.r[n + 1] << 32))
    if op.kind == "ureg":
        return _const(0) if n == 63 else (lambda t: t.warp.ur[n] | (t.warp.ur[n + 1] << 32))
    raise Unsupported(f"{op.kind} operand as a 64-bit value")


# ---------------------------------------------------------------------------
# Compiling instructions

ThreadFactory = Callable[[sass.Instruction, _Context], ThreadFn]
GroupFactory = Callable[[sass.Instruction, _Context], Step]
_THREAD: dict[str, ThreadFactory] = {}
_GROUP: dict[str, GroupFactory] = {}


def _per_thread(*names: str) -> Callable[[ThreadFactory], ThreadFactory]:
    """Register an instruction that each thread executes on its own."""

    def reg(f: ThreadFactory) -> ThreadFactory:
        for n in names:
            _THREAD[n] = f
        return f

    return reg


def _per_group(*names: str) -> Callable[[GroupFactory], GroupFactory]:
    """Register an instruction that acts on the threads executing it together (control flow, warp operations)."""

    def reg(f: GroupFactory) -> GroupFactory:
        for n in names:
            _GROUP[n] = f
        return f

    return reg


def _guard(ins: sass.Instruction) -> PredReader | None:
    return None if ins.guard is None else _psrc(ins.guard)


# Instructions of the uniform datapath. Their registers belong to the warp, so they run once for
# the threads that execute them together, not once per thread.
_UNIFORM = {"UIADD3", "UIMAD", "ULEA", "ULOP3", "USHF", "UISETP", "UMOV", "UFLO", "UPOPC", "UBREV", "USEL"}
_UNIFORM |= {"UPRMT", "ULDC"}


def _compile(ins: sass.Instruction, ctx: _Context) -> Step:
    if ins.name in _GROUP:
        return _GROUP[ins.name](ins, ctx)
    if ins.name not in _THREAD:
        raise Unsupported(f"no rule for {ins.name}")
    fn = _THREAD[ins.name](ins, ctx)
    guard = _guard(ins)
    if ins.name in _UNIFORM:

        def step_uniform(group: list[Thread], warp: Warp) -> None:
            if guard is None or guard(group[0]):
                fn(group[0])
            for t in group:
                t.pc += 1

        return step_uniform
    if guard is None:

        def step(group: list[Thread], warp: Warp) -> None:
            for t in group:
                fn(t)
                t.pc += 1

        return step

    def step_guarded(group: list[Thread], warp: Warp) -> None:
        for t in group:
            if guard(t):
                fn(t)
            t.pc += 1

    return step_guarded


def _mods(ins: sass.Instruction, allowed: set[str] = frozenset()) -> set[str]:
    mods = {m for m in ins.mods if m}
    extra = mods - allowed
    if extra:
        raise Unsupported(f"modifier {'.'.join(sorted(extra))}")
    return mods


def _rounding(mods: set[str]) -> int:
    found = [m for m in mods if m in _ROUNDING]
    return _ROUNDING[found[0]] if found else RN


def _combine(op: str) -> Callable[[bool, bool], bool]:
    if op == "AND":
        return lambda x, y: x and y
    if op == "OR":
        return lambda x, y: x or y
    if op == "XOR":
        return lambda x, y: x != y
    raise Unsupported(f"boolean operation {op}")


# -- floating point ----------------------------------------------------------


@_per_thread("FFMA")
def _ffma(ins: sass.Instruction, ctx: _Context) -> ThreadFn:
    mods = _mods(ins, {"FTZ", "RM", "RP", "RZ", "SAT"})
    d, a, b, c = ins.operands
    ra, rb, rc, w = _src(a, ctx, fp=True), _src(b, ctx, fp=True), _src(c, ctx, fp=True), _dst(d)
    rnd, flush, sat = _rounding(mods), "FTZ" in mods, "SAT" in mods
    return lambda t: w(t, fma32(ra(t), rb(t), rc(t), rnd, flush, sat))


@_per_thread("FADD")
def _fadd(ins: sass.Instruction, ctx: _Context) -> ThreadFn:
    mods = _mods(ins, {"FTZ", "RM", "RP", "RZ", "SAT"})
    d, a, b = ins.operands
    ra, rb, w = _src(a, ctx, fp=True), _src(b, ctx, fp=True), _dst(d)
    rnd, flush, sat = _rounding(mods), "FTZ" in mods, "SAT" in mods
    return lambda t: w(t, add32(ra(t), rb(t), rnd, flush, sat))


_FMUL_SCALE = {"D8": 0.125, "D4": 0.25, "D2": 0.5, "M2": 2.0, "M4": 4.0, "M8": 8.0}


@_per_thread("FMUL")
def _fmul(ins: sass.Instruction, ctx: _Context) -> ThreadFn:
    mods = _mods(ins, {"FTZ", "RM", "RP", "RZ", "SAT", *_FMUL_SCALE})
    d, a, b = ins.operands
    ra, rb, w = _src(a, ctx, fp=True), _src(b, ctx, fp=True), _dst(d)
    rnd, flush, sat = _rounding(mods), "FTZ" in mods, "SAT" in mods
    scale = next((_FMUL_SCALE[m] for m in mods if m in _FMUL_SCALE), 1.0)
    return lambda t: w(t, mul32(ra(t), rb(t), rnd, flush, sat, scale))


@_per_thread("FMNMX")
def _fmnmx(ins: sass.Instruction, ctx: _Context) -> ThreadFn:
    mods = _mods(ins, {"FTZ"})
    d, a, b, p = ins.operands
    ra, rb, rp, w = _src(a, ctx, fp=True), _src(b, ctx, fp=True), _psrc(p), _dst(d)
    fl = ftz if "FTZ" in mods else (lambda x: x)
    return lambda t: w(t, (fmin32 if rp(t) else fmax32)(fl(ra(t)), fl(rb(t))))


@_per_thread("FSEL", "SEL", "USEL")
def _sel(ins: sass.Instruction, ctx: _Context) -> ThreadFn:
    _mods(ins)
    d, a, b, p = ins.operands
    fp = ins.name == "FSEL"
    ra, rb, rp, w = _src(a, ctx, fp=fp), _src(b, ctx, fp=fp), _psrc(p), _dst(d)
    return lambda t: w(t, ra(t) if rp(t) else rb(t))


@_per_thread("FSETP")
def _fsetp(ins: sass.Instruction, ctx: _Context) -> ThreadFn:
    cmp_name, _, bop = ins.mods
    _mods(ins, {cmp_name, bop, "FTZ"})
    cmp, comb = _fcmp(cmp_name), _combine(bop)
    p0, p1, a, b, pin = ins.operands
    w0, w1, ra, rb, rin = _pdst(p0), _pdst(p1), _src(a, ctx, fp=True), _src(b, ctx, fp=True), _psrc(pin)
    fl = ftz if "FTZ" in ins.mods else (lambda x: x)

    def fn(t: Thread) -> None:
        c, q = cmp(f32(fl(ra(t))), f32(fl(rb(t)))), rin(t)
        w0(t, comb(c, q))
        w1(t, comb(not c, q))

    return fn


@_per_thread("FSET")
def _fset(ins: sass.Instruction, ctx: _Context) -> ThreadFn:
    _, cmp_name, _, bop = ins.mods
    _mods(ins, {"BF", cmp_name, bop, "FTZ"})
    cmp, comb = _fcmp(cmp_name), _combine(bop)
    d, a, b, pin = ins.operands
    w, ra, rb, rin = _dst(d), _src(a, ctx, fp=True), _src(b, ctx, fp=True), _psrc(pin)
    fl = ftz if "FTZ" in ins.mods else (lambda x: x)
    return lambda t: w(t, 0x3F800000 if comb(cmp(f32(fl(ra(t))), f32(fl(rb(t)))), rin(t)) else 0)


@_per_thread("MUFU")
def _mufu(ins: sass.Instruction, ctx: _Context) -> ThreadFn:
    func = ins.mods[0]
    _mods(ins, {func})
    d, a = ins.operands
    ra, w = _src(a, ctx, fp=True), _dst(d)
    mufu(func, 0)  # raises Unsupported for a function without a rule
    given, address = ctx.launch.mufu_results, ins.address
    if given is not None:

        def from_given(t: Thread) -> None:
            x = ra(t)
            r = given.get((address, x))
            w(t, mufu(func, x) if r is None else r)

        return from_given
    return lambda t: w(t, mufu(func, ra(t)))


@_per_thread("FRND")
def _frnd(ins: sass.Instruction, ctx: _Context) -> ThreadFn:
    mode = ins.mods[0]
    mods = _mods(ins, {mode, "FTZ"})
    d, a = ins.operands
    ra, w, flush = _src(a, ctx, fp=True), _dst(d), "FTZ" in mods
    return lambda t: w(t, frnd32(ra(t), mode, flush))


@_per_thread("F2I")
def _f2i(ins: sass.Instruction, ctx: _Context) -> ThreadFn:
    src_type, _, int_type, mode, _ = ins.mods
    if src_type:
        raise Unsupported(f"F2I from {src_type}")
    if int_type not in _INT_RANGE:
        raise Unsupported(f"F2I to {int_type}")
    mods = _mods(ins, {"FTZ", int_type, mode, "NTZ"})
    if "NTZ" not in mods:
        raise Unsupported("F2I without NTZ")
    d, a = ins.operands
    ra, w, flush = _src(a, ctx, fp=True), _dst(d), "FTZ" in mods
    return lambda t: w(t, f2i32(ra(t), mode, int_type, flush))


_INT_SRC = {"U8": (8, False), "S8": (8, True), "U16": (16, False), "S16": (16, True), "U32": (32, False)}
_INT_SRC["S32"] = (32, True)


def _int_value(v: int, size: int, is_signed: bool) -> int:
    v &= (1 << size) - 1
    return v - (1 << size) if is_signed and v >> (size - 1) else v


@_per_thread("I2FP")
def _i2fp(ins: sass.Instruction, ctx: _Context) -> ThreadFn:
    dst_type, src_type, rnd_name = ins.mods
    if dst_type != "F32" or src_type not in _INT_SRC:
        raise Unsupported(f"I2FP.{dst_type}.{src_type}")
    _mods(ins, {dst_type, src_type, rnd_name})
    size, is_signed = _INT_SRC[src_type]
    rnd = _ROUNDING[rnd_name]
    d, a = ins.operands
    ra, w = _src(a, ctx), _dst(d)
    return lambda t: w(t, to_f32(float(_int_value(ra(t), size, is_signed)), rnd))


@_per_thread("I2F")
def _i2f(ins: sass.Instruction, ctx: _Context) -> ThreadFn:
    (src_type,) = ins.mods
    if src_type not in _INT_SRC:
        raise Unsupported(f"I2F.{src_type}")
    size, is_signed = _INT_SRC[src_type]
    d, a = ins.operands
    sel = int(a.suffix[2:]) if a.suffix else 0
    a = sass.Operand(a.kind, a.value, a.bank, a.index, a.neg, a.abs)
    ra, w = _src(a, ctx), _dst(d)
    shift = sel * size
    return lambda t: w(t, to_f32(float(_int_value(ra(t) >> shift, size, is_signed))))


# -- moves, constants and special registers ----------------------------------


@_per_thread("MOV", "UMOV")
def _mov(ins: sass.Instruction, ctx: _Context) -> ThreadFn:
    _mods(ins)
    d, a = ins.operands
    ra, w = _src(a, ctx), _dst(d)
    return lambda t: w(t, ra(t))


_LDC_SIZE = {"U8": (1, False), "S8": (1, True), "U16": (2, False), "S16": (2, True), "": (4, False), "64": (8, False)}


@_per_thread("LDC", "ULDC")
def _ldc(ins: sass.Instruction, ctx: _Context) -> ThreadFn:
    (size_name,) = ins.mods
    if size_name not in _LDC_SIZE:
        raise Unsupported(f"{ins.name}.{size_name}")
    size, is_signed = _LDC_SIZE[size_name]
    d, c = ins.operands
    w = _dst(d, 2 if size == 8 else 1)
    bank, off = c.bank, c.value
    idx = _src(c.index, ctx) if c.index is not None else _const(0)

    def fn(t: Thread) -> None:
        v = ctx.cbuf(bank, idx(t) + off, size)
        if is_signed:
            v = _int_value(v, size * 8, True) & M32
        w(t, v)

    return fn


_SREGS: dict[str, Callable[[Thread], int]] = {
    "SR_TID.X": lambda t: t.tid[0],
    "SR_TID.Y": lambda t: t.tid[1],
    "SR_TID.Z": lambda t: t.tid[2],
    "SR_CTAID.X": lambda t: t.warp.cta.ctaid[0],
    "SR_CTAID.Y": lambda t: t.warp.cta.ctaid[1],
    "SR_CTAID.Z": lambda t: t.warp.cta.ctaid[2],
    "SR_LANEID": lambda t: t.lane,
    "SR_EQMASK": lambda t: 1 << t.lane,
    "SR_LTMASK": lambda t: (1 << t.lane) - 1,
    "SR_LEMASK": lambda t: (2 << t.lane) - 1,
    "SR_GTMASK": lambda t: M32 & ~((2 << t.lane) - 1),
    "SR_GEMASK": lambda t: M32 & ~((1 << t.lane) - 1),
    "SRZ": lambda t: 0,
}


@_per_thread("S2R")
def _s2r(ins: sass.Instruction, ctx: _Context) -> ThreadFn:
    _mods(ins)
    d, s = ins.operands
    if s.value not in _SREGS:
        raise Unsupported(f"special register {s.value}")
    f, w = _SREGS[s.value], _dst(d)
    return lambda t: w(t, f(t))


@_per_thread("CS2R")
def _cs2r(ins: sass.Instruction, ctx: _Context) -> ThreadFn:
    mods = _mods(ins, {"32"})
    d, s = ins.operands
    if s.value != "SRZ":
        raise Unsupported(f"CS2R from {s.value}")
    w = _dst(d, 1 if "32" in mods else 2)
    return lambda t: w(t, 0)


@_per_thread("P2R")
def _p2r(ins: sass.Instruction, ctx: _Context) -> ThreadFn:
    _mods(ins)
    d, _, a, m = ins.operands
    ra, w, mask = _src(a, ctx), _dst(d), m.value & 0x7F

    def fn(t: Thread) -> None:
        pr = sum(1 << i for i in range(7) if t.p[i])
        w(t, (ra(t) & ~mask & M32) | (pr & mask))

    return fn


@_per_thread("R2P")
def _r2p(ins: sass.Instruction, ctx: _Context) -> ThreadFn:
    _mods(ins)
    _, a, m = ins.operands
    ra, mask = _src(a, ctx), m.value & 0x7F

    def fn(t: Thread) -> None:
        v = ra(t)
        for i in range(7):
            if mask >> i & 1:
                t.p[i] = bool(v >> i & 1)

    return fn


@_per_thread("NOP", "DEPBAR")
def _nop(ins: sass.Instruction, ctx: _Context) -> ThreadFn:
    return lambda t: None


# -- integer arithmetic and logic ---------------------------------------------


@_per_thread("IADD3", "UIADD3")
def _iadd3(ins: sass.Instruction, ctx: _Context) -> ThreadFn:
    mods = _mods(ins, {"X"})
    x = "X" in mods
    d, p0, p1, a, b, c, *carry_in = ins.operands
    # A negated operand adds its bitwise NOT. Without X, each one also adds 1, so -x is ~x + 1 and
    # the carries come out as for a subtraction. With X, the carry-in predicates supply the 1.
    ra, rb, rc = (_src(s, ctx, not_neg=True) for s in (a, b, c))
    ones = 0 if x else sum(1 for s in (a, b, c) if s.neg)
    cins = [_psrc(p) for p in carry_in]
    w, w0, w1 = _dst(d), _pdst(p0), _pdst(p1)

    def fn(t: Thread) -> None:
        total = ra(t) + rb(t) + rc(t) + ones + sum(1 for f in cins if f(t))
        w(t, total & M32)
        w0(t, total >= 1 << 32)
        w1(t, total >= 2 << 32)

    return fn


@_per_thread("IMAD", "UIMAD")
def _imad(ins: sass.Instruction, ctx: _Context) -> ThreadFn:
    mods = _mods(ins, {"X", "MOV", "IADD", "SHL", "U32", "HI"})
    x, hi = "X" in mods, "HI" in mods
    d, a, b, c, *carry_in = ins.operands
    ra, rb, rc, w = _src(a, ctx), _src(b, ctx), _src(c, ctx, not_neg=x), _dst(d)
    cin = _psrc(carry_in[0]) if carry_in else None
    if hi:
        if not (c.kind == "reg" and c.value == 255):
            raise Unsupported("IMAD.HI with an addend")
        is_signed = "U32" not in mods
        if is_signed:
            return lambda t: w(t, ((signed(ra(t)) * signed(rb(t))) >> 32) & M32)
        return lambda t: w(t, (ra(t) * rb(t)) >> 32)
    if cin is not None:
        return lambda t: w(t, (ra(t) * rb(t) + rc(t) + cin(t)) & M32)
    return lambda t: w(t, (ra(t) * rb(t) + rc(t)) & M32)


@_per_thread("LEA", "ULEA")
def _lea(ins: sass.Instruction, ctx: _Context) -> ThreadFn:
    mods = _mods(ins, {"HI", "X"})
    ops = list(ins.operands)
    d, p, a, b = ops[:4]
    rest = ops[4:]
    hi = "HI" in mods
    c = rest.pop(0) if hi else None
    shift = rest.pop(0).value
    cin = _psrc(rest.pop(0)) if "X" in mods else None
    if a.neg:
        raise Unsupported("LEA with a negated operand")
    ra, rb, w, wp = _src(a, ctx), _src(b, ctx), _dst(d), _pdst(p)
    rc = _src(c, ctx) if c is not None else _const(0)

    def fn(t: Thread) -> None:
        if hi:
            v = ((((rc(t) << 32) | ra(t)) << shift) >> 32) & M32
        else:
            v = (ra(t) << shift) & M32
        total = v + rb(t) + (1 if cin is not None and cin(t) else 0)
        w(t, total & M32)
        wp(t, total >= 1 << 32)

    return fn


def _lut3(lut: int) -> Callable[[int, int, int], int]:
    def f(a: int, b: int, c: int) -> int:
        r = 0
        for i in range(8):
            if lut >> i & 1:
                r |= (a if i & 4 else ~a) & (b if i & 2 else ~b) & (c if i & 1 else ~c)
        return r & M32

    return f


@_per_thread("LOP3", "ULOP3")
def _lop3(ins: sass.Instruction, ctx: _Context) -> ThreadFn:
    _mods(ins, {"LUT"})
    p, d, a, b, c, lut, pin = ins.operands
    f = _lut3(lut.value)
    ra, rb, rc, w, wp, rin = _src(a, ctx), _src(b, ctx), _src(c, ctx), _dst(d), _pdst(p), _psrc(pin)

    def fn(t: Thread) -> None:
        v = f(ra(t), rb(t), rc(t))
        w(t, v)
        wp(t, v != 0 or rin(t))

    return fn


@_per_thread("PLOP3")
def _plop3(ins: sass.Instruction, ctx: _Context) -> ThreadFn:
    _mods(ins, {"LUT"})
    p0, p1, a, b, c, lut0, lut1 = ins.operands
    w0, w1 = _pdst(p0), _pdst(p1)
    ra, rb, rc = _psrc(a), _psrc(b), _psrc(c)
    l0, l1 = lut0.value, lut1.value

    def fn(t: Thread) -> None:
        i = (ra(t) << 2) | (rb(t) << 1) | rc(t)
        w0(t, bool(l0 >> i & 1))
        w1(t, bool(l1 >> i & 1))

    return fn


@_per_thread("SHF", "USHF")
def _shf(ins: sass.Instruction, ctx: _Context) -> ThreadFn:
    direction, wrap, dtype, hi = ins.mods
    _mods(ins, {direction, wrap, dtype, hi})
    if dtype not in ("U32", "S32"):
        raise Unsupported(f"SHF.{dtype}")
    right, is_signed, high = direction == "R", dtype == "S32", hi == "HI"
    d, a, b, c = ins.operands
    ra, rb, rc, w = _src(a, ctx), _src(b, ctx), _src(c, ctx), _dst(d)

    def fn(t: Thread) -> None:
        s = rb(t)
        s = s & 31 if wrap else min(s, 32)
        x = (rc(t) << 32) | ra(t)
        if right:
            if is_signed and x >> 63:
                x -= 1 << 64
            x >>= s
        else:
            x <<= s
        w(t, (x >> 32 if high else x) & M32)

    return fn


@_per_thread("SGXT")
def _sgxt(ins: sass.Instruction, ctx: _Context) -> ThreadFn:
    mods = _mods(ins, {"W", "U32"})
    d, a, b = ins.operands
    ra, rb, w = _src(a, ctx), _src(b, ctx), _dst(d)
    wrap, unsigned = "W" in mods, "U32" in mods

    def fn(t: Thread) -> None:
        n = rb(t)
        n = n & 31 if wrap else min(n, 32)
        v = ra(t) & ((1 << n) - 1)
        if not unsigned and n and v >> (n - 1):
            v -= 1 << n
        w(t, v & M32)

    return fn


@_per_thread("BMSK")
def _bmsk(ins: sass.Instruction, ctx: _Context) -> ThreadFn:
    mods = _mods(ins, {"W"})
    d, a, b = ins.operands
    ra, rb, w, wrap = _src(a, ctx), _src(b, ctx), _dst(d), "W" in mods

    def fn(t: Thread) -> None:
        pos, width = ra(t), rb(t)
        pos, width = (pos & 31, width & 31) if wrap else (min(pos, 32), min(width, 32))
        w(t, (((1 << width) - 1) << pos) & M32)

    return fn


@_per_thread("IMNMX")
def _imnmx(ins: sass.Instruction, ctx: _Context) -> ThreadFn:
    mods = _mods(ins, {"U32"})
    d, a, b, p = ins.operands
    ra, rb, rp, w = _src(a, ctx), _src(b, ctx), _psrc(p), _dst(d)
    key = (lambda v: v) if "U32" in mods else signed

    def fn(t: Thread) -> None:
        x, y = ra(t), rb(t)
        w(t, (min if rp(t) else max)(x, y, key=key))

    return fn


@_per_thread("PRMT")
def _prmt(ins: sass.Instruction, ctx: _Context) -> ThreadFn:
    _mods(ins)
    d, a, sel, b = ins.operands
    ra, rs, rb, w = _src(a, ctx), _src(sel, ctx), _src(b, ctx), _dst(d)

    def fn(t: Thread) -> None:
        x, s = ra(t) | (rb(t) << 32), rs(t)
        r = 0
        for i in range(4):
            n = s >> (4 * i) & 0xF
            byte = x >> (8 * (n & 7)) & 0xFF
            if n & 8:
                byte = 0xFF if byte & 0x80 else 0
            r |= byte << (8 * i)
        w(t, r)

    return fn


_ICMP: dict[str, Callable[[int, int], bool]] = {
    "F": lambda x, y: False,
    "LT": lambda x, y: x < y,
    "EQ": lambda x, y: x == y,
    "LE": lambda x, y: x <= y,
    "GT": lambda x, y: x > y,
    "NE": lambda x, y: x != y,
    "GE": lambda x, y: x >= y,
    "T": lambda x, y: True,
}


@_per_thread("ISETP", "UISETP")
def _isetp(ins: sass.Instruction, ctx: _Context) -> ThreadFn:
    cmp_name, _, bop, ex = ins.mods
    if ex:
        raise Unsupported("ISETP.EX")
    mods = _mods(ins, {cmp_name, bop, "U32"})
    cmp, comb = _ICMP[cmp_name], _combine(bop)
    p0, p1, a, b, pin = ins.operands
    w0, w1, ra, rb, rin = _pdst(p0), _pdst(p1), _src(a, ctx), _src(b, ctx), _psrc(pin)
    key = (lambda v: v) if "U32" in mods else signed

    def fn(t: Thread) -> None:
        c, q = cmp(key(ra(t)), key(rb(t))), rin(t)
        w0(t, comb(c, q))
        w1(t, comb(not c, q))

    return fn


@_per_thread("IABS")
def _iabs(ins: sass.Instruction, ctx: _Context) -> ThreadFn:
    _mods(ins)
    d, a = ins.operands
    ra, w = _src(a, ctx), _dst(d)
    return lambda t: w(t, abs(signed(ra(t))) & M32)


@_per_thread("POPC", "UPOPC")
def _popc(ins: sass.Instruction, ctx: _Context) -> ThreadFn:
    _mods(ins)
    d, a = ins.operands
    ra, w = _src(a, ctx), _dst(d)
    return lambda t: w(t, ra(t).bit_count())


@_per_thread("BREV", "UBREV")
def _brev(ins: sass.Instruction, ctx: _Context) -> ThreadFn:
    _mods(ins)
    d, a = ins.operands
    ra, w = _src(a, ctx), _dst(d)
    return lambda t: w(t, int(f"{ra(t):032b}"[::-1], 2))


@_per_thread("FLO", "UFLO")
def _flo(ins: sass.Instruction, ctx: _Context) -> ThreadFn:
    mods = _mods(ins, {"U32", "SH"})
    d, a = ins.operands
    ra, w = _src(a, ctx), _dst(d)
    unsigned, sh = "U32" in mods, "SH" in mods

    def fn(t: Thread) -> None:
        v = ra(t)
        if not unsigned and v & 0x80000000:
            v = ~v & M32
        if v == 0:
            w(t, M32)
        else:
            pos = v.bit_length() - 1
            w(t, 31 - pos if sh else pos)

    return fn


# -- memory -------------------------------------------------------------------

_MEM_SIZE = {"U8": (1, False), "S8": (1, True), "U16": (2, False), "S16": (2, True), "": (4, False)}
_MEM_SIZE |= {"64": (8, False), "128": (16, False)}


def _mem_size(name: str) -> tuple[int, bool]:
    if name not in _MEM_SIZE:
        raise Unsupported(f"memory size {name}")
    return _MEM_SIZE[name]


def _shared_addr(op: sass.Operand, ctx: _Context) -> Reader:
    if op.extra is not None:
        raise Unsupported("shared address with a uniform register")
    base, scale, off = _src(op.index, ctx), op.scale, op.value
    return lambda t: (base(t) * scale + off) & M32


def _shared(ctx: _Context, addr: int, size: int) -> bytearray:
    mem = ctx.cta.shared
    if addr % size:
        raise ExecutionError(f"misaligned shared memory access of {size} bytes at {addr:#x}")
    if addr + size > len(mem):
        raise ExecutionError(f"shared memory access at {addr:#x} is outside the {len(mem):#x} bytes")
    return mem


def _load(mem: bytearray, off: int, size: int, is_signed: bool) -> int:
    v = int.from_bytes(mem[off : off + size], "little")
    return _int_value(v, size * 8, True) & M32 if is_signed else v


@_per_thread("LDS")
def _lds(ins: sass.Instruction, ctx: _Context) -> ThreadFn:
    (size_name,) = ins.mods
    size, is_signed = _mem_size(size_name)
    d, addr = ins.operands
    ra, w = _shared_addr(addr, ctx), _dst(d, max(1, size // 4))

    def fn(t: Thread) -> None:
        a = ra(t)
        w(t, _load(_shared(ctx, a, size), a, size, is_signed))

    return fn


def _store_reader(op: sass.Operand, size: int) -> Reader:
    """The value a store writes: size bytes from a register and the ones after it."""
    n, count = op.value, max(1, size // 4)
    if op.kind != "reg":
        raise Unsupported(f"store of a {op.kind} operand")
    if n == 255:
        return _const(0)
    mask = (1 << (8 * size)) - 1
    return lambda t: sum((t.r[n + i] if n + i < 255 else 0) << (32 * i) for i in range(count)) & mask


@_per_thread("STS")
def _sts(ins: sass.Instruction, ctx: _Context) -> ThreadFn:
    (size_name,) = ins.mods
    size, _ = _mem_size(size_name)
    addr, s = ins.operands
    ra, rs = _shared_addr(addr, ctx), _store_reader(s, size)

    def fn(t: Thread) -> None:
        a = ra(t)
        _shared(ctx, a, size)[a : a + size] = rs(t).to_bytes(size, "little")

    return fn


_ATOM_OPS: dict[str, Callable[[int, int], int]] = {
    "ADD": lambda x, y: (x + y) & M32,
    "AND": lambda x, y: x & y,
    "OR": lambda x, y: x | y,
    "XOR": lambda x, y: x ^ y,
}


def _atom_op(name: str) -> Callable[[int, int], int]:
    if name not in _ATOM_OPS:
        raise Unsupported(f"atomic {name}")
    return _ATOM_OPS[name]


@_per_thread("ATOMS")
def _atoms(ins: sass.Instruction, ctx: _Context) -> ThreadFn:
    (op_name,) = ins.mods
    f = _atom_op(op_name)
    d, addr, s = ins.operands
    ra, rs, w = _shared_addr(addr, ctx), _src(s, ctx), _dst(d)

    def fn(t: Thread) -> None:
        a = ra(t)
        mem = _shared(ctx, a, 4)
        old = int.from_bytes(mem[a : a + 4], "little")
        mem[a : a + 4] = f(old, rs(t)).to_bytes(4, "little")
        w(t, old)

    return fn


def _global_addr(op: sass.Operand, ctx: _Context, mods: list[str]) -> Reader:
    if "E" not in mods:
        raise Unsupported("global access without 64-bit addresses (.E)")
    base, off = _reg_pair(op.index), op.value
    extra = _src(op.extra, ctx) if op.extra is not None else _const(0)
    return lambda t: (base(t) + extra(t) + off) & ((1 << 64) - 1)


_SEMANTICS = {"", "CONSTANT", "STRONG.SM", "STRONG.GPU"}


def _check_sem(name: str) -> None:
    if name not in _SEMANTICS:
        raise Unsupported(f"memory semantics {name}")


@_per_thread("LDG")
def _ldg(ins: sass.Instruction, ctx: _Context) -> ThreadFn:
    _, size_name, sem = ins.mods
    _check_sem(sem)
    size, is_signed = _mem_size(size_name)
    d, addr = ins.operands
    ra, w, mem = _global_addr(addr, ctx, ins.mods), _dst(d, max(1, size // 4)), ctx.launch.memory

    def fn(t: Thread) -> None:
        buf, off = mem.region(ra(t), size)
        w(t, _load(buf, off, size, is_signed))

    return fn


@_per_thread("STG")
def _stg(ins: sass.Instruction, ctx: _Context) -> ThreadFn:
    _, size_name, sem = ins.mods
    _check_sem(sem)
    size, _ = _mem_size(size_name)
    addr, s = ins.operands
    ra, rs, mem = _global_addr(addr, ctx, ins.mods), _store_reader(s, size), ctx.launch.memory

    def fn(t: Thread) -> None:
        buf, off = mem.region(ra(t), size)
        buf[off : off + size] = rs(t).to_bytes(size, "little")

    return fn


def _global_atomic(ctx: _Context, ra: Reader, rs: Reader, f: Callable[[int, int], int], t: Thread) -> int:
    buf, off = ctx.launch.memory.region(ra(t), 4)
    old = int.from_bytes(buf[off : off + 4], "little")
    buf[off : off + 4] = f(old, rs(t)).to_bytes(4, "little")
    return old


@_per_thread("ATOMG")
def _atomg(ins: sass.Instruction, ctx: _Context) -> ThreadFn:
    _, op_name, sem = ins.mods
    _check_sem(sem)
    f = _atom_op(op_name)
    p, d, addr, s = ins.operands
    if not sass._is_pt(p):
        raise Unsupported("ATOMG with a predicate destination")
    ra, rs, w = _global_addr(addr, ctx, ins.mods), _src(s, ctx), _dst(d)
    return lambda t: w(t, _global_atomic(ctx, ra, rs, f, t))


@_per_thread("RED")
def _red(ins: sass.Instruction, ctx: _Context) -> ThreadFn:
    _, op_name, dtype, sem = ins.mods
    _check_sem(sem)
    if dtype:
        raise Unsupported(f"RED.{dtype}")
    f = _atom_op(op_name)
    addr, s = ins.operands
    ra, rs = _global_addr(addr, ctx, ins.mods), _src(s, ctx)

    def fn(t: Thread) -> None:
        _global_atomic(ctx, ra, rs, f, t)

    return fn


# -- surfaces -------------------------------------------------------------------

_SU_COORDS = {"1D": 1, "1D_BUFFER": 1, "1D_ARRAY": 2, "2D": 2, "2D_ARRAY": 3, "3D": 3}
_SU_CHANNELS = {"R": 1, "RG": 2, "RGB": 3, "": 4}


def _surface(ins: sass.Instruction, ctx: _Context) -> Callable[[Thread], tuple[Surface, int | None]]:
    """For each thread: the surface named by the instruction's handle, and the texel's first word (None outside)."""
    dim = ins.mods[1]
    if dim not in _SU_COORDS:
        raise Unsupported(f"surface dimension {dim}")
    addr = next(op for op in ins.operands if op.kind == "addr")
    bank, index = (op.value for op in ins.operands[-2:])
    n = addr.index.value
    if addr.index.kind != "reg" or n == 255:
        raise Unsupported("surface coordinates without a register")
    key = (bank, index)

    def find(t: Thread) -> tuple[Surface, int | None]:
        surf = ctx.launch.surfaces.get(key)
        if surf is None:
            raise ExecutionError(f"no surface for handle {bank:#x}, {index:#x}")
        r = t.r
        c = [signed(r[n + i]) for i in range(_SU_COORDS[dim])]
        if dim == "1D" or dim == "1D_BUFFER":
            x, y, z = c[0], 0, 0
        elif dim == "1D_ARRAY":
            x, y, z = c[0], 0, c[1] & 0xFFFF
        elif dim == "2D":
            x, y, z = c[0], c[1], 0
        elif dim == "2D_ARRAY":
            x, y, z = c[0], c[1], c[2] & 0xFFFF
        else:
            x, y, z = c
        return surf, surf.index(x, y, z)

    return find


def _su_formatted(ins: sass.Instruction) -> int:
    kind, _, mask = ins.mods[:3]
    if kind != "P":
        raise Unsupported(f"{ins.name}.{kind}")
    if mask not in _SU_CHANNELS:
        raise Unsupported(f"surface component mask {mask}")
    return _SU_CHANNELS[mask]


@_per_thread("SUST")
def _sust(ins: sass.Instruction, ctx: _Context) -> ThreadFn:
    count = _su_formatted(ins)
    find = _surface(ins, ctx)
    s = ins.operands[1]
    n = s.value

    def fn(t: Thread) -> None:
        surf, i = find(t)
        if i is None:
            return
        for k in range(surf.channels):
            surf.data[i + k] = t.r[n + k] if k < count and n + k < 255 else 0

    return fn


@_per_thread("SULD")
def _suld(ins: sass.Instruction, ctx: _Context) -> ThreadFn:
    count = _su_formatted(ins)
    find = _surface(ins, ctx)
    w = _dst(ins.operands[0], count)

    def fn(t: Thread) -> None:
        surf, i = find(t)
        if i is None:
            w(t, 0)
            return
        words = [surf.data[i + k] if k < surf.channels else 0 for k in range(count)]
        w(t, sum(v << (32 * k) for k, v in enumerate(words)))

    return fn


def _su_atomic(ins: sass.Instruction, ctx: _Context, with_result: bool) -> ThreadFn:
    kind, _, _, op_name, sem = ins.mods[:5]
    if kind != "D":
        raise Unsupported(f"{ins.name}.{kind}")
    _check_sem(sem)
    f = _atom_op(op_name)
    find = _surface(ins, ctx)
    ops = ins.operands
    d = ops[0] if with_result else None
    s = ops[2] if with_result else ops[1]
    rs = _src(s, ctx)
    w = _dst(d) if d is not None else (lambda t, v: None)

    def fn(t: Thread) -> None:
        surf, i = find(t)
        if i is None:
            w(t, 0)
            return
        old = surf.data[i]
        surf.data[i] = f(old, rs(t))
        w(t, old)

    return fn


@_per_thread("SURED")
def _sured(ins: sass.Instruction, ctx: _Context) -> ThreadFn:
    return _su_atomic(ins, ctx, with_result=False)


@_per_thread("SUATOM")
def _suatom(ins: sass.Instruction, ctx: _Context) -> ThreadFn:
    return _su_atomic(ins, ctx, with_result=True)


# -- control flow -----------------------------------------------------------------


def _cond(ins: sass.Instruction) -> Callable[[Thread], bool]:
    """The guard and the instruction's own predicate (bit 87) together decide whether it acts."""
    guard = _guard(ins)
    pred = ins.operands[0] if ins.operands and ins.operands[0].kind == "pred" else None
    p = None if pred is None or sass._is_pt(pred) else _psrc(pred)
    if guard is None and p is None:
        return lambda t: True
    if p is None:
        return guard
    if guard is None:
        return p
    return lambda t: guard(t) and p(t)


def _label_index(ins: sass.Instruction) -> int:
    target = next(op for op in ins.operands if op.kind == "label")
    return target.value // INSTRUCTION_SIZE


@_per_group("BRA")
def _bra(ins: sass.Instruction, ctx: _Context) -> Step:
    _mods(ins)
    cond, target = _cond(ins), _label_index(ins)

    def step(group: list[Thread], warp: Warp) -> None:
        for t in group:
            t.pc = target if cond(t) else t.pc + 1

    return step


@_per_group("CALL")
def _call(ins: sass.Instruction, ctx: _Context) -> Step:
    # The sample has no RET; CALL.REL.NOINC leaves a loop, so it acts as a branch.
    _mods(ins, {"REL", "NOINC"})
    if "NOINC" not in ins.mods:
        raise Unsupported("CALL that returns")
    return _bra(sass.Instruction(ins.address, ins.opcode, "BRA", [], ins.guard, ins.operands), ctx)


@_per_group("EXIT")
def _exit(ins: sass.Instruction, ctx: _Context) -> Step:
    _mods(ins)
    cond = _cond(ins)

    def step(group: list[Thread], warp: Warp) -> None:
        for t in group:
            if cond(t):
                t.state = EXITED
            else:
                t.pc += 1

    return step


def _breg(ins: sass.Instruction) -> int:
    return next(op.value for op in ins.operands if op.kind == "breg")


@_per_group("BSSY")
def _bssy(ins: sass.Instruction, ctx: _Context) -> Step:
    _mods(ins)
    cond, n = _cond(ins), _breg(ins)

    def step(group: list[Thread], warp: Warp) -> None:
        warp.b[n] = warp.mask([t for t in group if cond(t)])
        for t in group:
            t.pc += 1

    return step


@_per_group("BSYNC")
def _bsync(ins: sass.Instruction, ctx: _Context) -> Step:
    _mods(ins)
    cond, n = _cond(ins), _breg(ins)

    def step(group: list[Thread], warp: Warp) -> None:
        for t in group:
            if cond(t):
                t.state, t.wait = WAIT_BSYNC, n
            else:
                t.pc += 1

    return step


@_per_group("BREAK")
def _break(ins: sass.Instruction, ctx: _Context) -> Step:
    _mods(ins)
    cond, n = _cond(ins), _breg(ins)

    def step(group: list[Thread], warp: Warp) -> None:
        for t in group:
            if cond(t):
                warp.b[n] &= ~(1 << t.lane)
            t.pc += 1

    return step


@_per_group("WARPSYNC")
def _warpsync(ins: sass.Instruction, ctx: _Context) -> Step:
    _mods(ins)
    cond = _cond(ins)
    mask = ins.operands[-1].value

    def step(group: list[Thread], warp: Warp) -> None:
        for t in group:
            if cond(t):
                t.state, t.wait = WAIT_WARPSYNC, mask
            else:
                t.pc += 1

    return step


@_per_group("BAR")
def _bar(ins: sass.Instruction, ctx: _Context) -> Step:
    _mods(ins, {"SYNC"})
    guard = _guard(ins)
    n = ins.operands[0].value

    def step(group: list[Thread], warp: Warp) -> None:
        for t in group:
            if guard is None or guard(t):
                t.state, t.wait = WAIT_BAR, n
            else:
                t.pc += 1

    return step


# -- warp-wide operations -------------------------------------------------------------


def _active(ins: sass.Instruction, group: list[Thread]) -> list[Thread]:
    guard = _guard(ins)
    return group if guard is None else [t for t in group if guard(t)]


def _advance(group: list[Thread]) -> None:
    for t in group:
        t.pc += 1


@_per_group("VOTE", "VOTEU")
def _vote(ins: sass.Instruction, ctx: _Context) -> Step:
    (mode,) = ins.mods
    if mode not in ("ALL", "ANY", "EQ"):
        raise Unsupported(f"VOTE.{mode}")
    d, p, pin = ins.operands
    w, wp, rin = _dst(d), _pdst(p), _psrc(pin)

    def step(group: list[Thread], warp: Warp) -> None:
        active = _active(ins, group)
        votes = [rin(t) for t in active]
        ballot = warp.mask([t for t, v in zip(active, votes, strict=True) if v])
        result = {"ALL": all(votes), "ANY": any(votes), "EQ": all(votes) or not any(votes)}[mode]
        for t in active:
            w(t, ballot)
            wp(t, result)
        _advance(group)

    return step


_SHFL_MODES = ("IDX", "UP", "DOWN", "BFLY")


@_per_group("SHFL")
def _shfl(ins: sass.Instruction, ctx: _Context) -> Step:
    (mode,) = ins.mods
    p, d, a, b, c = ins.operands
    wp, w, ra, rb, rc = _pdst(p), _dst(d), _src(a, ctx), _src(b, ctx), _src(c, ctx)

    def step(group: list[Thread], warp: Warp) -> None:
        active = _active(ins, group)
        lanes = {t.lane: t for t in warp.threads}
        out = []
        for t in active:
            lane, bval, cval = t.lane, rb(t) & 31, rc(t)
            seg = (cval >> 8) & 31
            max_lane = (lane & seg) | (cval & 31 & ~seg)
            min_lane = lane & seg
            if mode == "UP":
                j = lane - bval
                ok = j >= max_lane
            elif mode == "DOWN":
                j = lane + bval
                ok = j <= max_lane
            elif mode == "BFLY":
                j = lane ^ bval
                ok = j <= max_lane
            else:
                j = min_lane | (bval & ~seg)
                ok = j <= max_lane
            if not ok:
                j = lane
            src = lanes.get(j, t)
            out.append((t, ra(src), ok))
        for t, v, ok in out:
            w(t, v)
            wp(t, ok)
        _advance(group)

    if mode not in _SHFL_MODES:
        raise Unsupported(f"SHFL.{mode}")
    return step


@_per_group("MATCH")
def _match(ins: sass.Instruction, ctx: _Context) -> Step:
    mode, size = ins.mods
    p, d, a = ins.operands
    wp, w = _pdst(p), _dst(d)
    ra = _reg_pair(a) if size == "U64" else _src(a, ctx)

    def step(group: list[Thread], warp: Warp) -> None:
        active = _active(ins, group)
        values = {t.lane: ra(t) for t in active}
        same = len(set(values.values())) <= 1
        for t in active:
            if mode == "ALL":
                w(t, warp.mask(active) if same else 0)
                wp(t, same)
            else:
                w(t, warp.mask([u for u in active if values[u.lane] == values[t.lane]]))
        _advance(group)

    if mode not in ("ALL", "ANY"):
        raise Unsupported(f"MATCH.{mode}")
    return step


_REDUX: dict[str, Callable[[int, int], int]] = {
    "AND": lambda x, y: x & y,
    "OR": lambda x, y: x | y,
    "XOR": lambda x, y: x ^ y,
    "SUM": lambda x, y: (x + y) & M32,
}


@_per_group("REDUX")
def _redux(ins: sass.Instruction, ctx: _Context) -> Step:
    (op_name,) = ins.mods
    if op_name not in _REDUX:
        raise Unsupported(f"REDUX.{op_name}")
    f = _REDUX[op_name]
    d, a = ins.operands
    w, ra = _dst(d), _src(a, ctx)

    def step(group: list[Thread], warp: Warp) -> None:
        active = _active(ins, group)
        values = [ra(t) for t in active]
        acc = values[0] if values else 0
        for v in values[1:]:
            acc = f(acc, v)
        for t in active:
            w(t, acc)
        _advance(group)

    return step
