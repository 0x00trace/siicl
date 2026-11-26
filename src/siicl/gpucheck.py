"""Check translated compute programs on a GPU against the reference interpreter.

For each program, check() makes inputs from a seed, runs the SPIR-V translation
on the GPU (siicl.vk) and the program in siicl.interp, and compares every word
of every surface. The GPU records each MUFU input and result, and the
interpreter uses those results, because MUFU computes approximations whose
last bits the interpreter can't predict. check() reports separately how far
the recorded results are from exact values.
"""

from __future__ import annotations

import copy
import math
import random
import struct
from collections import Counter
from dataclasses import dataclass, field

from siicl import interp, sass, translate, vk

SURFACE_SIZE = {"1D": (256, 1, 1), "1D_ARRAY": (64, 1, 4), "2D": (32, 32, 1), "2D_ARRAY": (32, 32, 4)}
SURFACE_SIZE["3D"] = (16, 16, 8)
MAX_THREADS = 4096
COMPARE_OPS = {"ISETP", "UISETP"}
FLOAT_OPS = {"FFMA", "FADD", "FMUL", "FMNMX", "FSEL", "FSETP", "FSET", "MUFU", "FRND", "F2I"}
INT_OPS = {"IADD3", "IMAD", "LEA", "LOP3", "SHF", "ISETP", "IMNMX", "SEL", "SGXT", "PRMT", "I2FP", "I2F", "POPC"}
INT_OPS |= {"FLO", "BREV", "IABS", "UIADD3", "UISETP", "ULOP3", "USHF", "ULEA", "UIMAD"}
NO_REG_DEST = {"STS", "STG", "SUST", "SURED", "RED", "FSETP", "ISETP", "PLOP3", "UISETP", "BRA", "EXIT", "BAR"}
INTERP_STEP_LIMIT = 30_000_000
ABSOLUTE_ERROR = {"SIN", "COS", "LG2"}
LOG_CAPACITY = 1 << 22


def _dest(ins: sass.Instruction) -> sass.Operand | None:
    if ins.name in NO_REG_DEST or not ins.operands:
        return None
    op = ins.operands[1] if ins.name in ("LOP3", "ULOP3") else ins.operands[0]
    return op if op.kind in ("reg", "ureg") else None


def _reads(ins: sass.Instruction, kind: str, n: int) -> bool:
    dest = _dest(ins)
    for op in ins.operands:
        if op is dest:
            continue
        for o in (op, op.index):
            if o is not None and o.kind == kind and o.value == n:
                return True
    return False


def _use(instructions: list[sass.Instruction], start: int, kind: str, n: int) -> str | None:
    """How the first later instruction that reads a register uses it, unless the register is written first."""
    for ins in instructions[start + 1 :]:
        if _reads(ins, kind, n):
            return _kind(ins.name)
        d = _dest(ins)
        if d is not None and d.kind == kind and d.value == n:
            return None
    return None


def _kind(name: str) -> str | None:
    """'b' for a bound that an integer comparison uses, 'f' for a float, 'i' for another integer."""
    if name in COMPARE_OPS:
        return "b"
    return "f" if name in FLOAT_OPS else "i" if name in INT_OPS else None


def word_types(instructions: list[sass.Instruction]) -> tuple[dict[int, dict[int, str]], dict[int, dict[int, str]]]:
    """How the program uses constant buffer words: by (bank, offset), and for indexed reads by (bank, offset % 16)."""
    static: dict[int, dict[int, str]] = {}
    indexed: dict[int, dict[int, str]] = {}
    for i, ins in enumerate(instructions):
        for op in ins.operands:
            if op.kind != "cbuf":
                continue
            dynamic = op.index is not None and not sass._is_zero_reg(op.index)
            if ins.name in ("LDC", "ULDC", "MOV", "UMOV"):
                d = ins.operands[0]
                width = 2 if ins.mods and ins.mods[0] == "64" else 1
                for k in range(width):
                    kind = _use(instructions, i, d.kind, d.value + k)
                    if kind:
                        table = indexed if dynamic else static
                        key = (op.value + 4 * k) % 16 if dynamic else op.value + 4 * k
                        table.setdefault(op.bank, {}).setdefault(key, kind)
            elif not dynamic:
                kind = _kind(ins.name) or "i"
                static.setdefault(op.bank, {}).setdefault(op.value, kind)
            else:
                indexed.setdefault(op.bank, {}).setdefault(op.value % 16, "i")
    return static, indexed


def _float(rnd: random.Random, profile: int) -> float:
    """Uniform in [-2, 2] or [0, 1], or log-uniform between 2^-6 and 2^10 with either sign or positive."""
    if profile == 0:
        return rnd.uniform(-2.0, 2.0)
    if profile == 1:
        return rnd.random()
    magnitude = 2.0 ** rnd.uniform(-6, 10)
    return magnitude if profile == 3 or rnd.random() < 0.5 else -magnitude


def _float_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def make_launch(program: interp.Program, seed: int) -> interp.Launch:
    """Inputs for one check: constant buffers, surfaces filled with random words, and a grid."""
    rnd = random.Random(seed)
    int_profile = seed % 4  # integers in [0, 4), [0, 16), [0, 64), or a mix of small and large
    float_profile = (seed + seed // 4) % 4  # see _float
    static, indexed = word_types(program.instructions)
    banks = sorted(set(static) | set(indexed))

    def value(kind: str | None) -> int:
        kind = kind or rnd.choice("fi")
        if kind == "f":
            return _float_bits(_float(rnd, float_profile))
        if kind == "b" and int_profile:
            return rnd.randrange(16, 64)
        if int_profile == 3:
            return rnd.randrange(4) if rnd.random() < 0.5 else rnd.randrange(16, 64)
        return rnd.randrange((4, 16, 64)[int_profile])

    cbufs = {}
    for bank in banks:
        if bank == 1:
            continue
        end = max(static.get(bank, {0: ""})) + 16
        size = max(256, -(-end // 16) * 16, 4096 if bank in indexed else 0)
        words = []
        for off in range(0, size, 4):
            kind = static.get(bank, {}).get(off) or indexed.get(bank, {}).get(off % 16)
            words.append(value(kind))
        cbufs[bank] = struct.pack(f"<{len(words)}I", *words)
    consts = program.info.constants
    cbufs[1] = consts + bytes(-len(consts) % 16)
    surfaces = {}
    dims = translate.surfaces(program.instructions)
    for handle, (dim, channels) in dims.items():
        w, h, d = SURFACE_SIZE[dim]
        surfaces[handle] = interp.Surface(w, h, d, channels, [rnd.getrandbits(32) for _ in range(w * h * d * channels)])
    lx, ly, lz = program.info.local_size
    span = max((SURFACE_SIZE[d] for d, _ in dims.values()), default=(64, 1, 1))
    grid = [max(1, -(-span[k] // n)) for k, n in enumerate((lx, ly, lz))]
    while grid[0] * grid[1] * grid[2] * lx * ly * lz > MAX_THREADS and max(grid) > 1:
        k = grid.index(max(grid))
        grid[k] = -(-grid[k] // 2)
    return interp.Launch(tuple(grid), cbufs, surfaces)


def abs_error(a: int, b: int) -> float:
    """|a - b| for two 32-bit floats; 0 when they are equal or both NaN, infinity when only one is finite."""
    if a == b or (interp._is_nan(a) and interp._is_nan(b)):
        return 0.0
    x, y = interp.f32(a), interp.f32(b)
    return abs(x - y) if math.isfinite(x) and math.isfinite(y) else math.inf


def ulps(a: int, b: int) -> int:
    """Distance between two 32-bit floats in units in the last place (0 for two NaNs)."""
    if interp._is_nan(a) and interp._is_nan(b):
        return 0

    def key(x: int) -> int:
        return (x & 0x7FFFFFFF) if not x & 0x80000000 else -(x & 0x7FFFFFFF)

    return abs(key(a) - key(b))


@dataclass
class Check:
    seed: int
    grid: tuple[int, int, int]
    status: str  # "match", "mismatch", "interpreter error", "gpu step limit"
    words: int = 0  # surface words compared
    order_dependent: int = 0  # words whose value depends on the order in which threads run
    mismatches: list[tuple[tuple[int, int], int, int, int]] = field(default_factory=list)  # handle, index, cpu, gpu
    mismatch_count: int = 0
    instructions: int = 0  # thread-instructions the interpreter executed
    executed: set[int] = field(default_factory=set)  # addresses of the instructions that ran
    mufu_logged: int = 0
    # How far the GPU's MUFU results are from exact ones, by function: in ulps, or for the functions whose
    # accuracy NVIDIA gives as an absolute error (SIN, COS, LG2), as an absolute error.
    mufu_ulps: dict[str, int] = field(default_factory=dict)
    mufu_abs: dict[str, float] = field(default_factory=dict)
    error: str = ""


def check(program: interp.Program, tr: translate.Translation, gpu: vk.Gpu, seed: int) -> Check:
    launch = make_launch(program, seed)
    on_gpu = copy.deepcopy(launch)
    res = gpu.run(tr, on_gpu, log_capacity=LOG_CAPACITY)
    out = Check(seed, launch.grid, "match")
    if res.status & translate.STATUS_STEP_LIMIT:
        out.status = "gpu step limit"
        return out
    given: dict[tuple[int, int], int] = {}
    funcs = {ins.address: ins.mods[0] for ins in program.instructions if ins.name == "MUFU"}
    worst: Counter[str] = Counter()
    absolute: dict[str, float] = {}
    for address, x, r in res.mufu:
        given[(address, x)] = r
        func = funcs[address]
        exact = interp.mufu(func, x)
        if func in ABSOLUTE_ERROR:
            absolute[func] = max(absolute.get(func, 0.0), abs_error(r, exact))
        else:
            worst[func] = max(worst[func], ulps(r, exact))
    out.mufu_logged, out.mufu_ulps, out.mufu_abs = len(res.mufu), dict(worst), absolute
    if res.mufu_dropped:
        out.status, out.error = "interpreter error", f"{res.mufu_dropped:,} MUFU results did not fit the log"
        return out
    launch.mufu_results = given
    for surf in launch.surfaces.values():
        surf.track_writes()
    reversed_launch = copy.deepcopy(launch)
    try:
        result = interp.run(program, launch, step_limit=INTERP_STEP_LIMIT)
        interp.run(program, reversed_launch, step_limit=INTERP_STEP_LIMIT, reverse=True)
    except (interp.ExecutionError, interp.Unsupported) as err:
        out.status, out.error = "interpreter error", str(err)
        return out
    out.instructions, out.executed = result.instructions, result.executed
    for handle, surf in launch.surfaces.items():
        other = on_gpu.surfaces[handle].data
        backward = reversed_launch.surfaces[handle]
        racy = surf.conflicts | backward.conflicts
        for i, (a, b) in enumerate(zip(surf.data, other, strict=True)):
            if a != backward.data[i] or i in racy:
                out.order_dependent += 1
                continue
            out.words += 1
            if a != b and not (interp._is_nan(a) and interp._is_nan(b)):
                out.mismatch_count += 1
                if len(out.mismatches) < 8:
                    out.mismatches.append((handle, i, a, b))
    if out.mismatch_count:
        out.status = "mismatch"
    return out
