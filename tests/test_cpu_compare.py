"""Each program here runs on Unicorn (siicl.cpu) and on dynarmic (siicl.dynarmic). Both start from
the same registers and memory, and must end with the same registers, flags and data memory.

Unicorn is the reference: this is how instructions that siicl adds to its dynarmic fork are checked
(docs/cpu-backend.md). The programs cover the instruction groups that `siicl scan` reports in game
code. The tests need the C library (native/CMakeLists.txt) and skip without it.
"""

import random

import pytest
from unicorn import arm64_const as a64

from siicl import cpu, dynarmic
from tests.builders import AS, assemble

pytestmark = [
    pytest.mark.skipif(not AS, reason="aarch64-linux-gnu-binutils is not installed"),
    pytest.mark.skipif(not dynarmic.available(), reason="libsiicl_dynarmic.so is not built; see native/CMakeLists.txt"),
]

CODE, DATA, STACK = 0x10000, 0x20000, 0x30000
DATA_SIZE, STACK_SIZE = 0x1000, 0x1000
MASK64 = (1 << 64) - 1


def start_state(seed: int) -> dict:
    """Registers and data for a run. X16 and X17 point into the data page, for programs to use."""
    rng = random.Random(seed)
    x = [rng.getrandbits(64) for _ in range(31)]
    x[16], x[17] = DATA + 0x100, DATA + 0x800
    return {
        "x": x,
        "v": [rng.getrandbits(128) for _ in range(32)],
        "nzcv": rng.getrandbits(4),
        "data": rng.randbytes(DATA_SIZE),
    }


def on_unicorn(code: bytes, start: dict) -> dict:
    c = cpu.Cpu()
    c.map(CODE, 0x1000, cpu.READ | cpu.EXECUTE)
    c.map(DATA, DATA_SIZE, cpu.READ | cpu.WRITE)
    c.map(STACK, STACK_SIZE, cpu.READ | cpu.WRITE)
    c.write(CODE, code)
    c.write(DATA, start["data"])
    for i, value in enumerate(start["x"]):
        c.set_x(i, value)
    for i, value in enumerate(start["v"]):
        c.uc.reg_write(a64.UC_ARM64_REG_V0 + i, value)
    c.uc.reg_write(a64.UC_ARM64_REG_NZCV, start["nzcv"] << 28)
    c.pc, c.sp = CODE, STACK + STACK_SIZE
    stop = c.run(limit=100_000)
    return {
        "stop": (stop.reason, stop.svc, stop.pc - CODE),
        "x": [c.x(i) for i in range(31)],
        "sp": c.sp,
        "nzcv": c.uc.reg_read(a64.UC_ARM64_REG_NZCV) >> 28 & 0xF,
        "v": [c.uc.reg_read(a64.UC_ARM64_REG_V0 + i) for i in range(32)],
        "data": c.read(DATA, DATA_SIZE),
        "stack": c.read(STACK, STACK_SIZE),
    }


def on_dynarmic(code: bytes, start: dict) -> dict:
    with dynarmic.Machine() as m:
        m.map(CODE, 0x1000, cpu.READ | cpu.EXECUTE)
        m.map(DATA, DATA_SIZE, cpu.READ | cpu.WRITE)
        m.map(STACK, STACK_SIZE, cpu.READ | cpu.WRITE)
        m.write(CODE, code)
        m.write(DATA, start["data"])
        c = m.core()
        for i, value in enumerate(start["x"]):
            c.set_x(i, value)
        for i, value in enumerate(start["v"]):
            c.set_v(i, value)
        c.pstate = start["nzcv"] << 28
        c.pc, c.sp = CODE, STACK + STACK_SIZE
        stop = c.run(limit=100_000)
        return {
            "stop": (stop.reason, stop.svc, stop.pc - CODE),
            "x": [c.x(i) for i in range(31)],
            "sp": c.sp,
            "nzcv": c.pstate >> 28 & 0xF,
            "v": [c.v(i) for i in range(32)],
            "data": m.read(DATA, DATA_SIZE),
            "stack": m.read(STACK, STACK_SIZE),
        }


def differences(a: dict, b: dict) -> list[str]:
    out = []
    for key in a:
        if a[key] == b[key]:
            continue
        if key in ("x", "v"):
            out += [
                f"{key}{i}: {p:#x} != {q:#x}" for i, (p, q) in enumerate(zip(a[key], b[key], strict=True)) if p != q
            ]
        elif key in ("data", "stack"):
            at = next(i for i, (p, q) in enumerate(zip(a[key], b[key], strict=True)) if p != q)
            out.append(f"{key} from +{at:#x}: {a[key][at : at + 16].hex()} != {b[key][at : at + 16].hex()}")
        else:
            out.append(f"{key}: {a[key]} != {b[key]}")
    return out


def compare(tmp_path, source: str, *, seeds=range(4)) -> None:
    code = assemble(source + "\nsvc #0", tmp_path)
    for seed in seeds:
        start = start_state(seed)
        want, got = on_unicorn(code, start), on_dynarmic(code, start)
        assert want["stop"][0] == "svc", f"the program did not finish on Unicorn: {want['stop']}"
        assert not differences(want, got), f"seed {seed}: Unicorn != dynarmic: " + "; ".join(differences(want, got))


ATOMIC_OPS = ["ldadd", "ldclr", "ldeor", "ldset", "ldsmax", "ldsmin", "ldumax", "ldumin", "swp"]

# Unicorn 2.1.4 is not a reference for LDSMAX and LDSMIN at 8, 16 and 32 bits: it compares their
# operand as an unsigned number (see test_unicorn_signed_atomic_min_and_max below). Those forms are
# checked against the architecture's definition instead.
SIGNED_OPS = ("ldsmax", "ldsmin")


@pytest.mark.parametrize("op", ATOMIC_OPS)
def test_atomic_memory_operations(tmp_path, op):
    narrow = f"""
        {op}a w4, w5, [x17]
        {op}l w8, w9, [x17]
        {op}b w10, w11, [x16]
        {op}alb w12, w13, [x17]
        {op}h w14, w15, [x16]
        {op}alh w18, w19, [x17]
        {op} w21, wzr, [x17]       // the old value is discarded
    """
    source = f"""
        {op} x2, x3, [x16]
        {op}al x6, x7, [x16, #0]
        {"" if op in SIGNED_OPS else narrow}
        {op} x20, x20, [x16]       // the old value replaces the operand
        {op} xzr, x22, [x16]       // the operand is zero
        mov x23, x16
        {op} x24, x23, [x23]       // the old value replaces the address
        mov x25, sp
        sub sp, sp, #64
        {op}al x26, x27, [sp]      // the address is the stack pointer
        mov sp, x25
    """
    compare(tmp_path, source)


def _signed(value: int, bits: int) -> int:
    value &= (1 << bits) - 1
    return value - (1 << bits) if value >> (bits - 1) else value


MIN_MAX = {
    "ldsmax": lambda old, operand, bits: max(_signed(old, bits), _signed(operand, bits)),
    "ldsmin": lambda old, operand, bits: min(_signed(old, bits), _signed(operand, bits)),
    "ldumax": lambda old, operand, bits: max(old, operand),
    "ldumin": lambda old, operand, bits: min(old, operand),
}
EDGE_VALUES = [0, 1, 0x7F, 0x80, 0xFF, 0x100, 0x7FFF, 0x8000, 0xFFFF, 0x7FFF_FFFF, 0x8000_0000, 0xFFFF_FFFF]
EDGE_VALUES += [0x7FFF_FFFF_FFFF_FFFF, 0x8000_0000_0000_0000, MASK64, 0x1234_5678_9ABC_DEF0]


@pytest.mark.parametrize("op", MIN_MAX)
@pytest.mark.parametrize(("suffix", "reg", "bits"), [("b", "w", 8), ("h", "w", 16), ("", "w", 32), ("", "x", 64)])
def test_atomic_min_and_max_follow_the_architecture(tmp_path, op, suffix, reg, bits):
    """The operand is the low bits of the register, at the size of the access, signed for LDSMAX and LDSMIN."""
    mask = (1 << bits) - 1
    with dynarmic.Machine() as m:
        m.map(CODE, 0x1000, cpu.READ | cpu.EXECUTE)
        m.map(DATA, DATA_SIZE, cpu.READ | cpu.WRITE)
        m.write(CODE, assemble(f"{op}{suffix} {reg}2, {reg}3, [x1]\nsvc #0", tmp_path))
        c = m.core()
        for old in EDGE_VALUES:
            for operand in EDGE_VALUES:
                m.write(DATA, old.to_bytes(8, "little"))
                c.set_x(1, DATA)
                c.set_x(2, operand)
                c.pc = CODE
                assert c.run(limit=100).reason == "svc"
                result = MIN_MAX[op](old & mask, operand & mask, bits) & mask
                want = (old & mask, old & ~mask | result)
                assert (c.x(3), m.read_u64(DATA)) == want, f"old {old:#x}, operand {operand:#x}"


@pytest.mark.xfail(strict=True, reason="Unicorn 2.1.4 compares the operand of LDSMAX and LDSMIN as an unsigned number")
def test_unicorn_signed_atomic_min_and_max(tmp_path):
    """The larger of 0 and -1 is 0. This test passes once a Unicorn release computes that; then the
    signed forms can join test_atomic_memory_operations, and SIGNED_OPS can go."""
    c = cpu.Cpu()
    c.map(CODE, 0x1000, cpu.READ | cpu.EXECUTE)
    c.map(DATA, DATA_SIZE, cpu.READ | cpu.WRITE)
    c.write(CODE, assemble("ldsmax w2, w3, [x1]\nsvc #0", tmp_path))
    c.set_x(1, DATA)
    c.set_x(2, 0xFFFF_FFFF)
    c.pc = CODE
    assert c.run(limit=100).reason == "svc"
    assert c.read_u32(DATA) == 0


def test_compare_and_swap(tmp_path):
    source = """
        cas x2, x3, [x16]          // the comparison fails: memory holds other data
        ldr x4, [x16]
        casal x4, x5, [x16]        // the comparison succeeds
        casa w6, w7, [x17]
        ldr w8, [x17]
        casl w8, w9, [x17]
        casb w10, w11, [x16, #0]
        ldrb w12, [x16, #8]
        add x13, x16, #8
        casalb w12, w14, [x13]
        cash w15, w18, [x17]
        ldrh w19, [x17, #8]
        add x20, x17, #8
        casalh w19, w21, [x20]
        ldr x22, [x16]
        cas x22, x22, [x16]        // the new value is the compared one
        mov x23, x16
        ldr x24, [x16]
        casal x24, x23, [x23]      // the new value is the address
    """
    compare(tmp_path, source)


def test_compare_and_swap_pair(tmp_path):
    source = """
        casp x2, x3, x4, x5, [x16]       // fails
        ldp x6, x7, [x16]
        caspal x6, x7, x8, x9, [x16]     // succeeds
        ldp x10, x11, [x16]
        eor x11, x11, #1
        caspa x10, x11, x12, x13, [x16]  // fails: only the second doubleword differs
        casp w14, w15, w18, w19, [x17]   // fails
        ldp w20, w21, [x17]
        caspl w20, w21, w22, w23, [x17]  // succeeds
        ldp w24, w25, [x17]
        eor w24, w24, #1
        caspal w24, w25, w26, w27, [x17] // fails: only the first word differs
    """
    compare(tmp_path, source)


def test_load_acquire_rcpc(tmp_path):
    compare(
        tmp_path,
        "ldapr x2, [x16]\nldapr w3, [x17]\nldaprb w4, [x16]\nldaprh w5, [x17]\nsub sp, sp, #32\nldapr x7, [sp, #0]",
    )


def test_pointer_authentication_leaves_pointers_unchanged(tmp_path):
    source = """
        pacia x2, x3
        pacib x4, sp
        pacda x5, x6
        pacdb x7, x8
        paciza x9
        pacizb x10
        pacdza x11
        pacdzb x12
        autia x2, x3
        autib x4, sp
        autda x5, x6
        autdb x7, x8
        autiza x9
        autizb x10
        autdza x11
        autdzb x12
        mov x13, x16
        xpaci x13                  // a pointer without a code stays as it is
        mov x14, x17
        xpacd x14
        mov x30, x16
        xpaclri
        paciasp
        autiasp
        pacibsp
        autibsp
        pacia1716
        autib1716
    """
    compare(tmp_path, source)


def test_authenticated_branches_and_returns(tmp_path):
    source = """
        adr x2, 1f
        braa x2, x3
        udf #1
    1:  adr x4, 2f
        brabz x4
        udf #2
    2:  adr x5, 3f
        blraa x5, sp
        mov x6, x30                // where blraa returned to
        adr x7, 4f
        blrabz x7
        b 5f
    3:  mov x8, #3
        retaa
    4:  mov x9, #4
        retab
    5:  bl 6f
        b 7f
    6:  pacibsp
        stp x29, x30, [sp, #-16]!
        ldp x29, x30, [sp], #16
        retab
    7:  nop
    """
    compare(tmp_path, source)


def test_authenticated_loads(tmp_path):
    source = """
        ldraa x2, [x16]
        ldrab x3, [x16, #8]
        ldraa x4, [x17, #-16]
        mov x5, x16
        ldraa x6, [x5, #2040]!     // with writeback
        mov x7, x17
        ldrab x8, [x7, #-2048]!
        mov x9, sp
        sub sp, sp, #64
        ldraa x10, [sp, #8]
        mov sp, x9
    """
    compare(tmp_path, source)


def test_exclusive_and_ordered_accesses(tmp_path):
    source = """
        ldaxr x2, [x16]
        add x2, x2, #1
        stlxr w3, x2, [x16]
        ldxrb w4, [x17]
        stxrb w5, w4, [x17]
        ldaxp x6, x7, [x16]
        stlxp w8, x7, x6, [x16]
        stxr w9, x2, [x17]         // no reservation: fails
        ldar x10, [x16]
        stlr x11, [x17]
        ldarb w12, [x16]
        stlrh w13, [x17, #0]
        clrex
    """
    compare(tmp_path, source)


def test_dot_product_and_half_precision_negate(tmp_path):
    source = """
        udot v0.4s, v1.16b, v2.16b
        sdot v3.2s, v4.8b, v5.8b
        udot v6.4s, v7.16b, v8.4b[2]
        sdot v9.4s, v10.16b, v11.4b[1]
        fneg h12, h13
        fneg v14.8h, v15.8h
        fneg v16.4h, v17.4h
    """
    compare(tmp_path, source)


def test_crc32(tmp_path):
    source = """
        crc32b w2, w3, w4
        crc32h w5, w6, w7
        crc32w w8, w9, w10
        crc32x w11, w12, x13
        crc32cb w14, w15, w18
        crc32ch w19, w20, w21
        crc32cw w22, w23, w24
        crc32cx w25, w26, x27
    """
    compare(tmp_path, source)


def test_aes_sha_and_polynomial_multiply(tmp_path):
    source = """
        aese v0.16b, v1.16b
        aesmc v2.16b, v0.16b
        aesd v3.16b, v4.16b
        aesimc v5.16b, v3.16b
        sha1h s6, s7
        sha1c q8, s9, v10.4s
        sha1p q11, s12, v13.4s
        sha1m q14, s15, v16.4s
        sha1su0 v17.4s, v18.4s, v19.4s
        sha1su1 v20.4s, v21.4s
        sha256h q22, q23, v24.4s
        sha256h2 q25, q26, v27.4s
        sha256su0 v28.4s, v29.4s
        sha256su1 v30.4s, v31.4s, v0.4s
        pmull v1.1q, v2.1d, v3.1d
        pmull2 v4.1q, v5.2d, v6.2d
        pmull v7.8h, v8.8b, v9.8b
    """
    compare(tmp_path, source)


def test_common_vector_and_flag_instructions(tmp_path):
    source = """
        cmp x2, x3
        mrs x4, nzcv
        adds x5, x6, x7
        mrs x8, nzcv
        msr nzcv, x9
        movi v0.16b, #0x5a
        mov v1.16b, v2.16b
        dup v3.4s, w10
        fmov s4, #1.5
        fmov s5, #-2.0
        fmul s6, s4, s5
        fmul v7.4s, v7.4s, v7.4s
        ldr q8, [x16]
        str q9, [x17]
        ccmp x11, x12, #5, lo
        csel x13, x14, x15, ge
    """
    compare(tmp_path, source)
