"""Tests for the reference interpreter, with synthetic programs (tests/sassasm.py)."""

import math
import struct

import pytest

from siicl import interp
from siicl.nvshader import ComputeInfo
from tests.sassasm import bra_self, exit_, mov_imm, program, rel, s2r, store

PT_NOT = {87: 7, 90: 1}  # a trailing !PT operand


def fbits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def run(lines, *, local=(32, 1, 1), shared=0, outputs=1, cbufs=None, surfaces=None, step_limit=100_000, **kw):
    """Run a program on one workgroup; output k is the 1D surface with handle (0, k)."""
    threads = local[0] * local[1] * local[2]
    outs = {(0, k): interp.Surface(threads, channels=1) for k in range(1, outputs + 1)}
    launch = interp.Launch((1, 1, 1), cbufs or {}, outs | (surfaces or {}), mufu_results=kw.pop("mufu_results", None))
    prog = interp.Program(program(*lines), ComputeInfo(local, shared, 32, b""))
    interp.run(prog, launch, step_limit=step_limit, **kw)
    return [outs[(0, k)].data for k in range(1, outputs + 1)]


# -- arithmetic helpers ------------------------------------------------------------


def test_rounding_modes():
    x = 1 + 2.0**-30
    assert interp.to_f32(x) == 0x3F800000
    assert interp.to_f32(x, interp.RZ) == 0x3F800000
    assert interp.to_f32(x, interp.RP) == 0x3F800001
    assert interp.to_f32(-x, interp.RM) == 0xBF800001
    assert interp.to_f32(-x, interp.RP) == 0xBF800000
    assert interp.to_f32(1e39) == 0x7F800000
    assert interp.to_f32(1e39, interp.RZ) == 0x7F7FFFFF
    assert interp.to_f32(-1e39, interp.RP) == 0xFF7FFFFF
    assert interp.to_f32(1e-45) == 1  # the smallest subnormal
    assert interp.to_f32(math.nan) == interp.NAN


def test_fma_rounds_once():
    # a*b + c lies just below the midpoint between c and the next float. Rounding the exact
    # value to 64 bits first would land on the midpoint and then round up.
    a, b, c = fbits(2.0**26 + 8), fbits(2.0**26 - 8), 0x65800001
    assert interp.fma32(a, b, c) == c
    assert interp.add32(interp.mul32(a, b), c) == 0x65800002  # two roundings round up


def test_flush_to_zero_and_saturate():
    tiny = 0x00000005  # subnormal
    assert interp.add32(tiny, 0) == tiny
    assert interp.add32(tiny, 0, flush=True) == 0
    assert interp.mul32(fbits(-1.0), tiny | 0x80000000, flush=True) == 0x80000000 ^ 0x80000000
    assert interp.add32(fbits(0.75), fbits(0.5), sat=True) == fbits(1.0)
    assert interp.add32(fbits(-0.75), 0, sat=True) == 0
    assert interp.add32(interp.NAN, 0, sat=True) == 0


def test_min_and_max():
    pz, nz = fbits(0.0), fbits(-0.0)
    assert interp.fmin32(pz, nz) == nz and interp.fmax32(pz, nz) == pz
    assert interp.fmin32(interp.NAN, fbits(2.0)) == fbits(2.0)
    assert interp.fmax32(fbits(2.0), 0x7FC00000) == fbits(2.0)
    assert interp.fmin32(interp.NAN, 0x7FC00000) == interp.NAN


def test_float_to_integer():
    assert interp.f2i32(interp.NAN, "", "", False) == 0
    assert interp.f2i32(fbits(3e9), "", "", False) == 0x7FFFFFFF
    assert interp.f2i32(fbits(3e9), "", "U32", False) == 3_000_000_000
    assert interp.f2i32(fbits(-1.5), "", "U32", False) == 0
    assert interp.f2i32(fbits(2.5), "", "", False) == 2  # ties to even
    assert interp.f2i32(fbits(-2.5), "FLOOR", "", False) == (-3) & interp.M32
    assert interp.f2i32(fbits(70000.0), "TRUNC", "U16", False) == 0xFFFF


def test_mufu_uses_turns_for_sine():
    assert interp.mufu("SIN", fbits(0.25)) == fbits(1.0)
    assert interp.mufu("COS", fbits(0.5)) == fbits(-1.0)
    assert interp.mufu("RCP", fbits(-0.0)) == 0xFF800000
    assert interp.mufu("RSQ", fbits(4.0)) == fbits(0.5)
    assert interp.mufu("LG2", fbits(-1.0)) == interp.NAN
    with pytest.raises(interp.Unsupported):
        interp.mufu("TANH", 0)


# -- programs ---------------------------------------------------------------------


def test_integer_instructions():
    lines = [
        s2r(0, "SR_TID.X"),
        ("IMAD R1, R0, 0x3, R0", 0x824, {16: 1, 24: 0, 32: 3, 64: 0, 73: 1}),
        ("LOP3.LUT R2, R1, 0xff, RZ, 0xc0, !PT", 0x812, {16: 2, 24: 1, 32: 0xFF, 64: 255, 72: 0xC0, 81: 7, **PT_NOT}),
        ("SHF.L.U32 R3, R2, 0x4, RZ", 0x819, {16: 3, 24: 2, 32: 4, 64: 255, 73: 3}),
        ("IADD3 R5, -R0, RZ, RZ", 0x210, {16: 5, 24: 0, 32: 255, 64: 255, 72: 1, 81: 7, 84: 7}),
        ("SHF.R.S32.HI R6, RZ, 0x1, R5", 0x819, {16: 6, 24: 255, 32: 1, 64: 5, 73: 2, 76: 1, 80: 1}),
        ("LEA R7, R0, R3, 0x2", 0x211, {16: 7, 24: 0, 32: 3, 75: 2, 81: 7}),
        store(0, 1, 1),
        store(0, 3, 2),
        store(0, 6, 3),
        store(0, 7, 4),
        exit_(),
        bra_self(12),
    ]
    r1, r3, r6, r7 = run(lines, outputs=4)
    for tid in range(32):
        assert r1[tid] == 4 * tid
        assert r3[tid] == ((4 * tid) & 0xFF) << 4
        assert r6[tid] == (-tid >> 1) & interp.M32
        assert r7[tid] == (tid << 2) + r3[tid]


def test_carry_chain():
    lines = [
        s2r(0, "SR_TID.X"),
        mov_imm(1, 0xFFFFFFF0),
        mov_imm(8, 0xA),
        ("IADD3 R2, P0, R0, R1, RZ", 0x210, {16: 2, 24: 0, 32: 1, 64: 255, 81: 0, 84: 7}),
        (
            "IADD3.X R3, RZ, RZ, RZ, P0, !PT",
            0x210,
            {16: 3, 24: 255, 32: 255, 64: 255, 74: 1, 81: 7, 84: 7, 87: 0, 77: 7, 80: 1},
        ),
        ("IADD3 R4, P1, R0, -R8, RZ", 0x210, {16: 4, 24: 0, 32: 8, 63: 1, 64: 255, 81: 1, 84: 7}),
        (
            "IADD3.X R5, RZ, RZ, RZ, P1, !PT",
            0x210,
            {16: 5, 24: 255, 32: 255, 64: 255, 74: 1, 81: 7, 84: 7, 87: 1, 77: 7, 80: 1},
        ),
        store(0, 2, 1),
        store(0, 3, 2),
        store(0, 4, 3),
        store(0, 5, 4),
        exit_(),
        bra_self(12),
    ]
    low, carry, diff, no_borrow = run(lines, outputs=4)
    for tid in range(32):
        assert low[tid] == (tid + 0xFFFFFFF0) & interp.M32
        assert carry[tid] == (tid >= 16)
        assert diff[tid] == (tid - 10) & interp.M32
        assert no_borrow[tid] == (tid >= 10)


def test_divergent_branches_reconverge():
    lines = [
        s2r(0, "SR_TID.X"),
        ("BSSY B0, 0x70", 0x945, {16: 0, 32: rel(1, 7), 87: 7}),
        ("LOP3.LUT P0, RZ, R0, 0x1, RZ, 0xc0, !PT", 0x812, {16: 255, 24: 0, 32: 1, 64: 255, 72: 0xC0, 81: 0, **PT_NOT}),
        ("@P0 BRA 0x60", 0x947, {32: rel(3, 6), 87: 7}, 0),
        mov_imm(1, 2),
        ("BRA 0x70", 0x947, {32: rel(5, 7), 87: 7}),
        mov_imm(1, 1),
        ("BSYNC B0", 0x941, {16: 0, 87: 7}),
        ("VOTE.ANY R2, PT, PT", 0x806, {16: 2, 72: 1, 81: 7, 87: 7}),
        store(0, 1, 1),
        store(0, 2, 2),
        exit_(),
        bra_self(12),
    ]
    value, ballot = run(lines, outputs=2)
    assert value == [1 if tid & 1 else 2 for tid in range(32)]
    assert ballot == [0xFFFFFFFF] * 32  # every lane is back together after BSYNC


def test_loop():
    lines = [
        s2r(0, "SR_TID.X"),
        mov_imm(1, 0),
        mov_imm(2, 0),
        ("ISETP.GE.U32.AND P0, PT, R2, R0, PT", 0x20C, {24: 2, 32: 0, 76: 6, 81: 0, 84: 7, 87: 7}),
        ("@P0 BRA 0x80", 0x947, {32: rel(4, 8), 87: 7}, 0),
        ("IADD3 R1, R1, R2, RZ", 0x210, {16: 1, 24: 1, 32: 2, 64: 255, 81: 7, 84: 7}),
        ("IADD3 R2, R2, 0x1, RZ", 0x810, {16: 2, 24: 2, 32: 1, 64: 255, 81: 7, 84: 7}),
        ("BRA 0x30", 0x947, {32: rel(7, 3), 87: 7}),
        store(0, 1),
        exit_(),
        bra_self(10),
    ]
    (out,) = run(lines)
    assert out == [tid * (tid - 1) // 2 for tid in range(32)]


def test_branch_needs_guard_and_predicate():
    lines = [
        s2r(0, "SR_TID.X"),
        ("LOP3.LUT P1, RZ, R0, 0x1, RZ, 0xc0, !PT", 0x812, {16: 255, 24: 0, 32: 1, 64: 255, 72: 0xC0, 81: 1, **PT_NOT}),
        ("LOP3.LUT P2, RZ, R0, 0x2, RZ, 0xc0, !PT", 0x812, {16: 255, 24: 0, 32: 2, 64: 255, 72: 0xC0, 81: 2, **PT_NOT}),
        mov_imm(1, 0),
        ("@P1 BRA P2, 0x60", 0x947, {32: rel(4, 6), 87: 2}, 1),
        mov_imm(1, 1),
        store(0, 1),
        exit_(),
        bra_self(8),
    ]
    (out,) = run(lines)
    assert out == [0 if tid & 3 == 3 else 1 for tid in range(32)]


def test_shared_memory_and_barrier():
    lines = [
        s2r(0, "SR_TID.X"),
        ("STS [R0.X4], R0", 0x388, {24: 0, 32: 0, 73: 4, 78: 1}),
        ("BAR.SYNC 0x0", 0xB1D, {}),
        ("IADD3 R1, -R0, 0x3f, RZ", 0x810, {16: 1, 24: 0, 32: 0x3F, 64: 255, 72: 1, 81: 7, 84: 7}),
        ("LDS R2, [R1.X4]", 0x984, {16: 2, 24: 1, 73: 4, 78: 1}),
        store(0, 2),
        exit_(),
        bra_self(7),
    ]
    (out,) = run(lines, local=(64, 1, 1), shared=0x100)
    assert out == [63 - tid for tid in range(64)]  # the second warp's stores are visible to the first


def test_barrier_in_divergent_code_deadlocks():
    lines = [
        s2r(0, "SR_TID.X"),
        ("BSSY B0, 0x50", 0x945, {16: 0, 32: rel(1, 5), 87: 7}),
        ("ISETP.GE.U32.AND P0, PT, R0, 0x10, PT", 0x80C, {24: 0, 32: 0x10, 76: 6, 81: 0, 84: 7, 87: 7}),
        ("@P0 BRA 0x50", 0x947, {32: rel(3, 5), 87: 7}, 0),
        ("BAR.SYNC 0x0", 0xB1D, {}),
        ("BSYNC B0", 0x941, {16: 0, 87: 7}),
        exit_(),
        bra_self(7),
    ]
    with pytest.raises(interp.ExecutionError, match="deadlock"):
        run(lines)


SURFACE_ATOMICS = [
    s2r(0, "SR_TID.X"),
    ("LOP3.LUT R1, R0, 0x3, RZ, 0xc0, !PT", 0x812, {16: 1, 24: 0, 32: 3, 64: 255, 72: 0xC0, 81: 7, **PT_NOT}),
    mov_imm(2, 1),
    ("SUATOM.D.1D.ADD.STRONG.GPU.IGN R3, [R1], R2, 0x0, 0x2", 0x593, {16: 3, 24: 1, 32: 2, 40: 2, 76: 0xE, 87: 0}),
    ("SURED.D.1D.OR.STRONG.GPU.IGN [R1], R0, 0x0, 0x3", 0x59F, {24: 1, 32: 0, 40: 3, 76: 0xE, 87: 6}),
    store(0, 3),
    exit_(),
    bra_self(7),
]


def test_reverse_order_changes_only_order_dependent_results():
    old, counts, bits = run(SURFACE_ATOMICS, outputs=3, reverse=True)
    assert old == [7 - (tid >> 2) for tid in range(32)]  # the last lane now goes first
    assert counts[:4] == [8] * 4 and bits[:4] == [c | 0x1C for c in range(4)]


def test_recorded_mufu_results_replace_computed_ones():
    lines = [
        s2r(0, "SR_TID.X"),
        ("I2FP.F32.U32 R1, R0", 0x245, {16: 1, 32: 0, 75: 2, 84: 2}),
        ("MUFU.EX2 R2, R1", 0x308, {16: 2, 32: 1, 74: 2}),
        store(0, 2),
        exit_(),
        bra_self(5),
    ]
    (out,) = run(lines, mufu_results={(0x20, fbits(3.0)): 0x12345678})
    assert out[3] == 0x12345678
    assert out[4] == fbits(16.0)  # no recorded result: computed


def test_surface_atomics():
    old, counts, bits = run(SURFACE_ATOMICS, outputs=3)
    assert old == [tid >> 2 for tid in range(32)]  # lanes run in order
    assert counts[:4] == [8] * 4 and bits[:4] == [c | 0x1C for c in range(4)]


def test_float_instructions_and_constant_buffers():
    lines = [
        s2r(0, "SR_TID.X"),
        ("I2FP.F32.U32 R1, R0", 0x245, {16: 1, 32: 0, 75: 2, 84: 2}),
        ("FFMA R2, R1, c[0x3][0x0], -R1", 0xA23, {16: 2, 24: 1, 38: 0, 54: 3, 64: 1, 75: 1}),
        ("F2I.TRUNC.NTZ R3, R2", 0x305, {16: 3, 32: 2, 72: 1, 75: 2, 77: 1, 78: 3, 84: 2}),
        ("FSETP.GT.AND P0, PT, R2, 10, PT", 0x80B, {24: 2, 32: fbits(10.0), 76: 4, 81: 0, 84: 7, 87: 7}),
        ("FSEL R4, R2, RZ, P0", 0x208, {16: 4, 24: 2, 32: 255, 87: 0}),
        ("MUFU.RCP R5, R1", 0x308, {16: 5, 32: 1, 74: 4}),
        store(0, 2, 1),
        store(0, 3, 2),
        store(0, 4, 3),
        store(0, 5, 4),
        exit_(),
        bra_self(12),
    ]
    r2, r3, r4, r5 = run(lines, outputs=4, cbufs={3: struct.pack("<f", 2.5)})
    for tid in range(32):
        assert r2[tid] == fbits(1.5 * tid)
        assert r3[tid] == int(1.5 * tid)
        assert r4[tid] == (fbits(1.5 * tid) if 1.5 * tid > 10 else 0)
        assert r5[tid] == (0x7F800000 if tid == 0 else fbits(1 / tid))


def test_uniform_instructions_run_once_per_warp():
    # Every thread of the warp reaches the UIADD3, but uniform registers belong to the warp,
    # so the add happens once: 3 + 1, not 3 + 32 * 1.
    lines = [
        s2r(0, "SR_TID.X"),
        ("UMOV UR4, 0x3", 0x882, {16: 4, 32: 3}),
        ("UMOV UR5, 0x1", 0x882, {16: 5, 32: 1}),
        ("UIADD3 UR4, UR4, UR5, URZ", 0x290, {16: 4, 24: 4, 32: 5, 64: 63, 81: 7, 84: 7}),
        ("MOV R1, UR4", 0xC02, {16: 1, 32: 4}),
        store(0, 1),
        exit_(),
        bra_self(7),
    ]
    (out,) = run(lines)
    assert out == [4] * 32


def test_instruction_without_a_rule():
    lines = [("HADD2 R1, R2, R3", 0x230, {16: 1, 24: 2, 32: 3}), exit_(), bra_self(2)]
    prog = interp.Program(program(*lines), ComputeInfo((32, 1, 1), 0, 32, b""))
    assert [(ins.name, why) for ins, why in prog.unsupported()] == [("HADD2", "no rule for HADD2")]
    with pytest.raises(interp.Unsupported, match="0x0000 HADD2"):
        run(lines)


def test_step_limit():
    with pytest.raises(interp.ExecutionError, match="step limit"):
        run([bra_self(0)], step_limit=1000)
