"""Tests for the GPU check: input generation without a GPU, and whole checks when one is present."""

import struct

import pytest

from siicl import cli, gpucheck, interp, translate, vk
from siicl.nvshader import ComputeInfo
from tests.builders import build_vzip
from tests.sassasm import bra_self, exit_, program, s2r, store
from tests.test_nvshader import build_shader, compute_control
from tests.test_translate import INFO, INT_PROGRAM, PT_NOT

TYPED = [
    s2r(0, "SR_TID.X"),
    ("LDC R1, c[0x3][0x10]", 0xB82, {16: 1, 24: 255, 38: 0x10, 54: 3, 73: 4}),
    ("FADD R2, R1, R1", 0x221, {16: 2, 24: 1, 32: 1}),
    ("ISETP.GE.U32.AND P0, PT, R0, c[0x3][0x20], PT", 0xA0C, {24: 0, 38: 0x20, 54: 3, 76: 6, 81: 0, 84: 7, 87: 7}),
    ("IADD3 R3, R0, c[0x3][0x24], RZ", 0xA10, {16: 3, 24: 0, 38: 0x24, 54: 3, 64: 255, 81: 7, 84: 7}),
    ("LDC R4, c[0x3][R3+0x8]", 0xB82, {16: 4, 24: 3, 38: 0x8, 54: 3, 73: 4}),
    ("LOP3.LUT R5, R4, 0x1, RZ, 0xc0, !PT", 0x812, {16: 5, 24: 4, 32: 1, 64: 255, 72: 0xC0, 81: 7, **PT_NOT}),
    store(0, 2),
    exit_(),
    bra_self(9),
]


def test_word_types_follow_how_the_program_uses_them():
    prog = interp.Program(program(*TYPED), INFO)
    static, indexed = gpucheck.word_types(prog.instructions)
    assert static == {3: {0x10: "f", 0x20: "b", 0x24: "i"}}
    assert indexed == {3: {0x8: "i"}}


def test_inputs_are_reproducible_and_typed():
    prog = interp.Program(program(*TYPED), INFO)
    a, b = gpucheck.make_launch(prog, 5), gpucheck.make_launch(prog, 5)
    assert a.cbufs == b.cbufs and a.surfaces == b.surfaces
    for seed in range(8):
        launch = gpucheck.make_launch(prog, seed)
        words = struct.unpack(f"<{len(launch.cbufs[3]) // 4}I", launch.cbufs[3])
        assert words[0x24 // 4] < 64  # an integer
        assert 0x3C000000 <= words[0x10 // 4] & 0x7FFFFFFF <= 0x44800000 or words[0x10 // 4] & 0x7FFFFFFF < 0x40000001
        assert len(launch.cbufs[3]) % 16 == 0 and launch.cbufs[1] == b""
        assert launch.surfaces[(0, 1)].channels == 4
        assert launch.grid == (8, 1, 1)  # 256 texels of a 1D surface in groups of 32 threads


def test_abs_error():
    assert gpucheck.abs_error(0x3F800000, 0x3F800000) == 0.0
    assert gpucheck.abs_error(0x3F800000, 0x3F000000) == 0.5
    assert gpucheck.abs_error(0x7F800000, 0x3F800000) == float("inf")


def test_ulps():
    one = 0x3F800000
    assert gpucheck.ulps(one, one + 3) == 3
    assert gpucheck.ulps(0x00000001, 0x80000001) == 2  # across zero
    assert gpucheck.ulps(0x7FC00000, 0x7FFFFFFF) == 0  # two NaNs


@pytest.fixture(scope="module")
def gpu():
    try:
        g = vk.Gpu()
    except vk.VulkanError as err:
        pytest.skip(f"no Vulkan GPU: {err}")
    yield g
    g.close()


def test_check_on_a_gpu(gpu):
    prog = interp.Program(program(*INT_PROGRAM), INFO)
    tr = translate.translate(prog.instructions, INFO, mufu_log=True)
    result = gpucheck.check(prog, tr, gpu, seed=0)
    assert result.status == "match", result
    assert result.words > 0 and result.mismatch_count == 0
    assert len(result.executed) == len(prog.instructions) - 1  # all but the final branch to itself


def test_check_replays_mufu_results(gpu):
    lines = [
        s2r(0, "SR_TID.X"),
        ("I2FP.F32.U32 R1, R0", 0x245, {16: 1, 32: 0, 75: 2, 84: 2}),
        ("MUFU.LG2 R2, R1", 0x308, {16: 2, 32: 1, 74: 3}),
        ("MUFU.SIN R3, R2", 0x308, {16: 3, 32: 2, 74: 1}),
        ("MUFU.RSQ R4, R1", 0x308, {16: 4, 32: 1, 74: 5}),
        ("FFMA.FTZ R5, R3, R4, R2", 0x223, {16: 5, 24: 3, 32: 4, 64: 2, 80: 1}),
        store(0, 5),
        exit_(),
        bra_self(8),
    ]
    prog = interp.Program(program(*lines), INFO)
    result = gpucheck.check(prog, translate.translate(prog.instructions, INFO, mufu_log=True), gpu, seed=0)
    assert result.status == "match", result.mismatches
    assert result.mufu_logged == 3 * 256
    assert result.mufu_ulps["RSQ"] <= 2  # the right functions, approximately
    assert result.mufu_abs["LG2"] < 1e-5 and result.mufu_abs["SIN"] < 1e-5


def test_check_leaves_out_order_dependent_words(gpu):
    lines = [
        s2r(0, "SR_TID.X"),
        ("LOP3.LUT R1, R0, 0x1, RZ, 0xc0, !PT", 0x812, {16: 1, 24: 0, 32: 1, 64: 255, 72: 0xC0, 81: 7, **PT_NOT}),
        store(1, 0),  # every even thread stores its id to word 0, every odd one to word 4
        exit_(),
        bra_self(4),
    ]
    info = ComputeInfo((32, 1, 1), 0, 32, b"")
    prog = interp.Program(program(*lines), info)
    result = gpucheck.check(prog, translate.translate(prog.instructions, info, mufu_log=True), gpu, seed=0)
    assert result.status == "match" and result.order_dependent == 2


def test_cli_checks_compute_programs(gpu, tmp_path, capsys):
    shader = build_shader(program(*INT_PROGRAM), end=False, control=compute_control(local=(32, 1, 1), registers=32))
    path = tmp_path / "a.zip"
    path.write_bytes(build_vzip([("c1", shader, False)]))
    assert cli.main(["vzip", "check-gpu", str(path), "--seeds", "2"]) == 0
    out = capsys.readouterr().out
    assert "c1" in out and "2 match" in out
    assert "Checked 1 programs: 1 match on every seed, 0 have mismatches or errors" in out
