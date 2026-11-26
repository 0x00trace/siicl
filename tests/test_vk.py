"""Tests that run translated programs on a Vulkan GPU and compare them with the interpreter.

They skip when no Vulkan GPU is present. The programs are the synthetic ones
of tests/test_translate.py, with inputs chosen to hit edge cases.
"""

import copy
import struct

import pytest

from siicl import interp, translate, vk
from tests.sassasm import bra_self, exit_, program, s2r, store
from tests.test_translate import INFO, PROGRAMS, fbits, launch_for


@pytest.fixture(scope="module")
def gpu():
    try:
        g = vk.Gpu()
    except vk.VulkanError as err:
        pytest.skip(f"no Vulkan GPU: {err}")
    yield g
    g.close()


# FMUL and FFMA without .FTZ: some GPUs flush 32-bit subnormal numbers in shaders and Vulkan can't ask them
# not to (shaderDenormPreserveFloat32 is false on the RTX 5080). There the result may be the flushed one.
FLUSHING_GPUS_MAY_FLUSH = {
    12: lambda a, b: interp.mul32(a, b, flush=True),  # FMUL R15, R2, R3
    13: lambda a, b: interp.fma32(a, b, b ^ 0x80000000, flush=True),  # FFMA R16, R2, R3, -R3
}


@pytest.mark.parametrize("name", ["float", "int", "shared"])
def test_gpu_matches_the_interpreter(name, gpu):
    lines, cbufs, info = PROGRAMS[name]
    prog = interp.Program(program(*lines), info)
    tr = translate.translate(prog.instructions, info)
    cpu = launch_for(prog, cbufs)
    on_gpu = copy.deepcopy(cpu)
    interp.run(prog, cpu)
    assert gpu.run(tr, on_gpu).status == 0
    inputs = struct.unpack("<64I", cbufs[3]) if cbufs else ()
    threads = info.local_size[0]
    for handle, surf in cpu.surfaces.items():
        got = on_gpu.surfaces[handle].data
        flushed = FLUSHING_GPUS_MAY_FLUSH.get(handle[1]) if name == "float" else None
        bad = []
        for t in range(threads):
            want, have = surf.data[4 * t], got[4 * t]
            if want != have and not (flushed and have == flushed(inputs[2 * t], inputs[2 * t + 1])):
                bad.append((t, f"{want:#010x}", f"{have:#010x}"))
        text = next(ln[0] for ln in lines if ln[0].startswith("SUST") and ln[2][40] == handle[1])
        assert not bad, f"{text}: thread, cpu, gpu = {bad[:6]}"


def test_gpu_stops_at_the_step_limit(gpu):
    prog = interp.Program(program(bra_self(0)), INFO)
    tr = translate.translate(prog.instructions, INFO, step_limit=1000)
    assert gpu.run(tr, launch_for(prog, {})).status == translate.STATUS_STEP_LIMIT


def test_gpu_records_mufu_results(gpu):
    lines = [
        s2r(0, "SR_TID.X"),
        ("I2FP.F32.U32 R1, R0", 0x245, {16: 1, 32: 0, 75: 2, 84: 2}),
        ("MUFU.EX2 R2, R1", 0x308, {16: 2, 32: 1, 74: 2}),
        store(0, 2),
        exit_(),
        bra_self(5),
    ]
    prog = interp.Program(program(*lines), INFO)
    tr = translate.translate(prog.instructions, INFO, mufu_log=True)
    launch = launch_for(prog, {})
    res = gpu.run(tr, launch, log_capacity=64)
    assert sorted(res.mufu) == [(0x20, fbits(float(t)), launch.surfaces[(0, 1)].data[4 * t]) for t in range(32)]
    assert res.mufu_dropped == 0
    assert all(abs(fbits(2.0**t) - r) <= 2 for t, (_, _, r) in enumerate(sorted(res.mufu)))
