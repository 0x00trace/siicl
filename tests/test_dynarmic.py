"""Tests for siicl.dynarmic, on small AArch64 programs the tests assemble themselves.

They need the C library (native/CMakeLists.txt says how to build it) and skip without it.
"""

import struct
import threading
import time
import zlib

import pytest

from siicl import cpu, dynarmic
from tests.builders import AS, assemble

pytestmark = [
    pytest.mark.skipif(not AS, reason="aarch64-linux-gnu-binutils is not installed"),
    pytest.mark.skipif(not dynarmic.available(), reason="libsiicl_dynarmic.so is not built; see native/CMakeLists.txt"),
]

CODE, DATA, STACK = 0x10000, 0x20000, 0x30000


@pytest.fixture
def machine():
    with dynarmic.Machine(cores=2) as m:
        yield m


def load(m, tmp_path, source, *, data=b""):
    m.map(CODE, 0x10000, cpu.READ | cpu.EXECUTE)
    m.map(DATA, 0x1000, cpu.READ | cpu.WRITE)
    m.map(STACK, 0x10000, cpu.READ | cpu.WRITE)
    m.write(CODE, assemble(source, tmp_path))
    m.write(DATA, data)
    c = m.core()
    c.pc, c.sp = CODE, STACK + 0x10000
    return c


def test_stops_at_each_svc_and_resumes_after_it(machine, tmp_path):
    c = load(machine, tmp_path, "mov x0, #7\nsvc #0x21\nadd x0, x0, #1\nsvc #0x80")
    first = c.run()
    assert (first.reason, first.svc, first.pc, c.pc, c.x(0)) == ("svc", 0x21, CODE + 4, CODE + 8, 7)
    c.set_x(0, 40)
    second = c.run()
    assert (second.reason, second.svc, second.pc, c.x(0)) == ("svc", 0x80, CODE + 12, 41)


@pytest.mark.parametrize(
    ("source", "access", "kind", "address"),
    [
        ("mov x1, #0x50000\nldr x0, [x1]", "read", "unmapped", 0x50000),
        ("mov x1, #0x50000\nstr x0, [x1, #8]", "write", "unmapped", 0x50008),
        (f"mov x1, #{CODE:#x}\nstr x0, [x1]", "write", "protected", CODE),
    ],
)
def test_reports_a_bad_access_at_the_instruction_that_made_it(machine, tmp_path, source, access, kind, address):
    c = load(machine, tmp_path, source + "\nsvc #0")
    c.set_x(0, 0x55)
    stop = c.run()
    assert (stop.reason, stop.access, stop.kind, stop.address, stop.pc) == ("fault", access, kind, address, CODE + 4)
    assert (c.pc, c.x(0)) == (CODE + 4, 0x55)


def test_reports_a_jump_to_unmapped_memory_as_a_fetch_fault(machine, tmp_path):
    c = load(machine, tmp_path, "mov x1, #0x50000\nblr x1")
    stop = c.run()
    assert (stop.reason, stop.access, stop.address, stop.pc, c.x(30)) == ("fault", "fetch", 0x50000, 0x50000, CODE + 8)


def test_reports_undefined_instructions_and_breakpoints_at_the_instruction(machine, tmp_path):
    c = load(machine, tmp_path, "nop\nudf #0\nbrk #0x7d0")
    assert (c.run().reason, c.pc) == ("undefined", CODE + 4)
    c.pc = CODE + 8
    stop = c.run()
    assert (stop.reason, stop.svc, stop.pc, c.pc) == ("break", 0x7D0, CODE + 8, CODE + 8)


def test_hints_do_nothing(machine, tmp_path):
    c = load(machine, tmp_path, "yield\nwfe\nsevl\npacibsp\nautibsp\nxpaclri\nmov x0, #3\nsvc #0")
    assert (c.run().reason, c.x(0)) == ("svc", 3)


def test_instruction_budget_stops_a_loop_and_the_loop_can_continue(machine, tmp_path):
    c = load(machine, tmp_path, "mov x0, #0\n1: add x0, x0, #1\ncmp x0, #1000\nb.ne 1b\nsvc #1")
    stop = c.run(limit=100)
    # dynarmic checks the budget between blocks, so a run ends at the end of the loop's block.
    assert stop.reason == "limit" and 100 <= c.ticks < 103 and 0 < c.x(0) < 1000
    while (stop := c.run(limit=100)).reason == "limit":
        pass
    assert (stop.reason, c.x(0)) == ("svc", 1000)
    with pytest.raises(ValueError):
        c.run(limit=0)


def test_a_run_goes_on_to_the_end_of_its_block(machine, tmp_path):
    c = load(machine, tmp_path, "mov x0, #0\n" + "add x0, x0, #1\n" * 60 + "svc #1")
    stop = c.run(limit=10)
    assert (stop.reason, c.ticks, c.x(0)) == ("svc", 62, 60)


def test_runs_are_repeatable(tmp_path):
    source = "mov x0, #0\n1: add x0, x0, #1\ntst x0, #7\nb.ne 2f\nadd x1, x1, #1\n2: cmp x0, #900\nb.ne 1b\nsvc #1"

    def trace():
        with dynarmic.Machine() as m:
            c = load(m, tmp_path, source)
            seen = []
            for limit in [5, 17, 1, 40, 3] * 20:
                stop = c.run(limit=limit)
                seen.append((stop.reason, c.ticks, c.pc, c.x(0), c.x(1)))
            return seen

    assert trace() == trace()


def test_accesses_across_a_page_boundary_reach_both_host_pages(machine, tmp_path):
    source = f"""
        mov x1, #{DATA:#x}
        add x1, x1, #0xFFC
        ldr x0, [x1]
        mov x2, #0x7788
        movk x2, #0x5566, lsl #16
        str x2, [x1]
        svc #0
    """
    c = load(machine, tmp_path, source)
    machine.map(DATA + 0x1000, 0x1000, cpu.READ | cpu.WRITE)  # a separate host allocation
    machine.write(DATA + 0xFF8, bytes(range(16)))
    assert c.run().reason == "svc"
    assert c.x(0) == int.from_bytes(bytes(range(4, 12)), "little")
    assert machine.read(DATA + 0xFFC, 8) == struct.pack("<Q", 0x55667788)


def test_exclusive_increments_on_two_cores_in_parallel_lose_no_updates(machine, tmp_path):
    n = 200_000
    source = f"""
        mov x1, #{DATA:#x}
    1:  ldxr x2, [x1]
        add x2, x2, #1
        stxr w4, x2, [x1]
        cbnz w4, 1b
        subs x3, x3, #1
        b.ne 1b
        svc #0
    """
    first = load(machine, tmp_path, source)
    second = machine.core()
    for c in (first, second):
        c.pc = CODE
        c.set_x(3, n)
    stops = {}
    threads = [threading.Thread(target=lambda c=c: stops.setdefault(c.processor_id, c.run())) for c in (first, second)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert [s.reason for s in stops.values()] == ["svc", "svc"]
    assert machine.read_u64(DATA) == 2 * n


def test_counter_reads_return_the_counter_and_advance_it(machine, tmp_path):
    c = load(machine, tmp_path, "mrs x0, cntpct_el0\nmrs x1, cntpct_el0\nmrs x2, cntfrq_el0\nsvc #0")
    c.counter = 1000
    assert c.run().reason == "svc"
    assert (c.x(0), c.x(1), c.x(2), c.counter) == (1000, 1001, 19_200_000, 1002)


def test_thread_registers(machine, tmp_path):
    c = load(machine, tmp_path, "mrs x0, tpidrro_el0\nmrs x1, tpidr_el0\nmsr tpidr_el0, x2\nsvc #0")
    c.tpidrro_el0, c.tpidr_el0 = 0x1111, 0x2222
    c.set_x(2, 0x3333)
    c.run()
    assert (c.x(0), c.x(1), c.tpidr_el0) == (0x1111, 0x2222, 0x3333)


def test_runs_the_architecture_features_it_decodes(machine, tmp_path):
    source = f"""
        mov x1, #{DATA:#x}
        ldr q1, [x1, #16]
        ldr q2, [x1, #32]
        movi v0.4s, #0
        udot v0.4s, v1.16b, v2.16b // FEAT_DotProd
        mov w7, v0.s[0]
        ldr h3, [x1, #48]
        fneg h4, h3                // FEAT_FP16, the one instruction of it that games use
        umov w8, v4.h[0]
        ldr x9, [x1, #56]
        mov w10, #-1
        crc32x w10, w10, x9        // FEAT_CRC32
        mvn w10, w10
        svc #0
    """
    data = bytes(16) + bytes(range(1, 17)) + bytes([2] * 16)
    data += struct.pack("<e", 1.5) + bytes(6) + struct.pack("<Q", 0x0123456789ABCDEF)
    c = load(machine, tmp_path, source, data=data)
    assert c.run().reason == "svc"
    assert c.x(7) == 2 * (1 + 2 + 3 + 4)
    assert c.x(8) == struct.unpack("<H", struct.pack("<e", -1.5))[0]
    assert c.x(10) == zlib.crc32(struct.pack("<Q", 0x0123456789ABCDEF))


@pytest.mark.parametrize(
    "instruction",
    [
        "ldaddal x2, x3, [x1]",  # FEAT_LSE
        "casal x4, x5, [x1]",
        "swpal x4, x5, [x1]",
        "ldapr x6, [x1]",  # FEAT_LRCPC
        "pacia x0, x1",  # FEAT_PAuth, the forms that are not hints
        "autib x0, x1",
        "retab",
        "braa x0, x1",
        "ldraa x0, [x1]",
        "fadd h4, h3, h3",  # FEAT_FP16 arithmetic
        "mrs x0, cntvct_el0",  # the virtual counter; dynarmic has only CNTPCT_EL0
        "mrs x0, pmcr_el0",  # performance monitor registers
        "msr pmcntenset_el0, x0",
    ],
)
def test_instructions_dynarmic_does_not_decode_yet(machine, tmp_path, instruction):
    """Stage 2 of docs/cpu-backend.md adds these; this test records which ones are missing."""
    c = load(machine, tmp_path, f"mov x1, #{DATA:#x}\n{instruction}\nsvc #0")
    assert (c.run().reason, c.pc) == ("undefined", CODE + 4)


def test_reads_and_writes_guest_memory_and_registers(machine, tmp_path):
    c = load(machine, tmp_path, "svc #0")
    machine.write(DATA + 8, struct.pack("<Q", 0xFEDCBA9876543210))
    assert machine.read(DATA + 8, 4) == bytes.fromhex("10325476")
    assert (machine.read_u32(DATA + 12), machine.read_u64(DATA + 8)) == (0xFEDCBA98, 0xFEDCBA9876543210)
    c.set_v(5, 1 << 127 | 0x1234)
    assert c.v(5) == 1 << 127 | 0x1234
    with pytest.raises(cpu.GuestMemoryError):
        machine.read(0x50000, 4)
    with pytest.raises(ValueError, match="pages"):
        machine.map(0x60010, 0x1000, cpu.READ)


def test_saved_contexts_switch_one_core_between_threads(machine, tmp_path):
    c = load(machine, tmp_path, "add x0, x0, #1\nfmov d0, x0\nsvc #0\nb .-12")
    c.set_x(0, 100)
    c.tpidrro_el0 = 0xAAAA
    a = c.save_context()
    c.set_x(0, 200)
    c.tpidrro_el0 = 0xBBBB
    b = c.save_context()
    for ctx, want in [(a, 101), (b, 201), (a, 101)]:
        c.restore_context(ctx)
        assert c.run().reason == "svc"
        assert (c.x(0), c.v(0)) == (want, want)
        ctx.x[0] = want - 1  # the next turn of this thread starts from the same value
    c.restore_context(b)
    assert c.tpidrro_el0 == 0xBBBB


def test_rewritten_code_runs_after_a_write(machine, tmp_path):
    c = load(machine, tmp_path, "mov x0, #1\nsvc #0")
    assert (c.run().reason, c.x(0)) == ("svc", 1)
    machine.write(CODE, assemble("mov x0, #2", tmp_path))
    c.pc = CODE
    assert (c.run().reason, c.x(0)) == ("svc", 2)


def test_protect_and_unmap_change_what_code_can_do(machine, tmp_path):
    c = load(machine, tmp_path, f"mov x1, #{DATA:#x}\nstr x0, [x1]\nsvc #0")
    machine.protect(DATA, 0x1000, cpu.READ)
    assert c.run().kind == "protected"
    c.pc = CODE
    machine.unmap(DATA, 0x1000)
    assert c.run().kind == "unmapped"
    c.pc = CODE
    machine.protect(CODE, 0x10000, cpu.READ)
    stop = c.run()
    assert (stop.reason, stop.access, stop.kind) == ("fault", "fetch", "protected")


def test_halt_stops_a_run_from_another_thread(tmp_path):
    with dynarmic.Machine() as m:
        m.map(CODE, 0x1000, cpu.READ | cpu.EXECUTE)
        m.write(CODE, assemble("1: b 1b", tmp_path))
        c = m.core(count_ticks=False)
        c.pc = CODE
        with pytest.raises(ValueError):
            c.run(limit=10)
        _halt_and_check(c)


def _halt_and_check(c):
    timer = threading.Timer(0.05, c.halt)
    timer.start()
    start = time.monotonic()
    stop = c.run()
    timer.join()
    assert stop.reason == "halted" and time.monotonic() - start < 5
