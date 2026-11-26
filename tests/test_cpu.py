"""Tests for siicl.cpu, on small AArch64 programs the tests assemble themselves."""

import struct
import zlib

import pytest

from siicl import cpu
from tests.builders import AS, assemble

pytestmark = pytest.mark.skipif(not AS, reason="aarch64-linux-gnu-binutils is not installed")

CODE, DATA, STACK = 0x10000, 0x20000, 0x30000


def machine(tmp_path, source, *, data=b""):
    c = cpu.Cpu()
    c.map(CODE, 0x10000, cpu.READ | cpu.EXECUTE)
    c.map(DATA, 0x1000, cpu.READ | cpu.WRITE)
    c.map(STACK, 0x10000, cpu.READ | cpu.WRITE)
    c.write(CODE, assemble(source, tmp_path))
    c.write(DATA, data)
    c.pc, c.sp = CODE, STACK + 0x10000
    return c


def test_stops_at_each_svc_and_resumes_after_it(tmp_path):
    c = machine(tmp_path, "mov x0, #7\nsvc #0x21\nadd x0, x0, #1\nsvc #0x80")
    first = c.run()
    assert (first.reason, first.svc, first.pc, c.pc, c.x(0)) == ("svc", 0x21, CODE + 4, CODE + 8, 7)
    c.set_x(0, 40)
    second = c.run()
    assert (second.reason, second.svc, second.pc, c.x(0)) == ("svc", 0x80, CODE + 12, 41)
    assert second.describe() == "SVC 0x80"


@pytest.mark.parametrize(
    ("source", "access", "kind", "address"),
    [
        ("mov x1, #0x50000\nldr x0, [x1]", "read", "unmapped", 0x50000),
        ("mov x1, #0x50000\nstr x0, [x1, #8]", "write", "unmapped", 0x50008),
        (f"mov x1, #{CODE:#x}\nstr x0, [x1]", "write", "protected", CODE),
    ],
)
def test_reports_a_bad_access_at_the_instruction_that_made_it(tmp_path, source, access, kind, address):
    c = machine(tmp_path, source + "\nsvc #0")
    stop = c.run()
    assert (stop.reason, stop.access, stop.kind, stop.address, stop.pc) == ("fault", access, kind, address, CODE + 4)


def test_reports_a_jump_to_unmapped_memory_as_a_fetch_fault(tmp_path):
    c = machine(tmp_path, "mov x1, #0x50000\nblr x1")
    stop = c.run()
    assert (stop.reason, stop.access, stop.address, stop.pc, c.x(30)) == ("fault", "fetch", 0x50000, 0x50000, CODE + 8)


def test_reports_undefined_instructions_and_breakpoints(tmp_path):
    c = machine(tmp_path, "nop\nudf #0")
    assert (c.run().reason, c.pc) == ("undefined", CODE + 4)
    c = machine(tmp_path, "nop\nbrk #0x7d0")
    stop = c.run()
    assert (stop.reason, stop.svc, stop.pc) == ("break", 0x7D0, CODE + 4)


def test_instruction_budget_stops_a_loop_and_the_loop_can_continue(tmp_path):
    c = machine(tmp_path, "mov x0, #0\n1: add x0, x0, #1\ncmp x0, #1000\nb.ne 1b\nsvc #1")
    stop = c.run(limit=100)
    assert stop.reason == "limit" and 0 < c.x(0) < 1000
    while (stop := c.run(limit=100)).reason == "limit":
        pass
    assert (stop.reason, c.x(0)) == ("svc", 1000)
    with pytest.raises(ValueError):
        c.run(limit=0)


def test_runs_the_architecture_features_games_use(tmp_path):
    source = f"""
        mov x1, #{DATA:#x}
        mov x2, #5
        ldaddal x2, x3, [x1]       // FEAT_LSE: x3 = old value, memory += 5
        mov x4, #15
        mov x5, #99
        casal x4, x5, [x1]         // compare 15, swap in 99
        ldapr x6, [x1]             // FEAT_LRCPC
        ldr q1, [x1, #16]
        ldr q2, [x1, #32]
        movi v0.4s, #0
        udot v0.4s, v1.16b, v2.16b // FEAT_DotProd
        mov w7, v0.s[0]
        ldr h3, [x1, #48]
        fadd h4, h3, h3            // FEAT_FP16
        umov w8, v4.h[0]
        ldr x9, [x1, #56]
        mov w10, #-1
        crc32x w10, w10, x9        // FEAT_CRC32
        mvn w10, w10
        svc #0
    """
    data = struct.pack("<Q", 10) + bytes(8) + bytes(range(1, 17)) + bytes([2] * 16)
    data += struct.pack("<e", 1.5) + bytes(6) + struct.pack("<Q", 0x0123456789ABCDEF)
    c = machine(tmp_path, source, data=data)
    assert c.run().reason == "svc"
    assert (c.x(3), c.x(4), c.x(6), c.read_u64(DATA)) == (10, 15, 99, 99)
    assert c.x(7) == 2 * (1 + 2 + 3 + 4)
    assert c.x(8) == struct.unpack("<H", struct.pack("<e", 3.0))[0]
    assert c.x(10) == zlib.crc32(struct.pack("<Q", 0x0123456789ABCDEF))


def test_pointer_authentication_leaves_pointers_unchanged(tmp_path):
    source = """
        mov x0, #0x1234
        mov x1, #0x99
        pacia x0, x1
        mov x2, x0
        autib x0, x1
        bl fn
        svc #0
    fn:
        pacibsp
        stp x29, x30, [sp, #-16]!
        ldp x29, x30, [sp], #16
        retab
    """
    c = machine(tmp_path, source)
    stop = c.run()
    assert (stop.reason, c.x(2), c.x(0)) == ("svc", 0x1234, 0x1234)


def test_reads_and_writes_guest_memory_and_registers(tmp_path):
    c = machine(tmp_path, "mrs x0, tpidrro_el0\nmrs x1, tpidr_el0\nsvc #0")
    c.tpidrro_el0, c.tpidr_el0 = 0x1111, 0x2222
    c.write_u64(DATA + 8, 0xFEDCBA9876543210)
    assert c.read(DATA + 8, 4) == bytes.fromhex("10325476")
    assert (c.read_u32(DATA + 12), c.read_u64(DATA + 8)) == (0xFEDCBA98, 0xFEDCBA9876543210)
    c.run()
    assert (c.x(0), c.x(1)) == (0x1111, 0x2222)
    with pytest.raises(cpu.GuestMemoryError):
        c.read(0x50000, 4)
    with pytest.raises(ValueError, match="pages"):
        c.map(0x60010, 0x1000, cpu.READ)


def test_protect_and_unmap_change_what_code_can_do(tmp_path):
    c = machine(tmp_path, f"mov x1, #{DATA:#x}\nstr x0, [x1]\nsvc #0")
    c.protect(DATA, 0x1000, cpu.READ)
    assert c.run().kind == "protected"
    c.pc = CODE
    c.unmap(DATA, 0x1000)
    assert c.run().kind == "unmapped"
