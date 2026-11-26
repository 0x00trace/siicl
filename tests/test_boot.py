"""Tests for starting a process and writing the boot log, on synthetic modules."""

import io

import pytest

from siicl import addrspace, boot, cli, kernel
from siicl import npdm as npdm_mod
from siicl.addrspace import MemoryState
from tests.builders import AS, assemble, build_npdm, build_nso, build_nup

pytestmark = pytest.mark.skipif(not AS, reason="aarch64-linux-gnu-binutils is not installed")

BASE = addrspace.LAYOUT_39BIT.code.start
LAYOUT = addrspace.LAYOUT_39BIT

START = """
    mov x2, x0
    mov x3, x1
    mrs x4, tpidrro_el0
    mov x5, sp
    adr x6, .
    svc #0x01
"""


def process(tmp_path, rtld_source, **extra):
    mods = {"rtld": build_nso(assemble(rtld_source, tmp_path), bss=0), **extra}
    return kernel.Process(mods, npdm_mod.parse(build_npdm()))


def test_main_thread_starts_at_rtld_with_the_registers_the_system_gives_it(tmp_path):
    p = process(tmp_path, START, main=build_nso(b"\0" * 4))
    stop = p.cpu.run()
    t = p.main_thread
    assert (stop.reason, stop.svc, p.describe_address(stop.pc)) == ("svc", 0x01, "rtld+0x14")
    assert (p.cpu.x(2), p.cpu.x(3), p.cpu.x(6)) == (0, t.handle, BASE + 0x10)
    assert t.handle == 0x8000 and p.handles.get(t.handle) is t
    assert p.cpu.x(4) == t.tls and LAYOUT.kernel_map.contains(t.tls, kernel.TLS_SIZE)
    assert p.cpu.x(5) == t.stack.end and t.stack.size == 0x100000 and LAYOUT.stack.contains(t.stack.start)
    assert (t.priority, t.core) == (44, 0)
    assert p.space.query(t.tls).state == MemoryState.THREAD_LOCAL
    assert p.cpu.read(t.tls, kernel.TLS_SIZE) == bytes(kernel.TLS_SIZE)


def test_refuses_a_process_without_a_39_bit_address_space(tmp_path):
    raw = bytearray(build_npdm())
    raw[0x0C] = 0x33  # 64-bit, 36-bit address space
    with pytest.raises(kernel.ProcessError, match="36-bit"):
        kernel.Process({"rtld": build_nso(b"\0" * 4)}, npdm_mod.parse(bytes(raw)))


def test_handles_carry_a_serial_number_above_the_slot_index():
    table = kernel.HandleTable(size=4)
    a, b = table.add("a"), table.add("b")
    assert (a, b) == (0x8000, 0x10001)
    with pytest.raises(KeyError):
        table.get(0x18000)  # slot 0 with the wrong serial number


FAULT = """
    stp x29, x30, [sp, #-16]!
    mov x29, sp
    bl one
    svc #0
one:
    stp x29, x30, [sp, #-16]!
    mov x29, sp
    bl two
    ldp x29, x30, [sp], #16
    ret
two:
    mov x0, #0x40
    ldr x1, [x0]
    ret
"""


def test_boot_log_names_the_stop_and_walks_the_frame_pointers(tmp_path):
    p = process(tmp_path, FAULT, main=build_nso(b"\0" * 4))
    log = io.StringIO()
    stop = boot.boot(p, log, max_instructions=1000)
    text = log.getvalue()
    assert (stop.reason, stop.address) == ("fault", 0x40)
    assert "stopped: unmapped read at 0x40, by the instruction at rtld+0x28" in text
    assert [p.describe_address(a) for a, _ in boot.backtrace(p, stop.pc)] == ["rtld+0x28", "rtld+0x1c", "rtld+0xc"]
    assert "rtld+0x1c (link register)" in text
    assert f"  rtld     {BASE:#012x}-{BASE + 0x3000:#012x}" in text
    assert "r-x code         rtld .text" in text
    assert "main thread: handle 0x8000, entry rtld+0x0" in text


def test_boot_stops_when_the_instruction_budget_runs_out(tmp_path):
    p = process(tmp_path, "1: b 1b")
    stop = boot.boot(p, io.StringIO(), max_instructions=500)
    assert stop.reason == "limit"


def test_boot_command_writes_the_log_outside_the_repo(tmp_path, capsys):
    container = build_nup(
        [
            ("exefs/rtld", build_nso(assemble(START, tmp_path), bss=0), True),
            ("exefs/main", build_nso(b"\0" * 4), True),
            ("exefs/main.npdm", build_npdm(), True),
        ]
    )
    src = tmp_path / "game.nup"
    src.write_bytes(container)
    out = tmp_path / "out"
    assert cli.main(["boot", str(src), "--out", str(out)]) == 0
    assert "stopped: SVC 0x01 (SetHeapSize) at rtld+0x14" in capsys.readouterr().out
    text = (out / "boot.log").read_text()
    assert text.startswith("siicl boot log for game.nup, title 0400000000001000")
    assert "stopped: SVC 0x01 (SetHeapSize) at rtld+0x14" in text
