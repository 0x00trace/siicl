"""Tests for starting a process and writing the boot log, on synthetic modules."""

import io

import pytest

from siicl import addrspace, boot, cli, dynarmic, kernel
from siicl import npdm as npdm_mod
from siicl.addrspace import MemoryState
from tests.builders import AS, assemble, build_npdm, build_nso, build_nup

pytestmark = [
    pytest.mark.skipif(not AS, reason="aarch64-linux-gnu-binutils is not installed"),
    pytest.mark.usefixtures("cpu_backend"),  # each test runs on Unicorn and on dynarmic
]

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

CONNECT = """
    adr x1, name
    svc #0x1f
name:
    .asciz "dbg:"
"""


def process(tmp_path, rtld_source, **extra):
    mods = {"rtld": build_nso(assemble(rtld_source, tmp_path), bss=0), **extra}
    return kernel.Process(mods, npdm_mod.parse(build_npdm(svcs=range(1, 0x81))))


def test_main_thread_starts_at_rtld_with_the_registers_the_system_gives_it(tmp_path):
    p = process(tmp_path, START, main=build_nso(b"\0" * 4))
    stop = p.cpu.run()
    t = p.main_thread
    assert (stop.reason, stop.svc, p.describe_address(stop.pc)) == ("svc", 0x01, "rtld+0x14")
    assert (p.cpu.x(2), p.cpu.x(3), p.cpu.x(6)) == (0, t.handle, BASE + 0x10)
    assert t.handle == 0x8000 and p.handles.get(t.handle) is t
    assert p.cpu.x(4) == t.tls and LAYOUT.kernel_map.contains(t.tls, kernel.TLS_SIZE)
    stack = p.space.query(t.stack_top - 1)
    assert p.cpu.x(5) == t.stack_top == stack.end
    assert (stack.size, stack.state) == (0x100000, MemoryState.STACK) and LAYOUT.stack.contains(stack.start)
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
    end = boot.boot(p, log, max_seconds=1)
    text = log.getvalue()
    assert (end.reason, end.stop.address) == ("crash", 0x40)
    assert "the game crashed: unmapped read at 0x40, by the instruction at rtld+0x28 (0x8000028), in thread 0" in text
    assert [p.describe_address(a) for a, _ in boot.backtrace(p, end.stop.pc)] == ["rtld+0x28", "rtld+0x1c", "rtld+0xc"]
    assert "rtld+0x1c (link register)" in text
    assert f"  rtld     {BASE:#012x}-{BASE + 0x3000:#012x}" in text
    assert "r-x code         rtld .text" in text
    assert "main thread: handle 0x8000, entry rtld+0x0" in text
    assert "thread 0 (id 0x100, handle 0x8000, priority 44, core 0): runnable, at rtld+0x28" in text


def test_boot_stops_at_the_time_limit(tmp_path):
    p = process(tmp_path, "1: b 1b")
    end = boot.boot(p, io.StringIO(), max_seconds=0.001)
    assert (end.reason, end.message) == ("time", "the time limit of 0.001 s of guest time was reached")


def test_boot_log_lists_each_svc_and_the_one_that_is_missing(tmp_path):
    log = io.StringIO()
    end = boot.boot(process(tmp_path, "mov x1, #0x200000\nsvc #0x01\n" + CONNECT), log, max_seconds=1)
    text = log.getvalue()
    assert end.reason == "unimplemented"
    assert "t0  SetHeapSize(size=0x200000) -> ok, heap at 0x2000000000" in text
    assert 'SVC 0x1F (ConnectToNamedPort) to port "dbg:" is not implemented, at rtld+0xc' in text
    assert "        1 SetHeapSize" in text


def _game(tmp_path):
    container = build_nup(
        [
            ("exefs/rtld", build_nso(assemble(CONNECT, tmp_path), bss=0), True),
            ("exefs/main", build_nso(b"\0" * 4), True),
            ("exefs/main.npdm", build_npdm(svcs=(0x1F,)), True),
        ]
    )
    src = tmp_path / "game.nup"
    src.write_bytes(container)
    return src


def test_boot_command_writes_the_log_outside_the_repo(tmp_path, capsys):
    src, out = _game(tmp_path), tmp_path / "out"
    assert cli.main(["boot", str(src), "--out", str(out)]) == 0
    expected = 'SVC 0x1F (ConnectToNamedPort) to port "dbg:" is not implemented, at rtld+0x4 (0x8000004), in thread 0'
    assert expected in capsys.readouterr().out
    text = (out / "boot.log").read_text()
    assert text.startswith("siicl boot log for game.nup, title 0400000000001000")
    assert expected in text


def test_boot_command_runs_on_the_cpu_backend_it_is_given(tmp_path, cpu_backend):
    src, out = _game(tmp_path), tmp_path / "out"
    assert cli.main(["boot", str(src), "--out", str(out), "--cpu", cpu_backend]) == 0
    text = (out / "boot.log").read_text()
    assert f"CPU backend: {cpu_backend}" in text
    if cpu_backend == "dynarmic":
        assert "system counter reads: answered by the CPU backend" in text
    else:
        assert "system counter reads turned into SVCs: 0" in text


def test_boot_command_runs_on_dynarmic_unless_its_library_is_missing(tmp_path, capsys, monkeypatch):
    if not dynarmic.available():
        pytest.skip("libsiicl_dynarmic.so is not built; see native/CMakeLists.txt")
    src = _game(tmp_path)
    assert cli.main(["boot", str(src), "--out", str(tmp_path / "a")]) == 0
    assert "CPU backend: dynarmic" in (tmp_path / "a" / "boot.log").read_text()
    assert "Unicorn" not in capsys.readouterr().err

    monkeypatch.setenv("SIICL_DYNARMIC", str(tmp_path / "missing.so"))
    assert cli.main(["boot", str(src), "--out", str(tmp_path / "b")]) == 0
    assert "CPU backend: unicorn" in (tmp_path / "b" / "boot.log").read_text()
    assert "the dynarmic library is not built" in capsys.readouterr().err


def test_boot_command_says_when_the_dynarmic_library_is_missing(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("SIICL_DYNARMIC", str(tmp_path / "missing.so"))
    monkeypatch.setattr(dynarmic, "_lib", None)
    assert cli.main(["boot", str(_game(tmp_path)), "--out", str(tmp_path / "out"), "--cpu", "dynarmic"]) == 2
    assert "missing.so" in capsys.readouterr().err
