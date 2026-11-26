"""Run a game from its entry point and write a boot log of what it does, until it stops.

The log says where the modules and the main thread were placed, lists every SVC the threads make
with its arguments and result, then says where and why the run ended, with the registers, a
backtrace and the state of every thread. See docs/boot.md.
"""

from __future__ import annotations

from typing import TextIO

from siicl.addrspace import perm_text
from siicl.cpu import GuestMemoryError, Stop
from siicl.kernel import End, Process

MAX_FRAMES = 32


def boot(process: Process, log: TextIO, *, max_seconds: float) -> End:
    write_setup(process, log)
    print(f"\nrunning for at most {max_seconds:g} s of guest time", file=log)
    end = process.run(log, max_seconds=max_seconds)
    write_end(process, end, log)
    return end


def write_setup(process: Process, log: TextIO) -> None:
    print("modules:", file=log)
    for m in process.modules:
        print(f"  {m.name:<8} {m.base:#012x}-{m.end:#012x}  module id {m.module_id.hex()}", file=log)
    print("memory:", file=log)
    for b in process.space.blocks:
        print(f"  {b.start:#012x}-{b.end:#012x} {perm_text(b.perm)} {b.state.name.lower():<12} {b.name}", file=log)
    t = process.main_thread
    print(
        f"main thread: handle {t.handle:#x}, entry {process.describe_address(t.entry)}, "
        f"stack top {t.stack_top:#x}, thread-local storage {t.tls:#x}, priority {t.priority}, core {t.core}",
        file=log,
    )
    print(f"system counter reads turned into SVCs: {process.counter_reads}", file=log)


def describe_stop(process: Process, stop: Stop) -> str:
    where = f"{process.describe_address(stop.pc)} ({stop.pc:#x})"
    if stop.reason == "svc":
        return f"SVC 0x{stop.svc:02X} at {where}"
    if stop.reason == "fault" and stop.access == "fetch":
        return f"{stop.kind} fetch: jumped to {where}"
    if stop.reason == "fault":
        return f"{stop.kind} {stop.access} at {stop.address:#x}, by the instruction at {where}"
    return f"{stop.describe()} at {where}"


def describe_end(process: Process, end: End) -> str:
    text = end.message
    if end.stop is not None:
        text += f", at {process.describe_address(end.stop.pc)} ({end.stop.pc:#x})"
        if end.stop.reason != "svc":
            text = f"the game crashed: {describe_stop(process, end.stop)}"
    if end.thread is not None:
        text += f", in thread {end.thread.number}"
    return text


def write_end(process: Process, end: End, log: TextIO) -> None:
    print(f"\nstopped at {process.seconds:.6f} s: {describe_end(process, end)}", file=log)
    if end.thread is not None and end.thread is process.current:
        cpu = process.cpu
        print("registers:", file=log)
        regs = [(f"x{i}", cpu.x(i)) for i in range(31)] + [("sp", cpu.sp), ("pc", cpu.pc)]
        for i in range(0, len(regs), 4):
            print("  " + "  ".join(f"{name:>3} {value:016x}" for name, value in regs[i : i + 4]), file=log)
        print("backtrace:", file=log)
        pc = end.stop.pc if end.stop else cpu.pc
        for n, (address, how) in enumerate(backtrace(process, pc)):
            print(f"  #{n:<2} {address:#012x} {process.describe_address(address)}{how}", file=log)
    print("threads:", file=log)
    for t in process.threads:
        print(f"  {t.describe()}, at {process.describe_address(t.pc)}", file=log)
    if process.svc_counts:
        print("SVC calls:", file=log)
        for name, count in process.svc_counts.most_common():
            print(f"  {count:>9,} {name}", file=log)


def backtrace(process: Process, pc: int) -> list[tuple[int, str]]:
    """The stop address, the link register, then the return addresses along the frame-pointer chain."""
    cpu = process.cpu
    frames = [(pc, "")]
    lr = cpu.x(30)
    if process.module_at(lr):
        frames.append((lr, " (link register)"))
    fp = cpu.x(29)
    while len(frames) < MAX_FRAMES and fp and fp % 8 == 0:
        try:
            next_fp, ret = cpu.read_u64(fp), cpu.read_u64(fp + 8)
        except GuestMemoryError:
            break
        if not process.module_at(ret):
            break
        if ret != frames[-1][0]:
            frames.append((ret, ""))
        if next_fp <= fp:
            break
        fp = next_fp
    return frames
