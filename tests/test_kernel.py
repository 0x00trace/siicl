"""Tests for the kernel: SVCs and the scheduler, on AArch64 programs the tests assemble themselves.

Each program runs as the rtld module of a synthetic process. Most end with ExitProcess, after
which the main thread's registers hold what the test checks.
"""

import io
import struct

import pytest

from siicl import kernel, svcs
from siicl import npdm as npdm_mod
from siicl.addrspace import LAYOUT_39BIT, MemoryAttribute, MemoryState
from tests.builders import AS, assemble, build_npdm, build_nso

pytestmark = pytest.mark.skipif(not AS, reason="aarch64-linux-gnu-binutils is not installed")

HEAP = LAYOUT_39BIT.heap.start
ALL_SVCS = range(0x01, 0x81)

# A mutex and an unlock as the system's libraries write them: the mutex word holds the owner's
# handle, with bit 30 set while other threads wait. x0 is the mutex address, w1 the caller's handle.
MUTEX = """
lock:
    mov x9, x0
    mov w10, w1
2:  mov w11, #0
    casa w11, w10, [x9]
    cbz w11, 3f
    tbnz w11, #30, 4f
    orr w12, w11, #0x40000000
    mov w13, w11
    casa w13, w12, [x9]
    cmp w13, w11
    b.ne 2b
4:  and w0, w11, #0xbfffffff
    mov x1, x9
    mov w2, w10
    svc #0x1a
    ldr w11, [x9]
    and w11, w11, #0xbfffffff
    cmp w11, w10
    b.ne 2b
3:  ret

unlock:
    mov x9, x0
    mov w11, w1
    mov w12, #0
    casl w11, w12, [x9]
    cmp w11, w1
    b.eq 5f
    mov x0, x9
    svc #0x1b
5:  ret
"""

# Set up a 2 MiB heap and keep its address in x19.
HEAP_SETUP = """
    mov x28, x1
    mov x1, #0x200000
    svc #0x01
    mov x19, x1
"""


def npdm(*, svcs=ALL_SVCS, system_resource_size=0):
    raw = bytearray(build_npdm(svcs=svcs))
    struct.pack_into("<I", raw, 0x14, system_resource_size)
    return npdm_mod.parse(bytes(raw))


def run(tmp_path, source, *, seconds=1.0, **npdm_args):
    p = kernel.Process({"rtld": build_nso(assemble(source, tmp_path), bss=0)}, npdm(**npdm_args))
    log = io.StringIO()
    end = p.run(log, max_seconds=seconds)
    return p, end, log.getvalue()


def test_set_heap_size_grows_and_shrinks_the_heap(tmp_path):
    p, end, _ = run(
        tmp_path,
        """
        mov x1, #0x400000
        svc #0x01
        mov x19, x0
        mov x20, x1
        add x2, x1, #0x3ff000
        ldr x21, [x2]
        mov x1, #0x200000
        svc #0x01
        mov x22, x0
        mov x1, #0x1000
        svc #0x01
        mov x23, x0
        svc #0x07
        """,
    )
    assert end.reason == "exit"
    assert [p.cpu.x(n) for n in (19, 20, 21, 22, 23)] == [0, HEAP, 0, 0, svcs.INVALID_SIZE]
    heap = p.space.query(HEAP)
    assert (heap.size, heap.state, heap.perm) == (0x200000, MemoryState.NORMAL, 3)


def test_query_memory_describes_the_code_and_what_lies_past_the_address_space(tmp_path):
    p, end, _ = run(
        tmp_path,
        """
        sub sp, sp, #0x40
        mov x0, sp
        adr x2, .
        svc #0x06
        ldp x19, x20, [sp]
        ldp w21, w22, [sp, #16]
        ldr w23, [sp, #24]
        mov x0, sp
        mov x2, #0x8000000000
        svc #0x06
        ldp x24, x25, [sp]
        ldr w26, [sp, #16]
        svc #0x07
        """,
    )
    base = p.modules[0].base
    assert [p.cpu.x(n) for n in (19, 20, 21, 22, 23)] == [base, 0x1000, MemoryState.CODE, 0, 5]
    assert [p.cpu.x(n) for n in (24, 25, 26)] == [1 << 39, (1 << 64) - (1 << 39), MemoryState.INACCESSIBLE]


def test_map_memory_shows_heap_memory_in_the_stack_region_and_unmap_returns_it(tmp_path):
    dst = LAYOUT_39BIT.stack.start + 0x1000_0000
    p, end, _ = run(
        tmp_path,
        HEAP_SETUP
        + f"""
        mov x2, #77
        str x2, [x19, #0x1008]
        ldr x0, ={dst:#x}
        mov x1, x19
        mov x2, #0x2000
        svc #0x04
        mov x20, x0
        ldr x3, ={dst:#x}
        ldr x21, [x3, #0x1008]
        mov x4, #88
        str x4, [x3, #0x1010]
        mov x0, #0x1000
        mov x1, x19
        mov x2, #0x2000
        svc #0x04
        mov x22, x0
        ldr x0, ={dst:#x}
        mov x1, x19
        mov x2, #0x2000
        svc #0x05
        mov x23, x0
        ldr x24, [x19, #0x1010]
        svc #0x07
        """,
    )
    assert [p.cpu.x(n) for n in (20, 21, 22, 23, 24)] == [0, 77, svcs.INVALID_MEMORY_REGION, 0, 88]
    src = p.space.query(HEAP)
    assert (src.perm, src.attribute) == (3, 0)
    assert p.space.query(dst).state == MemoryState.FREE


def test_heap_memory_is_locked_while_mapped_elsewhere(tmp_path):
    dst = LAYOUT_39BIT.stack.start + 0x1000_0000
    p, end, _ = run(
        tmp_path,
        HEAP_SETUP
        + f"""
        ldr x0, ={dst:#x}
        mov x1, x19
        mov x2, #0x1000
        svc #0x04
        ldr x2, [x19]
        svc #0x07
        """,
    )
    assert (end.reason, end.stop.kind, end.stop.access) == ("crash", "protected", "read")
    assert p.space.query(HEAP).attribute == MemoryAttribute.LOCKED


def test_set_memory_permission_makes_heap_read_only(tmp_path):
    p, end, _ = run(
        tmp_path,
        HEAP_SETUP
        + """
        mov x0, x19
        mov x1, #0x1000
        mov w2, #1
        svc #0x02
        mov x20, x0
        mov x0, x19
        mov x1, #0x1000
        mov w2, #5
        svc #0x02
        mov x21, x0
        str x2, [x19]
        """,
    )
    assert [p.cpu.x(n) for n in (20, 21)] == [0, svcs.INVALID_NEW_MEMORY_PERMISSION]
    assert (end.reason, end.stop.kind, end.stop.access) == ("crash", "protected", "write")


def test_map_physical_memory_needs_a_system_resource_and_the_alias_region(tmp_path):
    alias = LAYOUT_39BIT.alias.start
    source = f"""
        ldr x0, ={alias:#x}
        mov x1, #0x200000
        svc #0x2c
        mov x19, x0
        cbnz x0, 1f
        mov x2, #5
        ldr x3, ={alias:#x}
        str x2, [x3, #0x100]
        ldr x20, [x3, #0x100]
        ldr x0, ={alias:#x}
        mov x1, #0x200000
        svc #0x2d
        mov x21, x0
    1:  svc #0x07
    """
    p, _, _ = run(tmp_path, source, system_resource_size=0x100000)
    assert [p.cpu.x(n) for n in (19, 20, 21)] == [0, 5, 0]
    assert p.space.query(alias).state == MemoryState.FREE
    p, _, _ = run(tmp_path, source)
    assert p.cpu.x(19) == svcs.INVALID_STATE


THREADS = (
    HEAP_SETUP
    + """
    mov x20, #0
1:  adr x1, worker
    add x2, x19, #16
    add x2, x2, x20, lsl #2
    add x3, x19, #0x100000
    add x3, x3, x20, lsl #16
    add x3, x3, #0x10000
    mov w4, #44
    mov w5, #-2
    svc #0x08
    add x6, x19, #16
    str w1, [x6, x20, lsl #2]
    add x20, x20, #1
    cmp x20, #2
    b.ne 1b
    ldr w0, [x19, #16]
    svc #0x09
    ldr w0, [x19, #20]
    svc #0x09
    add x1, x19, #16
    mov w2, #1
    mov x3, #-1
    svc #0x18
    add x1, x19, #20
    mov w2, #1
    mov x3, #-1
    svc #0x18
    ldr x21, [x19, #8]
    ldr w22, [x19]
    svc #0x07

worker:
    and x23, x0, #0xffffffffffe00000
    ldr w24, [x0]
    mov x25, #50
1:  mov x0, x23
    mov w1, w24
    bl lock
    ldr x26, [x23, #8]
    mov x0, #1000
    svc #0x0b
    add x26, x26, #1
    str x26, [x23, #8]
    mov x0, x23
    mov w1, w24
    bl unlock
    subs x25, x25, #1
    b.ne 1b
    svc #0x0a
"""
    + MUTEX
)


def test_threads_share_a_mutex_and_the_kernel_hands_it_over(tmp_path):
    p, end, log = run(tmp_path, THREADS)
    assert end.reason == "exit", end.message
    assert (p.cpu.x(21), p.cpu.x(22)) == (100, 0)
    assert p.svc_counts["ArbitrateLock"] > 10 and p.svc_counts["ArbitrateUnlock"] > 10
    assert [t.state for t in p.threads] == ["runnable", "exited", "exited"]
    assert "CreateThread(entry=rtld+" in log and "-> ok, thread 1, handle" in log


def test_runs_are_repeatable(tmp_path):
    logs = {run(tmp_path, THREADS)[2] for _ in range(2)}
    assert len(logs) == 1


def test_condition_variable_wakes_a_waiter_and_gives_it_the_mutex(tmp_path):
    p, end, _ = run(
        tmp_path,
        HEAP_SETUP
        + """
        mov x0, x19
        mov w1, w28
        bl lock
        adr x1, worker
        mov x2, x19
        add x3, x19, #0x20000
        mov w4, #44
        mov w5, #0
        svc #0x08
        str w1, [x19, #12]
        mov w0, w1
        svc #0x09
    1:  ldr w20, [x19, #8]
        cbnz w20, 2f
        mov x0, x19
        add x1, x19, #4
        mov w2, w28
        mov x3, #-1
        svc #0x1c
        mov x24, x0
        b 1b
    2:  ldr w21, [x19]
        ldr w22, [x19, #4]
        mov x0, x19
        mov w1, w28
        bl unlock
        ldr w23, [x19]
        svc #0x07

    worker:
        mov x20, x0
        ldr w21, [x20, #12]
        mov x0, x20
        mov w1, w21
        bl lock
        mov w2, #1
        str w2, [x20, #8]
        add x0, x20, #4
        mov w1, #1
        svc #0x1d
        mov x0, x20
        mov w1, w21
        bl unlock
        svc #0x0a
        """
        + MUTEX,
    )
    assert end.reason == "exit", end.message
    main = p.main_thread.handle
    assert [p.cpu.x(n) for n in (20, 21, 22, 23, 24)] == [1, main, 0, 0, 0]


def test_condition_variable_wait_times_out_and_time_moves_on(tmp_path):
    p, end, _ = run(
        tmp_path,
        HEAP_SETUP
        + """
        svc #0x1e
        mov x20, x0
        mov x0, x19
        add x1, x19, #4
        mov w2, w28
        ldr x3, =1000000
        svc #0x1c
        mov x21, x0
        ldr w22, [x19, #4]
        mrs x23, cntpct_el0
        svc #0x07
        """,
    )
    assert (p.cpu.x(21), p.cpu.x(22)) == (svcs.TIMED_OUT, 1)
    assert p.cpu.x(23) - p.cpu.x(20) >= kernel.ns_to_ticks(1_000_000)


def test_address_arbiter_waits_and_signals(tmp_path):
    p, end, _ = run(
        tmp_path,
        HEAP_SETUP
        + """
        adr x1, worker
        mov x2, x19
        add x3, x19, #0x20000
        mov w4, #44
        mov w5, #1
        svc #0x08
        mov w0, w1
        svc #0x09
        mov x0, x19
        mov w1, #2
        mov w2, #0
        mov x3, #-1
        svc #0x34
        mov x20, x0
        ldr w21, [x19]
        mov x0, x19
        mov w1, #0
        mov w2, #1
        mov x3, #-1
        svc #0x34
        mov x22, x0
        mov x0, x19
        mov w1, #1
        mov w2, #1
        mov w3, #1
        svc #0x35
        mov x23, x0
        ldr w24, [x19]
        mov x0, x19
        mov w1, #1
        mov w2, #9
        mov w3, #1
        svc #0x35
        mov x25, x0
        add x0, x19, #8
        mov w1, #1
        mov w2, #1
        mov x3, #0
        svc #0x34
        mov x26, x0
        ldr w27, [x19, #8]
        svc #0x07

    worker:
        mov x20, x0
        ldr x0, =1000000
        svc #0x0b
        mov x0, x20
        mov w1, #1
        str w1, [x0]
        mov w1, #0
        mov w2, #0
        mov w3, #1
        svc #0x35
        svc #0x0a
        """,
    )
    assert end.reason == "exit", end.message
    assert [p.cpu.x(n) for n in (20, 21, 22, 23, 24, 25, 26)] == [
        0,
        1,
        svcs.INVALID_STATE,
        0,
        2,
        svcs.INVALID_STATE,
        svcs.TIMED_OUT,
    ]
    assert p.cpu.x(27) == 0xFFFF_FFFF  # decremented from 0 before timing out


def test_sleep_and_the_counter_move_virtual_time(tmp_path):
    p, end, _ = run(
        tmp_path,
        """
        mrs x19, cntpct_el0
        mov x0, #0x4240
        movk x0, #0xf, lsl #16
        svc #0x0b
        mrs x20, cntpct_el0
        mrs x21, cntvct_el0
        svc #0x1e
        mov x22, x0
        svc #0x07
        """,
    )
    assert p.counter_reads == 3
    assert p.cpu.x(20) - p.cpu.x(19) >= kernel.ns_to_ticks(1_000_000)
    assert p.cpu.x(19) < p.cpu.x(20) < p.cpu.x(21) < p.cpu.x(22)


def test_thread_priority_and_core_svcs(tmp_path):
    p, end, _ = run(
        tmp_path,
        """
        mov x19, x1
        mov w0, w19
        mov w1, #50
        svc #0x0d
        mov w1, w19
        svc #0x0c
        mov x20, x1
        mov w0, w19
        mov w1, #2
        mov x2, #0xc
        svc #0x0f
        mov x21, x0
        svc #0x10
        mov x22, x0
        mov w2, w19
        svc #0x0e
        mov x23, x1
        mov x24, x2
        mov w4, #10
        mov w5, #0
        svc #0x08
        mov x25, x0
        mov w4, #44
        mov w5, #7
        svc #0x08
        mov x26, x0
        mov w0, w19
        mov w1, #0
        mov x2, #0x2
        svc #0x0f
        mov x27, x0
        svc #0x07
        """,
    )
    assert [p.cpu.x(n) for n in (20, 21, 22, 23, 24)] == [50, 0, 2, 2, 0xC]
    assert [p.cpu.x(n) for n in (25, 26, 27)] == [svcs.INVALID_PRIORITY, svcs.INVALID_CORE_ID, svcs.INVALID_COMBINATION]


def test_get_info_reports_the_layout_and_checks_handles(tmp_path):
    p, end, _ = run(
        tmp_path,
        """
        mov w1, #4
        mov w2, #0x8001
        movk w2, #0xffff, lsl #16
        mov x3, #0
        svc #0x29
        mov x19, x1
        mov w1, #11
        mov w2, #0
        mov x3, #3
        svc #0x29
        mov x20, x0
        mov w1, #11
        mov w2, #0x8001
        movk w2, #0xffff, lsl #16
        svc #0x29
        mov x21, x0
        mov w1, #33
        mov w2, #0x8001
        movk w2, #0xffff, lsl #16
        mov x3, #0
        svc #0x29
        mov x22, x1
        mov w1, #99
        svc #0x29
        """,
    )
    assert [p.cpu.x(n) for n in (19, 20, 21, 22)] == [HEAP, 0, svcs.INVALID_HANDLE, 0x1000]
    assert (end.reason, end.message) == ("unimplemented", "GetInfo type 99 (sub 0x0) is not implemented")


def test_break_ends_the_run_unless_it_only_notifies(tmp_path):
    p, end, log = run(
        tmp_path,
        """
        ldr x0, =0x80000004
        mov x1, #0
        mov x2, #0
        svc #0x26
        adr x0, message
        mov x1, #5
        svc #0x27
        mov x0, #1
        sub sp, sp, #16
        mov w2, #0x2a
        str w2, [sp]
        mov x1, sp
        mov x2, #4
        svc #0x26
    message:
        .ascii "hello"
        """,
    )
    assert (end.reason, end.message) == ("break", "the game called Break: assert, data 2a000000")
    assert "OutputDebugString('hello') -> ok" in log
    assert "Break(reason=0x80000004 post-load DLL" in log


def test_waiting_for_a_thread_that_never_starts_is_a_deadlock(tmp_path):
    p, end, _ = run(
        tmp_path,
        HEAP_SETUP
        + """
        adr x1, worker
        mov x2, #0
        add x3, x19, #0x10000
        mov w4, #44
        mov w5, #0
        svc #0x08
        str w1, [x19]
        mov x1, x19
        mov w2, #1
        mov x3, #-1
        svc #0x18
    worker:
        svc #0x0a
        """,
    )
    assert end.reason == "deadlock"


def test_cancel_synchronization_cancels_the_next_wait(tmp_path):
    p, end, _ = run(
        tmp_path,
        """
        mov x19, x1
        mov w0, w19
        svc #0x19
        sub sp, sp, #16
        str w19, [sp]
        mov x1, sp
        mov w2, #1
        mov x3, #-1
        svc #0x18
        mov x20, x0
        mov x1, sp
        mov w2, #1
        mov x3, #0
        svc #0x18
        mov x21, x0
        svc #0x07
        """,
    )
    assert (p.cpu.x(20), p.cpu.x(21)) == (svcs.CANCELLED, svcs.TIMED_OUT)


def test_a_crash_and_svcs_the_npdm_forbids_end_the_run(tmp_path):
    _, end, _ = run(tmp_path, "nop\nudf #0")
    assert (end.reason, end.stop.reason) == ("crash", "undefined")
    _, end, _ = run(tmp_path, "svc #0x01", svcs=(0x07,))
    assert (end.reason, end.message) == ("crash", "the game called SetHeapSize, which its NPDM does not allow")
    _, end, _ = run(tmp_path, "svc #0x7f")
    assert (end.reason, end.message) == ("unimplemented", "SVC 0x7F (CallSecureMonitor) is not implemented")


def test_busy_threads_on_two_cores_both_make_progress(tmp_path):
    p, end, _ = run(
        tmp_path,
        HEAP_SETUP
        + """
        adr x1, worker
        mov x2, x19
        add x3, x19, #0x10000
        mov w4, #44
        mov w5, #1
        svc #0x08
        mov w0, w1
        svc #0x09
    1:  ldr x2, [x19]
        cbz x2, 1b
        svc #0x07
    worker:
        mov x1, #1
        str x1, [x0]
    2:  b 2b
        """,
        seconds=0.1,
    )
    assert end.reason == "exit"


def _bytes(data: bytes) -> str:
    return ".byte " + ", ".join(str(b) for b in data)


def test_a_program_reaches_a_service_and_maps_its_shared_memory(tmp_path):
    from siicl import ipc

    get_time = ipc.build(type=ipc.TIPC_FIRST + 1, data=b"time:u".ljust(8, b"\0"), align_data=False)
    get_shared = ipc.build(type=4, data=struct.pack("<IIII", ipc.CMIF_IN, 0, 20, 0))
    shared_at = LAYOUT_39BIT.code.start + 0x4000_0000
    p, end, log = run(
        tmp_path,
        f"""
        adr x1, port
        svc #0x1f
        mov w19, w1
        mrs x20, tpidrro_el0
        adr x1, get_time
        mov x2, #{len(get_time)}
        bl copy
        mov w0, w19
        svc #0x21
        ldr w21, [x20, #12]
        ldr w22, [x20, #16]
        adr x1, get_shared
        mov x2, #{len(get_shared)}
        bl copy
        mov w0, w21
        svc #0x21
        ldr w23, [x20, #12]
        mov w0, w23
        ldr x1, ={shared_at:#x}
        mov x2, #0x1000
        mov w3, #1
        svc #0x13
        mov x24, x0
        ldr x2, ={shared_at:#x}
        ldr x25, [x2, #0x100]
        svc #0x07
    copy:
        ldrb w3, [x1], #1
        strb w3, [x20], #1
        subs x2, x2, #1
        b.ne copy
        mrs x20, tpidrro_el0
        ret
    port:
        .asciz "sm:"
        .balign 4
    get_time:
        {_bytes(get_time)}
        .balign 4
    get_shared:
        {_bytes(get_shared)}
        """,
    )
    assert end.reason == "exit", end.message
    assert p.cpu.x(22) == 0  # the TIPC result
    assert p.cpu.x(24) == 0
    from siicl.services import time

    assert p.cpu.x(25).to_bytes(8, "little") == time.CLOCK_SOURCE_ID[:8]
    assert "sm: GetServiceHandle('time:u') -> time" in log
    assert "MapSharedMemory(" in log and "-> ok, time" in log


def test_events_signal_waiters_and_reset(tmp_path):
    event = kernel.Event("test")
    p = kernel.Process(
        {
            "rtld": build_nso(
                assemble(
                    """
        sub sp, sp, #16
        mov w0, #1
        movk w0, #1, lsl #16
        str w0, [sp]
        mov x1, sp
        mov w2, #1
        mov x3, #0
        svc #0x18
        mov x19, x0
        mov w0, #1
        movk w0, #1, lsl #16
        svc #0x17
        mov x20, x0
        mov x1, sp
        mov w2, #1
        mov x3, #0
        svc #0x18
        mov x21, x0
        mov w0, #1
        movk w0, #1, lsl #16
        svc #0x17
        mov x22, x0
        svc #0x07
        """,
                    tmp_path,
                ),
                bss=0,
            )
        },
        npdm(),
    )
    assert p.handles.add(event.readable) == 0x10001
    event.signal(p)
    end = p.run(io.StringIO(), max_seconds=1)
    assert end.reason == "exit"
    assert [p.cpu.x(n) for n in (19, 20, 21, 22)] == [0, 0, svcs.TIMED_OUT, svcs.INVALID_STATE]
