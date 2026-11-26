"""The SVCs siicl handles, as the Horizon kernel does them.

`call()` runs the handler for an SVC that stopped the CPU and logs the call. Each handler reads its
arguments from the registers, writes its results back, and returns the text for the log. Results
go in W0: 0 for success, otherwise a kernel result code. An SVC without a handler ends the run,
so the boot log shows the next thing to write.
"""

from __future__ import annotations

import hashlib
import struct
from collections.abc import Callable

from siicl import ipc, services
from siicl.addrspace import AddressSpaceError, MemoryAttribute, MemoryState, perm_text
from siicl.cpu import PAGE, READ, WRITE, Stop
from siicl.kernel import (
    COUNTER_SVC,
    PROCESS_ID,
    End,
    Event,
    Process,
    ReadableEvent,
    SharedMemory,
    SyncObject,
    Thread,
    TransferMemory,
    Unimplemented,
)
from siicl.svc import SVC_NAMES

OK = 0
INVALID_SIZE = 0xCA01
INVALID_ADDRESS = 0xCC01
OUT_OF_MEMORY = 0xD001
INVALID_CURRENT_MEMORY = 0xD401
INVALID_NEW_MEMORY_PERMISSION = 0xD801
INVALID_MEMORY_REGION = 0xDC01
INVALID_PRIORITY = 0xE001
INVALID_CORE_ID = 0xE201
INVALID_HANDLE = 0xE401
INVALID_COMBINATION = 0xE801
TIMED_OUT = 0xEA01
CANCELLED = 0xEC01
OUT_OF_RANGE = 0xEE01
INVALID_ENUM_VALUE = 0xF001
INVALID_STATE = 0xFA01

RESULT_NAMES = {
    INVALID_SIZE: "invalid size",
    INVALID_ADDRESS: "invalid address",
    OUT_OF_MEMORY: "out of memory",
    INVALID_CURRENT_MEMORY: "invalid current memory",
    INVALID_NEW_MEMORY_PERMISSION: "invalid new memory permission",
    INVALID_MEMORY_REGION: "invalid memory region",
    INVALID_PRIORITY: "invalid priority",
    INVALID_CORE_ID: "invalid core ID",
    INVALID_HANDLE: "invalid handle",
    INVALID_COMBINATION: "invalid combination",
    TIMED_OUT: "timed out",
    CANCELLED: "cancelled",
    OUT_OF_RANGE: "out of range",
    INVALID_ENUM_VALUE: "invalid enum value",
    INVALID_STATE: "invalid state",
}

HANDLE_WAIT_MASK = 0x4000_0000
HEAP_ALIGN = 0x20_0000
TOTAL_MEMORY = 0x2_4000_0000  # the memory GetInfo says the process has: siicl's choice, 9 GiB
RW = READ | WRITE

Handler = Callable[[Process, Thread], "str | End"]
_HANDLERS: dict[int, Handler] = {}


def handles(number: int) -> Callable[[Handler], Handler]:
    def register(fn: Handler) -> Handler:
        _HANDLERS[number] = fn
        return fn

    return register


def implemented() -> set[int]:
    return set(_HANDLERS)


def call(p: Process, t: Thread, stop: Stop) -> End | None:
    number = stop.svc
    if number & 0xFFE0 == COUNTER_SVC:
        if number & 0x1F != 31:
            p.cpu.set_x(number & 0x1F, p.ticks)
        p.ticks += 1
        return None
    name = SVC_NAMES.get(number, f"SVC 0x{number:02X}")
    p.svc_counts[name] += 1
    if p.allowed_svcs is not None and number not in p.allowed_svcs:
        return End("crash", f"the game called {name}, which its NPDM does not allow", t, stop)
    handler = _HANDLERS.get(number)
    if handler is None:
        raise Unimplemented(f"SVC 0x{number:02X} ({name}) is not implemented")
    out = handler(p, t)
    if isinstance(out, End):
        return out
    p.trace(t, f"{name}{out}")
    return None


def time_out(p: Process, t: Thread) -> None:
    """End a wait whose deadline has passed."""
    if t.wait_kind == "sleep":
        p.wake(t, {})
        return
    if t.wait_kind == "sync":
        t.stop_sync_wait()
    elif t.wait_kind in ("condvar", "mutex"):
        if t.lock_owner is not None:
            t.lock_owner.mutex_waiters.remove(t)
            t.lock_owner = None
        elif t in p.condvar_waiters:
            p.condvar_waiters.remove(t)
        t.condvar_key = None
    elif t.wait_kind == "arbiter":
        p.arbiter_waiters.remove(t)
        t.arbiter_addr = None
    p.wake(t, {0: TIMED_OUT})


# Registers and results


def _w(p: Process, n: int) -> int:
    return p.cpu.x(n) & 0xFFFF_FFFF


def _s32(v: int) -> int:
    v &= 0xFFFF_FFFF
    return v - (1 << 32) if v & 0x8000_0000 else v


def _s64(v: int) -> int:
    return v - (1 << 64) if v >> 63 else v


def describe_result(code: int) -> str:
    return "ok" if code == OK else f"{code:#x} ({RESULT_NAMES.get(code, 'error')})"


def _result(p: Process, code: int, extra: str = "") -> str:
    p.cpu.set_x(0, code)
    return f" -> {describe_result(code)}{extra}"


def _pages(address: int, size: int) -> int | None:
    """The result for a range that is not page-aligned or empty, or None for a good one."""
    if address % PAGE:
        return INVALID_ADDRESS
    if size % PAGE or size == 0:
        return INVALID_SIZE
    return None


def _blocks(p: Process, address: int, size: int):
    try:
        return p.space.range(address, size)
    except AddressSpaceError:
        return None


# Memory


@handles(0x01)
def set_heap_size(p: Process, t: Thread) -> str:
    size = p.cpu.x(1)
    args = f"(size={size:#x})"
    heap = p.space.layout.heap
    if size % HEAP_ALIGN or size > heap.size:
        return args + _result(p, INVALID_SIZE)
    current = p.heap_size
    if size > current:
        if used_memory(p) + size - current > TOTAL_MEMORY:
            return args + _result(p, OUT_OF_MEMORY)
        p.space.map(heap.start + current, size - current, RW, MemoryState.NORMAL, "heap")
    elif size < current:
        tail = _blocks(p, heap.start + size, current - size)
        if tail is None or any((b.state, b.perm, b.attribute) != (MemoryState.NORMAL, RW, 0) for b in tail):
            return args + _result(p, INVALID_CURRENT_MEMORY)
        p.space.unmap(heap.start + size, current - size)
    p.heap_size = size
    p.cpu.set_x(1, heap.start)
    return args + _result(p, OK, f", heap at {heap.start:#x}")


# Memory states SetMemoryPermission may change: the heap, and the .data and .bss of the modules, which
# the libraries make partly read-only after relocating them.
_REPROTECTABLE = {MemoryState.NORMAL, MemoryState.CODE_DATA, MemoryState.ALIAS_CODE_DATA}


@handles(0x02)
def set_memory_permission(p: Process, t: Thread) -> str:
    address, size, perm = p.cpu.x(0), p.cpu.x(1), _w(p, 2)
    args = f"(address={address:#x}, size={size:#x}, perm={perm_text(perm)})"
    if (bad := _pages(address, size)) is not None:
        return args + _result(p, bad)
    if perm not in (0, READ, RW):
        return args + _result(p, INVALID_NEW_MEMORY_PERMISSION)
    blocks = _blocks(p, address, size)
    if blocks is None or any(b.state not in _REPROTECTABLE or b.attribute for b in blocks):
        return args + _result(p, INVALID_CURRENT_MEMORY)
    p.space.update(address, size, perm=perm)
    return args + _result(p, OK)


_ATTRIBUTE_STATES = {MemoryState.NORMAL, MemoryState.CODE_DATA, MemoryState.ALIAS_CODE_DATA}


@handles(0x03)
def set_memory_attribute(p: Process, t: Thread) -> str:
    address, size, mask, value = p.cpu.x(0), p.cpu.x(1), _w(p, 2), _w(p, 3)
    args = f"(address={address:#x}, size={size:#x}, mask={mask:#x}, value={value:#x})"
    if (bad := _pages(address, size)) is not None:
        return args + _result(p, bad)
    supported = MemoryAttribute.UNCACHED | MemoryAttribute.PERMISSION_LOCKED
    if (mask | value) != mask or (mask | value) & ~supported:
        return args + _result(p, INVALID_COMBINATION)
    blocks = _blocks(p, address, size)
    if blocks is None or any(b.state not in _ATTRIBUTE_STATES for b in blocks):
        return args + _result(p, INVALID_CURRENT_MEMORY)
    for b in blocks:
        p.space.update(b.start, b.size, attribute=(b.attribute & ~mask) | value)
    return args + _result(p, OK)


# Memory states MapMemory may take pages from: the heap, and the .data and .bss of the modules.
_ALIASABLE = {MemoryState.NORMAL, MemoryState.CODE_DATA, MemoryState.ALIAS_CODE_DATA}


@handles(0x04)
def map_memory(p: Process, t: Thread) -> str:
    dst, src, size = p.cpu.x(0), p.cpu.x(1), p.cpu.x(2)
    args = f"(dst={dst:#x}, src={src:#x}, size={size:#x})"
    if (bad := _pages(dst, size)) is not None or (bad := _pages(src, size)) is not None:
        return args + _result(p, bad)
    if not p.space.layout.stack.contains(dst, size):
        return args + _result(p, INVALID_MEMORY_REGION)
    blocks = _blocks(p, src, size)
    if not p.space.is_free(dst, size) or blocks is None:
        return args + _result(p, INVALID_CURRENT_MEMORY)
    if any(b.state not in _ALIASABLE or (b.perm, b.attribute) != (RW, 0) for b in blocks):
        return args + _result(p, INVALID_CURRENT_MEMORY)
    for b in blocks:
        p.space.map(
            dst + b.start - src, b.size, RW, MemoryState.STACK, f"alias of {b.name}", memory=b.memory, offset=b.offset
        )
    p.space.update(src, size, perm=0, attribute=MemoryAttribute.LOCKED)
    return args + _result(p, OK)


@handles(0x05)
def unmap_memory(p: Process, t: Thread) -> str:
    dst, src, size = p.cpu.x(0), p.cpu.x(1), p.cpu.x(2)
    args = f"(dst={dst:#x}, src={src:#x}, size={size:#x})"
    if (bad := _pages(dst, size)) is not None or (bad := _pages(src, size)) is not None:
        return args + _result(p, bad)
    dst_blocks, src_blocks = _blocks(p, dst, size), _blocks(p, src, size)
    if dst_blocks is None or src_blocks is None:
        return args + _result(p, INVALID_CURRENT_MEMORY)
    if any(b.state != MemoryState.STACK for b in dst_blocks) or any(
        b.attribute != MemoryAttribute.LOCKED for b in src_blocks
    ):
        return args + _result(p, INVALID_CURRENT_MEMORY)
    for b in dst_blocks:
        s = p.space.query(src + b.start - dst)
        if s.host_address + (src + b.start - dst - s.start) != b.host_address:
            return args + _result(p, INVALID_MEMORY_REGION)
    p.space.unmap(dst, size)
    p.space.update(src, size, perm=RW, attribute=0)
    return args + _result(p, OK)


@handles(0x06)
def query_memory(p: Process, t: Thread) -> str:
    info, address = p.cpu.x(0), p.cpu.x(2)
    b = p.space.query(address)
    p.cpu.write(info, struct.pack("<QQIIIIII", b.start, b.size, b.state, b.attribute, b.perm, 0, 0, 0))
    p.cpu.set_x(1, 0)
    return f"(address={address:#x})" + _result(
        p, OK, f": {b.start:#x}+{b.size:#x} {b.state.name.lower()} {perm_text(b.perm)}"
    )


@handles(0x2C)
def map_physical_memory(p: Process, t: Thread) -> str:
    address, size = p.cpu.x(0), p.cpu.x(1)
    args = f"(address={address:#x}, size={size:#x})"
    if (bad := _pages(address, size)) is not None:
        return args + _result(p, bad)
    if not p.space.layout.alias.contains(address, size):
        return args + _result(p, INVALID_MEMORY_REGION)
    if not (p.npdm and p.npdm.system_resource_size):
        return args + _result(p, INVALID_STATE)
    gaps = p.space.free_ranges(address, size)
    mapped = size - sum(g.size for g in gaps)
    if mapped and any(
        b.state != MemoryState.NORMAL for b in p.space.blocks if b.start < address + size and b.end > address
    ):
        return args + _result(p, INVALID_CURRENT_MEMORY)
    if used_memory(p) + size - mapped > TOTAL_MEMORY:
        return args + _result(p, OUT_OF_MEMORY)
    for g in gaps:
        p.space.map(g.start, g.size, RW, MemoryState.NORMAL, "physical memory")
    return args + _result(p, OK)


@handles(0x2D)
def unmap_physical_memory(p: Process, t: Thread) -> str:
    address, size = p.cpu.x(0), p.cpu.x(1)
    args = f"(address={address:#x}, size={size:#x})"
    if (bad := _pages(address, size)) is not None:
        return args + _result(p, bad)
    if not p.space.layout.alias.contains(address, size):
        return args + _result(p, INVALID_MEMORY_REGION)
    inside = [b for b in p.space.blocks if b.start < address + size and b.end > address]
    if any(b.state != MemoryState.NORMAL or b.attribute for b in inside):
        return args + _result(p, INVALID_CURRENT_MEMORY)
    for b in inside:
        start, end = max(b.start, address), min(b.end, address + size)
        p.space.unmap(start, end - start)
    return args + _result(p, OK)


def used_memory(p: Process) -> int:
    return sum(b.size for b in p.space.blocks if not b.name.startswith("alias"))


# Threads


@handles(0x08)
def create_thread(p: Process, t: Thread) -> str:
    entry, arg, stack_top, priority, core = p.cpu.x(1), p.cpu.x(2), p.cpu.x(3), _w(p, 4), _s32(p.cpu.x(5))
    args = (
        f"(entry={p.describe_address(entry)}, arg={arg:#x}, stack_top={stack_top:#x}, priority={priority}, core={core})"
    )
    if core == -2:
        core = p.ideal_core
    if core not in p.cores:
        return args + _result(p, INVALID_CORE_ID)
    highest, lowest = p.priorities
    if not highest <= priority <= lowest:
        return args + _result(p, INVALID_PRIORITY)
    thread = p.new_thread(entry=entry, arg=arg, stack_top=stack_top, priority=priority, core=core)
    p.cpu.set_x(1, thread.handle)
    return args + _result(p, OK, f", thread {thread.number}, handle {thread.handle:#x}")


@handles(0x09)
def start_thread(p: Process, t: Thread) -> str:
    handle = _w(p, 0)
    thread = p.thread_for(handle)
    args = f"(handle={handle:#x})"
    if thread is None:
        return args + _result(p, INVALID_HANDLE)
    if thread.state != "created":
        return args + _result(p, INVALID_STATE)
    p.start_thread(thread)
    return args + _result(p, OK, f", thread {thread.number}")


@handles(0x0A)
def exit_thread(p: Process, t: Thread) -> str:
    p.exit_thread(t)
    return "()"


@handles(0x0B)
def sleep_thread(p: Process, t: Thread) -> str:
    ns = _s64(p.cpu.x(0))
    if ns <= 0:
        p.yield_thread(t)
    else:
        p.block(t, "sleep", ns)
    return f"(ns={ns})"


@handles(0x0C)
def get_thread_priority(p: Process, t: Thread) -> str:
    handle = _w(p, 1)
    thread = p.thread_for(handle)
    if thread is None:
        return f"(handle={handle:#x})" + _result(p, INVALID_HANDLE)
    p.cpu.set_x(1, thread.priority)
    return f"(handle={handle:#x})" + _result(p, OK, f", priority {thread.priority}")


@handles(0x0D)
def set_thread_priority(p: Process, t: Thread) -> str:
    handle, priority = _w(p, 0), _w(p, 1)
    args = f"(handle={handle:#x}, priority={priority})"
    thread = p.thread_for(handle)
    if thread is None:
        return args + _result(p, INVALID_HANDLE)
    if not p.priorities[0] <= priority <= p.priorities[1]:
        return args + _result(p, INVALID_PRIORITY)
    thread.priority = priority
    return args + _result(p, OK)


@handles(0x0E)
def get_thread_core_mask(p: Process, t: Thread) -> str:
    handle = _w(p, 2)
    thread = p.thread_for(handle)
    if thread is None:
        return f"(handle={handle:#x})" + _result(p, INVALID_HANDLE)
    p.cpu.set_x(1, thread.core)
    p.cpu.set_x(2, thread.affinity)
    return f"(handle={handle:#x})" + _result(p, OK, f", core {thread.core}, mask {thread.affinity:#x}")


@handles(0x0F)
def set_thread_core_mask(p: Process, t: Thread) -> str:
    handle, core, mask = _w(p, 0), _s32(p.cpu.x(1)), p.cpu.x(2)
    args = f"(handle={handle:#x}, core={core}, mask={mask:#x})"
    thread = p.thread_for(handle)
    if thread is None:
        return args + _result(p, INVALID_HANDLE)
    allowed = sum(1 << c for c in p.cores)
    if core == -2:
        core, mask = p.ideal_core, 1 << p.ideal_core
    else:
        if mask == 0 or mask & ~allowed:
            return args + _result(p, INVALID_CORE_ID)
        if core >= 0:
            if not mask & (1 << core):
                return args + _result(p, INVALID_COMBINATION)
        elif core in (-1, -3):
            core = thread.core if core == -3 and mask & (1 << thread.core) else (mask & -mask).bit_length() - 1
        else:
            return args + _result(p, INVALID_CORE_ID)
    thread.affinity = mask
    p.move_to_core(thread, core)
    return args + _result(p, OK)


@handles(0x10)
def get_current_processor_number(p: Process, t: Thread) -> str:
    p.cpu.set_x(0, t.core)
    return f"() -> {t.core}"


@handles(0x25)
def get_thread_id(p: Process, t: Thread) -> str:
    handle = _w(p, 1)
    thread = p.thread_for(handle)
    if thread is None:
        return f"(handle={handle:#x})" + _result(p, INVALID_HANDLE)
    p.cpu.set_x(1, thread.id)
    return f"(handle={handle:#x})" + _result(p, OK, f", id {thread.id:#x}")


@handles(0x24)
def get_process_id(p: Process, t: Thread) -> str:
    handle = _w(p, 1)
    if not isinstance(p.object_for(handle), Process | Thread):
        return f"(handle={handle:#x})" + _result(p, INVALID_HANDLE)
    p.cpu.set_x(1, PROCESS_ID)
    return f"(handle={handle:#x})" + _result(p, OK, f", id {PROCESS_ID:#x}")


@handles(0x16)
def close_handle(p: Process, t: Thread) -> str:
    handle = _w(p, 0)
    try:
        obj = p.handles.get(handle)
        p.handles.close(handle)
    except KeyError:
        return f"(handle={handle:#x})" + _result(p, INVALID_HANDLE)
    if isinstance(obj, TransferMemory) and not obj.kept:  # the lent pages come back
        p.space.update(obj.address, obj.size, perm=RW, attribute=0)
    return f"(handle={handle:#x})" + _result(p, OK)


# Synchronization


@handles(0x18)
def wait_synchronization(p: Process, t: Thread) -> str:
    pointer, count, timeout = p.cpu.x(1), _s32(p.cpu.x(2)), _s64(p.cpu.x(3))
    if not 0 <= count <= 0x40:
        return f"(count={count})" + _result(p, OUT_OF_RANGE)
    handles_ = struct.unpack(f"<{count}I", p.cpu.read(pointer, 4 * count)) if count else ()
    args = f"(handles=[{', '.join(f'{h:#x}' for h in handles_)}], timeout={timeout})"
    objects = [p.object_for(h) for h in handles_]
    if not all(isinstance(o, SyncObject) for o in objects):
        return args + _result(p, INVALID_HANDLE)
    if t.cancel_pending:
        t.cancel_pending = False
        return args + _result(p, CANCELLED)
    for i, o in enumerate(objects):
        if o.is_signaled():
            p.cpu.set_x(1, i)
            return args + _result(p, OK, f", index {i}")
    if timeout == 0:
        return args + _result(p, TIMED_OUT)
    t.sync_objects = objects
    for o in objects:
        o.sync_waiters.append(t)
    p.block(t, "sync", timeout)
    return args + " -> waiting"


@handles(0x19)
def cancel_synchronization(p: Process, t: Thread) -> str:
    handle = _w(p, 0)
    thread = p.thread_for(handle)
    if thread is None:
        return f"(handle={handle:#x})" + _result(p, INVALID_HANDLE)
    if thread.state == "waiting" and thread.wait_kind == "sync":
        thread.stop_sync_wait()
        p.wake(thread, {0: CANCELLED})
    else:
        thread.cancel_pending = True
    return f"(handle={handle:#x})" + _result(p, OK)


def _release_mutex(p: Process, owner: Thread, address: int) -> int:
    """Hand the mutex at address to the highest-priority thread waiting for it; return the new mutex word."""
    waiters = [w for w in owner.mutex_waiters if w.mutex_addr == address]
    if not waiters:
        return 0
    successor = min(waiters, key=lambda w: w.priority)
    others = [w for w in waiters if w is not successor]
    owner.mutex_waiters = [w for w in owner.mutex_waiters if w.mutex_addr != address]
    for w in others:
        w.lock_owner = successor
        successor.mutex_waiters.append(w)
    successor.lock_owner = None
    p.wake(successor, {0: OK})
    return successor.tag | (HANDLE_WAIT_MASK if others else 0)


@handles(0x1A)
def arbitrate_lock(p: Process, t: Thread) -> str:
    owner_handle, address, tag = _w(p, 0), p.cpu.x(1), _w(p, 2)
    args = f"(owner={owner_handle:#x}, address={address:#x}, tag={tag:#x})"
    if address % 4:
        return args + _result(p, INVALID_ADDRESS)
    if p.cpu.read_u32(address) != owner_handle | HANDLE_WAIT_MASK:
        return args + _result(p, OK, ", already released")
    owner = p.thread_for(owner_handle)
    if owner is None:
        return args + _result(p, INVALID_HANDLE)
    t.mutex_addr, t.tag, t.lock_owner = address, tag, owner
    owner.mutex_waiters.append(t)
    p.block(t, "mutex")
    return args + f" -> waiting for thread {owner.number}"


@handles(0x1B)
def arbitrate_unlock(p: Process, t: Thread) -> str:
    address = p.cpu.x(0)
    if address % 4:
        return f"(address={address:#x})" + _result(p, INVALID_ADDRESS)
    value = _release_mutex(p, t, address)
    p.cpu.write_u32(address, value)
    return f"(address={address:#x})" + _result(p, OK, f", mutex word {value:#x}")


@handles(0x1C)
def wait_process_wide_key_atomic(p: Process, t: Thread) -> str:
    mutex, key, tag, timeout = p.cpu.x(0), p.cpu.x(1), _w(p, 2), _s64(p.cpu.x(3))
    args = f"(mutex={mutex:#x}, key={key:#x}, tag={tag:#x}, timeout={timeout})"
    if mutex % 4:
        return args + _result(p, INVALID_ADDRESS)
    p.cpu.write_u32(mutex, _release_mutex(p, t, mutex))
    p.cpu.write_u32(key, 1)
    if timeout == 0:
        return args + _result(p, TIMED_OUT)
    t.mutex_addr, t.tag, t.condvar_key = mutex, tag, key
    p.condvar_waiters.append(t)
    p.block(t, "condvar", timeout)
    return args + " -> waiting"


@handles(0x1D)
def signal_process_wide_key(p: Process, t: Thread) -> str:
    key, count = p.cpu.x(0), _s32(p.cpu.x(1))
    waiters = sorted((w for w in p.condvar_waiters if w.condvar_key == key), key=lambda w: w.priority)
    chosen = waiters if count <= 0 else waiters[:count]
    for w in chosen:
        p.condvar_waiters.remove(w)
        w.condvar_key = None
        _acquire_for(p, w)
    if not any(w.condvar_key == key for w in p.condvar_waiters):
        p.cpu.write_u32(key, 0)
    return f"(key={key:#x}, count={count}) -> woke {len(chosen)}"


def _acquire_for(p: Process, w: Thread) -> None:
    """A thread woken from a condition variable takes its mutex back, or waits for its owner."""
    value = p.cpu.read_u32(w.mutex_addr)
    if value == 0:
        p.cpu.write_u32(w.mutex_addr, w.tag)
        p.wake(w, {0: OK})
        return
    p.cpu.write_u32(w.mutex_addr, value | HANDLE_WAIT_MASK)
    owner = p.thread_for(value & ~HANDLE_WAIT_MASK)
    if owner is None:
        p.wake(w, {0: INVALID_STATE})
        return
    w.lock_owner = owner
    owner.mutex_waiters.append(w)
    w.wait_kind = "mutex"


ARBITRATION_TYPES = ("wait if less than", "decrement and wait if less than", "wait if equal", "wait if equal (64-bit)")
SIGNAL_TYPES = ("signal", "signal and increment if equal", "signal and modify by waiting count if equal")


@handles(0x34)
def wait_for_address(p: Process, t: Thread) -> str:
    address, kind, timeout = p.cpu.x(0), _w(p, 1), _s64(p.cpu.x(3))
    wide = kind == 3
    value = p.cpu.x(2) if wide else _s32(p.cpu.x(2))
    args = f"(address={address:#x}, type={kind}, value={value:#x}, timeout={timeout})"
    if kind >= len(ARBITRATION_TYPES):
        return args + _result(p, INVALID_ENUM_VALUE)
    if address % (8 if wide else 4):
        return args + _result(p, INVALID_ADDRESS)
    current = p.cpu.read_u64(address) if wide else _s32(p.cpu.read_u32(address))
    if kind in (0, 1) and not current < value or kind in (2, 3) and current != value:
        return args + _result(p, INVALID_STATE)
    if kind == 1:
        p.cpu.write_u32(address, current - 1)
    if timeout == 0:
        return args + _result(p, TIMED_OUT)
    t.arbiter_addr = address
    p.arbiter_waiters.append(t)
    p.block(t, "arbiter", timeout)
    return args + " -> waiting"


@handles(0x35)
def signal_to_address(p: Process, t: Thread) -> str:
    address, kind, value, count = p.cpu.x(0), _w(p, 1), _s32(p.cpu.x(2)), _s32(p.cpu.x(3))
    args = f"(address={address:#x}, type={kind}, value={value:#x}, count={count})"
    if kind >= len(SIGNAL_TYPES):
        return args + _result(p, INVALID_ENUM_VALUE)
    if address % 4:
        return args + _result(p, INVALID_ADDRESS)
    waiters = sorted((w for w in p.arbiter_waiters if w.arbiter_addr == address), key=lambda w: w.priority)
    if kind > 0:
        if _s32(p.cpu.read_u32(address)) != value:
            return args + _result(p, INVALID_STATE)
        if kind == 1:
            new = value + 1
        elif count <= 0:
            new = value - 2 if waiters else value + 1
        elif waiters:
            extra = 0
            for _ in waiters[1:]:
                before, extra = extra, extra + 1
                if before >= count:
                    break
            new = value - 1 if extra < count else value
        else:
            new = value + 1
        p.cpu.write_u32(address, new)
    chosen = waiters if count <= 0 else waiters[:count]
    for w in chosen:
        p.arbiter_waiters.remove(w)
        w.arbiter_addr = None
        p.wake(w, {0: OK})
    return args + _result(p, OK, f", woke {len(chosen)}")


@handles(0x36)
def synchronize_preemption_state(p: Process, t: Thread) -> str:
    return "()"


# Time, information and debugging


@handles(0x1E)
def get_system_tick(p: Process, t: Thread) -> str:
    p.cpu.set_x(0, p.ticks)
    p.ticks += 1
    return f"() -> {p.ticks - 1}"


INFO_NAMES = {
    0: "CoreMask",
    1: "PriorityMask",
    2: "AliasRegionAddress",
    3: "AliasRegionSize",
    4: "HeapRegionAddress",
    5: "HeapRegionSize",
    6: "TotalMemorySize",
    7: "UsedMemorySize",
    8: "DebuggerAttached",
    9: "ResourceLimit",
    10: "IdleTickCount",
    11: "RandomEntropy",
    12: "AslrRegionAddress",
    13: "AslrRegionSize",
    14: "StackRegionAddress",
    15: "StackRegionSize",
    16: "SystemResourceSizeTotal",
    17: "SystemResourceSizeUsed",
    18: "ProgramId",
    20: "UserExceptionContextAddress",
    21: "TotalNonSystemMemorySize",
    22: "UsedNonSystemMemorySize",
    23: "IsApplication",
    24: "FreeThreadCount",
    25: "ThreadTickCount",
    26: "IsSvcPermitted",
    27: "IoRegionHint",
    28: "AliasRegionExtraSize",
    29: "RemoteRegionAddress",
    30: "RemoteRegionSize",
    31: "RemoteMemoryUsage",
    32: "RemoteMemoryUsagePeak",
    33: "ProcessPageSize",
    34: "TransferMemoryHint",
    36: "AddressSpaceSize",
    37: "ShadowStackRegionAddress",
    38: "ShadowStackRegionSize",
}


# GetInfo types that take handle 0 instead of a process or thread handle (switchbrew's "Zero").
_INFO_WITHOUT_HANDLE = {8, 9, 10, 11, 19}


def _info_value(p: Process, kind: int, sub: int) -> int | None:
    layout = p.space.layout
    resource = p.npdm.system_resource_size if p.npdm else 0
    simple = {
        0: sum(1 << c for c in p.cores),
        1: sum(1 << i for i in range(p.priorities[0], p.priorities[1] + 1)),
        2: layout.alias.start,
        3: layout.alias.size,
        4: layout.heap.start,
        5: layout.heap.size,
        6: TOTAL_MEMORY,
        7: used_memory(p),
        8: 0,
        9: 0,
        12: layout.space.start,
        13: layout.space.size,
        14: layout.stack.start,
        15: layout.stack.size,
        16: resource,
        17: 0,
        18: p.npdm.program_id if p.npdm else 0,
        21: TOTAL_MEMORY - resource,
        22: used_memory(p),
        23: 1,
        24: 0x200 - len(p.threads),
        26: None,
        28: 0,
        29: layout.remote.start,
        30: layout.remote.size,
        31: 0,
        32: 0,
        33: PAGE,
    }
    if kind in simple and kind != 26:
        return simple[kind] if sub == 0 else None
    if kind == 10:
        return 0
    if kind == 11:
        if not 0 <= sub < 4:
            return None
        return int.from_bytes(hashlib.sha256(b"siicl entropy %d" % sub).digest()[:8], "little")
    if kind == 20:
        if not p.user_exception_context:
            p.user_exception_context = p.new_tls()
        return p.user_exception_context
    if kind == 25:
        return p.ticks
    if kind == 26:
        return int(p.allowed_svcs is None or sub in p.allowed_svcs)
    raise Unimplemented(f"GetInfo type {kind} (sub {sub:#x}) is not implemented")


@handles(0x29)
def get_info(p: Process, t: Thread) -> str:
    kind, handle, sub = _w(p, 1), _w(p, 2), p.cpu.x(3)
    name = INFO_NAMES.get(kind, f"type {kind}")
    args = f"({name}, handle={handle:#x}, sub={sub:#x})"
    if kind == 25:
        if not isinstance(p.object_for(handle), Thread):
            return args + _result(p, INVALID_HANDLE)
    elif kind in _INFO_WITHOUT_HANDLE:
        if handle != 0:
            return args + _result(p, INVALID_HANDLE)
    elif p.object_for(handle) is not p:
        return args + _result(p, INVALID_HANDLE)
    value = _info_value(p, kind, sub)
    if value is None:
        return args + _result(p, INVALID_COMBINATION)
    p.cpu.set_x(1, value)
    return args + _result(p, OK, f", {value:#x}")


BREAK_REASONS = ("panic", "assert", "user", "pre-load DLL", "post-load DLL", "pre-unload DLL", "post-unload DLL")


@handles(0x26)
def break_(p: Process, t: Thread) -> str | End:
    reason, arg, size = p.cpu.x(0), p.cpu.x(1), p.cpu.x(2)
    kind = reason & 0x7FFF_FFFF
    name = BREAK_REASONS[kind] if kind < len(BREAK_REASONS) else f"reason {kind:#x}"
    detail = ""
    if 0 < size <= 0x100:
        try:
            detail = ", data " + p.cpu.read(arg, size).hex()
        except Exception:
            detail = ", data unreadable"
    text = f"(reason={reason:#x} {name}, arg={arg:#x}, size={size:#x}){detail}"
    if reason & 0x8000_0000:
        return text + _result(p, OK)
    return End("break", f"the game called Break: {name}{detail}", t)


@handles(0x27)
def output_debug_string(p: Process, t: Thread) -> str:
    pointer, size = p.cpu.x(0), p.cpu.x(1)
    text = p.cpu.read(pointer, min(size, 0x1000)).decode("utf-8", "replace").rstrip("\n")
    return f"({text!r})" + _result(p, OK)


@handles(0x07)
def exit_process(p: Process, t: Thread) -> End:
    return End("exit", "the game called ExitProcess", t)


# IPC


@handles(0x1F)
def connect_to_named_port(p: Process, t: Thread) -> str:
    raw = p.cpu.read(p.cpu.x(1), 12)
    name = raw.split(b"\0", 1)[0].decode("ascii", "replace")
    if name != "sm:":
        raise Unimplemented(f'SVC 0x1F (ConnectToNamedPort) to port "{name}" is not implemented')
    handle = p.handles.add(services.Session(services.sm.ServiceManager(p)))
    p.cpu.set_x(1, handle)
    return f"({name!r})" + _result(p, OK, f", handle {handle:#x}")


def _request(p: Process, t: Thread, handle: int, address: int, size: int) -> str:
    try:
        session = p.handles.get(handle)
    except KeyError:
        session = None
    if not isinstance(session, services.Session):
        return f"(handle={handle:#x})" + _result(p, INVALID_HANDLE)
    text = services.handle_request(p, t, session, address, size)
    p.cpu.set_x(0, OK)
    return f"(handle={handle:#x}) {text}"


@handles(0x21)
def send_sync_request(p: Process, t: Thread) -> str:
    return _request(p, t, _w(p, 0), t.tls, ipc.MESSAGE_SIZE)


@handles(0x22)
def send_sync_request_with_user_buffer(p: Process, t: Thread) -> str:
    return _request(p, t, _w(p, 2), p.cpu.x(0), p.cpu.x(1))


# Events and shared memory


def _event(p: Process, handle: int) -> Event | ReadableEvent | None:
    obj = p.object_for(handle)
    return obj if isinstance(obj, Event | ReadableEvent) else None


@handles(0x11)
def signal_event(p: Process, t: Thread) -> str:
    handle = _w(p, 0)
    event = _event(p, handle)
    if not isinstance(event, Event):
        return f"(handle={handle:#x})" + _result(p, INVALID_HANDLE)
    event.signal(p)
    return f"(handle={handle:#x})" + _result(p, OK)


@handles(0x12)
def clear_event(p: Process, t: Thread) -> str:
    handle = _w(p, 0)
    event = _event(p, handle)
    if event is None:
        return f"(handle={handle:#x})" + _result(p, INVALID_HANDLE)
    (event if isinstance(event, Event) else event.event).clear()
    return f"(handle={handle:#x})" + _result(p, OK)


@handles(0x17)
def reset_signal(p: Process, t: Thread) -> str:
    handle = _w(p, 0)
    event = _event(p, handle)
    if not isinstance(event, ReadableEvent):
        return f"(handle={handle:#x})" + _result(p, INVALID_HANDLE)
    if not event.signaled:
        return f"(handle={handle:#x})" + _result(p, INVALID_STATE)
    event.signaled = False
    return f"(handle={handle:#x})" + _result(p, OK)


# Memory states CreateTransferMemory may lend: the heap, physical memory and the modules' .data.
_TRANSFERABLE = {MemoryState.NORMAL, MemoryState.CODE_DATA, MemoryState.ALIAS_CODE_DATA}


@handles(0x15)
def create_transfer_memory(p: Process, t: Thread) -> str:
    address, size, perm = p.cpu.x(1), p.cpu.x(2), _w(p, 3)
    args = f"(address={address:#x}, size={size:#x}, perm={perm_text(perm)})"
    if (bad := _pages(address, size)) is not None:
        return args + _result(p, bad)
    if perm not in (0, READ, RW):
        return args + _result(p, INVALID_NEW_MEMORY_PERMISSION)
    blocks = _blocks(p, address, size)
    if blocks is None or any(b.state not in _TRANSFERABLE or (b.perm, b.attribute) != (RW, 0) for b in blocks):
        return args + _result(p, INVALID_CURRENT_MEMORY)
    p.space.update(address, size, perm=perm, attribute=MemoryAttribute.LOCKED)
    handle = p.handles.add(TransferMemory(address, size, perm))
    p.cpu.set_x(1, handle)
    return args + _result(p, OK, f", handle {handle:#x}")


@handles(0x80)
def set_gpu_memory_attribute(p: Process, t: Thread) -> str:
    """A Switch 2 SVC (X0 address, X1 size, W2 mask, W3 value; see switchbrew). switchbrew does not say
    what the bits mean; siicl records them for each range and changes nothing else."""
    address, size, mask, value = p.cpu.x(0), p.cpu.x(1), _w(p, 2), _w(p, 3)
    args = f"(address={address:#x}, size={size:#x}, mask={mask:#x}, value={value:#x})"
    if (bad := _pages(address, size)) is not None:
        return args + _result(p, bad)
    if _blocks(p, address, size) is None:
        return args + _result(p, INVALID_CURRENT_MEMORY)
    attributes = p.services.setdefault("gpu memory attributes", {})
    old = attributes.get((address, size), 0)
    attributes[(address, size)] = (old & ~mask) | (value & mask)
    return args + _result(p, OK)


@handles(0x13)
def map_shared_memory(p: Process, t: Thread) -> str:
    handle, address, size, perm = _w(p, 0), p.cpu.x(1), p.cpu.x(2), _w(p, 3)
    args = f"(handle={handle:#x}, address={address:#x}, size={size:#x}, perm={perm_text(perm)})"
    shared = p.object_for(handle)
    if not isinstance(shared, SharedMemory):
        return args + _result(p, INVALID_HANDLE)
    if (bad := _pages(address, size)) is not None:
        return args + _result(p, bad)
    if shared.size is None:
        shared.allocate(size)
    if size != shared.size:
        return args + _result(p, INVALID_SIZE)
    if perm not in (READ, RW):
        return args + _result(p, INVALID_NEW_MEMORY_PERMISSION)
    layout = p.space.layout
    if any(
        r.contains(address, 1) or r.contains(address + size - 1, 1) for r in (layout.heap, layout.alias, layout.stack)
    ):
        return args + _result(p, INVALID_MEMORY_REGION)
    if not p.space.is_free(address, size):
        return args + _result(p, INVALID_CURRENT_MEMORY)
    p.space.map(address, size, perm, MemoryState.SHARED, f"shared memory ({shared.name})", memory=shared.memory)
    return args + _result(p, OK, f", {shared.name}")


@handles(0x14)
def unmap_shared_memory(p: Process, t: Thread) -> str:
    handle, address, size = _w(p, 0), p.cpu.x(1), p.cpu.x(2)
    args = f"(handle={handle:#x}, address={address:#x}, size={size:#x})"
    shared = p.object_for(handle)
    if not isinstance(shared, SharedMemory):
        return args + _result(p, INVALID_HANDLE)
    blocks = _blocks(p, address, size)
    if blocks is None or any(b.state != MemoryState.SHARED or b.memory is not shared.memory for b in blocks):
        return args + _result(p, INVALID_CURRENT_MEMORY)
    p.space.unmap(address, size)
    return args + _result(p, OK)
