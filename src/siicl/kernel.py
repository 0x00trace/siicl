"""The process siicl runs a game in, and the kernel that runs its threads.

`Process` sets the process up as the system does when it launches a game: it loads the modules,
gives the main thread a stack and thread-local storage, and sets the registers the main thread
starts with. `Process.run()` then runs the threads and handles their SVCs (siicl.svcs) until
the game needs something siicl does not implement, crashes, exits or runs out of time.

Threads take turns on the one CPU. The scheduler keeps a run queue for each core, and gives the
cores turns in a fixed order: on its turn, a core runs its highest-priority runnable thread until
the thread makes an SVC or has run SLICE instructions. Threads of equal priority on a core take
turns too. Nothing depends on the host's timing, so two runs of the same game do the same thing.

Time is virtual. The system counter (19.2 MHz) advances by one nanosecond for each instruction of
a full slice, by SVC_NS for each SVC, and by one tick for each read of the counter. When every
thread is waiting, it jumps to the next timeout.
"""

from __future__ import annotations

import re
import struct
from collections import Counter
from dataclasses import dataclass
from typing import TextIO

from siicl import loader
from siicl.addrspace import LAYOUT_39BIT, AddressSpace, Layout, MemoryState
from siicl.cpu import PAGE, READ, WRITE, Cpu, GuestMemoryError, Stop
from siicl.npdm import Npdm

TLS_SIZE = 0x200  # each thread's thread-local region; a page holds eight
ADDRESS_SPACE_39BIT = 3  # the NPDM's address space type for a 64-bit process with a 39-bit address space
DEFAULT_STACK_SIZE = 0x100000
DEFAULT_PRIORITY = 44
PROCESS_ID = 0x51
FIRST_THREAD_ID = 0x100

TICK_HZ = 19_200_000
SLICE = 200_000
SVC_NS = 1_000

CURRENT_THREAD = 0xFFFF8000
CURRENT_PROCESS = 0xFFFF8001

# siicl turns each read of the system counter (MRS Xt, CNTPCT_EL0 or CNTVCT_EL0) into SVC #(COUNTER_SVC | t),
# so the game reads virtual time. Real SVC numbers are below 0x100.
COUNTER_SVC = 0xF000
_COUNTER_READ = re.compile(rb"[\x20-\x5f]\xe0\x3b\xd5", re.DOTALL)


def ns_to_ticks(ns: int) -> int:
    return ns * 12 // 625  # 19.2 MHz


class ProcessError(ValueError):
    pass


class Unimplemented(Exception):
    """The game asked for something siicl does not implement yet."""


@dataclass
class End:
    """Why Process.run() returned.

    reason is one of: "unimplemented", "crash" (a fault, an undefined instruction or a BRK the
    game did not handle), "break" (the game called Break), "exit" (ExitProcess), "deadlock"
    (every thread waits, with no timeout) or "time" (the time limit was reached).
    """

    reason: str
    message: str
    thread: Thread | None = None
    stop: Stop | None = None


class SyncObject:
    """A kernel object that WaitSynchronization can wait for."""

    def __init__(self) -> None:
        self.sync_waiters: list[Thread] = []

    def is_signaled(self) -> bool:
        raise NotImplementedError

    def wake_waiters(self, process: Process) -> None:
        for t in list(self.sync_waiters):
            index = t.sync_objects.index(self)
            t.stop_sync_wait()
            process.wake(t, {0: 0, 1: index})


class Thread(SyncObject):
    def __init__(self, process: Process, *, entry: int, arg: int, stack_top: int, priority: int, core: int, tls: int):
        super().__init__()
        self.process = process
        self.id = FIRST_THREAD_ID + len(process.threads)
        self.number = len(process.threads)
        self.handle = 0
        self.entry, self.arg, self.stack_top = entry, arg, stack_top
        self.priority = priority
        self.core = core
        self.affinity = 1 << core
        self.tls = tls
        self.state = "created"  # then "runnable", "waiting" or "exited"
        self.context: object | None = None
        self.pc = entry
        self.pending: dict[int, int] = {}  # registers to set before the thread runs again
        self.wait_kind: str | None = None
        self.deadline: int | None = None  # in ticks
        self.cancel_pending = False
        self.sync_objects: list[SyncObject] = []
        self.mutex_waiters: list[Thread] = []  # threads waiting for a mutex this thread holds
        self.lock_owner: Thread | None = None
        self.mutex_addr = 0
        self.tag = 0
        self.condvar_key: int | None = None
        self.arbiter_addr: int | None = None

    def is_signaled(self) -> bool:
        return self.state == "exited"

    def stop_sync_wait(self) -> None:
        for o in self.sync_objects:
            o.sync_waiters.remove(self)
        self.sync_objects = []

    def describe(self) -> str:
        wait = f", waiting ({self.wait_kind})" if self.state == "waiting" else ""
        return (
            f"thread {self.number} (id {self.id:#x}, handle {self.handle:#x}, priority {self.priority}, "
            f"core {self.core}): {self.state}{wait}"
        )


class ReadableEvent(SyncObject):
    """The side of an event that threads wait for."""

    def __init__(self, event: Event) -> None:
        super().__init__()
        self.event = event
        self.signaled = False

    def is_signaled(self) -> bool:
        return self.signaled


class Event:
    """An event, as a service hands one out: it signals its readable side."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.readable = ReadableEvent(self)

    def signal(self, process: Process) -> None:
        if not self.readable.signaled:
            self.readable.signaled = True
            self.readable.wake_waiters(process)

    def clear(self) -> None:
        self.readable.signaled = False


class SharedMemory:
    """Memory a service shares with the game, which maps it with MapSharedMemory."""

    def __init__(self, name: str, size: int) -> None:
        from siicl.addrspace import HostMemory

        self.name = name
        self.size = size
        self.memory = HostMemory(size)
        self.view = memoryview(self.memory._map)


class TransferMemory:
    """Memory the game lends to a service. Its pages stay locked to the permission the game keeps
    (often none) until the handle is closed."""

    def __init__(self, address: int, size: int, perm: int) -> None:
        self.address, self.size, self.perm = address, size, perm
        self.kept = False  # a service holds it, so closing the game's handle does not give the pages back


class HandleTable:
    """Handles as the kernel makes them: a slot index in the low 15 bits and a serial number above it."""

    def __init__(self, size: int = 1024) -> None:
        self.size = size
        self._slots: dict[int, tuple[int, object]] = {}
        self._serial = 0

    def add(self, obj: object) -> int:
        index = next((i for i in range(self.size) if i not in self._slots), None)
        if index is None:
            raise ProcessError("the handle table is full")
        self._serial = self._serial % 0x7FFF + 1
        self._slots[index] = (self._serial, obj)
        return (self._serial << 15) | index

    def get(self, handle: int) -> object:
        entry = self._slots.get(handle & 0x7FFF)
        if entry is None or entry[0] != handle >> 15:
            raise KeyError(f"no handle {handle:#x}")
        return entry[1]

    def close(self, handle: int) -> None:
        self.get(handle)
        del self._slots[handle & 0x7FFF]


class Process:
    def __init__(self, modules: dict[str, bytes], npdm: Npdm | None = None, *, layout: Layout = LAYOUT_39BIT):
        if npdm is not None and npdm.address_space_type != ADDRESS_SPACE_39BIT:
            raise ProcessError(
                f"siicl runs only 64-bit processes with a 39-bit address space, not {npdm.address_space}"
            )
        self.npdm = npdm
        caps = npdm.caps if npdm else None
        self.cpu = Cpu()
        self._blank = self.cpu.save_context()
        self.space = AddressSpace(self.cpu, layout)
        self.modules = loader.load(self.space, modules)
        self.counter_reads = self._patch_counter_reads()
        self.handles = HandleTable()
        self.threads: list[Thread] = []
        self._tls_free: list[int] = []
        self.ticks = 0
        self.heap_size = 0
        if caps and caps.min_core is not None:
            self.cores = list(range(caps.min_core, caps.max_core + 1))
            self.priorities = (caps.highest_priority, caps.lowest_priority)
        else:
            self.cores, self.priorities = [0, 1, 2, 3], (0, 63)
        self.allowed_svcs = set(caps.svcs) if caps and caps.svcs else None
        self.ideal_core = npdm.main_thread_core if npdm else self.cores[0]
        self.queues: dict[int, list[Thread]] = {c: [] for c in self.cores}
        self.services: dict[str, object] = {}  # state the services keep for the process
        self.romfs = None  # the game's romfs image (siicl.romfs.RomFsImage), if it has one
        self.stub_unknown = False  # answer unknown service commands with success instead of ending the run
        self.stubbed: list[str] = []
        self.condvar_waiters: list[Thread] = []
        self.arbiter_waiters: list[Thread] = []
        self.current: Thread | None = None
        self.log: TextIO | None = None
        self.svc_counts: Counter[str] = Counter()
        self._next_core = 0
        self._symbols = None
        self._last_trace: tuple[int, str] | None = None
        self._repeats = 0
        self.user_exception_context = 0
        stack_size = npdm.main_thread_stack_size if npdm else DEFAULT_STACK_SIZE
        stack = self.space.map_free(
            layout.stack, (stack_size + PAGE - 1) // PAGE * PAGE, READ | WRITE, MemoryState.STACK, "main thread stack"
        )
        self.main_thread = self.new_thread(
            entry=self.modules[0].base,
            arg=0,
            stack_top=stack.end,
            priority=npdm.main_thread_priority if npdm else DEFAULT_PRIORITY,
            core=self.ideal_core,
        )
        self.main_thread.pending[1] = self.main_thread.handle
        self.start_thread(self.main_thread)
        self.switch_to(self.main_thread)

    # Setup

    def _patch_counter_reads(self) -> int:
        count = 0
        for m in self.modules:
            text = self.cpu.read(m.text.start, m.text.size)
            for match in _COUNTER_READ.finditer(text):
                offset = match.start()
                word = struct.unpack_from("<I", text, offset)[0]
                if offset % 4 or word & 0xFFFFFFE0 not in (0xD53BE020, 0xD53BE040):
                    continue
                svc = 0xD4000001 | ((COUNTER_SVC | (word & 0x1F)) << 5)
                self.cpu.write_u32(m.text.start + offset, svc)
                count += 1
        return count

    def new_thread(self, *, entry: int, arg: int, stack_top: int, priority: int, core: int) -> Thread:
        thread = Thread(
            self, entry=entry, arg=arg, stack_top=stack_top, priority=priority, core=core, tls=self.new_tls()
        )
        thread.handle = self.handles.add(thread)
        self.threads.append(thread)
        return thread

    def new_tls(self) -> int:
        if not self._tls_free:
            page = self.space.map_free(
                self.space.layout.kernel_map, PAGE, READ | WRITE, MemoryState.THREAD_LOCAL, "thread-local storage"
            )
            self._tls_free = [page.start + i * TLS_SIZE for i in range(PAGE // TLS_SIZE)]
        return self._tls_free.pop(0)

    # Objects

    def thread_for(self, handle: int) -> Thread | None:
        if handle == CURRENT_THREAD:
            return self.current
        try:
            obj = self.handles.get(handle)
        except KeyError:
            return None
        return obj if isinstance(obj, Thread) else None

    def object_for(self, handle: int) -> object | None:
        if handle == CURRENT_THREAD:
            return self.current
        if handle == CURRENT_PROCESS:
            return self
        try:
            return self.handles.get(handle)
        except KeyError:
            return None

    # Scheduling

    def start_thread(self, thread: Thread) -> None:
        thread.state = "runnable"
        self.queues[thread.core].append(thread)

    def exit_thread(self, thread: Thread) -> None:
        thread.state = "exited"
        self._dequeue(thread)
        thread.wake_waiters(self)

    def block(self, thread: Thread, kind: str, timeout_ns: int | None = None) -> None:
        """Make thread wait. A timeout below zero, or None, waits for ever."""
        thread.state = "waiting"
        thread.wait_kind = kind
        thread.deadline = None if timeout_ns is None or timeout_ns < 0 else self.ticks + max(1, ns_to_ticks(timeout_ns))
        self._dequeue(thread)

    def wake(self, thread: Thread, registers: dict[int, int]) -> None:
        if thread.state != "waiting":
            return
        thread.pending.update(registers)
        thread.state, thread.wait_kind, thread.deadline = "runnable", None, None
        self.queues[thread.core].append(thread)

    def yield_thread(self, thread: Thread) -> None:
        queue = self.queues[thread.core]
        if thread in queue:
            queue.remove(thread)
            queue.append(thread)

    def move_to_core(self, thread: Thread, core: int) -> None:
        if thread.core == core:
            return
        was_queued = thread in self.queues[thread.core]
        self._dequeue(thread)
        thread.core = core
        if was_queued:
            self.queues[core].append(thread)

    def _dequeue(self, thread: Thread) -> None:
        queue = self.queues[thread.core]
        if thread in queue:
            queue.remove(thread)

    def _pick(self) -> Thread | None:
        n = len(self.cores)
        for i in range(n):
            core = self.cores[(self._next_core + i) % n]
            queue = self.queues[core]
            if queue:
                self._next_core = (self._next_core + i + 1) % n
                return min(queue, key=lambda t: t.priority)
        return None

    def switch_to(self, thread: Thread) -> None:
        if self.current is not thread:
            if thread.context is None:
                self.cpu.restore_context(self._blank)
                self.cpu.pc, self.cpu.sp = thread.entry, thread.stack_top
                self.cpu.set_x(0, thread.arg)
                self.cpu.tpidrro_el0 = thread.tls
            else:
                self.cpu.restore_context(thread.context)
            self.current = thread
        for reg, value in thread.pending.items():
            self.cpu.set_x(reg, value)
        thread.pending.clear()

    def _time_out(self) -> None:
        for t in self.threads:
            if t.state == "waiting" and t.deadline is not None and t.deadline <= self.ticks:
                from siicl import svcs  # here, because siicl.svcs imports this module

                svcs.time_out(self, t)

    # Running

    @property
    def seconds(self) -> float:
        return self.ticks / TICK_HZ

    def trace(self, thread: Thread, text: str) -> None:
        """Log a line. A run of the same line from the same thread is written once, then counted."""
        if not self.log:
            return
        key = (thread.number, text)
        if key == self._last_trace:
            self._repeats += 1
            return
        self._flush_repeats()
        self._last_trace = key
        print(f"{self.seconds:11.6f} t{thread.number:<2} {text}", file=self.log)

    def _flush_repeats(self) -> None:
        if self._repeats and self.log:
            number, _ = self._last_trace
            print(f"{self.seconds:11.6f} t{number:<2} (the line above {self._repeats:,} more times)", file=self.log)
        self._repeats = 0

    def run(self, log: TextIO | None = None, *, max_seconds: float = 60.0) -> End:
        self.log = log
        try:
            return self._run(max_seconds)
        finally:
            self._flush_repeats()

    def _run(self, max_seconds: float) -> End:
        from siicl import svcs  # here, because siicl.svcs imports this module

        max_ticks = int(max_seconds * TICK_HZ)
        while True:
            if self.ticks >= max_ticks:
                return End("time", f"the time limit of {max_seconds:g} s of guest time was reached")
            thread = self._pick()
            if thread is None:
                deadlines = [t.deadline for t in self.threads if t.state == "waiting" and t.deadline is not None]
                if not deadlines:
                    return End("deadlock", "every thread is waiting, and none has a timeout")
                self.ticks = max(self.ticks, min(deadlines))
                self._time_out()
                continue
            self.switch_to(thread)
            stop = self.cpu.run(limit=SLICE)
            thread.pc = self.cpu.pc
            if stop.reason == "limit":
                self.ticks += ns_to_ticks(SLICE)
                self.yield_thread(thread)
            elif stop.reason == "svc":
                self.ticks += ns_to_ticks(SVC_NS)
                try:
                    end = svcs.call(self, thread, stop)
                except Unimplemented as err:
                    return End("unimplemented", str(err), thread, stop)
                except GuestMemoryError as err:
                    return End("crash", f"an SVC argument points to bad memory: {err}", thread, stop)
                if end is not None:
                    return end
            else:
                return End("crash", f"the game crashed: {stop.describe()}", thread, stop)
            thread.context = self.cpu.save_context()
            self._time_out()

    # Logs

    @property
    def symbols(self):
        """The modules' dynamic symbols, read once, when a log first needs a name."""
        if self._symbols is None:
            from siicl.symbols import SymbolTable, module_symbols

            found = []
            for m in self.modules:
                try:
                    found += module_symbols(self.cpu.read, m.base, self.cpu.read(m.text.start, 8))
                except GuestMemoryError, struct.error:
                    continue
            self._symbols = SymbolTable(found)
        return self._symbols

    def symbol_name(self, address: int, *, demangle: bool = True) -> str:
        """The function that holds address, as "name+0x10", or an empty string."""
        found = self.symbols.lookup(address)
        if found is None:
            return ""
        sym, offset = found
        name = sym.name
        if demangle:
            from siicl.symbols import demangle as demangle_names

            name = demangle_names([name])[0]
        return f"{name}+{offset:#x}"

    def module_at(self, address: int) -> loader.Module | None:
        return next((m for m in self.modules if m.contains(address)), None)

    def describe_address(self, address: int) -> str:
        """An address as a module and offset, such as "main+0x1a4", or as the block that holds it."""
        module = self.module_at(address)
        if module:
            return f"{module.name}+{address - module.base:#x}"
        block = self.space.query(address)
        if block.state != MemoryState.FREE and block.name:
            return f"{block.name}+{address - block.start:#x}"
        return f"{address:#x}"
