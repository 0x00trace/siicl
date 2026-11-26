"""The process siicl runs a game in: its CPU, memory, modules, threads and handles.

`Process` sets the process up as the system does when it launches a game: it loads the modules,
gives the main thread a stack and thread-local storage, and sets the registers the main thread
starts with. The SVCs the game calls are not handled yet (stage 2 of docs/roadmap.md).
"""

from __future__ import annotations

from dataclasses import dataclass

from siicl import loader
from siicl.addrspace import LAYOUT_39BIT, AddressSpace, Block, Layout, MemoryState
from siicl.cpu import PAGE, READ, WRITE, Cpu
from siicl.npdm import Npdm

TLS_SIZE = 0x200  # each thread's thread-local region; a page holds eight
ADDRESS_SPACE_39BIT = 3  # the NPDM's address space type for a 64-bit process with a 39-bit address space
DEFAULT_STACK_SIZE = 0x100000
DEFAULT_PRIORITY = 44


class ProcessError(ValueError):
    pass


@dataclass
class Thread:
    handle: int
    entry: int
    stack: Block
    tls: int
    priority: int
    core: int


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


class Process:
    def __init__(self, modules: dict[str, bytes], npdm: Npdm | None = None, *, layout: Layout = LAYOUT_39BIT):
        if npdm is not None and npdm.address_space_type != ADDRESS_SPACE_39BIT:
            raise ProcessError(
                f"siicl runs only 64-bit processes with a 39-bit address space, not {npdm.address_space}"
            )
        self.npdm = npdm
        self.cpu = Cpu()
        self.space = AddressSpace(self.cpu, layout)
        self.modules = loader.load(self.space, modules)
        self.handles = HandleTable()
        self.threads: list[Thread] = []
        self._tls_free: list[int] = []
        stack_size = npdm.main_thread_stack_size if npdm else DEFAULT_STACK_SIZE
        self.main_thread = self._new_thread(
            entry=self.modules[0].base,
            stack_size=(stack_size + PAGE - 1) // PAGE * PAGE,
            priority=npdm.main_thread_priority if npdm else DEFAULT_PRIORITY,
            core=npdm.main_thread_core if npdm else 0,
        )
        self.cpu.pc = self.main_thread.entry
        self.cpu.sp = self.main_thread.stack.end
        self.cpu.set_x(0, 0)
        self.cpu.set_x(1, self.main_thread.handle)
        self.cpu.tpidrro_el0 = self.main_thread.tls

    def _new_thread(self, *, entry: int, stack_size: int, priority: int, core: int) -> Thread:
        n = len(self.threads)
        stack = self.space.map_free(
            self.space.layout.stack, stack_size, READ | WRITE, MemoryState.STACK, f"thread {n} stack"
        )
        thread = Thread(0, entry, stack, self._new_tls(), priority, core)
        thread.handle = self.handles.add(thread)
        self.threads.append(thread)
        return thread

    def _new_tls(self) -> int:
        if not self._tls_free:
            page = self.space.map_free(
                self.space.layout.kernel_map, PAGE, READ | WRITE, MemoryState.THREAD_LOCAL, "thread-local storage"
            )
            self._tls_free = [page.start + i * TLS_SIZE for i in range(PAGE // TLS_SIZE)]
        return self._tls_free.pop(0)

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
