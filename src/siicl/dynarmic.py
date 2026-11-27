"""Run AArch64 guest code in dynarmic, through siicl's C library (native/siicl_dynarmic.h).

This is the CPU backend that docs/cpu-backend.md plans; siicl.cpu, on Unicorn, is still the one
`siicl boot` uses. Build the library before using this module:

    git submodule update --init native/dynarmic
    cmake -S native -B build/native -G Ninja -DCMAKE_BUILD_TYPE=Release
    cmake --build build/native

A `Machine` holds a guest address space and the exclusive monitor that its cores share. Each
`Core` is one AArch64 CPU; cores of one machine can run at the same time on different host
threads, because ctypes releases the GIL while a core runs. `Core.run()` returns the same
`siicl.cpu.Stop` as `Cpu.run()`, so the kernel can use either backend.

The instruction budget is counted in dynarmic's ticks, one for each instruction. dynarmic checks
the budget between blocks of translated code, so a run can go past the budget to the end of a
block; `Core.ticks` says how many instructions the last run executed. A run that ends at an
exception is charged its whole block, and one that ends at a data fault is charged nothing.
"""

from __future__ import annotations

import ctypes
import mmap
import os
from ctypes import POINTER, Structure, byref, c_char_p, c_int, c_uint32, c_uint64, c_void_p
from pathlib import Path

from siicl.cpu import EXECUTE, PAGE, READ, WRITE, GuestMemoryError, Stop

assert (READ, WRITE, EXECUTE) == (1, 2, 4), "the C library uses Unicorn's permission values"

# The C library's stop reasons (SD_* in siicl_dynarmic.h). An interpreter fallback (6) means that
# dynarmic can't run the instruction, so it counts as undefined.
_REASONS = {0: "limit", 1: "svc", 2: "fault", 3: "undefined", 4: "break", 5: "exception", 6: "undefined", 7: "halted"}
_ACCESS = {0: "read", 1: "write", 2: "fetch"}
_KIND = {0: "unmapped", 1: "protected"}
_UNLIMITED = 1 << 62


class DynarmicError(RuntimeError):
    """The C library is missing, or refused a request."""


class _Stop(Structure):
    _fields_ = [
        ("reason", c_uint32),
        ("number", c_uint32),
        ("pc", c_uint64),
        ("address", c_uint64),
        ("access", c_uint32),
        ("kind", c_uint32),
        ("ticks", c_uint64),
    ]


class Context(Structure):
    """All of a core's registers, as `Core.save_context()` returns them."""

    _fields_ = [
        ("x", c_uint64 * 31),
        ("sp", c_uint64),
        ("pc", c_uint64),
        ("v", (c_uint64 * 2) * 32),
        ("pstate", c_uint32),
        ("fpcr", c_uint32),
        ("fpsr", c_uint32),
        ("reserved", c_uint32),
        ("tpidr_el0", c_uint64),
        ("tpidrro_el0", c_uint64),
    ]


_U64x2 = c_uint64 * 2
_SIGNATURES = {
    "sd_version": ([], c_char_p),
    "sd_machine_create": ([c_uint32, c_uint32], c_void_p),
    "sd_machine_destroy": ([c_void_p], None),
    "sd_map": ([c_void_p, c_uint64, c_uint64, c_uint32, c_void_p], c_int),
    "sd_protect": ([c_void_p, c_uint64, c_uint64, c_uint32], c_int),
    "sd_unmap": ([c_void_p, c_uint64, c_uint64], c_int),
    "sd_read": ([c_void_p, c_uint64, c_void_p, c_uint64], c_int),
    "sd_write": ([c_void_p, c_uint64, c_void_p, c_uint64], c_int),
    "sd_core_create": ([c_void_p, c_uint32, c_int], c_void_p),
    "sd_core_destroy": ([c_void_p], None),
    "sd_get_x": ([c_void_p, c_uint32], c_uint64),
    "sd_set_x": ([c_void_p, c_uint32, c_uint64], None),
    "sd_get_v": ([c_void_p, c_uint32, POINTER(c_uint64)], None),
    "sd_set_v": ([c_void_p, c_uint32, POINTER(c_uint64)], None),
    "sd_save_context": ([c_void_p, POINTER(Context)], None),
    "sd_restore_context": ([c_void_p, POINTER(Context)], None),
    "sd_run": ([c_void_p, c_uint64, POINTER(_Stop)], None),
    "sd_halt": ([c_void_p], None),
    "sd_invalidate": ([c_void_p, c_uint64, c_uint64], None),
}
for _name, _type in [
    ("sp", c_uint64),
    ("pc", c_uint64),
    ("tpidr_el0", c_uint64),
    ("tpidrro_el0", c_uint64),
    ("counter", c_uint64),
    ("pstate", c_uint32),
    ("fpcr", c_uint32),
    ("fpsr", c_uint32),
]:
    _SIGNATURES[f"sd_get_{_name}"] = ([c_void_p], _type)
    _SIGNATURES[f"sd_set_{_name}"] = ([c_void_p, _type], None)

_lib: ctypes.CDLL | None = None


def library_path() -> Path:
    """Where the library is: $SIICL_DYNARMIC, or build/native in the repository."""
    if env := os.environ.get("SIICL_DYNARMIC"):
        return Path(env)
    return Path(__file__).resolve().parents[2] / "build" / "native" / "libsiicl_dynarmic.so"


def available() -> bool:
    return library_path().is_file()


def _load() -> ctypes.CDLL:
    global _lib
    if _lib is None:
        path = library_path()
        try:
            lib = ctypes.CDLL(str(path))
        except OSError as err:
            raise DynarmicError(f"cannot load {path} ({err}); build it as siicl.dynarmic's docstring says") from None
        for name, (args, res) in _SIGNATURES.items():
            fn = getattr(lib, name)
            fn.argtypes, fn.restype = args, res
        _lib = lib
    return _lib


def version() -> str:
    return _load().sd_version().decode()


def _check_pages(address: int, size: int) -> None:
    if address % PAGE or size % PAGE or size <= 0:
        raise ValueError(f"range {address:#x}+{size:#x} is not whole pages")


class Machine:
    """A guest address space and the cores that run in it."""

    def __init__(self, *, cores: int = 1, address_bits: int = 39) -> None:
        self._lib = _load()
        self._m = self._lib.sd_machine_create(address_bits, cores)
        if not self._m:
            raise DynarmicError(f"cannot create a machine with {cores} cores and {address_bits} address bits")
        self._memory: dict[int, tuple[mmap.mmap, ctypes.Array]] = {}
        self._cores: list[Core] = []

    def close(self) -> None:
        if self._m:
            for core in self._cores:
                core._destroy()
            self._cores.clear()
            self._lib.sd_machine_destroy(self._m)
            self._m = None
            self._memory.clear()

    def __enter__(self) -> Machine:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def core(self, processor_id: int | None = None, *, count_ticks: bool = True) -> Core:
        """A new core; processor_id defaults to the next unused one.

        With count_ticks, `Core.run(limit=...)` stops after about limit instructions, as the
        repeatable mode needs. Without it, a run ends only at a stop or at `Core.halt()` from
        another thread, as real-time play needs. dynarmic supports one or the other on a core.
        """
        pid = len(self._cores) if processor_id is None else processor_id
        handle = self._lib.sd_core_create(self._m, pid, int(count_ticks))
        if not handle:
            raise DynarmicError(f"cannot create core {pid}")
        core = Core(self, handle, pid, count_ticks)
        self._cores.append(core)
        return core

    # Memory

    def map_host(self, address: int, size: int, perm: int, host_address: int) -> None:
        """Map guest pages onto host memory at host_address, which must stay allocated."""
        _check_pages(address, size)
        if self._lib.sd_map(self._m, address, size, perm, host_address):
            raise DynarmicError(f"cannot map {address:#x}+{size:#x}")

    def map(self, address: int, size: int, perm: int) -> int:
        """Map new zeroed host memory at address, and return its host address."""
        _check_pages(address, size)
        buf = mmap.mmap(-1, size)
        view = (ctypes.c_char * size).from_buffer(buf)
        host = ctypes.addressof(view)
        self.map_host(address, size, perm, host)
        self._memory[address] = (buf, view)
        return host

    def protect(self, address: int, size: int, perm: int) -> None:
        _check_pages(address, size)
        if self._lib.sd_protect(self._m, address, size, perm):
            raise DynarmicError(f"cannot protect {address:#x}+{size:#x}: not all of it is mapped")

    def unmap(self, address: int, size: int) -> None:
        _check_pages(address, size)
        if self._lib.sd_unmap(self._m, address, size):
            raise DynarmicError(f"cannot unmap {address:#x}+{size:#x}")

    def read(self, address: int, size: int) -> bytes:
        buf = ctypes.create_string_buffer(size)
        if self._lib.sd_read(self._m, address, buf, size):
            raise GuestMemoryError(f"cannot read {size:#x} bytes at {address:#x}: not mapped")
        return buf.raw

    def write(self, address: int, data: bytes) -> None:
        data = bytes(data)
        if self._lib.sd_write(self._m, address, data, len(data)):
            raise GuestMemoryError(f"cannot write {len(data):#x} bytes at {address:#x}: not mapped")

    def read_u32(self, address: int) -> int:
        return int.from_bytes(self.read(address, 4), "little")

    def read_u64(self, address: int) -> int:
        return int.from_bytes(self.read(address, 8), "little")


def _register(name: str) -> property:
    def get(self: Core) -> int:
        return getattr(self._lib, f"sd_get_{name}")(self._c)

    def set_(self: Core, value: int) -> None:
        getattr(self._lib, f"sd_set_{name}")(self._c, value)

    return property(get, set_)


class Core:
    """One AArch64 CPU of a machine."""

    def __init__(self, machine: Machine, handle: int, processor_id: int, count_ticks: bool) -> None:
        self.machine = machine
        self.processor_id = processor_id
        self.count_ticks = count_ticks
        self._lib = machine._lib
        self._c = handle
        self.ticks = 0  # instructions the last run executed

    def _destroy(self) -> None:
        if self._c:
            self._lib.sd_core_destroy(self._c)
            self._c = None

    pc = _register("pc")
    sp = _register("sp")
    pstate = _register("pstate")
    fpcr = _register("fpcr")
    fpsr = _register("fpsr")
    tpidr_el0 = _register("tpidr_el0")
    tpidrro_el0 = _register("tpidrro_el0")
    counter = _register("counter")  # CNTPCT_EL0; each read returns it and then adds 1

    def x(self, index: int) -> int:
        return self._lib.sd_get_x(self._c, index)

    def set_x(self, index: int, value: int) -> None:
        self._lib.sd_set_x(self._c, index, value & 0xFFFF_FFFF_FFFF_FFFF)

    def v(self, index: int) -> int:
        """Vector register Vn as a 128-bit integer."""
        out = _U64x2()
        self._lib.sd_get_v(self._c, index, out)
        return out[0] | out[1] << 64

    def set_v(self, index: int, value: int) -> None:
        self._lib.sd_set_v(self._c, index, _U64x2(value & (1 << 64) - 1, value >> 64 & (1 << 64) - 1))

    def save_context(self) -> Context:
        ctx = Context()
        self._lib.sd_save_context(self._c, byref(ctx))
        return ctx

    def restore_context(self, ctx: Context) -> None:
        """Load all registers; this also clears the core's exclusive reservation."""
        self._lib.sd_restore_context(self._c, byref(ctx))

    def run(self, *, limit: int | None = None) -> Stop:
        """Run from the current PC until the code needs siicl, or for about limit instructions."""
        if limit is not None and limit <= 0:
            raise ValueError("limit must be positive")
        if limit is not None and not self.count_ticks:
            raise ValueError("this core counts no instructions; create it with count_ticks=True for a limit")
        out = _Stop()
        self._lib.sd_run(self._c, limit or _UNLIMITED, byref(out))
        self.ticks = out.ticks
        reason = _REASONS.get(out.reason, "exception")
        if reason == "svc":
            return Stop("svc", out.pc, svc=out.number)
        if reason == "break":
            return Stop("break", out.pc, svc=out.number)
        if reason == "fault":
            return Stop("fault", out.pc, address=out.address, access=_ACCESS[out.access], kind=_KIND[out.kind])
        if reason == "exception":
            return Stop("exception", out.pc, exception=out.number)
        return Stop(reason, out.pc)

    def halt(self) -> None:
        """Make a run on another thread return with reason "halted" (cores without count_ticks)."""
        self._lib.sd_halt(self._c)

    def invalidate(self, address: int, size: int) -> None:
        """Translate the code in the range again on this core."""
        self._lib.sd_invalidate(self._c, address, size)
