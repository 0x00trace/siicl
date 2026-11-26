"""Run AArch64 guest code in Unicorn, the CPU emulator built on QEMU.

`Cpu.run()` runs guest code until siicl has to act, and returns a `Stop` that says why: an SVC,
a memory fault, an undefined instruction, a BRK, or the end of an instruction budget. The caller
handles the stop and calls `run()` again. After an SVC the PC already points past it.

Unicorn's "max" CPU model implements every architecture feature the scanner finds in game code:
LSE atomics, LDAPR, dot product, half precision, CRC32, AES and SHA. It runs code at EL1 with the
pointer authentication keys off (SCTLR_EL1.EnIA, EnIB, EnDA and EnDB are clear), so PAC
instructions leave pointers unchanged, AUT instructions always succeed, and RETAB returns like RET.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import unicorn
from unicorn import arm64_const as a64

PAGE = 0x1000
READ, WRITE, EXECUTE = unicorn.UC_PROT_READ, unicorn.UC_PROT_WRITE, unicorn.UC_PROT_EXEC

_X = [getattr(a64, f"UC_ARM64_REG_X{i}") for i in range(31)]
_RUN_END = 0xFFFF_FFFF_FFFF_FFFC  # never mapped, so a run never ends by reaching it

# QEMU's exception numbers, as Unicorn passes them to an interrupt hook.
_EXCP_UDEF, _EXCP_SWI, _EXCP_BKPT = 1, 2, 7
_SVC_MASK, _SVC_BITS = 0xFFE0001F, 0xD4000001
_BRK_MASK, _BRK_BITS = 0xFFE0001F, 0xD4200000

_BAD_ACCESS = {
    unicorn.UC_MEM_READ_UNMAPPED: ("read", "unmapped"),
    unicorn.UC_MEM_WRITE_UNMAPPED: ("write", "unmapped"),
    unicorn.UC_MEM_FETCH_UNMAPPED: ("fetch", "unmapped"),
    unicorn.UC_MEM_READ_PROT: ("read", "protected"),
    unicorn.UC_MEM_WRITE_PROT: ("write", "protected"),
    unicorn.UC_MEM_FETCH_PROT: ("fetch", "protected"),
}


class CpuError(RuntimeError):
    """Unicorn reported an error that is not something the guest code did."""


class GuestMemoryError(ValueError):
    """siicl tried to read or write guest memory that is not mapped."""


@dataclass(frozen=True)
class Stop:
    """Why a run ended.

    reason is one of:
    - "svc": an SVC instruction at pc, with number svc. The CPU's PC is past it.
    - "fault": the instruction at pc made a read, write or fetch (access) at address, which was
      unmapped or protected (kind). For a fetch, pc and address are the same.
    - "undefined": the instruction at pc is not one the CPU implements.
    - "break": a BRK instruction at pc, with its immediate in svc.
    - "limit": the instruction budget ran out; pc is the next instruction to run.
    - "exception": another CPU exception, with QEMU's number in exception.
    """

    reason: str
    pc: int
    svc: int | None = None
    address: int | None = None
    access: str | None = None
    kind: str | None = None
    exception: int | None = None

    def describe(self) -> str:
        if self.reason == "svc":
            return f"SVC 0x{self.svc:02X}"
        if self.reason == "fault":
            return f"{self.kind} {self.access} at {self.address:#x}"
        if self.reason == "undefined":
            return "undefined instruction"
        if self.reason == "break":
            return f"BRK #{self.svc:#x}"
        if self.reason == "limit":
            return "instruction budget used up"
        return f"CPU exception {self.exception}"


class Cpu:
    """One AArch64 CPU and the guest memory it sees."""

    def __init__(self) -> None:
        self.uc = unicorn.Uc(unicorn.UC_ARCH_ARM64, unicorn.UC_MODE_ARM)
        self.uc.ctl_set_cpu_model(a64.UC_CPU_ARM64_MAX)
        self._stop: Stop | None = None
        self.uc.hook_add(unicorn.UC_HOOK_INTR, self._on_interrupt)
        self.uc.hook_add(unicorn.UC_HOOK_MEM_INVALID, self._on_bad_access)
        self.uc.hook_add(unicorn.UC_HOOK_INSN_INVALID, self._on_undefined)

    # Memory

    def map(self, address: int, size: int, perm: int) -> None:
        _check_pages(address, size)
        self.uc.mem_map(address, size, perm)

    def map_host(self, address: int, size: int, perm: int, host_address: int) -> None:
        """Map guest pages straight onto host memory at host_address, which must stay allocated."""
        _check_pages(address, size)
        self.uc.mem_map_ptr(address, size, perm, host_address)

    def protect(self, address: int, size: int, perm: int) -> None:
        _check_pages(address, size)
        self.uc.mem_protect(address, size, perm)

    def unmap(self, address: int, size: int) -> None:
        _check_pages(address, size)
        self.uc.mem_unmap(address, size)

    def read(self, address: int, size: int) -> bytes:
        try:
            return bytes(self.uc.mem_read(address, size))
        except unicorn.UcError as err:
            raise GuestMemoryError(f"cannot read {size:#x} bytes at {address:#x}: {err}") from None

    def write(self, address: int, data: bytes) -> None:
        try:
            self.uc.mem_write(address, bytes(data))
        except unicorn.UcError as err:
            raise GuestMemoryError(f"cannot write {len(data):#x} bytes at {address:#x}: {err}") from None

    def read_u32(self, address: int) -> int:
        return struct.unpack("<I", self.read(address, 4))[0]

    def read_u64(self, address: int) -> int:
        return struct.unpack("<Q", self.read(address, 8))[0]

    def write_u32(self, address: int, value: int) -> None:
        self.write(address, struct.pack("<I", value & 0xFFFF_FFFF))

    def write_u64(self, address: int, value: int) -> None:
        self.write(address, struct.pack("<Q", value & 0xFFFF_FFFF_FFFF_FFFF))

    # Registers

    def x(self, n: int) -> int:
        return self.uc.reg_read(_X[n])

    def set_x(self, n: int, value: int) -> None:
        self.uc.reg_write(_X[n], value & 0xFFFF_FFFF_FFFF_FFFF)

    @property
    def pc(self) -> int:
        return self.uc.reg_read(a64.UC_ARM64_REG_PC)

    @pc.setter
    def pc(self, value: int) -> None:
        self.uc.reg_write(a64.UC_ARM64_REG_PC, value)

    @property
    def sp(self) -> int:
        return self.uc.reg_read(a64.UC_ARM64_REG_SP)

    @sp.setter
    def sp(self, value: int) -> None:
        self.uc.reg_write(a64.UC_ARM64_REG_SP, value)

    @property
    def tpidrro_el0(self) -> int:
        return self.uc.reg_read(a64.UC_ARM64_REG_TPIDRRO_EL0)

    @tpidrro_el0.setter
    def tpidrro_el0(self, value: int) -> None:
        self.uc.reg_write(a64.UC_ARM64_REG_TPIDRRO_EL0, value)

    @property
    def tpidr_el0(self) -> int:
        return self.uc.reg_read(a64.UC_ARM64_REG_TPIDR_EL0)

    @tpidr_el0.setter
    def tpidr_el0(self, value: int) -> None:
        self.uc.reg_write(a64.UC_ARM64_REG_TPIDR_EL0, value)

    def save_context(self) -> object:
        """The registers, system registers included, as an object restore_context() takes back."""
        return self.uc.context_save()

    def restore_context(self, context: object) -> None:
        self.uc.context_restore(context)

    # Running

    def run(self, *, limit: int | None = None) -> Stop:
        """Run from the current PC until the code needs siicl, or for at most limit instructions."""
        if limit is not None and limit <= 0:
            raise ValueError("limit must be positive")
        self._stop = None
        try:
            self.uc.emu_start(self.pc, _RUN_END, count=limit or 0)
        except unicorn.UcError as err:
            if self._stop is None and err.errno == unicorn.UC_ERR_INSN_INVALID:
                self._stop = Stop("undefined", self.pc)
            if self._stop is None:
                raise CpuError(f"Unicorn stopped at {self.pc:#x}: {err}") from None
        return self._stop or Stop("limit", self.pc)

    def _on_interrupt(self, uc, number: int, _data) -> None:
        pc = self.pc
        if number == _EXCP_SWI and self._word(pc - 4) & _SVC_MASK == _SVC_BITS:
            self._stop = Stop("svc", pc - 4, svc=(self._word(pc - 4) >> 5) & 0xFFFF)
        elif number == _EXCP_BKPT and self._word(pc) & _BRK_MASK == _BRK_BITS:
            self._stop = Stop("break", pc, svc=(self._word(pc) >> 5) & 0xFFFF)
        elif number == _EXCP_UDEF:
            self._stop = Stop("undefined", pc)
        else:
            self._stop = Stop("exception", pc, exception=number)
        uc.emu_stop()

    def _on_bad_access(self, uc, access: int, address: int, _size, _value, _data) -> bool:
        what, kind = _BAD_ACCESS.get(access, ("access", "invalid"))
        pc = address if what == "fetch" else self.pc
        self._stop = Stop("fault", pc, address=address, access=what, kind=kind)
        return False

    def _on_undefined(self, uc, _data) -> bool:
        self._stop = Stop("undefined", self.pc)
        return False

    def _word(self, address: int) -> int:
        try:
            return self.read_u32(address)
        except GuestMemoryError:
            return 0


def _check_pages(address: int, size: int) -> None:
    if address % PAGE or size % PAGE or size <= 0:
        raise ValueError(f"{address:#x}+{size:#x} is not a whole number of {PAGE:#x}-byte pages")
