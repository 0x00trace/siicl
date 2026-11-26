"""Reader for NPDM process descriptors (magic `META`).

The layout follows https://switchbrew.org/wiki/NPDM. The kernel capability
descriptors say which SVCs, cores and priorities the process may use.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

MAGIC = b"META"
ADDRESS_SPACE_TYPES = {
    0: "32-bit",
    1: "64-bit, 36-bit address space",
    2: "32-bit, no reserved region",
    3: "64-bit, 39-bit address space",
}

KAC_THREAD_INFO = 3
KAC_ENABLE_SYSTEM_CALLS = 4
KAC_MEMORY_MAP = 6
KAC_IO_MEMORY_MAP = 7
KAC_MEMORY_REGION_MAP = 10
KAC_ENABLE_INTERRUPTS = 11
KAC_MISC_PARAMS = 13
KAC_KERNEL_VERSION = 14
KAC_HANDLE_TABLE_SIZE = 15
KAC_MISC_FLAGS = 16
KAC_NAMES = {
    KAC_THREAD_INFO: "ThreadInfo",
    KAC_ENABLE_SYSTEM_CALLS: "EnableSystemCalls",
    KAC_MEMORY_MAP: "MemoryMap",
    KAC_IO_MEMORY_MAP: "IoMemoryMap",
    KAC_MEMORY_REGION_MAP: "MemoryRegionMap",
    KAC_ENABLE_INTERRUPTS: "EnableInterrupts",
    KAC_MISC_PARAMS: "MiscParams",
    KAC_KERNEL_VERSION: "KernelVersion",
    KAC_HANDLE_TABLE_SIZE: "HandleTableSize",
    KAC_MISC_FLAGS: "MiscFlags",
}


class NpdmError(ValueError):
    """The data is not an NPDM, or it is damaged."""


@dataclass
class KernelCaps:
    svcs: set[int] = field(default_factory=set)
    lowest_priority: int | None = None
    highest_priority: int | None = None
    min_core: int | None = None
    max_core: int | None = None
    kernel_version: tuple[int, int] | None = None
    handle_table_size: int | None = None
    program_type: int | None = None
    misc_flags: int | None = None
    descriptor_counts: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class ServiceAccess:
    name: str
    server: bool


@dataclass
class Npdm:
    flags: int
    main_thread_priority: int
    main_thread_core: int
    system_resource_size: int
    version: int
    main_thread_stack_size: int
    name: str
    product_code: str
    program_id: int
    fs_permissions: int | None
    services: list[ServiceAccess]
    caps: KernelCaps
    acid_program_id_min: int | None = None
    acid_program_id_max: int | None = None
    acid_flags: int | None = None
    acid_caps: KernelCaps | None = None

    @property
    def is_64bit(self) -> bool:
        return bool(self.flags & 1)

    @property
    def address_space_type(self) -> int:
        return (self.flags >> 1) & 7

    @property
    def address_space(self) -> str:
        return ADDRESS_SPACE_TYPES.get(self.address_space_type, f"unknown ({self.address_space_type})")


def trailing_ones(value: int) -> int:
    n = 0
    while value & 1:
        value >>= 1
        n += 1
    return n


def parse_kernel_caps(data: bytes) -> KernelCaps:
    caps = KernelCaps()
    for (desc,) in struct.iter_unpack("<I", data[: len(data) // 4 * 4]):
        kind = trailing_ones(desc)
        if kind == 32:
            continue  # 0xFFFFFFFF is padding
        label = KAC_NAMES.get(kind, f"type {kind}")
        caps.descriptor_counts[label] = caps.descriptor_counts.get(label, 0) + 1
        if kind == KAC_THREAD_INFO:
            caps.lowest_priority = (desc >> 4) & 0x3F
            caps.highest_priority = (desc >> 10) & 0x3F
            caps.min_core = (desc >> 16) & 0xFF
            caps.max_core = (desc >> 24) & 0xFF
        elif kind == KAC_ENABLE_SYSTEM_CALLS:
            mask, index = (desc >> 5) & 0xFFFFFF, (desc >> 29) & 7
            caps.svcs.update(index * 24 + bit for bit in range(24) if mask & (1 << bit))
        elif kind == KAC_MISC_PARAMS:
            caps.program_type = (desc >> 14) & 7
        elif kind == KAC_KERNEL_VERSION:
            caps.kernel_version = ((desc >> 19) & 0x1FFF, (desc >> 15) & 0xF)
        elif kind == KAC_HANDLE_TABLE_SIZE:
            caps.handle_table_size = (desc >> 16) & 0x3FF
        elif kind == KAC_MISC_FLAGS:
            caps.misc_flags = desc >> 17
    return caps


def parse_services(data: bytes) -> list[ServiceAccess]:
    out = []
    i = 0
    while i < len(data):
        ctrl = data[i]
        i += 1
        if ctrl == 0:
            break
        length = (ctrl & 7) + 1
        out.append(ServiceAccess(data[i : i + length].decode("ascii", "replace"), bool(ctrl & 0x80)))
        i += length
    return out


def _section(raw: bytes, off: int, size: int, what: str) -> bytes:
    if off + size > len(raw):
        raise NpdmError(f"{what} runs past the end of the NPDM")
    return raw[off : off + size]


def _cstr(b: bytes) -> str:
    return b.split(b"\0", 1)[0].decode("utf-8", "replace")


def parse(raw: bytes) -> Npdm:
    try:
        return _parse(raw)
    except struct.error as e:
        raise NpdmError(f"truncated NPDM: {e}") from None


def _parse(raw: bytes) -> Npdm:
    if raw[:4] != MAGIC:
        raise NpdmError("not an NPDM (magic is not 'META')")
    aci_off, aci_size, acid_off, acid_size = struct.unpack_from("<IIII", raw, 0x70)

    aci = _section(raw, aci_off, aci_size, "ACI0")
    if aci[:4] != b"ACI0":
        raise NpdmError("ACI0 magic not found")
    program_id = struct.unpack_from("<Q", aci, 0x10)[0]
    fac_off, fac_size, sac_off, sac_size, kac_off, kac_size = struct.unpack_from("<IIIIII", aci, 0x20)
    fac = _section(aci, fac_off, fac_size, "ACI0 FS access")
    services = parse_services(_section(aci, sac_off, sac_size, "ACI0 service access"))
    caps = parse_kernel_caps(_section(aci, kac_off, kac_size, "ACI0 kernel capabilities"))

    npdm = Npdm(
        flags=raw[0x0C],
        main_thread_priority=raw[0x0E],
        main_thread_core=raw[0x0F],
        system_resource_size=struct.unpack_from("<I", raw, 0x14)[0],
        version=struct.unpack_from("<I", raw, 0x18)[0],
        main_thread_stack_size=struct.unpack_from("<I", raw, 0x1C)[0],
        name=_cstr(raw[0x20:0x30]),
        product_code=_cstr(raw[0x30:0x40]),
        program_id=program_id,
        fs_permissions=struct.unpack_from("<Q", fac, 4)[0] if len(fac) >= 12 else None,
        services=services,
        caps=caps,
    )

    if acid_size >= 0x240:
        acid = _section(raw, acid_off, acid_size, "ACID")
        if acid[0x200:0x204] == b"ACID":
            npdm.acid_flags = struct.unpack_from("<I", acid, 0x20C)[0]
            npdm.acid_program_id_min, npdm.acid_program_id_max = struct.unpack_from("<QQ", acid, 0x210)
            k_off, k_size = struct.unpack_from("<II", acid, 0x230)
            npdm.acid_caps = parse_kernel_caps(_section(acid, k_off, k_size, "ACID kernel capabilities"))
    return npdm
