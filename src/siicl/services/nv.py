"""The GPU driver service, "nvsrv" on the Switch 2, which the graphics supervisor hands out.

In the sample it uses the commands of the Switch 1's "nvdrv" (see switchbrew's NV_services page):
Open a device by path, send it ioctls, Close it. An ioctl's request number has the Linux layout:
the direction in bits 30-31, the argument size in bits 16-29, a type byte in bits 8-15 and the
number in bits 0-7. siicl finds a device's handler by type and number, since argument sizes
differ between the Switch 1 and the Switch 2.

A device or ioctl that siicl does not implement ends the run, with its request and its input.
"""

from __future__ import annotations

import struct
from collections.abc import Callable
from typing import ClassVar

from siicl.kernel import Event, Unimplemented
from siicl.services.base import Call, Service, command, stub

NV_OK = 0
NV_NOT_SUPPORTED = 2
NV_BAD_PARAMETER = 4


def describe_request(request: int) -> str:
    direction = ("none", "write", "read", "read and write")[request >> 30]
    kind = (request >> 8) & 0xFF
    label = chr(kind) if 0x20 <= kind < 0x7F else f"{kind:#x}"
    return f"{request:#010x} (type '{label}', number {request & 0xFF}, size {(request >> 16) & 0x3FFF:#x}, {direction})"


def ioctl(kind: str, number: int):
    """Mark a device method as the handler of ioctls with this type character and number."""

    def mark(fn):
        fn._ioctl = (ord(kind), number)
        return fn

    return mark


class Device:
    """An open device. Its ioctl handlers take (call, request, data) and return (error, data, data2):
    the argument written back, and for Ioctl3 a second output."""

    path: ClassVar[str] = "?"
    ioctls: ClassVar[dict[tuple[int, int], Callable]] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.ioctls = {}
        for klass in reversed(cls.__mro__):
            for attr in vars(klass).values():
                if hasattr(attr, "_ioctl"):
                    cls.ioctls[attr._ioctl] = attr

    def __init__(self, process) -> None:
        self.process = process


DEVICES: dict[str, type[Device]] = {}

# Devices a game may not open, as on a retail console: Open fails. If the sample opens the GPU
# profiler, NVN closes it and every other device at once, so a retail console evidently refuses it.
UNAVAILABLE = {"/dev/nvhost-prof-gpu"}


def device(cls: type[Device]) -> type[Device]:
    DEVICES[cls.path] = cls
    return cls


def state(process) -> dict:
    st = process.services.setdefault("nv", {})
    if not st:
        st["fds"] = {}
        st["next_fd"] = 1
    return st


class NvService(Service):
    name = "nvsrv"

    @command(0)
    def Open(self, call: Call) -> int:
        path = call.in_buffer(0).split(b"\0", 1)[0].decode("ascii", "replace")
        call.note(repr(path))
        if path in UNAVAILABLE:
            call.note("not available")
            call.push("II", 0, NV_NOT_SUPPORTED)
            return 0
        cls = DEVICES.get(path)
        if cls is None:
            raise Unimplemented(f'the GPU driver device "{path}" is not implemented')
        st = state(self.process)
        fd = st["next_fd"]
        st["next_fd"] += 1
        st["fds"][fd] = cls(self.process)
        call.note(f"fd {fd}")
        call.push("II", fd, NV_OK)
        return 0

    def _ioctl(self, call: Call, outputs: int) -> int:
        fd, request = call.args("II")
        dev = state(self.process)["fds"].get(fd)
        call.note(f"fd {fd}, {describe_request(request)}")
        if dev is None:
            call.push("I", NV_BAD_PARAMETER)
            return 0
        data = call.in_buffer(0)
        handler = dev.ioctls.get(((request >> 8) & 0xFF, request & 0xFF))
        if handler is None:
            raise Unimplemented(
                f"GPU driver ioctl {describe_request(request)} on {dev.path} is not implemented "
                f"(input {data.hex() or 'none'})"
            )
        error, out, out2 = handler(dev, call, request, bytearray(data))
        call.write(call.out_buffer(0), bytes(out))
        if outputs == 2 and out2 is not None:
            call.write(call.out_buffer(1), bytes(out2))
        call.push("I", error)
        return 0

    @command(1)
    def Ioctl(self, call: Call) -> int:
        return self._ioctl(call, 1)

    @command(2)
    def Close(self, call: Call) -> int:
        fd = call.arg("I")
        call.note(f"fd {fd}")
        state(self.process)["fds"].pop(fd, None)
        call.push("I", NV_OK)
        return 0

    @command(11)
    def Ioctl2(self, call: Call) -> int:
        return self._ioctl(call, 1)

    @command(12)
    def Ioctl3(self, call: Call) -> int:
        return self._ioctl(call, 2)

    # An event a device signals, such as an error notifier: the fd and the event's number.
    @command(4)
    def QueryEvent(self, call: Call) -> int:
        fd, number = call.args("II")
        call.note(f"fd {fd}, event {number}")
        events = state(self.process).setdefault("events", {})
        event = events.setdefault((fd, number), Event(f"GPU driver fd {fd} event {number}"))
        call.copy(event.readable)
        call.push("I", NV_OK)
        return 0

    SetAruid = stub(8, "SetAruid", None, "I", NV_OK)
    SetGraphicsFirmwareMemoryMarginEnabled = stub(13, "SetGraphicsFirmwareMemoryMarginEnabled", "B")


# The GPU siicl describes: the Switch 2's GA10F. The Switch 2's characteristics structure is 0xD0
# bytes. Up to 0x70 it has the fields of the Switch 1's 0xA0-byte structure (Linux nvgpu's
# nvgpu_gpu_characteristics). At 0x70 come nine signed 16-bit values that NVN reads and compares
# like nvgpu's interface levels (the last ioctl number of each driver interface): it refuses a value
# at 0x78 below 6 and enables features above 7, 0xA, 0x12, 0x1F, 0x24, 0x26 and 0x2A. siicl gives 0x40
# to all but the first, and puts the Switch 1's remaining fields after them. See docs/boot.md.
CHARACTERISTICS = dict(
    arch=0x170,  # NVN looks up arch | impl in its chip table: 0x17E is GA10F, the Switch 2's GPU
    impl=0xE,
    rev=0xA1,
    num_gpc=1,
    l2_cache_size=0x100000,
    video_memory_size=0,
    num_tpc_per_gpc=6,
    bus_type=0x20,  # AXI
    big_page_size=0x20000,
    compression_page_size=0x20000,
    pde_coverage_bit_count=0x1B,
    available_big_page_sizes=0x30000,
    gpc_mask=1,
    sm_version=0x807,
    spa_version=0x807,
    warp_count=0x30,
    va_bit_count=0x28,
    flags=0x55,
    twod_class=0x902D,
    threed_class=0xC797,  # AMPERE_B
    compute_class=0xC7C0,  # AMPERE_COMPUTE_B
    gpfifo_class=0xC76F,  # AMPERE_CHANNEL_GPFIFO_B
    inline_to_memory_class=0xA140,
    dma_copy_class=0xC7B5,  # AMPERE_DMA_COPY_B
    # The first is 0x12: from 0x13 on, NVN maps its SMs with the VsmsMapping ioctl and checks the
    # result, and siicl does not know that ioctl's format yet.
    interface_levels=(0x12,) + (0x40,) * 8,
    max_fbps_count=1,
    max_ltc_per_fbp=2,
    max_lts_per_ltc=2,
    max_gpc_count=1,
    rop_l2_en_mask=0x21D70,
    chipname=b"ga10f",
)
CHARACTERISTICS_SIZE = 0xD0


def characteristics() -> bytes:
    c = CHARACTERISTICS
    raw = struct.pack(
        "<IIIIQQIIIIIIIIIIIIQIIIIII9hxxIIIIIIII8sQ",
        c["arch"], c["impl"], c["rev"], c["num_gpc"],
        c["l2_cache_size"], c["video_memory_size"],
        c["num_tpc_per_gpc"], c["bus_type"], c["big_page_size"], c["compression_page_size"],
        c["pde_coverage_bit_count"], c["available_big_page_sizes"], c["gpc_mask"],
        c["sm_version"], c["spa_version"], c["warp_count"], c["va_bit_count"], 0,
        c["flags"],
        c["twod_class"], c["threed_class"], c["compute_class"], c["gpfifo_class"],
        c["inline_to_memory_class"], c["dma_copy_class"],
        *c["interface_levels"],
        c["max_fbps_count"], 0, c["max_ltc_per_fbp"], c["max_lts_per_ltc"], 0, c["max_gpc_count"],
        c["rop_l2_en_mask"], 0,
        c["chipname"], 0,
    )  # fmt: skip
    assert len(raw) <= CHARACTERISTICS_SIZE
    return raw.ljust(CHARACTERISTICS_SIZE, b"\0")


@device
class NvHostCtrlGpu(Device):
    """Queries about the GPU itself."""

    path = "/dev/nvhost-ctrl-gpu"

    # The argument is the buffer size and address, then the characteristics (on the Switch 1 the
    # same, with 0xA0 bytes); Ioctl3 also returns the characteristics in its second output.
    @ioctl("G", 5)
    def GetCharacteristics(self, call: Call, request: int, data: bytearray):
        info = characteristics()
        size = min(len(info), max(0, len(data) - 0x10))
        struct.pack_into("<Q", data, 0, size)
        data[0x10 : 0x10 + size] = info[:size]
        call.note(f"{size:#x} bytes")
        return NV_OK, data, info[:size]

    # On the Switch 1 this is VsmsMapping, with a 4-byte argument; here it is 8 bytes. siicl leaves it
    # zeroed: every SM on GPC 0, TPC 0.
    @ioctl("G", 19)
    def VsmsMapping(self, call: Call, request: int, data: bytearray):
        return NV_OK, data, None

    # The argument is the mask buffer's size and address, then the mask (as on the Switch 1); Ioctl3
    # also returns the mask in its second output. One GPC with six TPCs.
    @ioctl("G", 6)
    def GetTpcMasks(self, call: Call, request: int, data: bytearray):
        mask = (1 << CHARACTERISTICS["num_tpc_per_gpc"]) - 1
        size = struct.unpack_from("<I", data, 0)[0] or 4
        if len(data) >= 0x18:
            struct.pack_into("<Q", data, 0x10, mask)
        return NV_OK, data, mask.to_bytes(8, "little")[:size]


@device
class NvHostAsGpu(Device):
    """A GPU virtual address space: the range the game sets up, the parts it reserves, and what is mapped."""

    path = "/dev/nvhost-as-gpu"

    def __init__(self, process) -> None:
        super().__init__(process)
        self.start = self.end = 0
        self.next = 0
        self.mappings: dict[int, tuple[int, int]] = {}  # GPU address -> (size, what it maps)

    # As on the Switch 1 (InitializeEx): a first u32 (5 in the sample, where the Switch 1 has the big
    # page size), the ctrl-gpu fd, flags, then the range of GPU addresses: start, end, split.
    @ioctl("A", 9)
    def InitializeEx(self, call: Call, request: int, data: bytearray):
        first, as_fd, flags, _, start, end, split = struct.unpack_from("<IiIIQQQ", data)
        self.start, self.end, self.next = start, end, start
        call.note(f"{first:#x}, range {start:#x}-{end:#x}, split {split:#x}, flags {flags:#x}")
        return NV_OK, data, None

    SMALL_PAGE, BIG_PAGE, SPLIT = 0x1000, 0x20000, 0x4_0000_0000

    # As on the Switch 1 (GetVaRegions): the buffer size, then two regions, each an offset, a page
    # size and a page count. siicl splits the range at 16 GiB: small pages below, big pages above.
    @ioctl("A", 8)
    def GetVaRegions(self, call: Call, request: int, data: bytearray):
        split = min(max(self.SPLIT, self.start), self.end)
        regions = (
            (self.start, self.SMALL_PAGE, (split - self.start) // self.SMALL_PAGE),
            (split, self.BIG_PAGE, (self.end - split) // self.BIG_PAGE),
        )
        struct.pack_into("<QI4x", data, 0, 0, 0x30)
        for i, (offset, page, pages) in enumerate(regions):
            struct.pack_into("<QI4xQ", data, 0x10 + 0x18 * i, offset, page, pages)
        call.note(", ".join(f"{o:#x}+{p * n:#x} in {p:#x} pages" for o, p, n in regions))
        return NV_OK, data, bytes(data[0x10:0x40])  # Ioctl3 also returns the regions in its second output
