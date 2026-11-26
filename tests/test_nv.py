"""Tests for the GPU driver service (nvsrv) and the GPU memory manager, with requests the tests build."""

import struct

import pytest

from siicl import ipc, services
from siicl.kernel import Unimplemented
from siicl.services import gfx, nv
from tests.test_services import process, response, send, tipc


def ioctl3(fd, request, data=b"", out_sizes=(0x100, 0x100)):
    """An Ioctl3 request: the argument in a send static, the two outputs in receive statics."""
    return ipc.build(
        type=4,
        data=struct.pack("<IIII", ipc.CMIF_IN, 0, 12, 0) + struct.pack("<II", fd, request),
        statics=[ipc.Buffer(0x4800_0000, len(data), 0)],
        receive_list=[ipc.Buffer(0x4800_1000, out_sizes[0]), ipc.Buffer(0x4800_2000, out_sizes[1])],
    )


def open_device(p, session, path):
    p.cpu.write(0x4800_3000, path.encode() + b"\0")
    msg = ipc.build(
        type=4,
        data=struct.pack("<IIII", ipc.CMIF_IN, 0, 0, 0),
        statics=[ipc.Buffer(0x4800_3000, len(path) + 1, 0)],
    )
    text, reply = send(p, session, msg)
    return text, struct.unpack_from("<II", response(reply)[1])


def call_ioctl(p, session, fd, request, data):
    p.cpu.write(0x4800_0000, data)
    text, reply = send(p, session, ioctl3(fd, request, data))
    error = struct.unpack_from("<I", response(reply)[1])[0]
    return text, error, p.cpu.read(0x4800_1000, len(data)), p.cpu.read(0x4800_2000, 0x100)


@pytest.fixture
def driver():
    p = process()
    p.space.map(0x4800_0000, 0x4000, 3, 5, "buffers")
    return p, services.Session(nv.NvService(p))


def test_open_hands_out_fds_and_refuses_the_profiler(driver):
    p, session = driver
    text, (fd, error) = open_device(p, session, "/dev/nvhost-ctrl-gpu")
    assert (fd, error, text) == (1, 0, "nvsrv Open('/dev/nvhost-ctrl-gpu', fd 1)")
    _, (fd, error) = open_device(p, session, "/dev/nvhost-prof-gpu")
    assert error == nv.NV_NOT_SUPPORTED
    with pytest.raises(Unimplemented, match='device "/dev/nothing" is not implemented'):
        open_device(p, session, "/dev/nothing")


def test_characteristics_describe_the_switch_2_gpu(driver):
    p, session = driver
    fd = open_device(p, session, "/dev/nvhost-ctrl-gpu")[1][0]
    _, error, arg, out2 = call_ioctl(p, session, fd, 0xC0E04705, bytes(0xE0))
    arch, impl = struct.unpack_from("<II", out2)
    assert (error, arch | impl) == (0, 0x17E)  # GA10F in NVN's chip table
    assert struct.unpack_from("<Q", arg)[0] == nv.CHARACTERISTICS_SIZE and arg[0x10:0xE0] == out2[:0xD0]
    levels = struct.unpack_from("<9h", out2, 0x70)
    assert levels[0] < 0x13 and levels[4] >= 6  # below the SM mapping feature, at least NVN's minimum
    assert struct.unpack_from("<IIII", out2, 0x58)[1:] == (0xC797, 0xC7C0, 0xC76F)


def test_unknown_ioctls_end_the_run_with_their_request(driver):
    p, session = driver
    fd = open_device(p, session, "/dev/nvhost-ctrl-gpu")[1][0]
    with pytest.raises(Unimplemented, match=r"ioctl 0xc0084799 \(type 'G', number 153, size 0x8"):
        call_ioctl(p, session, fd, 0xC0084799, bytes(8))


def test_the_address_space_reports_two_regions(driver):
    p, session = driver
    fd = open_device(p, session, "/dev/nvhost-as-gpu")[1][0]
    call_ioctl(p, session, fd, 0x40284109, struct.pack("<IiIIQQQ", 5, 1, 0, 0, 0x800_0000, 0xC0_0000_0000, 0))
    _, error, arg, out2 = call_ioctl(p, session, fd, 0xC0404108, bytes(0x40))
    small = struct.unpack_from("<QI4xQ", out2, 0)
    big = struct.unpack_from("<QI4xQ", out2, 0x18)
    assert error == 0 and arg[0x10:0x40] == out2[:0x30]
    assert small == (0x800_0000, 0x1000, (0x4_0000_0000 - 0x800_0000) // 0x1000)
    assert big[0] + big[1] * big[2] == 0xC0_0000_0000


def test_the_supervisor_reports_partitions_that_hold_the_cpu_mirror():
    p = process()
    session = services.Session(gfx.Session497(p))
    send(p, session, tipc(203, b"Gpu\0"))
    present, low, high = struct.unpack_from("<QQQ", p.cpu.read(p.main_thread.tls, 40), 12)
    assert present and low <= 0x800_0000 and high > 0x80_0000_0000 + (4 << 30)
    send(p, session, tipc(202, struct.pack("<I", 1)))
    assert struct.unpack_from("<QQ", p.cpu.read(p.main_thread.tls, 28), 12) == gfx.GSV_PARTITION


def test_gpu_memory_is_created_mapped_and_signals_its_address_space():
    p = process()
    spaces = gfx.Session497(p)
    _, msg = send(p, services.Session(spaces), tipc(12, struct.pack("<I", 1)))
    sync = p.handles.get(msg.move_handles[0])
    manager = services.Session(gfx.GpuMemoryManager(p, spaces))
    send(p, manager, tipc(101, struct.pack("<QQ", 0x105_0000_0000, 0x105_0000_1FFF) + bytes(8)))
    handle = struct.unpack_from("<I", p.cpu.read(p.main_thread.tls, 16), 12)[0]
    send(p, manager, tipc(211, struct.pack("<IQQQQ", 1, 0x105_0000_0000, 0x105_0000_1FFF, handle, 0)))
    memory = gfx.gpu_memory(p)
    assert memory.cpu_address(1, 0x105_0000_0010) == 0x105_0000_0010
    assert memory.cpu_address(1, 0x105_0000_2000) is None
    assert sync.signaled
