"""The Switch 2's graphics services, as far as the sample has used them.

The SDK boots the graphics system (`nn::gs::sdk::BootGraphicsSystem`) through a graphics
supervisor session, which the applet's ISelfController GetGraphicsSupervisor (command 2000) hands
out as a copied handle. The supervisor speaks TIPC, and its client in the SDK (`nn::gsv`) writes
the messages by hand. No public documentation describes its commands, so siicl names them by
number and by what the sample sends:

- 4: the process ID, a u64 (0x8000000000000006 in the sample), zeros, and a copy of the process handle
- 11: a copy of the process handle
- 101: no arguments
- 1501, 1502: no arguments (from `nn::gsv::shim::detail::InitializeInterModuleImageContext`)
- 1402: no arguments; the response carries one handle, a session for the GPU driver service
  (`nn::nvsrv`)
"""

from __future__ import annotations

import struct

from siicl.kernel import Event
from siicl.services.base import Call, Service, Session, command

# The header the supervisor writes at the start of the read-only transfer memory of command 204:
# 1, a value, a size below 0x1000, then offsets into the read-only memory (3rd, 5th and 7th words
# after the size) and into the read-write memory (4th and 6th). The values are guesses for now.
QUEUE_HEADER = (1, 0, 0x800, 0x40, 0x0, 0x80, 0x40, 0x1000)


class NvService(Service):
    """The GPU driver service the supervisor's command 1402 hands out (`nn::nvsrv`)."""

    name = "nvsrv"


class SupervisorObject(Service):
    """A placeholder for what a supervisor command hands out, until the sample shows what it is."""

    def __init__(self, process, number: int) -> None:
        super().__init__(process)
        self.name = f"gsv object from command {number}"


class Session202(Service):
    """The session supervisor command 202 hands out. It speaks TIPC. Its client reads a number from
    command 1 (it must be below 0x400), then sends it, capped at 63, with command 21. siicl answers 63."""

    name = "gsv session 202"

    @command(1, "command 1")
    def Command1(self, call: Call) -> int:
        call.push("I", 63)
        return 0

    @command(21, "command 21")
    def Command21(self, call: Call) -> int:
        call.note(f"{call.arg('I')}")
        return 0


class Session497(Service):
    """The session supervisor command 497 hands out. Its command 11 returns a range of bit positions,
    low then high; the client makes a 64-bit mask of the bits from low to high (low may not be above
    high or above 64). siicl offers the whole range, 0 to 63."""

    name = "gsv session 497"

    def __init__(self, process) -> None:
        super().__init__(process)
        self.events: dict[int, Event] = {}

    @command(11, "command 11 (bit range)")
    def BitRange(self, call: Call) -> int:
        call.push("II", 0, 63)
        return 0

    # A u32 index. In the sample the client then waits on the event command 12 gave for that index,
    # so siicl signals it at once, as if the work had finished.
    @command(21, "command 21")
    def Command21(self, call: Call) -> int:
        index = call.arg("I")
        call.note(f"index {index}")
        if index in self.events:
            self.events[index].signal(self.process)
        return 0

    # No arguments; the client reads one handle, then the result. What the handle is for is not known yet.
    @command(22, "command 22")
    def Command22(self, call: Call) -> int:
        call.move(Session(SupervisorObject(self.process, "497/22")))
        return 0

    # A u32 index (0 in the sample); the client reads one handle, then the result. The game then
    # waits on the handle, so it is an event's readable side.
    @command(12, "command 12 (event)")
    def Command12(self, call: Call) -> int:
        index = call.arg("I")
        call.note(f"index {index}")
        self.events.setdefault(index, Event(f"gsv session 497 event {index}"))
        call.move(self.events[index].readable)
        return 0


class DisplaySession(Service):
    """The session supervisor command 305 hands out, which the SDK's display library (`nn::vi2`) uses."""

    name = "vi2 session"

    def __init__(self, process) -> None:
        super().__init__(process)
        self.events: dict[int, Event] = {}

    # The client stores the u32 after the result and requires it to be below 0x400. siicl answers 1.
    @command(1, "command 1 (count)")
    def Count1(self, call: Call) -> int:
        call.push("I", 1)
        return 0

    # The client stores the u32 after the result and requires it to be below 8. siicl answers 1.
    @command(3, "command 3 (count)")
    def Count(self, call: Call) -> int:
        call.push("I", 1)
        return 0

    @command(21, "command 21")
    def Command21(self, call: Call) -> int:
        return 0

    # A display name ("default" in the sample) and a u32 (1); the client reads only the result.
    @command(101, "command 101 (open display)")
    def OpenDisplay(self, call: Call) -> int:
        name = call.payload[:8].split(b"\0", 1)[0].decode("ascii", "replace")
        call.note(f"{name!r}, {call.arg('I', 8)}")
        return 0

    # A u32 (1 in the sample); the client reads one handle, then the result, and polls the handle until
    # it is signaled. siicl hands out an event that is already signaled.
    @command(11, "command 11 (event)")
    def Event11(self, call: Call) -> int:
        index = call.arg("I")
        call.note(f"{index}")
        event = self.events.setdefault(index, Event(f"vi2 event {index}"))
        event.signal(self.process)
        call.move(event.readable)
        return 0


class GraphicsSupervisor(Service):
    name = "gsv IGraphicsSupervisor"

    @command(4, "command 4 (boot)")
    def Boot(self, call: Call) -> int:
        pid, region = call.args("QQ")
        call.note(f"process ID {pid:#x}, {region:#x}")
        return 0

    @command(11, "command 11 (process handle)")
    def ProcessHandle(self, call: Call) -> int:
        return 0

    @command(101, "command 101")
    def Command101(self, call: Call) -> int:
        return 0

    @command(1501, "command 1501 (image context)")
    def ImageContext1(self, call: Call) -> int:
        return 0

    @command(1502, "command 1502 (image context)")
    def ImageContext2(self, call: Call) -> int:
        return 0

    @command(201, "command 201")
    def Command201(self, call: Call) -> int:
        return 0

    # The client of 202 reads its result after one handle, so the response carries one. What the
    # handle is for is not known yet: siicl gives a session to a placeholder interface.
    @command(202, "command 202")
    def Command202(self, call: Call) -> int:
        call.move(Session(Session202(self.process)))
        return 0

    # 204 takes two transfer memories (copied) with their sizes, and returns two moved handles: its
    # client reads them at offsets 12 and 16 and the result at 20. What they are is not known yet.
    @command(204, "command 204")
    def Command204(self, call: Call) -> int:
        call.note(call.payload[:24].hex())
        shared, server = (self.process.handles.get(h) for h in call.message.copy_handles[:2])
        shared.kept = server.kept = True
        self.queues = (shared, server)
        self.process.cpu.write(server.address, struct.pack("<8I", *QUEUE_HEADER))
        call.move(Session(SupervisorObject(self.process, 204)))
        call.move(Session(SupervisorObject(self.process, 204)))
        return 0

    # From nn::gs::sdk::BootNativeMemoryManagement(MemoryBlockModeBits): the requested mode bits; the
    # client keeps the u64 after the result as the mode it may use (CreateNativeMemory tests its bit 1).
    # siicl grants what was asked.
    @command(115, "command 115 (memory block mode)")
    def MemoryBlockMode(self, call: Call) -> int:
        bits = call.arg("Q")
        call.note(f"mode bits {bits:#x}")
        call.push("Q", bits)
        return 0

    # From nn::gs::sdk::BootGraphicsFrameManagement: no arguments, only a result.
    @command(1101, "command 1101 (frame management)")
    def FrameManagement1(self, call: Call) -> int:
        return 0

    @command(1102, "command 1102 (frame management)")
    def FrameManagement2(self, call: Call) -> int:
        return 0

    # From nn::gpu::sdk::BootWithRemoteMemory: a u64 (4 in the sample); the client reads one handle,
    # then the result.
    @command(497, "command 497")
    def Command497(self, call: Call) -> int:
        call.note(f"{call.arg('Q'):#x}")
        call.move(Session(Session497(self.process)))
        return 0

    @command(401, "command 401")
    def Command401(self, call: Call) -> int:
        return 0

    @command(402, "command 402")
    def Command402(self, call: Call) -> int:
        return 0

    @command(301, "command 301")
    def Command301(self, call: Call) -> int:
        return 0

    # Two u32s (0x20000 and 3 in the sample); the client reads one handle, then the result: a session
    # for the display library.
    @command(305, "command 305 (display session)")
    def Command305(self, call: Call) -> int:
        a, b = call.args("II")
        call.note(f"{a:#x}, {b}")
        call.move(Session(DisplaySession(self.process)))
        return 0

    @command(1402, "command 1402 (GPU driver session)")
    def OpenNvService(self, call: Call) -> int:
        call.move(Session(NvService(self.process)))
        return 0
