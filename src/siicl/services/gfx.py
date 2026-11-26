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

from siicl.services.base import Call, Service, Session, command


class NvService(Service):
    """The GPU driver service the supervisor's command 1402 hands out (`nn::nvsrv`)."""

    name = "nvsrv"


class SupervisorObject(Service):
    """A placeholder for what a supervisor command hands out, until the sample shows what it is."""

    def __init__(self, process, number: int) -> None:
        super().__init__(process)
        self.name = f"gsv object from command {number}"


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
        call.move(Session(SupervisorObject(self.process, 202)))
        return 0

    # 204 takes two transfer memories (copied) with their sizes, and returns two moved handles: its
    # client reads them at offsets 12 and 16 and the result at 20. What they are is not known yet.
    @command(204, "command 204")
    def Command204(self, call: Call) -> int:
        call.note(call.payload[:24].hex())
        call.move(Session(SupervisorObject(self.process, 204)))
        call.move(Session(SupervisorObject(self.process, 204)))
        return 0

    @command(1402, "command 1402 (GPU driver session)")
    def OpenNvService(self, call: Call) -> int:
        call.move(Session(NvService(self.process)))
        return 0
