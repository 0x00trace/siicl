"""The file system service, "fsp-srv": the game's romfs, and later its save data."""

from __future__ import annotations

from siicl.services.base import Call, Service, command, service, stub

RESULT_OUT_OF_RANGE = (3005 << 9) | 2  # fs module (2), description 3005


class Storage(Service):
    """An IStorage over an image with read(offset, size) and size, such as the romfs."""

    name = "fs IStorage"

    def __init__(self, process, image, label: str) -> None:
        super().__init__(process)
        self.image = image
        self.label = label

    @command(0)
    def Read(self, call: Call) -> int:
        offset, size = call.args("qq")
        out = call.out_buffer(0)
        size = min(size, out.size)
        call.note(f"{self.label}, offset {offset:#x}, size {size:#x}")
        if offset < 0 or size < 0:
            return RESULT_OUT_OF_RANGE
        call.write(out, self.image.read(offset, size))
        return 0

    @command(4)
    def GetSize(self, call: Call) -> int:
        call.push("q", self.image.size)
        return 0

    Flush = stub(2, "Flush")


@service("fsp-srv")
class FileSystemProxy(Service):
    name = "fsp-srv"

    SetCurrentProcess = stub(1, "SetCurrentProcess", "Q")
    DisableAutoSaveDataCreation = stub(1003, "DisableAutoSaveDataCreation")
    GetGlobalAccessLogMode = stub(1005, "GetGlobalAccessLogMode", None, "I", 0)
    OutputAccessLogToSdCard = stub(1006, "OutputAccessLogToSdCard")
    GetProgramIndexForAccessLog = stub(1011, "GetProgramIndexForAccessLog", None, "II", 0, 1)
    FlushAccessLogOnSdCard = stub(1016, "FlushAccessLogOnSdCard")

    @command(200)
    def OpenDataStorageByCurrentProcess(self, call: Call) -> int:
        romfs = getattr(self.process, "romfs", None)
        if romfs is None:
            return (1 << 9) | 2  # fs: path not found
        call.push_object(Storage(self.process, romfs, "romfs"))
        return 0

    @command(202)
    def OpenDataStorageWithProgramIndex(self, call: Call) -> int:
        call.note(f"program index {call.arg('B')}")
        return self.OpenDataStorageByCurrentProcess(call)
