"""The time services, "time:u", "time:a" and "time:s".

Games read the clocks mostly from the time shared memory, which holds the steady clock's base and
the system clocks' contexts. Each value is kept twice, with a counter whose low bit says which copy
is current. siicl's clocks start at a fixed date, BOOT_POSIX_TIME, so runs are repeatable, and the
steady clock counts from zero at boot.
"""

from __future__ import annotations

import datetime
import struct

from siicl.kernel import TICK_HZ, Event, SharedMemory
from siicl.services.base import Call, Service, command, service, stub

BOOT_POSIX_TIME = 1_767_225_600  # 2026-01-01 00:00:00 UTC
CLOCK_SOURCE_ID = bytes.fromhex("5349494342530000a1b2c3d4e5f60718")
SHARED_MEMORY_SIZE = 0x1000
LOCATION = b"UTC"


def steady_seconds(process) -> int:
    return process.ticks // TICK_HZ


def system_clock_context() -> bytes:
    """offset (seconds), then the steady clock time point it counts from (seconds, source ID)."""
    return struct.pack("<qq", BOOT_POSIX_TIME, 0) + CLOCK_SOURCE_ID


INT64_MAX = (1 << 63) - 1


def time_shared_memory() -> SharedMemory:
    """The time shared memory, as the Switch 2 SDK reads it. Each value has a counter and two copies:

    - 0x00: the steady clock's base time (ns) and source ID, as on the Switch 1 (copies at 0x08, 0x20)
    - 0x38: the user system clock's context (copies of 0x20 bytes at 0x40 and 0x60)
    - 0x80: the network system clock's context (copies at 0x88 and 0xA8)
    - 0xC8: whether the user clock corrects itself automatically (bytes at 0xCC and 0xCD)
    - 0xD0: the steady clock's adjustment (copies of 0x38 bytes at 0xD8 and 0x110): base (ns), multiplier,
      shift, offset (ns), bound, source ID. The steady time in ns is the time since boot plus
      ((time since boot - base) * multiplier >> shift) + offset, which is capped at bound when the
      multiplier is not negative and kept at least bound when it is. The source ID must not be zero.
    """
    shared = SharedMemory("time", SHARED_MEMORY_SIZE)
    view = shared.view
    view[0x08 : 0x08 + 0x18] = struct.pack("<q", 0) + CLOCK_SOURCE_ID
    for base in (0x38, 0x80):  # the user and the network system clock
        view[base + 8 : base + 8 + 0x20] = system_clock_context()
    view[0xCC] = 0
    view[0xD8 : 0xD8 + 0x38] = struct.pack("<qqQqq", 0, 0, 0, 0, INT64_MAX) + CLOCK_SOURCE_ID
    return shared


def state(process) -> dict:
    st = process.services.setdefault("time", {})
    if not st:
        st["shared"] = time_shared_memory()
        st["operation"] = Event("clock operation")
    return st


def calendar(posix: int) -> bytes:
    """CalendarTime (year, month, day, hour, minute, second) then CalendarAdditionalInfo, for UTC."""
    t = datetime.datetime.fromtimestamp(posix, datetime.UTC)
    cal = struct.pack("<hbbbbbx", t.year, t.month, t.day, t.hour, t.minute, t.second)
    info = struct.pack("<II8sBxxxi", (t.weekday() + 1) % 7, t.timetuple().tm_yday - 1, b"UTC", 0, 0)
    return cal + info


def clock_snapshot(process, kind: int) -> bytes:
    """A ClockSnapshot (0xD0 bytes): both system clocks' contexts, times and calendar times, the steady
    clock's time point, the location name, whether automatic correction is on, and the type."""
    now = BOOT_POSIX_TIME + steady_seconds(process)
    cal = calendar(now)
    out = system_clock_context() * 2 + struct.pack("<qq", now, now)
    out += cal[:8] + cal[:8] + cal[8:] + cal[8:]
    out += struct.pack("<q", steady_seconds(process)) + CLOCK_SOURCE_ID
    out += LOCATION.ljust(0x24, b"\0") + struct.pack("<BBH", 0, kind, 0)
    return out


class SystemClock(Service):
    name = "time ISystemClock"

    @command(0)
    def GetCurrentTime(self, call: Call) -> int:
        now = BOOT_POSIX_TIME + steady_seconds(self.process)
        call.push("q", now)
        return 0

    @command(2)
    def GetSystemClockContext(self, call: Call) -> int:
        call.out += system_clock_context()
        return 0

    @command(4)
    def GetOperationEventReadableHandle(self, call: Call) -> int:
        call.copy(state(self.process)["operation"].readable)
        return 0


class SteadyClock(Service):
    name = "time ISteadyClock"

    @command(0)
    def GetCurrentTimePoint(self, call: Call) -> int:
        call.push("q", steady_seconds(self.process))
        call.out += CLOCK_SOURCE_ID
        return 0

    GetTestOffset = stub(2, "GetTestOffset", "", "q", 0)
    GetRtcValue = stub(100, "GetRtcValue", "", "q", BOOT_POSIX_TIME)
    IsRtcResetDetected = stub(101, "IsRtcResetDetected", "", "B", 0)
    GetSetupResultValue = stub(102, "GetSetupResultValue", "", "I", 0)
    GetInternalOffset = stub(200, "GetInternalOffset", "", "q", 0)


class TimeZoneService(Service):
    name = "time ITimeZoneService"

    GetDeviceLocationName = stub(0, "GetDeviceLocationName", "", "36s", LOCATION)
    GetTotalLocationNameCount = stub(2, "GetTotalLocationNameCount", "", "i", 1)

    @command(3)
    def LoadLocationNameList(self, call: Call) -> int:
        call.write(call.out_buffer(0), LOCATION.ljust(0x24, b"\0"))
        call.push("i", 1)
        return 0

    @command(4)
    def LoadTimeZoneRule(self, call: Call) -> int:
        call.note(repr(call.payload[:0x24].split(b"\0")[0].decode("ascii", "replace")))
        out = call.out_buffer(0)
        call.write(out, bytes(out.size))
        return 0

    GetTimeZoneRuleVersion = stub(5, "GetTimeZoneRuleVersion", "", "16s", b"2026a")

    @command(100)
    def ToCalendarTime(self, call: Call) -> int:
        call.out += calendar(call.arg("q"))
        return 0

    @command(101)
    def ToCalendarTimeWithMyRule(self, call: Call) -> int:
        call.out += calendar(call.arg("q"))
        return 0

    @command(201)
    def ToPosixTime(self, call: Call) -> int:
        year, month, day, hour, minute, second = call.args("hbbbbb")
        posix = int(datetime.datetime(year, month, day, hour, minute, second, tzinfo=datetime.UTC).timestamp())
        call.write(call.out_buffer(0), struct.pack("<q", posix))
        call.push("i", 1)
        return 0

    @command(202)
    def ToPosixTimeWithMyRule(self, call: Call) -> int:
        return self.ToPosixTime(call)


@service("time:u", "time:a", "time:s")
class Time(Service):
    name = "time"

    def _clock(self, call: Call, cls) -> int:
        call.push_object(cls(self.process))
        return 0

    @command(0)
    def GetStandardUserSystemClock(self, call: Call) -> int:
        return self._clock(call, SystemClock)

    @command(1)
    def GetStandardNetworkSystemClock(self, call: Call) -> int:
        return self._clock(call, SystemClock)

    @command(2)
    def GetStandardSteadyClock(self, call: Call) -> int:
        return self._clock(call, SteadyClock)

    @command(3)
    def GetTimeZoneService(self, call: Call) -> int:
        return self._clock(call, TimeZoneService)

    @command(4)
    def GetStandardLocalSystemClock(self, call: Call) -> int:
        return self._clock(call, SystemClock)

    @command(5)
    def GetEphemeralNetworkSystemClock(self, call: Call) -> int:
        return self._clock(call, SystemClock)

    @command(20)
    def GetSharedMemoryNativeHandle(self, call: Call) -> int:
        call.copy(state(self.process)["shared"])
        return 0

    @command(400)
    def GetClockSnapshot(self, call: Call) -> int:
        kind = call.arg("B")
        call.note(f"type {kind}")
        call.write(call.out_buffer(0), clock_snapshot(self.process, kind))
        return 0

    IsStandardUserSystemClockAutomaticCorrectionEnabled = stub(
        100, "IsStandardUserSystemClockAutomaticCorrectionEnabled", "", "B", 0
    )
    IsStandardNetworkSystemClockAccuracySufficient = stub(
        200, "IsStandardNetworkSystemClockAccuracySufficient", "", "B", 1
    )
