"""The input service, "hid".

The game reads controller state from a shared memory that `hid` hands out (through IAppletResource),
and sends commands to choose which controllers and features it supports. siicl's shared memory
stays zeroed, so the game sees no controllers connected. The memory takes the size the game maps it
with, since the Switch 2's layout is not documented.
"""

from __future__ import annotations

from siicl.kernel import Event, SharedMemory
from siicl.services.base import Call, Service, command, service, stub


def state(process) -> dict:
    st = process.services.setdefault("hid", {})
    if not st:
        st["shared"] = SharedMemory("hid", None)
        st["events"] = {}
    return st


def event(process, name: str) -> Event:
    events = state(process)["events"]
    return events.setdefault(name, Event(f"hid {name}"))


class AppletResource(Service):
    name = "hid IAppletResource"

    @command(0)
    def GetSharedMemoryHandle(self, call: Call) -> int:
        call.copy(state(self.process)["shared"])
        return 0


@service("hid")
class Hid(Service):
    name = "hid"

    @command(0)
    def CreateAppletResource(self, call: Call) -> int:
        call.push_object(AppletResource(self.process))
        return 0

    ActivateDebugPad = stub(1, "ActivateDebugPad")
    ActivateTouchScreen = stub(11, "ActivateTouchScreen")
    ActivateMouse = stub(21, "ActivateMouse")
    ActivateKeyboard = stub(31, "ActivateKeyboard")
    SendKeyboardLockKeyEvent = stub(32, "SendKeyboardLockKeyEvent")

    @command(40)
    def AcquireXpadIdEventHandle(self, call: Call) -> int:
        call.copy(event(self.process, "xpad id").readable)
        return 0

    ActivateXpad = stub(41, "ActivateXpad")
    StartSixAxisSensor = stub(66, "StartSixAxisSensor")
    StopSixAxisSensor = stub(67, "StopSixAxisSensor")
    SetSupportedNpadStyleSet = stub(100, "SetSupportedNpadStyleSet")
    GetSupportedNpadStyleSet = stub(101, "GetSupportedNpadStyleSet", None, "I", 0)
    SetSupportedNpadIdType = stub(102, "SetSupportedNpadIdType")
    ActivateNpad = stub(103, "ActivateNpad")
    DeactivateNpad = stub(104, "DeactivateNpad")

    @command(106)
    def AcquireNpadStyleSetUpdateEventHandle(self, call: Call) -> int:
        npad = call.arg("I")
        call.note(f"npad {npad:#x}")
        call.copy(event(self.process, f"npad {npad:#x} style set").readable)
        return 0

    DisconnectNpad = stub(107, "DisconnectNpad")
    GetPlayerLedPattern = stub(108, "GetPlayerLedPattern", None, "Q", 0)
    ActivateNpadWithRevision = stub(109, "ActivateNpadWithRevision")
    SetNpadJoyHoldType = stub(120, "SetNpadJoyHoldType")
    GetNpadJoyHoldType = stub(121, "GetNpadJoyHoldType", None, "Q", 0)
    SetNpadJoyAssignmentModeSingleByDefault = stub(122, "SetNpadJoyAssignmentModeSingleByDefault")
    SetNpadJoyAssignmentModeSingle = stub(123, "SetNpadJoyAssignmentModeSingle")
    SetNpadJoyAssignmentModeDual = stub(124, "SetNpadJoyAssignmentModeDual")
    MergeSingleJoyAsDualJoy = stub(125, "MergeSingleJoyAsDualJoy")
    SetNpadHandheldActivationMode = stub(128, "SetNpadHandheldActivationMode")
    GetNpadHandheldActivationMode = stub(129, "GetNpadHandheldActivationMode", None, "Q", 0)
    SetNpadAnalogStickUseCenterClamp = stub(134, "SetNpadAnalogStickUseCenterClamp")
    GetVibrationDeviceInfo = stub(200, "GetVibrationDeviceInfo", None, "II", 0, 0)
    SendVibrationValue = stub(201, "SendVibrationValue")
    PermitVibration = stub(204, "PermitVibration")
    IsVibrationPermitted = stub(205, "IsVibrationPermitted", None, "B", 1)
    SendVibrationValues = stub(206, "SendVibrationValues")
    IsVibrationDeviceMounted = stub(211, "IsVibrationDeviceMounted", None, "B", 0)
    ActivateConsoleSixAxisSensor = stub(300, "ActivateConsoleSixAxisSensor")
    StartConsoleSixAxisSensor = stub(301, "StartConsoleSixAxisSensor")
    SetNpadCommunicationMode = stub(1000, "SetNpadCommunicationMode")
    GetNpadCommunicationMode = stub(1001, "GetNpadCommunicationMode", None, "Q", 0)
    SetTouchScreenConfiguration = stub(1002, "SetTouchScreenConfiguration")
