"""The applet services an application uses: "appletOE" and the interfaces it returns.

An application opens IApplicationProxy and, through it, the interfaces for its state, its window,
its display and its functions. The system talks back through a message queue: on start the
queue holds a focus-change message, and the application reads it to learn it is in focus.
"""

from __future__ import annotations

import struct

from siicl.kernel import Event
from siicl.services.base import Call, Service, Session, command, service, stub

RESULT_NO_MESSAGE = 0x680  # applet module (128), description 3
RESULT_NOT_AVAILABLE = 0x2A8  # applet module (128), description 1: no launch parameter of that kind
MESSAGE_FOCUS_STATE_CHANGED = 0xF
FOCUS_IN_FOCUS = 1
OPERATION_MODE_CONSOLE = 1
USER_ID = struct.pack("<QQ", 0x0000_0000_0000_0001, 0x5349_4943_4C00_0001)  # siicl's one user account
PRESELECTED_USER_MAGIC = 0xC79497CA
APPLET_RESOURCE_USER_ID = 0x51
DISPLAY_SIZE = (1920, 1080)


class AppletState:
    def __init__(self) -> None:
        self.messages = [MESSAGE_FOCUS_STATE_CHANGED]
        self.message_event = Event("applet messages")
        self.message_event.readable.signaled = True
        self.launchable_event = Event("library applet launchable")
        self.resolution_event = Event("display resolution changed")
        self.gpu_error_event = Event("GPU error detected")
        self.friend_invitation_event = Event("friend invitation")
        self.notification_event = Event("notification storage channel")
        self.suspended_tick_event = Event("accumulated suspended tick changed")
        self.health_warning_event = Event("health warning disappeared")


def state(process) -> AppletState:
    return process.services.setdefault("applet", AppletState())


class Storage(Service):
    name = "applet IStorage"

    def __init__(self, process, data: bytes) -> None:
        super().__init__(process)
        self.data = bytearray(data)

    @command(0)
    def Open(self, call: Call) -> int:
        call.push_object(StorageAccessor(self.process, self))
        return 0


class StorageAccessor(Service):
    name = "applet IStorageAccessor"

    def __init__(self, process, storage: Storage) -> None:
        super().__init__(process)
        self.storage = storage

    @command(0)
    def GetSize(self, call: Call) -> int:
        call.push("q", len(self.storage.data))
        return 0

    @command(10)
    def Write(self, call: Call) -> int:
        offset = call.arg("q")
        data = call.in_buffer(0)
        self.storage.data[offset : offset + len(data)] = data
        return 0

    @command(11)
    def Read(self, call: Call) -> int:
        offset = call.arg("q")
        out = call.out_buffer(0)
        call.write(out, bytes(self.storage.data[offset : offset + out.size]))
        return 0


class CommonStateGetter(Service):
    name = "applet ICommonStateGetter"

    @command(0)
    def GetEventHandle(self, call: Call) -> int:
        call.copy(state(self.process).message_event.readable)
        return 0

    @command(1)
    def ReceiveMessage(self, call: Call) -> int:
        st = state(self.process)
        if not st.messages:
            st.message_event.clear()
            return RESULT_NO_MESSAGE
        message = st.messages.pop(0)
        if not st.messages:
            st.message_event.clear()
        call.note(f"message {message:#x}")
        call.push("I", message)
        return 0

    GetOperationMode = stub(5, "GetOperationMode", "", "B", OPERATION_MODE_CONSOLE)
    GetPerformanceMode = stub(6, "GetPerformanceMode", "", "I", OPERATION_MODE_CONSOLE)
    GetBootMode = stub(8, "GetBootMode", "", "B", 0)
    GetCurrentFocusState = stub(9, "GetCurrentFocusState", "", "B", FOCUS_IN_FOCUS)
    RequestToAcquireSleepLock = stub(10, "RequestToAcquireSleepLock")
    IsVrModeEnabled = stub(50, "IsVrModeEnabled", "", "B", 0)
    GetDefaultDisplayResolution = stub(60, "GetDefaultDisplayResolution", "", "ii", *DISPLAY_SIZE)

    @command(61)
    def GetDefaultDisplayResolutionChangeEvent(self, call: Call) -> int:
        call.copy(state(self.process).resolution_event.readable)
        return 0

    SetCpuBoostMode = stub(66, "SetCpuBoostMode", "I")
    SetPerformanceConfigurationChangedNotification = stub(90, "SetPerformanceConfigurationChangedNotification", "B")
    EnableStartupLogoDisappearedMessage = stub(130, "EnableStartupLogoDisappearedMessage")
    GetSettingsPlatformRegion = stub(300, "GetSettingsPlatformRegion", "", "B", 1)
    SetRequestExitToLibraryAppletAtExecuteNextProgramEnabled = stub(
        900, "SetRequestExitToLibraryAppletAtExecuteNextProgramEnabled"
    )
    SetRtcModeChangedMessageEnabled = stub(2000, "SetRtcModeChangedMessageEnabled")


class GraphicsSupervisor(Service):
    name = "applet IGraphicsSupervisor"


class SelfController(Service):
    name = "applet ISelfController"

    LockExit = stub(1, "LockExit")
    UnlockExit = stub(2, "UnlockExit")

    @command(9)
    def GetLibraryAppletLaunchableEvent(self, call: Call) -> int:
        call.copy(state(self.process).launchable_event.readable)
        return 0

    SetScreenShotPermission = stub(10, "SetScreenShotPermission", "I")
    SetOperationModeChangedNotification = stub(11, "SetOperationModeChangedNotification", "B")
    SetPerformanceModeChangedNotification = stub(12, "SetPerformanceModeChangedNotification", "B")
    SetFocusHandlingMode = stub(13, "SetFocusHandlingMode", "BBB")
    SetRestartMessageEnabled = stub(14, "SetRestartMessageEnabled", "B")
    SetOutOfFocusSuspendingEnabled = stub(16, "SetOutOfFocusSuspendingEnabled", "B")
    SetAlbumImageOrientation = stub(19, "SetAlbumImageOrientation", "I")
    CreateManagedDisplayLayer = stub(40, "CreateManagedDisplayLayer", "", "Q", 1)
    SetHandlesRequestToDisplay = stub(50, "SetHandlesRequestToDisplay", "B")
    SetIdleTimeDetectionExtension = stub(62, "SetIdleTimeDetectionExtension", "I")
    ReportUserIsActive = stub(65, "ReportUserIsActive")
    SetAutoSleepDisabled = stub(68, "SetAutoSleepDisabled", "B")
    SetInputDetectionPolicy = stub(72, "SetInputDetectionPolicy", "I")

    @command(91)
    def GetAccumulatedSuspendedTickChangedEvent(self, call: Call) -> int:
        call.copy(state(self.process).suspended_tick_event.readable)
        return 0

    SetAlbumImageTakenNotificationEnabled = stub(100, "SetAlbumImageTakenNotificationEnabled", "B")

    # A Switch 2 addition. switchbrew gives its name but not what the interface it returns does. The
    # sample expects a session handle back, copied, not an object in the domain.
    @command(2000)
    def GetGraphicsSupervisor(self, call: Call) -> int:
        call.copy(Session(GraphicsSupervisor(self.process)))
        return 0

    SetRtcScreenSharingAudioEnabled = stub(2400, "SetRtcScreenSharingAudioEnabled", "B")


class WindowController(Service):
    name = "applet IWindowController"

    GetAppletResourceUserId = stub(1, "GetAppletResourceUserId", "", "Q", APPLET_RESOURCE_USER_ID)
    AcquireForegroundRights = stub(10, "AcquireForegroundRights")


class AudioController(Service):
    name = "applet IAudioController"

    SetExpectedMasterVolume = stub(0, "SetExpectedMasterVolume", "ff")
    GetMainAppletExpectedMasterVolume = stub(1, "GetMainAppletExpectedMasterVolume", "", "f", 1.0)
    GetLibraryAppletExpectedMasterVolume = stub(2, "GetLibraryAppletExpectedMasterVolume", "", "f", 1.0)


class DisplayController(Service):
    name = "applet IDisplayController"


class LibraryAppletCreator(Service):
    name = "applet ILibraryAppletCreator"

    @command(10)
    def CreateStorage(self, call: Call) -> int:
        size = call.arg("q")
        call.note(f"size={size:#x}")
        call.push_object(Storage(self.process, bytes(size)))
        return 0


class ApplicationFunctions(Service):
    name = "applet IApplicationFunctions"

    @command(1)
    def PopLaunchParameter(self, call: Call) -> int:
        kind = call.arg("I")
        call.note(f"kind={kind}")
        if kind != 2:  # only the preselected user
            return RESULT_NOT_AVAILABLE
        data = struct.pack("<II", PRESELECTED_USER_MAGIC, 1) + USER_ID
        call.push_object(Storage(self.process, data.ljust(0x88, b"\0")))
        return 0

    EnsureSaveData = stub(20, "EnsureSaveData", "16s", "Q", 0)
    GetDesiredLanguage = stub(21, "GetDesiredLanguage", "", "8s", b"en-US")
    SetTerminateResult = stub(22, "SetTerminateResult", "I")
    GetDisplayVersion = stub(23, "GetDisplayVersion", "", "16s", b"1.0.0")
    NotifyRunning = stub(40, "NotifyRunning", "", "B", 0)
    # Switch 2 additions (20.0.0+) without names on switchbrew. The sample sends them without
    # arguments; siicl answers them with success and no data.
    Command320 = stub(320, "command 320")
    Command321 = stub(321, "command 321")
    GetPseudoDeviceId = stub(50, "GetPseudoDeviceId", "", "16s", bytes(range(16)))
    SetGamePlayRecordingState = stub(67, "SetGamePlayRecordingState", "I")
    EnableApplicationCrashReport = stub(90, "EnableApplicationCrashReport", "B")
    GetPreviousProgramIndex = stub(123, "GetPreviousProgramIndex", "", "i", -1)

    @command(130)
    def GetGpuErrorDetectedSystemEvent(self, call: Call) -> int:
        call.copy(state(self.process).gpu_error_event.readable)
        return 0

    @command(140)
    def GetFriendInvitationStorageChannelEvent(self, call: Call) -> int:
        call.copy(state(self.process).friend_invitation_event.readable)
        return 0

    TryPopFromFriendInvitationStorageChannel = stub(
        141, "TryPopFromFriendInvitationStorageChannel", result=RESULT_NOT_AVAILABLE
    )

    @command(150)
    def GetNotificationStorageChannelEvent(self, call: Call) -> int:
        call.copy(state(self.process).notification_event.readable)
        return 0

    @command(160)
    def GetHealthWarningDisappearedSystemEvent(self, call: Call) -> int:
        call.copy(state(self.process).health_warning_event.readable)
        return 0


class DebugFunctions(Service):
    name = "applet IDebugFunctions"


class PerformanceFunctions(Service):
    name = "applet IPerformanceFunctions"


class GraphicsScalingParametersFunctions(Service):
    name = "applet IGraphicsScalingParametersFunctions"


class ApplicationProxy(Service):
    name = "applet IApplicationProxy"

    def _open(self, call: Call, cls) -> int:
        call.push_object(cls(self.process))
        return 0

    @command(0)
    def GetCommonStateGetter(self, call: Call) -> int:
        return self._open(call, CommonStateGetter)

    @command(1)
    def GetSelfController(self, call: Call) -> int:
        return self._open(call, SelfController)

    @command(2)
    def GetWindowController(self, call: Call) -> int:
        return self._open(call, WindowController)

    @command(3)
    def GetAudioController(self, call: Call) -> int:
        return self._open(call, AudioController)

    @command(4)
    def GetDisplayController(self, call: Call) -> int:
        return self._open(call, DisplayController)

    @command(11)
    def GetLibraryAppletCreator(self, call: Call) -> int:
        return self._open(call, LibraryAppletCreator)

    @command(20)
    def GetApplicationFunctions(self, call: Call) -> int:
        return self._open(call, ApplicationFunctions)

    @command(1000)
    def GetDebugFunctions(self, call: Call) -> int:
        return self._open(call, DebugFunctions)

    # Switch 2 additions. switchbrew names only the interfaces these return.
    @command(40)
    def GetPerformanceFunctions(self, call: Call) -> int:
        return self._open(call, PerformanceFunctions)

    @command(60)
    def GetGraphicsScalingParametersFunctions(self, call: Call) -> int:
        return self._open(call, GraphicsScalingParametersFunctions)


@service("appletOE")
class AppletOE(Service):
    name = "appletOE"

    @command(0)
    def OpenApplicationProxy(self, call: Call) -> int:
        call.push_object(ApplicationProxy(self.process))
        return 0

    # A Switch 2 addition (switchbrew's name, which is unofficial): a u32 and the same arguments as
    # command 0, the process ID and a copy of the current process handle.
    @command(10)
    def OpenApplicationProxy2(self, call: Call) -> int:
        return self.OpenApplicationProxy(call)
