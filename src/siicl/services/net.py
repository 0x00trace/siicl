"""Network, socket and account services. siicl has no network: requests for a connection fail."""

from __future__ import annotations

from siicl.kernel import Event
from siicl.services.applet import USER_ID
from siicl.services.base import Call, Service, command, service, stub

NIFM_REQUEST_STATE_FREE = 1
RESULT_NO_NETWORK = (1111 << 9) | 110  # nifm module (110): network communication disabled
USER_NICKNAME = b"siicl"


class NetworkRequest(Service):
    name = "nifm IRequest"

    def __init__(self, process) -> None:
        super().__init__(process)
        self.events = (Event("network request 0"), Event("network request 1"))

    GetRequestState = stub(0, "GetRequestState", None, "I", NIFM_REQUEST_STATE_FREE)
    GetResult = stub(1, "GetResult", result=RESULT_NO_NETWORK)

    @command(2)
    def GetSystemEventReadableHandles(self, call: Call) -> int:
        for e in self.events:
            call.copy(e.readable)
        return 0

    Cancel = stub(3, "Cancel")

    @command(4)
    def Submit(self, call: Call) -> int:
        self.events[0].signal(self.process)
        return 0

    SetRequirement = stub(5, "SetRequirement")
    SetRequirementPreset = stub(6, "SetRequirementPreset", "I")
    SetPriority = stub(8, "SetPriority", "I")
    SetNetworkProfileId = stub(9, "SetNetworkProfileId")
    SetRejectable = stub(10, "SetRejectable", "B")
    SetConnectionConfirmationOption = stub(11, "SetConnectionConfirmationOption", "b")
    SetPersistent = stub(12, "SetPersistent", "B")
    SetInstant = stub(13, "SetInstant", "B")
    SetSustainable = stub(14, "SetSustainable", "B")
    SetRawPriority = stub(15, "SetRawPriority", "B")
    SetGreedy = stub(16, "SetGreedy", "B")
    SetSharable = stub(17, "SetSharable", "B")
    SetRequirementByRevision = stub(18, "SetRequirementByRevision", "I")
    GetRevision = stub(20, "GetRevision", None, "I", 0)
    SetKeptInSleep = stub(23, "SetKeptInSleep", "B")
    RegisterSocketDescriptor = stub(24, "RegisterSocketDescriptor", "i")
    UnregisterSocketDescriptor = stub(25, "UnregisterSocketDescriptor", "i")


class GeneralService(Service):
    name = "nifm IGeneralService"

    @command(1)
    def GetClientId(self, call: Call) -> int:
        call.write(call.out_buffer(0), (1).to_bytes(4, "little"))
        return 0

    @command(4)
    def CreateRequest(self, call: Call) -> int:
        call.note(f"requirement preset {call.arg('I')}")
        call.push_object(NetworkRequest(self.process))
        return 0

    GetCurrentNetworkProfile = stub(5, "GetCurrentNetworkProfile", result=RESULT_NO_NETWORK)
    GetCurrentIpAddress = stub(12, "GetCurrentIpAddress", result=RESULT_NO_NETWORK)
    GetInternetConnectionStatus = stub(18, "GetInternetConnectionStatus", result=RESULT_NO_NETWORK)
    IsWirelessCommunicationEnabled = stub(17, "IsWirelessCommunicationEnabled", None, "B", 0)
    IsEthernetCommunicationEnabled = stub(20, "IsEthernetCommunicationEnabled", None, "B", 0)
    IsAnyInternetRequestAccepted = stub(21, "IsAnyInternetRequestAccepted", None, "B", 0)
    IsAnyForegroundRequestAccepted = stub(22, "IsAnyForegroundRequestAccepted", None, "B", 0)


@service("nifm:u", "nifm:a", "nifm:s")
class NetworkInterface(Service):
    name = "nifm"

    @command(4)
    def CreateGeneralServiceOld(self, call: Call) -> int:
        call.push_object(GeneralService(self.process))
        return 0

    @command(5)
    def CreateGeneralService(self, call: Call) -> int:
        call.push_object(GeneralService(self.process))
        return 0


@service("ldr:ro", "ro:1")
class RelocatableObjects(Service):
    """Loading of modules at run time (NRO files). siicl only takes the process handle for now."""

    name = "ldr:ro"

    RegisterProcessHandle = stub(4, "RegisterProcessHandle", "Q")
    RegisterProcessModuleInfo = stub(10, "RegisterProcessModuleInfo")


class Profile(Service):
    name = "acc IProfile"

    @command(0)
    def Get(self, call: Call) -> int:
        call.out += self._base()
        call.write(call.out_buffer(0), bytes(0x80))  # user data
        return 0

    @command(1)
    def GetBase(self, call: Call) -> int:
        call.out += self._base()
        return 0

    GetImageSize = stub(10, "GetImageSize", None, "I", 0)
    LoadImage = stub(11, "LoadImage", None, "I", 0)

    def _base(self) -> bytes:
        """ProfileBase: user ID, last edit time, nickname (0x21 bytes, padded to 0x38 bytes in all)."""
        return USER_ID + (1_767_225_600).to_bytes(8, "little") + USER_NICKNAME.ljust(0x20, b"\0")


class ManagerForApplication(Service):
    name = "acc IManagerForApplication"

    CheckAvailability = stub(0, "CheckAvailability", result=RESULT_NO_NETWORK)
    GetAccountId = stub(1, "GetAccountId", None, "Q", 0x5349_4943_4C00_0001)


@service("acc:u0")
class Account(Service):
    name = "acc:u0"

    GetUserCount = stub(0, "GetUserCount", None, "i", 1)

    @command(1)
    def GetUserExistence(self, call: Call) -> int:
        call.push("B", int(call.payload[:16] == USER_ID))
        return 0

    @command(2)
    def ListAllUsers(self, call: Call) -> int:
        call.write(call.out_buffer(0), USER_ID)
        return 0

    @command(3)
    def ListOpenUsers(self, call: Call) -> int:
        call.write(call.out_buffer(0), USER_ID)
        return 0

    @command(4)
    def GetLastOpenedUser(self, call: Call) -> int:
        call.out += USER_ID
        return 0

    @command(5)
    def GetProfile(self, call: Call) -> int:
        call.push_object(Profile(self.process))
        return 0

    IsUserRegistrationRequestPermitted = stub(50, "IsUserRegistrationRequestPermitted", None, "B", 0)
    InitializeApplicationInfoV0 = stub(100, "InitializeApplicationInfoV0", "Q")

    @command(101)
    def GetBaasAccountManagerForApplication(self, call: Call) -> int:
        call.push_object(ManagerForApplication(self.process))
        return 0

    InitializeApplicationInfo = stub(140, "InitializeApplicationInfo", "Q")

    @command(141)
    def ListQualifiedUsers(self, call: Call) -> int:
        call.write(call.out_buffer(0), USER_ID)
        return 0

    IsUserAccountSwitchLocked = stub(150, "IsUserAccountSwitchLocked", None, "B", 0)
    InitializeApplicationInfoV2 = stub(160, "InitializeApplicationInfoV2", "Q")


@service("bsd:u", "bsd:s")
class Sockets(Service):
    """BSD sockets. siicl registers the client and has no network, so sockets cannot connect."""

    name = "bsd"

    @command(0)
    def RegisterClient(self, call: Call) -> int:
        call.push("i", 0)
        return 0

    @command(1)
    def StartMonitoring(self, call: Call) -> int:
        return 0

    # A Switch 2 command the sample sends instead of RegisterClient: a 16-byte configuration and the
    # process ID, without the transfer memory RegisterClient takes.
    @command(36)
    def RegisterClient36(self, call: Call) -> int:
        call.note(call.payload[:16].hex())
        call.push("i", 0)
        return 0
