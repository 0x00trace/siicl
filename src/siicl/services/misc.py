"""Small services a game opens at startup: parental controls, add-on content and error contexts."""

from __future__ import annotations

from siicl.kernel import Event
from siicl.services.base import Call, Service, command, service, stub


class ParentalControlService(Service):
    name = "pctl IParentalControlService"

    Initialize = stub(1, "Initialize")
    CheckFreeCommunicationPermission = stub(1001, "CheckFreeCommunicationPermission")
    ConfirmStereoVisionPermission = stub(1013, "ConfirmStereoVisionPermission")
    IsFreeCommunicationAvailable = stub(1018, "IsFreeCommunicationAvailable")
    IsRestrictionEnabled = stub(1031, "IsRestrictionEnabled", "", "B", 0)
    GetSafetyLevel = stub(1032, "GetSafetyLevel", "", "I", 0)
    IsStereoVisionPermitted = stub(1065, "IsStereoVisionPermitted", "", "B", 1)
    IsPairingActive = stub(1403, "IsPairingActive", "", "B", 0)
    IsRestrictedByPlayTimer = stub(1473, "IsRestrictedByPlayTimer", "", "B", 0)

    @command(1457)
    def GetPlayTimerEventToRequestSuspension(self, call: Call) -> int:
        call.copy(self.process.services.setdefault("pctl suspension", Event("play timer suspension")).readable)
        return 0


@service("pctl", "pctl:a", "pctl:s", "pctl:r")
class ParentalControl(Service):
    name = "pctl"

    @command(0)
    def CreateService(self, call: Call) -> int:
        call.push_object(ParentalControlService(self.process))
        return 0

    @command(1)
    def CreateServiceWithoutInitialize(self, call: Call) -> int:
        call.push_object(ParentalControlService(self.process))
        return 0


@service("aoc:u")
class AddOnContent(Service):
    """The game's add-on content: siicl reports none."""

    name = "aoc:u"

    CountAddOnContent = stub(2, "CountAddOnContent", None, "I", 0)
    ListAddOnContent = stub(3, "ListAddOnContent", None, "I", 0)
    GetAddOnContentBaseId = stub(5, "GetAddOnContentBaseId", None, "Q", 0x0400_0000_0000_1000)
    PrepareAddOnContent = stub(7, "PrepareAddOnContent")

    @command(8)
    def GetAddOnContentListChangedEvent(self, call: Call) -> int:
        call.copy(self.process.services.setdefault("aoc changed", Event("add-on content list changed")).readable)
        return 0

    @command(10)
    def GetAddOnContentListChangedEventWithProcessId(self, call: Call) -> int:
        return self.GetAddOnContentListChangedEvent(call)

    @command(100)
    def CreateEcPurchasedEventManager(self, call: Call) -> int:
        call.push_object(PurchaseEventManager(self.process))
        return 0

    @command(101)
    def CreatePermanentEcPurchasedEventManager(self, call: Call) -> int:
        call.push_object(PurchaseEventManager(self.process))
        return 0


class PurchaseEventManager(Service):
    name = "aoc IPurchaseEventManager"

    SetDefaultDeliveryTarget = stub(0, "SetDefaultDeliveryTarget")
    SetDeliveryTarget = stub(1, "SetDeliveryTarget")

    @command(2)
    def GetPurchasedEventReadableHandle(self, call: Call) -> int:
        call.copy(self.process.services.setdefault("aoc purchased", Event("add-on content purchased")).readable)
        return 0


class ContextRegistrar(Service):
    name = "ectx IContextRegistrar"

    Complete = stub(0, "Complete")


@service("ectx:aw", "ectx:w")
class ErrorContext(Service):
    name = "ectx"

    @command(0)
    def CreateContextRegistrar(self, call: Call) -> int:
        call.push_object(ContextRegistrar(self.process))
        return 0
