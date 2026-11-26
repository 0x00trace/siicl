"""The service manager, reached through the "sm:" port. It speaks TIPC."""

from __future__ import annotations

from siicl.kernel import Unimplemented
from siicl.services.base import REGISTRY, Call, Service, command


def service_name(value: int) -> str:
    return value.to_bytes(8, "little").split(b"\0", 1)[0].decode("ascii", "replace")


class UnknownService(Service):
    """A service siicl does not implement, handed out under --stub-unknown. It has no commands."""

    def __init__(self, process, name: str) -> None:
        super().__init__(process)
        self.name = name


class ServiceManager(Service):
    """The service manager. It does not check the NPDM's list of services: the sample needs services
    the list does not name (such as "bsd:s"), and stops if the service manager refuses them."""

    name = "sm:"

    @command(0)
    def RegisterClient(self, call: Call) -> int:
        call.note(f"pid={call.pid:#x}" if call.pid is not None else "no pid")
        return 0

    @command(1)
    def GetServiceHandle(self, call: Call) -> int:
        name = service_name(call.arg("Q"))
        call.note(repr(name))
        factory = REGISTRY.get(name)
        if factory is None:
            if not self.process.stub_unknown:
                raise Unimplemented(f'the service "{name}" is not implemented')
            self.process.stubbed.append(f'service "{name}"')
            call.note("not implemented: every command will be answered with success")
            call.push_object(UnknownService(self.process, name))
            return 0
        cached = self.process.services.get(("service", name))
        obj = cached or factory(self.process)
        call.push_object(obj)
        return 0

    @command(4)
    def DetachClient(self, call: Call) -> int:
        return 0
