"""siicl's system services. Importing this package registers every service with the service manager."""

from siicl.services import applet, fs, hid, lm, misc, net, sm, time  # noqa: F401
from siicl.services.base import REGISTRY, Service, Session, handle_request

__all__ = ["REGISTRY", "Service", "Session", "handle_request"]
