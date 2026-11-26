"""The framework siicl's system services are written in.

A game reaches a service by connecting to the service manager port ("sm:") and asking it for the
service by name. Each answer is a session: a handle the game sends requests on with
SendSyncRequest. siicl answers the requests itself, in Python: a session leads to a `Service`
object, whose methods marked with @command handle the requests. A command can return more objects
(interfaces), which the game reaches through new sessions, or, once the game has turned a session
into a domain, through object IDs in the same session.

A request for a command or a service that siicl does not implement ends the run, so the boot log
names the next one to write.
"""

from __future__ import annotations

import struct
from collections.abc import Callable
from typing import ClassVar

from siicl import ipc
from siicl.kernel import PROCESS_ID, Process, Thread, Unimplemented

POINTER_BUFFER_SIZE = 0x8000

REGISTRY: dict[str, Callable[[Process], Service]] = {}


def service(*names: str):
    """Register a Service class under the names the service manager knows it by."""

    def register(cls):
        for n in names:
            REGISTRY[n] = cls
        return cls

    return register


def command(number: int, name: str | None = None):
    def mark(fn):
        fn._command = (number, name or fn.__name__)
        return fn

    return mark


def stub(number: int, name: str, fmt: str | None = None, *out, result: int = 0):
    """A command that logs its arguments, pushes out (a format and values) and returns result.

    fmt unpacks the arguments for the log; with None, the log shows them in hex, if there are any.
    """

    def method(self, call: Call) -> int:
        if fmt:
            call.note(", ".join(f"{v:#x}" if isinstance(v, int) else repr(v) for v in call.args(fmt)))
        elif fmt is None and call.payload.strip(b"\0"):
            call.note(call.payload.hex())
        if out:
            call.push(*out)
        return result

    method.__name__ = name
    return command(number, name)(method)


class Service:
    """An object a session leads to: a service, or an interface a command returned."""

    name: ClassVar[str] = "?"
    commands: ClassVar[dict[int, tuple[str, Callable]]] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.commands = {}
        for klass in reversed(cls.__mro__):
            for attr in vars(klass).values():
                if hasattr(attr, "_command"):
                    number, name = attr._command
                    cls.commands[number] = (name, attr)

    def __init__(self, process: Process) -> None:
        self.process = process


class Session:
    """A session handle's kernel object. With a domain, it holds several objects, by ID."""

    def __init__(self, obj: Service) -> None:
        self.obj = obj
        self.domain: dict[int, Service] | None = None
        self.next_id = 1

    def add(self, obj: Service) -> int:
        object_id = self.next_id
        self.next_id += 1
        self.domain[object_id] = obj
        return object_id


class Call:
    """One request to one command: its arguments, buffers and handles, and what it sends back."""

    def __init__(self, process: Process, thread: Thread, message: ipc.Message, payload: bytes, in_objects=()):
        self.process = process
        self.thread = thread
        self.message = message
        self.payload = payload
        self.in_objects = list(in_objects)
        self.out = bytearray()
        self.copy_handles: list[int] = []
        self.move_handles: list[int] = []
        self.objects: list[Service] = []
        self.notes: list[str] = []

    # Arguments

    def args(self, fmt: str, offset: int = 0) -> tuple:
        return struct.unpack_from("<" + fmt, self.payload.ljust(offset + struct.calcsize("<" + fmt), b"\0"), offset)

    def arg(self, fmt: str, offset: int = 0):
        return self.args(fmt, offset)[0]

    @property
    def pid(self) -> int | None:
        return self.message.pid

    def in_handle(self, i: int = 0) -> int:
        handles = self.message.copy_handles + self.message.move_handles
        return handles[i]

    def read(self, buffer: ipc.Buffer) -> bytes:
        return self.process.cpu.read(buffer.address, buffer.size) if buffer.size else b""

    def write(self, buffer: ipc.Buffer, data: bytes) -> None:
        data = data[: buffer.size]
        if data:
            self.process.cpu.write(buffer.address, data)

    def static(self, index: int) -> ipc.Buffer:
        """The send static (X descriptor) with this index."""
        return next((b for b in self.message.statics if b.index == index), ipc.Buffer(0, 0, index))

    def receive_static(self, index: int) -> ipc.Buffer:
        return self.message.receive_list[index] if index < len(self.message.receive_list) else ipc.Buffer(0, 0)

    def send_buffer(self, index: int) -> ipc.Buffer:
        return self.message.sends[index] if index < len(self.message.sends) else ipc.Buffer(0, 0)

    def receive_buffer(self, index: int) -> ipc.Buffer:
        return self.message.receives[index] if index < len(self.message.receives) else ipc.Buffer(0, 0)

    def in_buffer(self, index: int) -> bytes:
        """An input buffer that may come either way: a send buffer (A) or a send static (X)."""
        a = self.send_buffer(index)
        return self.read(a) if a.size else self.read(self.static(index))

    def out_buffer(self, index: int) -> ipc.Buffer:
        """An output buffer that may come either way: a receive buffer (B) or a receive static (C)."""
        b = self.receive_buffer(index)
        return b if b.size else self.receive_static(index)

    # Results

    def push(self, fmt: str, *values) -> None:
        self.out += struct.pack("<" + fmt, *values)

    def push_object(self, obj: Service) -> None:
        self.objects.append(obj)

    def copy(self, kernel_object: object) -> int:
        handle = self.process.handles.add(kernel_object)
        self.copy_handles.append(handle)
        return handle

    def move(self, kernel_object: object) -> int:
        handle = self.process.handles.add(kernel_object)
        self.move_handles.append(handle)
        return handle

    def note(self, text: str) -> None:
        self.notes.append(text)


def handle_request(p: Process, t: Thread, session: Session, address: int, size: int) -> str:
    """Answer the request at address and write the response over it. Returns the text for the log."""
    msg = ipc.parse(p.cpu.read(address, min(size, ipc.MESSAGE_SIZE)))
    if msg.pid is not None:
        msg.pid = PROCESS_ID  # the kernel writes the sender's process ID over what the sender put there
    if msg.type >= ipc.TIPC_FIRST:
        call = Call(p, t, msg, msg.data)
        text, result = _dispatch(session.obj, msg.type - ipc.TIPC_FIRST, call)
        for obj in call.objects:
            call.move(Session(obj))
        response = ipc.build_tipc_response(
            result, bytes(call.out), copy_handles=call.copy_handles, move_handles=call.move_handles
        )
    elif msg.type in (ipc.TYPE_CLOSE, ipc.TYPE_TIPC_CLOSE):
        return f"{session.obj.name} close"
    elif msg.type in (ipc.TYPE_CONTROL, ipc.TYPE_CONTROL_WITH_CONTEXT):
        text, response = _control(p, t, session, msg)
    elif msg.type in (ipc.TYPE_REQUEST, ipc.TYPE_REQUEST_WITH_CONTEXT):
        req = ipc.parse_cmif(msg, domain=session.domain is not None)
        if session.domain is not None:
            target = session.domain.get(req.object_id)
            if target is None:
                raise Unimplemented(f"a request for object {req.object_id}, which the domain does not have")
            if req.domain_command == ipc.DOMAIN_CLOSE:
                del session.domain[req.object_id]
                response = ipc.build_cmif_response(0, domain_objects=[])
                p.cpu.write(address, response)
                return f"{target.name} close object {req.object_id}"
        else:
            target = session.obj
        in_objects = [session.domain[i] for i in req.input_objects] if session.domain else []
        call = Call(p, t, msg, req.payload, in_objects)
        text, result = _dispatch(target, req.command, call)
        if session.domain is not None:
            ids = [session.add(obj) for obj in call.objects]
            response = ipc.build_cmif_response(
                result,
                bytes(call.out),
                copy_handles=call.copy_handles,
                move_handles=call.move_handles,
                domain_objects=ids,
            )
        else:
            for obj in call.objects:
                call.move(Session(obj))
            response = ipc.build_cmif_response(
                result, bytes(call.out), copy_handles=call.copy_handles, move_handles=call.move_handles
            )
    else:
        raise Unimplemented(f"IPC message type {msg.type} is not implemented")
    p.cpu.write(address, response)
    return text


def _dispatch(target: Service, number: int, call: Call) -> tuple[str, int]:
    entry = target.commands.get(number)
    if entry is None:
        detail = describe_request(call)
        if not call.process.stub_unknown:
            raise Unimplemented(f"{target.name} command {number} is not implemented ({detail})")
        call.process.stubbed.append(f"{target.name} command {number}")
        return f"{target.name} command {number} ({detail}) -> not implemented, answered with success", 0
    name, method = entry
    result = method(target, call) or 0
    text = f"{target.name} {name}"
    if call.notes:
        text += f"({', '.join(call.notes)})"
    if result:
        text += f" -> {result:#x}"
    if call.objects:
        text += " -> " + ", ".join(o.name for o in call.objects)
    return text, result


def describe_request(call: Call) -> str:
    """What a request carries, for the log line of a command siicl does not implement."""
    m = call.message
    parts = [f"arguments {call.payload.hex() or 'none'}"]
    if m.pid is not None:
        parts.append("process ID")
    if m.copy_handles or m.move_handles:
        parts.append(f"handles {[hex(h) for h in m.copy_handles]} copied, {[hex(h) for h in m.move_handles]} moved")
    for label, buffers in (
        ("X", m.statics),
        ("A", m.sends),
        ("B", m.receives),
        ("W", m.exchanges),
        ("C", m.receive_list),
    ):
        for b in buffers:
            parts.append(f"{label} buffer {b.address:#x}+{b.size:#x}")
    if call.in_objects:
        parts.append(f"{len(call.in_objects)} objects")
    return ", ".join(parts)


def _control(p: Process, t: Thread, session: Session, msg: ipc.Message) -> tuple[str, bytes]:
    req = ipc.parse_cmif(msg, domain=False)
    name = session.obj.name
    if req.command == 0:  # ConvertCurrentObjectToDomain
        if session.domain is None:
            session.domain = {}
            session.add(session.obj)
        return f"{name} ConvertCurrentObjectToDomain", ipc.build_cmif_response(0, struct.pack("<I", 1))
    if req.command == 1:  # CopyFromCurrentDomain
        object_id = struct.unpack_from("<I", req.payload)[0]
        handle = p.handles.add(Session(session.domain[object_id]))
        return f"{name} CopyFromCurrentDomain({object_id})", ipc.build_cmif_response(0, move_handles=[handle])
    if req.command in (2, 4):  # CloneCurrentObject, CloneCurrentObjectEx
        clone = Session(session.obj)
        clone.domain, clone.next_id = session.domain, session.next_id
        handle = p.handles.add(clone)
        return f"{name} CloneCurrentObject", ipc.build_cmif_response(0, move_handles=[handle])
    if req.command == 3:  # QueryPointerBufferSize
        return f"{name} QueryPointerBufferSize", ipc.build_cmif_response(0, struct.pack("<H", POINTER_BUFFER_SIZE))
    raise Unimplemented(f"IPC control command {req.command} is not implemented")
