"""Read and write the IPC messages a game sends to system services.

A message is the HIPC layout the kernel copies between processes, at most 0x100 bytes, in the
sending thread's thread-local storage (or in a buffer, for SendSyncRequestWithUserBuffer):

- two header words: the message type, the counts of each descriptor, the number of data words,
  the receive-list mode, and whether a special header follows
- the special header, if any: whether the sender's process ID follows, and the numbers of handles
  to copy and to move; then the process ID and the handles
- the buffer descriptors: statics (X), sends (A), receives (B) and exchanges (W)
- the data words
- the receive list (C descriptors)

Inside the data words, most services use CMIF: padding to a 16-byte boundary, an optional domain
header, a 16-byte header with the magic "SFCI" (requests) or "SFCO" (responses), then the
arguments. The service manager uses TIPC instead: the message type is the command ID plus 16, and
the data words hold the arguments directly. The layout follows switchbrew's IPC_Marshalling page
and libnx's hipc.h, cmif.h and tipc.h.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

MESSAGE_SIZE = 0x100
CMIF_IN = 0x49434653  # "SFCI"
CMIF_OUT = 0x4F434653  # "SFCO"

TYPE_CLOSE = 2
TYPE_REQUEST = 4
TYPE_CONTROL = 5
TYPE_REQUEST_WITH_CONTEXT = 6
TYPE_CONTROL_WITH_CONTEXT = 7
TYPE_TIPC_CLOSE = 15
TIPC_FIRST = 16

DOMAIN_SEND = 1
DOMAIN_CLOSE = 2


class IpcError(ValueError):
    pass


@dataclass(frozen=True)
class Buffer:
    address: int
    size: int
    index: int = 0  # statics: the index; sends, receives and exchanges: the mode


@dataclass
class Message:
    type: int
    pid: int | None = None
    copy_handles: list[int] = field(default_factory=list)
    move_handles: list[int] = field(default_factory=list)
    statics: list[Buffer] = field(default_factory=list)
    sends: list[Buffer] = field(default_factory=list)
    receives: list[Buffer] = field(default_factory=list)
    exchanges: list[Buffer] = field(default_factory=list)
    data: bytes = b""
    data_offset: int = 0  # where the data words start in the message
    receive_list: list[Buffer] = field(default_factory=list)


def _static(w0: int, w1: int) -> Buffer:
    address = w1 | ((w0 >> 12) & 0xF) << 32 | ((w0 >> 6) & 0x3F) << 36
    return Buffer(address, w0 >> 16, w0 & 0x3F)


def _buffer(w0: int, w1: int, w2: int) -> Buffer:
    address = w1 | ((w2 >> 28) & 0xF) << 32 | ((w2 >> 2) & 0x3FFFFF) << 36
    size = w0 | ((w2 >> 24) & 0xF) << 32
    return Buffer(address, size, w2 & 3)


def parse(raw: bytes) -> Message:
    if len(raw) < 8:
        raise IpcError("a message is at least two words")
    words = struct.unpack_from(f"<{len(raw) // 4}I", raw)
    h0, h1 = words[0], words[1]
    msg = Message(type=h0 & 0xFFFF)
    n_statics, n_sends, n_receives, n_exchanges = (h0 >> 16) & 0xF, (h0 >> 20) & 0xF, (h0 >> 24) & 0xF, h0 >> 28
    n_data, receive_mode = h1 & 0x3FF, (h1 >> 10) & 0xF
    i = 2
    if h1 >> 31:
        special = words[i]
        i += 1
        if special & 1:
            msg.pid = words[i] | words[i + 1] << 32
            i += 2
        n_copy, n_move = (special >> 1) & 0xF, (special >> 5) & 0xF
        msg.copy_handles = list(words[i : i + n_copy])
        i += n_copy
        msg.move_handles = list(words[i : i + n_move])
        i += n_move
    for _ in range(n_statics):
        msg.statics.append(_static(words[i], words[i + 1]))
        i += 2
    for target, count in ((msg.sends, n_sends), (msg.receives, n_receives), (msg.exchanges, n_exchanges)):
        for _ in range(count):
            target.append(_buffer(words[i], words[i + 1], words[i + 2]))
            i += 3
    msg.data_offset = 4 * i
    msg.data = raw[4 * i : 4 * (i + n_data)]
    i += n_data
    n_list = 0 if receive_mode < 2 else 1 if receive_mode == 2 else receive_mode - 2
    for _ in range(n_list):
        msg.receive_list.append(Buffer(words[i] | (words[i + 1] & 0xFFFF) << 32, words[i + 1] >> 16))
        i += 2
    if 4 * i > len(raw):
        raise IpcError("the message runs past its buffer")
    return msg


def build(
    *,
    type: int = 0,
    data: bytes = b"",
    copy_handles=(),
    move_handles=(),
    pid: int | None = None,
    statics=(),
    sends=(),
    receives=(),
    receive_list=(),
    align_data: bool = True,
) -> bytes:
    """Build a message. With align_data, data starts at the next 16-byte boundary, as CMIF needs."""
    special = pid is not None or copy_handles or move_handles
    words: list[int] = [0, 0]
    if special:
        words.append(int(pid is not None) | len(copy_handles) << 1 | len(move_handles) << 5)
        if pid is not None:
            words += [pid & 0xFFFF_FFFF, pid >> 32]
        words += list(copy_handles) + list(move_handles)
    for b in statics:
        words += [
            (b.index & 0x3F) | ((b.address >> 36) & 0x3F) << 6 | ((b.address >> 32) & 0xF) << 12 | b.size << 16,
            b.address & 0xFFFF_FFFF,
        ]
    for b in list(sends) + list(receives):
        words += [
            b.size & 0xFFFF_FFFF,
            b.address & 0xFFFF_FFFF,
            (b.index & 3)
            | ((b.address >> 36) & 0x3FFFFF) << 2
            | ((b.size >> 32) & 0xF) << 24
            | ((b.address >> 32) & 0xF) << 28,
        ]
    pad = (-(4 * len(words)) % 16) // 4 if align_data else 0
    body = bytes(4 * pad) + data
    n_data = (len(body) + 3) // 4
    if align_data:
        n_data = max(n_data, (16 + len(data) + 3) // 4)  # libnx reserves 16 bytes for the padding
    body = body.ljust(4 * n_data, b"\0")
    receive_mode = 0 if not receive_list else 2 + len(receive_list)
    words[0] = type | len(statics) << 16 | len(sends) << 20 | len(receives) << 24
    words[1] = n_data | receive_mode << 10 | int(bool(special)) << 31
    out = struct.pack(f"<{len(words)}I", *words) + body
    for b in receive_list:
        out += struct.pack("<II", b.address & 0xFFFF_FFFF, (b.address >> 32) & 0xFFFF | b.size << 16)
    if len(out) > MESSAGE_SIZE:
        raise IpcError(f"the message is {len(out):#x} bytes, more than {MESSAGE_SIZE:#x}")
    return out


@dataclass
class CmifRequest:
    command: int
    payload: bytes
    token: int = 0
    domain_command: int | None = None  # DOMAIN_SEND or DOMAIN_CLOSE, for a message to a domain
    object_id: int | None = None
    input_objects: list[int] = field(default_factory=list)


def _cmif_start(msg: Message) -> int:
    return (-msg.data_offset) % 16


def parse_cmif(msg: Message, *, domain: bool) -> CmifRequest:
    data = msg.data
    pos = _cmif_start(msg)
    req = CmifRequest(command=0, payload=b"")
    end = len(data)
    if domain:
        dcmd, n_in, size, object_id = struct.unpack_from("<BBHI", data, pos)
        req.domain_command, req.object_id = dcmd, object_id
        pos += 16
        req.input_objects = list(struct.unpack_from(f"<{n_in}I", data, pos + size)) if n_in else []
        end = pos + size
        if dcmd == DOMAIN_CLOSE:
            return req
    magic, _version, command, token = struct.unpack_from("<IIII", data, pos)
    if magic != CMIF_IN:
        raise IpcError(f"no CMIF header: magic {magic:#x}")
    req.command, req.token = command, token
    req.payload = data[pos + 16 : end]
    return req


def build_cmif_response(
    result: int,
    payload: bytes = b"",
    *,
    copy_handles=(),
    move_handles=(),
    domain_objects: list[int] | None = None,
    statics=(),
) -> bytes:
    data = b""
    if domain_objects is not None:
        data += struct.pack("<I12x", len(domain_objects))
    data += struct.pack("<IIII", CMIF_OUT, 0, result, 0) + payload
    if domain_objects:
        data += struct.pack(f"<{len(domain_objects)}I", *domain_objects)
    return build(data=data, copy_handles=copy_handles, move_handles=move_handles, statics=statics)


def build_tipc_response(result: int, payload: bytes = b"", *, copy_handles=(), move_handles=()) -> bytes:
    return build(
        data=struct.pack("<I", result) + payload,
        copy_handles=copy_handles,
        move_handles=move_handles,
        align_data=False,
    )
