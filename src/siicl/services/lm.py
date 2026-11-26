"""The log manager, "lm". The game sends its log messages here, as binary log packets.

A packet is a 0x18-byte header (process ID, thread ID, flags, severity, verbosity, payload size)
and then fields, each a one-byte key, a ULEB128 length and the data. A message too long for one
packet is split over several; the header flags mark the first (bit 0) and the last (bit 1).
"""

from __future__ import annotations

import struct

from siicl.services.base import Call, Service, command, service

SEVERITIES = ("trace", "info", "warning", "error", "fatal")
FIELDS = {
    2: "text",
    3: "line",
    4: "file",
    5: "function",
    6: "module",
    7: "thread",
    10: "process",
}


def parse_packet(raw: bytes) -> tuple[dict, int, int]:
    """The fields of one packet, its flags and its severity."""
    if len(raw) < 0x18:
        return {}, 0, 0
    _pid, _tid, flags, _pad, severity, _verbosity, size = struct.unpack_from("<QQBBBBI", raw)
    fields: dict = {}
    pos, end = 0x18, min(len(raw), 0x18 + size)
    while pos < end:
        key = raw[pos]
        pos += 1
        length = shift = 0
        while pos < end:
            byte = raw[pos]
            pos += 1
            length |= (byte & 0x7F) << shift
            shift += 7
            if not byte & 0x80:
                break
        value = raw[pos : pos + length]
        pos += length
        name = FIELDS.get(key)
        if name == "line":
            fields[name] = int.from_bytes(value[:4], "little")
        elif name:
            fields[name] = value.decode("utf-8", "replace").rstrip("\0")
    return fields, flags, severity


class Logger(Service):
    name = "lm ILogger"

    def __init__(self, process) -> None:
        super().__init__(process)
        self.pending: dict = {}

    @command(0)
    def Log(self, call: Call) -> int:
        fields, flags, severity = parse_packet(call.in_buffer(0))
        if flags & 1:
            self.pending = {}
        text = self.pending.get("text", "") + fields.pop("text", "")
        self.pending.update(fields)
        self.pending["text"] = text
        if flags & 2 or not flags & 3:
            f = self.pending
            where = " ".join(
                part
                for part in (
                    f.get("module", ""),
                    f"{f['file']}:{f.get('line', '?')}" if "file" in f else "",
                    f.get("function", ""),
                )
                if part
            )
            level = SEVERITIES[severity] if severity < len(SEVERITIES) else str(severity)
            call.note(f"{level}{' ' + where if where else ''}: {f['text'].rstrip()!r}")
            self.pending = {}
        return 0

    @command(1)
    def SetDestination(self, call: Call) -> int:
        call.note(f"destination={call.arg('I'):#x}")
        return 0


@service("lm")
class LogService(Service):
    name = "lm"

    @command(0)
    def OpenLogger(self, call: Call) -> int:
        call.push_object(Logger(self.process))
        return 0
