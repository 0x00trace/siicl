"""Tests for IPC messages and the service framework, with messages the tests build themselves."""

import struct

import pytest

from siicl import ipc, kernel, services
from siicl import npdm as npdm_mod
from siicl.kernel import Unimplemented
from siicl.services import applet, lm, time
from siicl.services.sm import ServiceManager
from tests.builders import build_npdm, build_nso

pytestmark = pytest.mark.usefixtures("cpu_backend")  # each test runs on Unicorn and on dynarmic

POINTER = 0x8000


def test_messages_survive_a_round_trip():
    raw = ipc.build(
        type=4,
        data=b"payload!",
        copy_handles=[0x1234],
        move_handles=[0x5678, 0x9ABC],
        pid=0,
        statics=[ipc.Buffer(0x12_3456_7890, 0x40, 3)],
        sends=[ipc.Buffer(0x4_0000_1000, 0x2_0000_0010, 1)],
        receives=[ipc.Buffer(0x3000, 0x20, 0)],
        receive_list=[ipc.Buffer(0x1_0000_8000, 0x100)],
    )
    msg = ipc.parse(raw.ljust(ipc.MESSAGE_SIZE, b"\0"))
    assert (msg.type, msg.pid, msg.copy_handles, msg.move_handles) == (4, 0, [0x1234], [0x5678, 0x9ABC])
    assert msg.statics == [ipc.Buffer(0x12_3456_7890, 0x40, 3)]
    assert msg.sends == [ipc.Buffer(0x4_0000_1000, 0x2_0000_0010, 1)]
    assert msg.receives == [ipc.Buffer(0x3000, 0x20, 0)]
    assert msg.receive_list == [ipc.Buffer(0x1_0000_8000, 0x100)]
    start = (-msg.data_offset) % 16
    assert msg.data[start : start + 8] == b"payload!"


def test_cmif_requests_with_and_without_a_domain_header():
    header = struct.pack("<IIII", ipc.CMIF_IN, 1, 42, 7) + b"\x01\x02"
    plain = ipc.parse_cmif(ipc.parse(ipc.build(type=4, data=header)), domain=False)
    assert (plain.command, plain.token, plain.payload[:2]) == (42, 7, b"\x01\x02")
    domain_header = struct.pack("<BBHI4xI", ipc.DOMAIN_SEND, 1, len(header), 5, 0)
    req = ipc.parse_cmif(ipc.parse(ipc.build(type=4, data=domain_header + header + struct.pack("<I", 9))), domain=True)
    assert (req.command, req.object_id, req.input_objects, req.payload) == (42, 5, [9], b"\x01\x02")


def test_cmif_responses_start_their_data_on_a_16_byte_boundary():
    raw = ipc.build_cmif_response(0xABC, b"\x55" * 4, copy_handles=[7], domain_objects=[3])
    msg = ipc.parse(raw.ljust(ipc.MESSAGE_SIZE, b"\0"))
    start = (msg.data_offset + (-msg.data_offset) % 16) - msg.data_offset
    count = struct.unpack_from("<I", msg.data, start)[0]
    magic, _, result, _ = struct.unpack_from("<IIII", msg.data, start + 16)
    assert (msg.copy_handles, count, magic, result) == ([7], 1, ipc.CMIF_OUT, 0xABC)
    assert msg.data[start + 32 : start + 36] == b"\x55" * 4
    assert struct.unpack_from("<I", msg.data, start + 36)[0] == 3


def process():
    return kernel.Process({"rtld": build_nso(b"\0" * 4)}, npdm_mod.parse(build_npdm()))


def send(p, session, raw):
    t = p.main_thread
    p.cpu.write(t.tls, raw.ljust(ipc.MESSAGE_SIZE, b"\0"))
    text = services.handle_request(p, t, session, t.tls, ipc.MESSAGE_SIZE)
    return text, ipc.parse(p.cpu.read(t.tls, ipc.MESSAGE_SIZE))


def cmif(command, payload=b"", *, domain_object=None, domain_command=ipc.DOMAIN_SEND, **kw):
    data = struct.pack("<IIII", ipc.CMIF_IN, 0, command, 0) + payload
    if domain_object is not None:
        data = struct.pack("<BBHI4xI", domain_command, 0, len(data), domain_object, 0) + data
    return ipc.build(type=4, data=data, **kw)


def response(msg, *, domain=False, payload_size=0):
    """(result, payload, domain object IDs) of a CMIF response. The IDs follow payload_size bytes of payload."""
    start = (-msg.data_offset) % 16
    ids = []
    if domain:
        count = struct.unpack_from("<I", msg.data, start)[0]
        start += 16
    magic, _, result, _ = struct.unpack_from("<IIII", msg.data, start)
    assert magic == ipc.CMIF_OUT
    payload = msg.data[start + 16 :]
    if domain and count:
        ids = list(struct.unpack_from(f"<{count}I", payload, payload_size))
    return result, payload, ids


def test_the_service_manager_hands_out_sessions_over_tipc():
    p = process()
    sm = services.Session(ServiceManager(p))
    text, msg = send(p, sm, ipc.build(type=ipc.TIPC_FIRST + 1, data=b"lm".ljust(8, b"\0"), align_data=False))
    assert text == "sm: GetServiceHandle('lm') -> lm"
    assert struct.unpack_from("<I", msg.data)[0] == 0
    session = p.handles.get(msg.move_handles[0])
    assert isinstance(session.obj, lm.LogService)
    with pytest.raises(Unimplemented, match='the service "no:such" is not implemented'):
        send(p, sm, ipc.build(type=ipc.TIPC_FIRST + 1, data=b"no:such", align_data=False))


def test_domains_hold_the_objects_commands_return():
    p = process()
    session = services.Session(lm.LogService(p))
    text, msg = send(p, session, ipc.build(type=ipc.TYPE_CONTROL, data=struct.pack("<IIII", ipc.CMIF_IN, 0, 3, 0)))
    assert text == "lm QueryPointerBufferSize" and response(msg)[1][:2] == struct.pack("<H", POINTER)
    _, msg = send(p, session, ipc.build(type=ipc.TYPE_CONTROL, data=struct.pack("<IIII", ipc.CMIF_IN, 0, 0, 0)))
    assert response(msg)[1][:4] == struct.pack("<I", 1)
    text, msg = send(p, session, cmif(0, domain_object=1))
    assert text == "lm OpenLogger -> lm ILogger"
    assert response(msg, domain=True)[2] == [2]
    assert isinstance(session.domain[2], lm.Logger)
    text, _ = send(p, session, cmif(0, domain_object=2, domain_command=ipc.DOMAIN_CLOSE))
    assert text == "lm ILogger close object 2" and 2 not in session.domain


def test_without_a_domain_returned_objects_get_their_own_sessions():
    p = process()
    session = services.Session(applet.AppletOE(p))
    text, msg = send(p, session, cmif(0, pid=0))
    assert text == "appletOE OpenApplicationProxy -> applet IApplicationProxy"
    assert isinstance(p.handles.get(msg.move_handles[0]).obj, applet.ApplicationProxy)
    _, msg = send(p, session, ipc.build(type=ipc.TYPE_CONTROL, data=struct.pack("<IIII", ipc.CMIF_IN, 0, 2, 0)))
    assert p.handles.get(msg.move_handles[0]).obj is session.obj


def test_unknown_commands_end_the_run_or_are_answered_when_asked():
    p = process()
    session = services.Session(lm.LogService(p))
    with pytest.raises(Unimplemented, match=r"lm command 77 is not implemented \(arguments 0102"):
        send(p, session, cmif(77, b"\x01\x02"))
    p.stub_unknown = True
    text, msg = send(p, session, cmif(77))
    assert text.endswith("-> not implemented, answered with success") and response(msg)[0] == 0
    assert p.stubbed == ["lm command 77"]


def test_the_application_starts_with_a_focus_message_and_then_has_none():
    p = process()
    session = services.Session(applet.CommonStateGetter(p))
    text, msg = send(p, session, cmif(1))
    assert text == "applet ICommonStateGetter ReceiveMessage(message 0xf)"
    assert response(msg)[1][:4] == struct.pack("<I", applet.MESSAGE_FOCUS_STATE_CHANGED)
    _, msg = send(p, session, cmif(1))
    assert response(msg)[0] == applet.RESULT_NO_MESSAGE


def test_log_packets_are_decoded_into_one_message():
    def field(key, data):
        return bytes([key, len(data)]) + data

    body = field(2, b"hello ") + field(3, struct.pack("<I", 12)) + field(4, b"a.cpp") + field(6, b"core")
    head = struct.pack("<QQBBBBI", 0, 0, 1, 0, 1, 0, len(body)) + body
    fields, flags, severity = lm.parse_packet(head)
    assert (fields, flags, severity) == ({"text": "hello ", "line": 12, "file": "a.cpp", "module": "core"}, 1, 1)


def test_time_shared_memory_has_the_switch_2_steady_clock_block():
    view = time.time_shared_memory().view
    assert struct.unpack_from("<q", view, 0x40)[0] == time.BOOT_POSIX_TIME  # user clock offset
    assert bytes(view[0x50:0x60]) == time.CLOCK_SOURCE_ID
    base, mult, shift, offset, bound = struct.unpack_from("<qqQqq", view, 0xD8)
    assert (base, mult, shift, offset, bound) == (0, 0, 0, 0, time.INT64_MAX)
    assert bytes(view[0x100:0x110]) == time.CLOCK_SOURCE_ID


def test_calendar_times_and_clock_snapshots():
    cal = time.calendar(time.BOOT_POSIX_TIME + 86400 + 3661)
    assert struct.unpack_from("<hbbbbb", cal) == (2026, 1, 2, 1, 1, 1)
    assert struct.unpack_from("<II", cal, 8) == (5, 1)  # Friday, the year's second day
    assert len(time.clock_snapshot(process(), 1)) == 0xD0


def tipc(command, data=b"", **kw):
    return ipc.build(type=ipc.TIPC_FIRST + command, data=data, align_data=False, **kw)


def test_the_graphics_supervisor_answers_with_the_handles_its_clients_read():
    from siicl.services import gfx

    p = process()
    session = services.Session(gfx.GraphicsSupervisor(p))
    text, msg = send(
        p, session, tipc(4, struct.pack("<QQ", 0x51, 0x8000000000000006) + bytes(32), copy_handles=[0xFFFF8001])
    )
    assert text == "gsv IGraphicsSupervisor command 4 (boot)(process ID 0x51, 0x8000000000000006)"
    assert (msg.move_handles, struct.unpack_from("<I", msg.data)[0]) == ([], 0)
    _, msg = send(p, session, tipc(1402))
    assert isinstance(p.handles.get(msg.move_handles[0]).obj, gfx.NvService)
    _, msg = send(p, session, tipc(201))
    assert struct.unpack_from("<I", p.cpu.read(p.main_thread.tls, 12), 8)[0] == 0 and not msg.move_handles


def test_the_supervisor_writes_its_queue_header_and_keeps_the_transfer_memories():
    from siicl.kernel import TransferMemory
    from siicl.services import gfx

    p = process()
    p.space.map(0x1000_0000, 0x3000, 3, 5, "lent")
    shared, server = TransferMemory(0x1000_0000, 0x1000, 3), TransferMemory(0x1000_1000, 0x2000, 1)
    handles = [p.handles.add(shared), p.handles.add(server)]
    session = services.Session(gfx.GraphicsSupervisor(p))
    _, msg = send(p, session, tipc(204, struct.pack("<IIQQ", 1, 0x1FF, 0x1000, 0x2000), copy_handles=handles))
    raw = p.cpu.read(p.main_thread.tls, 24)  # two moved handles, then the result, at offsets 12, 16 and 20
    assert struct.unpack_from("<III", raw, 12) == (msg.move_handles[0], msg.move_handles[1], 0)
    assert struct.unpack("<8I", p.cpu.read(0x1000_1000, 32)) == gfx.QUEUE_HEADER
    assert shared.kept and server.kept


def test_supervisor_sessions_hand_out_events_and_signal_them():
    from siicl.kernel import ReadableEvent
    from siicl.services import gfx

    p = process()
    session = services.Session(gfx.Session497(p))
    _, msg = send(p, session, tipc(12, struct.pack("<I", 0)))
    event = p.handles.get(msg.move_handles[0])
    assert isinstance(event, ReadableEvent) and not event.signaled
    send(p, session, tipc(21, struct.pack("<I", 0)))
    assert event.signaled
    _, msg = send(p, session, tipc(11))
    assert struct.unpack_from("<III", p.cpu.read(p.main_thread.tls, 20), 8) == (0, 0, 63)


def test_the_display_session_opens_the_default_display():
    from siicl.services import gfx

    p = process()
    session = services.Session(gfx.DisplaySession(p))
    text, _ = send(p, session, tipc(101, b"default\0" + struct.pack("<I", 1)))
    assert text == "vi2 session command 101 (open display)('default', 1)"
    _, msg = send(p, session, tipc(11, struct.pack("<I", 1)))
    assert p.handles.get(msg.move_handles[0]).signaled


def test_hid_hands_out_shared_memory_that_takes_the_size_the_game_maps(tmp_path):
    from siicl.services import hid

    p = process()
    session = services.Session(hid.Hid(p))
    _, msg = send(p, session, cmif(0, struct.pack("<Q", 0x51), pid=0))
    resource = p.handles.get(msg.move_handles[0])
    _, msg = send(p, resource, cmif(0))
    shared = p.handles.get(msg.copy_handles[0])
    assert shared.size is None
    shared.allocate(0x40000)
    assert shared.size == 0x40000 and len(shared.view) == 0x40000


def test_the_application_gets_an_operation_mode_event_and_a_graphics_scale():
    p = process()
    functions = services.Session(applet.ApplicationFunctions(p))
    _, msg = send(p, functions, cmif(210))
    assert p.handles.get(msg.copy_handles[0]).signaled
    scaling = services.Session(applet.GraphicsScalingParametersFunctions(p))
    _, msg = send(p, scaling, cmif(10))
    assert response(msg)[1][:4] == struct.pack("<I", 10)
