import struct

from siicl import cli, nvshader, vipr, vzip
from tests.builders import build_vzip


def make_vipr(buffer_sizes, *, descriptor=(12 << 26) | (1 << 22)) -> bytes:
    """A minimal VIPR image that declares a GPU buffer type and lists external buffers."""
    head = b"VIPR" + struct.pack("<III", 2, 0x40, 8) + struct.pack("<QII", 1, descriptor, 1)
    body = b"GARD" + bytes(12)
    count = len(buffer_sizes)
    for index, size in enumerate(buffer_sizes):
        body += struct.pack("<IIIIQQ", count, 0xBFFF0000, index, 0xCFFF0000, 0, size)
    return head + body + b"GARDBYE!"


def make_shader(n: int) -> bytes:
    code = b"\x18\x79" + bytes(14 + 16 * n)
    control = struct.pack("<I", nvshader.CONTROL_MAGIC) + bytes(0x1C)
    return struct.pack("<IIIIQ", nvshader.MAGIC, len(code), len(control), 0, n) + code + control


def test_reads_header_type_and_buffer_records():
    data = make_vipr([100, 200])
    t = vipr.gpu_buffer_type(data)
    assert t is not None and (t.backend, t.kind) == (12, 1) and t.label == "NVN gen2 buffer"
    assert vipr.buffer_refs(data) == [vipr.BufferRef(2, 0, 100), vipr.BufferRef(2, 1, 200)]
    assert vipr.gpu_buffer_type(b"VIPR" + bytes(0x20)) is None


def test_matches_buffers_stored_before_their_image(tmp_path):
    s1, s2, s3 = make_shader(1), make_shader(3), make_shader(5)
    files = [
        ("b1", s1, True),
        ("b2", s2, True),
        ("img1", make_vipr([len(s2), len(s1)]), True),  # records need not follow archive order
        ("b3", s3, False),
        ("img2", make_vipr([len(s3), 12345]), True),  # 12345: stored elsewhere, no match
        ("json", b'{\n"a": 1}', False),
    ]
    p = tmp_path / "a.zip"
    p.write_bytes(build_vzip(files))
    with vzip.VzipReader.open(p) as r:
        m = vipr.match_buffers(r)
        owner = {r.find(n).offset: n for n, _, _ in files}
        got = {owner[off]: (o.resource.name, o.ref.index) for off, o in m.owners.items()}
    assert got == {"b2": ("img1", 0), "b1": ("img1", 1), "b3": ("img2", 0)}
    assert m.resources == 2 and m.refs == 4
    assert [(e.name, ref.size) for e, ref in m.unmatched] == [("img2", 12345)]


def test_cli_buffers(tmp_path, capsys):
    s1 = make_shader(2)
    p = tmp_path / "a.zip"
    p.write_bytes(build_vzip([("b1", s1, True), ("img", make_vipr([len(s1)]), True)]))
    assert cli.main(["vzip", "buffers", str(p), "--list"]) == 0
    out = capsys.readouterr().out
    assert "b1 " in out and "img" in out
    assert "NVN gen2 buffer" in out
