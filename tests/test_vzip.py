import os
import struct

import pytest

from siicl import cli, lz4, vzip
from tests.builders import build_nup, build_vzip

BLOCK = os.urandom(0x10000)
# The second block starts with the last half of the first, so it must copy from the previous block.
LINKED = BLOCK + BLOCK[0x8000:] + BLOCK[0x8000:] + b"tail" * 300
FILES = [
    ("957d2f1371b01b2c", LINKED, True),
    ("a0", b"RIFF\x10\x00\x00\x00WAVEfmt " + bytes(40), False),
    ("e5ec17c7be03271", b'{\n\t"key": 1\n}\n' * 20, True),
    ("bad/../name", b"VIPR\x02\x00\x00\x00" + bytes(100), True),
    ("empty", b"", False),
]
ALIASES = {"1234abcd": "e5ec17c7be03271", "beef": "e5ec17c7be03271"}


@pytest.fixture
def archive(tmp_path):
    p = tmp_path / "data.zip"
    p.write_bytes(build_vzip(FILES, aliases=ALIASES))
    return p


def test_lists_index_entries_and_aliases(archive):
    with vzip.VzipReader.open(archive) as r:
        assert len(r.entries) == len(FILES) + len(ALIASES)
        assert r.keys_sorted
        assert len(r.unique_data()) == len(FILES)
        target = r.find("e5ec17c7be03271")
        assert {e.name for e in r.aliases(target)} == {"e5ec17c7be03271", "1234abcd", "beef"}
        assert all(r.find(a).offset == target.offset for a in ALIASES)
        assert r.lookup(target.key) == [target]
        assert set(r.zip_entries) == {n for n, _, _ in FILES}


def test_reads_linked_lz4_blocks(archive):
    with vzip.VzipReader.open(archive) as r:
        e = r.find("957d2f1371b01b2c")
        assert e.compression == vzip.COMPRESSION_LZ4_LINKED
        assert r.read(e) == LINKED
        raw = r.read_at(e.offset, e.stored_size)
        blocks = list(r._blocks(raw))
        assert len(blocks) == 3
        with pytest.raises(ValueError):  # the second block cannot be decoded without the first
            lz4.decompress_block(blocks[1], 0x10000)


def test_reads_stored_and_aliased_data(archive):
    with vzip.VzipReader.open(archive) as r:
        assert r.read(r.find("a0")) == FILES[1][1]
        assert r.read(r.find("beef")) == FILES[2][1]
        assert r.read(r.find("empty")) == b""


def test_head_decodes_only_the_start(archive):
    with vzip.VzipReader.open(archive) as r:
        assert r.head(r.find("957d2f1371b01b2c"), 16) == LINKED[:16]
        assert r.head(r.find("a0"), 4) == b"RIFF"


def test_crc_mismatch_is_an_error(archive):
    raw = bytearray(archive.read_bytes())
    with vzip.VzipReader.open(archive) as r:
        e = r.find("a0")
    raw[e.offset + 20] ^= 0xFF
    archive.write_bytes(bytes(raw))
    with vzip.VzipReader.open(archive) as r:
        with pytest.raises(vzip.VzipError, match="CRC-32"):
            r.read(r.find("a0"))
        assert r.read(r.find("a0"), verify=False)[20] == raw[e.offset + 20]


def test_zip64_header_offsets(tmp_path):
    p = tmp_path / "z.zip"
    p.write_bytes(build_vzip(FILES, zip64_offsets=True))
    with vzip.VzipReader.open(p) as r:
        for e in r.unique_data():
            lho = r.zip_entries[e.name].header_offset
            assert r.read_at(lho, 4) == vzip.ZIP_LOCAL
            assert r.read(e) == dict((n, d) for n, d, _ in FILES)[e.name]


def test_opens_the_archive_inside_a_nup(tmp_path):
    nup_path = tmp_path / "game.nup"
    nup_path.write_bytes(build_nup([("exefs/main", b"code", True), ("romfs/data.zip", build_vzip(FILES), False)]))
    with vzip.VzipReader.open(nup_path) as r:
        assert r.read(r.find("957d2f1371b01b2c")) == LINKED


def test_rejects_other_files(tmp_path):
    p = tmp_path / "x"
    p.write_bytes(b"PK\x03\x04" + bytes(64))
    with pytest.raises(vzip.VzipError):
        vzip.VzipReader.open(p)


def test_rejects_record_pointing_outside_the_archive(tmp_path):
    raw = bytearray(build_vzip(FILES))
    struct.pack_into("<Q", raw, vzip.HEADER_SIZE + 8, 1 << 40)  # first record's data offset
    p = tmp_path / "bad.zip"
    p.write_bytes(bytes(raw))
    with pytest.raises(vzip.VzipError, match="outside"):
        vzip.VzipReader.open(p)


def test_output_names_are_safe(archive):
    with vzip.VzipReader.open(archive) as r:
        assert vzip.output_name(r.find("957d2f1371b01b2c")) == "957d2f1371b01b2c"
        assert vzip.output_name(r.find("bad/../name")).startswith("entry")


@pytest.mark.parametrize(
    "head, label",
    [
        (b"RIFF\x00\x00\x00\x00WAVEfmt ", "RIFF/WAVE"),
        (b"VIPR\x02\x00", "VIPR"),
        (b'{\n\t"a": 1}', "text ({)"),
        (b"\x20\x00\x00\x00\xff", "0x20000000"),
        (b"", "empty"),
    ],
)
def test_file_type(head, label):
    assert vzip.file_type(head) == label


@pytest.mark.parametrize("jobs", [1, 2])
def test_check_archive_verifies_and_greps(archive, jobs):
    res = vzip.check_archive(archive, jobs=jobs, pattern=b"WAVE")
    assert res.checked == len(FILES)
    assert res.failures == []
    assert res.matches == [("a0", 8)]


def test_lz4_linked_round_trip():
    data = os.urandom(5000) * 40
    blocks = lz4.compress_linked(data, 0x10000)
    assert lz4.decompress_linked(blocks, len(data), 0x10000) == data
    assert lz4.decompress_head(blocks[0], 10) == data[:10]


def test_cli_commands(archive, tmp_path, capsys):
    assert cli.main(["vzip", "stats", str(archive)]) == 0
    out = capsys.readouterr().out
    assert "Index records:          7" in out
    assert "RIFF/WAVE" in out

    assert cli.main(["vzip", "list", str(archive), "--sort", "size"]) == 0
    assert "957d2f1371b01b2c" in capsys.readouterr().out

    dest = tmp_path / "out"
    assert (
        cli.main(["vzip", "extract", str(archive), "--out", str(dest), "--type", "RIFF/WAVE", "--name", "bad/../name"])
        == 0
    )
    files = sorted(p.name for p in dest.iterdir())
    assert "a0" in files and "index.tsv" in files
    assert all("/" not in f and ".." not in f for f in files)
    assert (dest / "a0").read_bytes() == FILES[1][1]

    assert cli.main(["vzip", "verify", str(archive), "--jobs", "1"]) == 0
    assert "5 files" in capsys.readouterr().out


def test_cli_extract_refuses_the_repository(archive):
    from siicl import paths

    root = paths.project_root()
    assert cli.main(["vzip", "extract", str(archive), "--out", str(root / "out"), "--all"]) == 1


def test_cli_verify_reports_damage(archive, capsys):
    raw = bytearray(archive.read_bytes())
    with vzip.VzipReader.open(archive) as r:
        e = r.find("a0")
    raw[e.offset + 20] ^= 0xFF
    archive.write_bytes(bytes(raw))
    assert cli.main(["vzip", "verify", str(archive), "--jobs", "1"]) == 1
    assert "1 failed" in capsys.readouterr().out


@pytest.mark.parametrize(
    "data, h64, h32",
    [
        (b"", 0xCBF29CE484222325, 0x811C9DC5),
        (b"a", 0xAF63DC4C8601EC8C, 0xE40C292C),
        (b"foobar", 0x85944171F73967E8, 0xBF9CF968),
    ],
)
def test_fnv1a_reference_values(data, h64, h32):
    assert vzip.fnv1a64(data) == h64
    assert vzip.fnv1a32(data) == h32


def test_names_and_keys_follow_the_hashes():
    assert vzip.name_for_path("/data/a.png.import") == vzip.name_for_path("data/a.png.import")
    assert vzip.name_for_path("data/a.png.import") == format(vzip.fnv1a64(b"data/a.png.import"), "x")
    assert vzip.key_for_name("abc") == vzip.fnv1a32(b"abc")


def test_recovers_paths_named_inside_resources(tmp_path):
    shader = vzip.name_for_path("data/shaders/x.frag.glsl.import")
    texture = vzip.name_for_path("data/tex/y.png.import")
    resource = b"VIPR\x02\x00\x00\x00" + b"\x00/data/shaders/x.frag.glsl\x00" + b"\x00/data/tex/y.png.import\x00"
    p = tmp_path / "a.zip"
    p.write_bytes(build_vzip([(shader, resource, True), (texture, b"pixels", False), ("ffff", b"{\n}", False)]))
    with vzip.VzipReader.open(p) as r:
        assert vzip.recover_paths(r) == {shader: "data/shaders/x.frag.glsl.import", texture: "data/tex/y.png.import"}
        assert r.find_path("/data/tex/y.png.import").name == texture
        assert all(vzip.key_for_name(e.name) == e.key for e in r.entries)


def test_cli_paths_and_extract_by_path(tmp_path, capsys):
    texture = vzip.name_for_path("data/tex/y.png.import")
    resource = b"VIPR\x00\x00\x00\x00\x00/data/tex/y.png.import\x00"
    src = tmp_path / "a.zip"
    src.write_bytes(build_vzip([("ab", resource, True), (texture, b"pixels", False)]))
    table = tmp_path / "paths.tsv"
    assert cli.main(["vzip", "paths", str(src), "--out", str(table)]) == 0
    assert table.read_text() == f"name\tpath\n{texture}\tdata/tex/y.png.import\n"
    assert cli.main(["vzip", "list", str(src), "--paths", str(table)]) == 0
    assert "data/tex/y.png.import" in capsys.readouterr().out
    dest = tmp_path / "out"
    assert cli.main(["vzip", "extract", str(src), "--out", str(dest), "--path", "data/tex/y.png.import"]) == 0
    assert (dest / texture).read_bytes() == b"pixels"


def test_path_names_ignore_letter_case():
    # The game's archive lookup lowercases A-Z before hashing a path.
    assert vzip.name_for_path("Data/Shaders/A.GLSL.import") == vzip.name_for_path("data/shaders/a.glsl.import")
