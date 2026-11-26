import pytest

from siicl import nup
from tests.builders import build_nup

FILES = [
    ("exefs/main", b"main code " * 100, True),
    ("exefs/main.npdm", b"META....", True),
    ("romfs/data.bin", b"\x00\x01\x02" * 50, False),
]


@pytest.fixture
def nup_path(tmp_path):
    p = tmp_path / "sample.nup"
    p.write_bytes(build_nup(FILES, metadata=[(1, b"META"), (5, b"Name")]))
    return p


def test_lists_files_with_directory_paths(nup_path):
    with nup.NupReader(nup_path) as r:
        assert [f.path for f in r.files] == ["exefs/main", "exefs/main.npdm", "romfs/data.bin"]
        assert r.info.title_id == 0x0400_0000_0000_1000
        assert r.info.version == 1
        assert r.find("romfs/data.bin").compression == nup.COMPRESSION_NONE


def test_reads_and_verifies_each_file(nup_path):
    with nup.NupReader(nup_path) as r:
        for path, data, _ in FILES:
            assert r.read(r.find(path)) == data


def test_reads_metadata_records(nup_path):
    with nup.NupReader(nup_path) as r:
        assert [(m.kind, m.data) for m in r.metadata()] == [(1, b"META"), (5, b"Name")]


def test_rejects_damaged_table_of_contents(tmp_path):
    raw = bytearray(build_nup(FILES))
    raw[0x80 + 0x30] ^= 0xFF  # title ID byte; the TOC hash no longer matches
    p = tmp_path / "bad.nup"
    p.write_bytes(raw)
    with pytest.raises(nup.NupError, match="SHA-256"):
        nup.NupReader(p)


def test_rejects_file_whose_hash_does_not_match(tmp_path):
    raw = bytearray(build_nup(FILES))
    p = tmp_path / "bad.nup"
    with nup.NupReader(_write(p, raw)) as r:
        entry = r.find("romfs/data.bin")
    raw[entry.offset] ^= 0xFF
    with nup.NupReader(_write(p, raw)) as r, pytest.raises(nup.NupError, match="SHA-256"):
        r.read(r.find("romfs/data.bin"))


def test_refuses_reads_over_the_limit(nup_path):
    with nup.NupReader(nup_path) as r, pytest.raises(nup.NupError, match="read limit"):
        r.read(r.find("exefs/main"), max_size=10)


def test_rejects_other_files(tmp_path):
    p = tmp_path / "x.nup"
    p.write_bytes(b"PFS0" + bytes(200))
    with pytest.raises(nup.NupError, match="not a NUP"):
        nup.NupReader(p)


def _write(p, raw):
    p.write_bytes(bytes(raw))
    return p
