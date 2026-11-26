import pytest

from siicl import nso
from tests.builders import build_nso

TEXT = bytes(range(256)) * 64


@pytest.mark.parametrize("compress", [True, False])
def test_reads_segments(compress):
    exe = nso.parse(build_nso(TEXT, b"rodata!", b"data!!", compress=compress))
    assert exe.segment_data("text") == TEXT
    assert exe.segment_data("rodata") == b"rodata!"
    assert exe.segment_data("data") == b"data!!"
    assert exe.segments["text"].compressed is compress
    assert exe.segments["rodata"].memory_offset == 0x4000
    assert exe.module_id == b"\x11" * 32


def test_rejects_segment_whose_hash_does_not_match():
    raw = bytearray(build_nso(TEXT, compress=False))
    raw[0x100] ^= 0xFF
    with pytest.raises(nso.NsoError, match="SHA-256"):
        nso.parse(bytes(raw)).segment_data("text")


def test_rejects_other_files():
    with pytest.raises(nso.NsoError, match="NSO0"):
        nso.parse(b"NRO0" + bytes(0x100))
