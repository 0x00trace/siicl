"""Tests for the RomFS image builder: the tests read the image back with their own small reader."""

import struct

from siicl import romfs


def lookup(image: romfs.RomFsImage, path: str) -> tuple[int, int] | None:
    """Find a file through the hash tables, as the game's library does: (data offset, size)."""
    header = struct.unpack("<10Q", image.read(0, 0x50))
    _, dh_off, dh_size, dm_off, _, fh_off, fh_size, fm_off, _, data_off = header
    parent = 0
    parts = path.encode().split(b"/")
    for part in parts[:-1]:
        buckets = struct.unpack(f"<{dh_size // 4}I", image.read(dh_off, dh_size))
        entry = buckets[romfs.path_hash(parent, part) % len(buckets)]
        while entry != romfs.NONE:
            p, _, _, _, nxt, n = struct.unpack("<6I", image.read(dm_off + entry, 0x18))
            if p == parent and image.read(dm_off + entry + 0x18, n) == part:
                break
            entry = nxt
        if entry == romfs.NONE:
            return None
        parent = entry
    buckets = struct.unpack(f"<{fh_size // 4}I", image.read(fh_off, fh_size))
    entry = buckets[romfs.path_hash(parent, parts[-1]) % len(buckets)]
    while entry != romfs.NONE:
        p, _, off, size, nxt, n = struct.unpack("<IIQQII", image.read(fm_off + entry, 0x20))
        if p == parent and image.read(fm_off + entry + 0x20, n) == parts[-1]:
            return data_off + off, size
        entry = nxt
    return None


def file(path, data):
    return romfs.RomFile(path, len(data), lambda off, n: data[off : off + n])


def test_files_are_found_through_the_hash_tables_and_read_back():
    files = {
        "data.zip": bytes(range(256)) * 3,
        "settings.cvar": b"cvar=1",
        "dir/sub/deep.bin": b"\xaa" * 40,
        "dir/top.txt": b"top",
    }
    image = romfs.RomFsImage([file(p, d) for p, d in files.items()])
    for path, data in files.items():
        offset, size = lookup(image, path)
        assert (size, image.read(offset, size)) == (len(data), data)
    assert lookup(image, "dir/missing.txt") is None
    assert image.size % 16 == 0 or image.size == lookup(image, "settings.cvar")[0] + 6


def test_a_read_can_span_files_padding_and_the_end_of_the_image():
    image = romfs.RomFsImage([file("a", b"A" * 5), file("b", b"B" * 3)])
    a, _ = lookup(image, "a")
    b, _ = lookup(image, "b")
    assert b - a == 16  # each file starts on a 16-byte boundary
    assert image.read(a, 24) == b"A" * 5 + bytes(11) + b"B" * 3 + bytes(5)


def test_hash_bucket_counts_follow_the_standard_rule():
    assert [romfs.bucket_count(n) for n in (0, 2, 3, 4, 18, 19, 20)] == [3, 3, 3, 5, 19, 19, 23]
