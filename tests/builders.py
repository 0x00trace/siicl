"""Build synthetic NUP, NSO and NPDM files for tests.

Tests never use real game data (see CLAUDE.md); everything here is made up.
"""

from __future__ import annotations

import hashlib
import shutil
import struct
import subprocess
import zlib
from compression import zstd
from pathlib import Path

from siicl import lz4, vzip

AS = shutil.which("aarch64-linux-gnu-as")
OBJCOPY = shutil.which("aarch64-linux-gnu-objcopy")
MARCH = (
    "armv9.5-a+sve2+sme+memtag+rng+ls64+mops+cssc+sm4+sha3+lse128+rcpc3+the+gcs+fp16fml+i8mm+bf16"
    "+crypto+flagm+predres+sb+ssbs+wfxt+hbc+fp16+dotprod+rcpc+lor+rdma+lse+crc+pauth+rcpc2+flagm2"
    "+frintts+jscvt+fcma"
)


def assemble(source: str, workdir: Path) -> bytes:
    """Assemble AArch64 source and return the raw bytes of its .text section."""
    src, obj, out = workdir / "t.s", workdir / "t.o", workdir / "t.bin"
    src.write_text(".text\n" + source + "\n")
    subprocess.run([AS, f"-march={MARCH}", "-o", str(obj), str(src)], check=True, capture_output=True)
    subprocess.run([OBJCOPY, "-O", "binary", "-j", ".text", str(obj), str(out)], check=True)
    return out.read_bytes()


def build_nup(files, *, title_id=0x0400_0000_0000_1000, metadata=()):
    """files: list of (path, data, zstd_compress). Paths must be 'dir/name'."""
    dirs: list[str] = []
    for path, _, _ in files:
        d = path.split("/", 1)[0]
        if d not in dirs:
            dirs.append(d)
    by_dir = {d: [f for f in files if f[0].split("/", 1)[0] == d] for d in dirs}
    ordered = [f for d in dirs for f in by_dir[d]]

    strings = bytearray()

    def add(name: str) -> tuple[int, int]:
        off = len(strings)
        strings.extend(name.encode())
        return off, len(name.encode())

    dir_names = [add(d) for d in dirs]
    file_names = [add(p.split("/", 1)[1]) for p, _, _ in ordered]
    n_dirs = len(dirs) + 1
    files_off = 0x70 + n_dirs * 0x18
    nodes_off = files_off + len(ordered) * 0x50
    str_off = nodes_off + 4 * (len(dirs) + len(ordered))
    toc_size = (str_off + len(strings) + 15) & ~15
    data_off = 0x80 + toc_size

    pos = data_off
    stored_list = []
    for _, data, compress in ordered:
        stored = zstd.compress(data) if compress else data
        stored_list.append((pos, stored))
        pos = (pos + len(stored) + 15) & ~15
    meta = b"".join(struct.pack("<II", kind, len(d)) + d for kind, d in metadata)
    meta_off = pos
    total = meta_off + len(meta)

    toc = bytearray(toc_size)
    toc[0:4] = b"TOC\x01"
    struct.pack_into("<III", toc, 4, 1, n_dirs, 0)
    struct.pack_into("<QQQ", toc, 0x10, len(ordered), str_off, len(strings))
    struct.pack_into("<Q", toc, 0x30, title_id)
    struct.pack_into("<IIIII", toc, 0x38, 0, 1, 2, 0x20000, 0x101)
    struct.pack_into("<QQ", toc, 0x50, meta_off if meta else 0, len(meta))
    struct.pack_into("<IIII", toc, 0x70, 0, 0, 0, len(dirs))
    first = len(dirs)
    for i, d in enumerate(dirs):
        struct.pack_into("<IIII", toc, 0x70 + (i + 1) * 0x18, *dir_names[i], first, len(by_dir[d]))
        first += len(by_dir[d])
    for i, ((_, data, compress), (off, stored)) in enumerate(zip(ordered, stored_list, strict=True)):
        o = files_off + i * 0x50
        struct.pack_into("<IIQQQ", toc, o, *file_names[i], len(data), off, len(stored))
        toc[o + 0x20 : o + 0x40] = hashlib.sha256(data).digest()
        struct.pack_into("<I", toc, o + 0x40, 1 if compress else 0)
    for i in range(len(dirs) + len(ordered)):
        struct.pack_into("<I", toc, nodes_off + 4 * i, i + 1)
    toc[str_off : str_off + len(strings)] = strings

    header = bytearray(0x80)
    header[0:4] = b"NUP\x01"
    struct.pack_into("<HHQQ", header, 4, 0x80, 1, 0x80, toc_size)
    header[0x18:0x38] = hashlib.sha256(toc).digest()
    struct.pack_into("<QQI", header, 0x38, data_off, total, 1)

    out = bytearray(header + toc)
    for off, stored in stored_list:
        out.extend(b"\0" * (off - len(out)))
        out.extend(stored)
    out.extend(b"\0" * (meta_off - len(out)))
    out.extend(meta)
    return bytes(out)


def build_nso(
    text: bytes,
    rodata: bytes = b"ro",
    data: bytes = b"data",
    *,
    compress=True,
    hashed=True,
    bss=0x1000,
    module_id=b"\x11" * 32,
) -> bytes:
    segs = [text, rodata, data]
    stored = [lz4.compress_block(s) if compress else s for s in segs]
    header = bytearray(0x100)
    header[0:4] = b"NSO0"
    struct.pack_into("<I", header, 0x0C, (0b111 if compress else 0) | (0b111000 if hashed else 0))
    file_off, mem_off = 0x100, 0
    for i, s in enumerate(segs):
        struct.pack_into("<III", header, 0x10 + i * 0x10, file_off, mem_off, len(s))
        file_off += len(stored[i])
        mem_off += (len(s) + 0xFFF) & ~0xFFF
    struct.pack_into("<I", header, 0x3C, bss)
    header[0x40:0x60] = module_id
    struct.pack_into("<III", header, 0x60, *(len(s) for s in stored))
    for i, s in enumerate(segs):
        header[0xA0 + i * 0x20 : 0xC0 + i * 0x20] = hashlib.sha256(s).digest()
    return bytes(header) + b"".join(stored)


def kac_svcs(svcs) -> list[int]:
    groups: dict[int, int] = {}
    for s in svcs:
        groups[s // 24] = groups.get(s // 24, 0) | (1 << (s % 24))
    return [(index << 29) | (mask << 5) | 0b1111 for index, mask in sorted(groups.items())]


def build_npdm(
    *,
    svcs=(0x01, 0x21, 0x80),
    services=(("fsp-srv", False), ("my:srv", True)),
    program_id=0x0400_0000_0000_1000,
    name="Application",
    acid_svcs=None,
) -> bytes:
    kac = [
        (3 << 24) | (0 << 16) | (24 << 10) | (59 << 4) | 0b0111,  # ThreadInfo
        *kac_svcs(svcs),
        (6 << 14) | 0x1FFF,  # MiscParams: program type 6
        (20 << 19) | (5 << 15) | 0x3FFF,  # KernelVersion 20.5
        (1023 << 16) | 0x7FFF,  # HandleTableSize
        (1 << 17) | 0xFFFF,  # MiscFlags
        0xFFFFFFFF,  # padding
    ]
    kac_raw = struct.pack(f"<{len(kac)}I", *kac)
    sac = b"".join(bytes([(len(n) - 1) | (0x80 if srv else 0)]) + n.encode() for n, srv in services)
    fac = struct.pack("<B3xQIIII", 1, 0x8000_0000_0000_0001, 0x1C, 0, 0x1C, 0)

    aci = bytearray(0x40)
    aci[0:4] = b"ACI0"
    struct.pack_into("<Q", aci, 0x10, program_id)
    fac_off = 0x40
    sac_off = fac_off + len(fac)
    kac_off = (sac_off + len(sac) + 15) & ~15
    struct.pack_into("<IIIIII", aci, 0x20, fac_off, len(fac), sac_off, len(sac), kac_off, len(kac_raw))
    aci.extend(fac + sac)
    aci.extend(b"\0" * (kac_off - len(aci)))
    aci.extend(kac_raw)

    acid_list = kac if acid_svcs is None else [kac[0], *kac_svcs(acid_svcs)]
    acid_kac = struct.pack(f"<{len(acid_list)}I", *acid_list)
    acid = bytearray(0x240)
    acid[0x200:0x204] = b"ACID"
    struct.pack_into("<IQQ", acid, 0x20C, 0, program_id, program_id)
    struct.pack_into("<IIIIII", acid, 0x220, 0x240, 0, 0x240, 0, 0x240, len(acid_kac))
    acid.extend(acid_kac)

    meta = bytearray(0x80)
    meta[0:4] = b"META"
    meta[0x0C] = 0x37
    meta[0x0E] = 44
    meta[0x0F] = 0
    struct.pack_into("<III", meta, 0x14, 0, 1, 0x100000)
    meta[0x20 : 0x20 + len(name)] = name.encode()
    aci_off = 0x80
    acid_off = (aci_off + len(aci) + 15) & ~15
    struct.pack_into("<IIII", meta, 0x70, aci_off, len(aci), acid_off, len(acid))
    meta.extend(aci)
    meta.extend(b"\0" * (acid_off - len(meta)))
    meta.extend(acid)
    return bytes(meta)


def build_vzip(files, *, aliases=None, keys=None, zip64_offsets=False) -> bytes:
    """files: list of (name, data, compress). aliases: {alias name: name of the file it shares data with}."""
    aliases = aliases or {}
    keys = keys or {}
    names = [n for n, _, _ in files] + list(aliases)
    strtab = bytearray()
    name_off = {}
    for n in names:
        name_off[n] = len(strtab)
        strtab += n.encode() + b"\0"
    index_size = vzip.HEADER_SIZE + vzip.RECORD_SIZE * len(names) + len(strtab)
    pos = (index_size + 15) & ~15

    body = bytearray()
    placed = {}  # name -> (data offset, stored, size, comp, crc, header offset)
    for name, data, compress in files:
        payload = b"".join(struct.pack("<I", len(blk)) + blk for blk in lz4.compress_linked(data)) if compress else data
        lho = pos + len(body)
        nb = name.encode()
        k = 2
        while (lho + 30 + len(nb) + 20 + 4 + k) % 16:
            k += 1
        extra = struct.pack("<HHQQ", 1, 16, len(data), len(payload)) + struct.pack("<HHH", 0xA11E, k, 16) + bytes(k - 2)
        body += struct.pack("<IHHHHHIIIHH", 0x04034B50, 45, 8, 0, 0, 0, 0, 0xFFFFFFFF, 0xFFFFFFFF, len(nb), len(extra))
        body += nb + extra
        data_off = pos + len(body)
        crc = zlib.crc32(data)
        body += payload
        body += struct.pack("<IIQQ", 0x08074B50, crc, len(payload), len(data))
        placed[name] = (data_off, len(payload), len(data), 1 if compress else 0, crc, lho)

    cd = bytearray()
    for name, _, _ in files:
        data_off, stored, size, comp, crc, lho = placed[name]
        nb = name.encode()
        z64 = struct.pack("<QQ", size, stored) + (struct.pack("<Q", lho) if zip64_offsets else b"")
        extra = struct.pack("<HH", 1, len(z64)) + z64 + struct.pack("<HHH", 0xA11E, 2, 16)
        field = 0xFFFFFFFF if zip64_offsets else lho
        cd += struct.pack(
            "<IHHHHHHIIIHHHHHII",
            0x02014B50,
            0x0A2D,
            45,
            8,
            0,
            0,
            0,
            crc,
            0xFFFFFFFF,
            0xFFFFFFFF,
            len(nb),
            len(extra),
            0,
            0,
            0,
            0,
            field,
        )
        cd += nb + extra
    cd_off = pos + len(body)
    z64_off = cd_off + len(cd)
    tail = struct.pack("<IQHHIIQQQQ", 0x06064B50, 44, 0x0A2D, 45, 0, 0, len(files), len(files), len(cd), cd_off)
    tail += struct.pack("<IIQI", 0x07064B50, 0, z64_off, 1)
    tail += struct.pack("<IHHHHIIH", 0x06054B50, 0, 0, 0xFFFF, 0xFFFF, len(cd), 0xFFFFFFFF, 0)

    records = []
    for n in names:
        data_off, stored, size, comp, _, _ = placed[aliases.get(n, n)]
        key = keys.get(n, vzip.key_for_name(n))
        records.append((key, struct.pack("<IIQQQII", name_off[n], 0x65FFFFFF, data_off, stored, size, comp, key)))
    records.sort(key=lambda kv: kv[0])
    index = struct.pack("<4sII", b"VZIP", len(names), len(strtab)) + b"".join(r for _, r in records) + strtab
    return index + bytes(pos - len(index)) + bytes(body) + bytes(cd) + tail
