"""Build synthetic NUP, NSO and NPDM files for tests.

Tests never use real game data (see CLAUDE.md); everything here is made up.
"""

from __future__ import annotations

import hashlib
import shutil
import struct
import subprocess
from compression import zstd
from pathlib import Path

from siicl import lz4

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
