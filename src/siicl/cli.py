"""Command-line interface: `siicl info` and `siicl scan`."""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from siicl import a64scan, npdm, nso, nup, nvshader, paths, vipr, vzip
from siicl.report import ModuleScan, Report, to_dict, to_markdown

EXEFS_MODULES = (
    "rtld",
    "main",
    "sdk",
    "subsdk0",
    "subsdk1",
    "subsdk2",
    "subsdk3",
    "subsdk4",
    "subsdk5",
    "subsdk6",
    "subsdk7",
    "subsdk8",
    "subsdk9",
)


def cmd_info(args: argparse.Namespace) -> int:
    with nup.NupReader(args.file) as r:
        i = r.info
        print(f"NUP version {i.version}, title ID {i.title_id:016X}")
        print(f"File size {i.file_size:,} bytes; table of contents hash checked")
        print(f"Unknown fields: {', '.join(f'{k}={v:#x}' for k, v in i.unknown.items())}")
        print()
        print(f"{'Path':<34} {'Size':>16} {'Stored':>16} Compression")
        for e in r.files:
            comp = nup.COMPRESSION_NAMES.get(e.compression, str(e.compression))
            print(f"{e.path:<34} {e.size:>16,} {e.stored_size:>16,} {comp}")
        print()
        print("Metadata records:")
        for rec in r.metadata():
            print(f"  kind {rec.kind} ({rec.description}): {len(rec.data):,} bytes")
        if args.verify:
            print()
            for e in r.files:
                if e.size > args.verify_limit:
                    print(f"  skipped {e.path} ({e.size:,} bytes, over --verify-limit)")
                    continue
                r.read(e, max_size=args.verify_limit)
                print(f"  {e.path}: hash matches")
    return 0


def _load_exefs(source: Path):
    """Return (title_id, {module name: NSO bytes}, NPDM bytes or None)."""
    if source.is_dir():
        mods = {p.name: p.read_bytes() for p in sorted(source.iterdir()) if p.name in EXEFS_MODULES}
        npdm_path = source / "main.npdm"
        return None, mods, npdm_path.read_bytes() if npdm_path.exists() else None
    with nup.NupReader(source) as r:
        mods, npdm_raw = {}, None
        for e in r.files:
            if not e.path.startswith("exefs/"):
                continue
            if e.name == "main.npdm":
                npdm_raw = r.read(e)
            elif e.name in EXEFS_MODULES:
                mods[e.name] = r.read(e)
        return r.info.title_id, mods, npdm_raw


def cmd_scan(args: argparse.Namespace) -> int:
    source = Path(args.file).expanduser()
    title_id, mods, npdm_raw = _load_exefs(source)
    if args.module:
        missing = set(args.module) - set(mods)
        if missing:
            print(f"siicl: modules not found: {', '.join(sorted(missing))}", file=sys.stderr)
            return 2
        mods = {k: v for k, v in mods.items() if k in args.module}
    if not mods:
        print("siicl: no exefs modules found", file=sys.stderr)
        return 2

    label = f"{title_id:016X}" if title_id is not None else source.stem
    out_dir = paths.ensure_outside_repo(Path(args.out) if args.out else paths.cache_dir() / label)
    out_dir.mkdir(parents=True, exist_ok=True)

    report = Report(source=source.name, title_id=title_id, npdm=npdm.parse(npdm_raw) if npdm_raw else None)
    order = {name: i for i, name in enumerate(EXEFS_MODULES)}
    for name in sorted(mods, key=lambda n: order.get(n, 99)):
        exe = nso.parse(mods[name])
        text = exe.segment_data("text")
        t0 = time.monotonic()
        scan = a64scan.scan_code(text, jobs=args.jobs, workdir=out_dir)
        print(f"scanned {name}: {len(text):,} bytes of text in {time.monotonic() - t0:.1f} s", file=sys.stderr)
        report.modules.append(ModuleScan(name, len(text), exe.module_id.hex(), scan))

    md = to_markdown(report)
    (out_dir / "report.md").write_text(md + "\n")
    (out_dir / "report.json").write_text(json.dumps(to_dict(report), indent=2) + "\n")
    print(md)
    print(f"\nWrote {out_dir / 'report.md'} and {out_dir / 'report.json'}", file=sys.stderr)
    return 0


def _gib(n: int) -> str:
    return f"{n / (1 << 30):.2f} GiB"


def cmd_vzip_list(args: argparse.Namespace) -> int:
    with vzip.VzipReader.open(args.file) as r:
        entries = r.entries
        if args.sort == "size":
            entries = sorted(entries, key=lambda e: -e.size)
        elif args.sort == "offset":
            entries = sorted(entries, key=lambda e: (e.offset, e.index))
        known = _read_paths(Path(args.paths)) if args.paths else {}
        if args.limit:
            entries = entries[: args.limit]
        print(f"{'Name':<17} {'Key':<8} {'Offset':>14} {'Stored':>14} {'Size':>14} Comp  {'Type':<16} Path")
        for e in entries:
            kind = vzip.file_type(r.head(e, 64))
            comp = vzip.COMPRESSION_NAMES.get(e.compression, str(e.compression))
            print(
                f"{e.name:<17} {e.key:08x} {e.offset:>#14x} {e.stored_size:>14,} {e.size:>14,} {comp:<5} "
                f"{kind:<16} {known.get(e.name, '')}"
            )
    return 0


def _read_paths(path: Path) -> dict[str, str]:
    out = {}
    for line in path.read_text().splitlines()[1:]:
        name, _, p = line.partition("\t")
        out[name] = p
    return out


def _default_out(source: str, leaf: str) -> Path:
    """~/.cache/siicl/<title id or file stem>/<leaf>."""
    label = Path(source).stem
    with open(source, "rb") as f:
        is_nup = f.read(3) == nup.MAGIC
    if is_nup:
        with nup.NupReader(source) as r:
            label = f"{r.info.title_id:016X}"
    return paths.cache_dir() / label / leaf


def cmd_vzip_paths(args: argparse.Namespace) -> int:
    out = paths.ensure_outside_repo(Path(args.out) if args.out else _default_out(args.file, "vzip-paths.tsv"))
    with vzip.VzipReader.open(args.file) as r:
        found = vzip.recover_paths(r)
        total = len(r.entries)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        f.write("name\tpath\n")
        for name, p in sorted(found.items(), key=lambda kv: kv[1]):
            f.write(f"{name}\t{p}\n")
    print(f"Found paths for {len(found):,} of {total:,} names; wrote {out}")
    return 0


def cmd_vzip_stats(args: argparse.Namespace) -> int:
    with vzip.VzipReader.open(args.file) as r:
        unique = r.unique_data()
        zips = r.zip_entries
        keys = [e.key for e in r.entries]
        print(
            f"Index records:          {len(r.entries):,} (keys sorted: {'yes' if r.keys_sorted else 'no'}, "
            f"{len(keys) - len(set(keys)):,} duplicate keys)"
        )
        print(f"Stored copies of data:  {len(unique):,} ({len(r.entries) - len(unique):,} names are aliases)")
        print(
            f"ZIP central directory:  {len(zips):,} entries; {sum(1 for n in zips if n not in r.by_name):,} "
            "not in the index"
        )
        print(f"Compressed (LZ4):       {sum(1 for e in unique if e.compression == 1):,} of {len(unique):,}")
        print(
            f"Size:                   {_gib(sum(e.size for e in unique))} "
            f"({_gib(sum(e.stored_size for e in unique))} stored)"
        )
        print(f"Index flags field:      {', '.join(f'{v:#x}' for v in sorted({e.flags for e in r.entries}))}")
        fnv = sum(1 for e in r.entries if vzip.key_for_name(e.name) == e.key)
        print(f"Keys = FNV-1a 32 of name: {fnv:,} of {len(r.entries):,}")
        types: dict[str, list[int]] = {}
        for e in unique:
            t = types.setdefault(vzip.file_type(r.head(e, 64)), [0, 0])
            t[0] += 1
            t[1] += e.size
        print(f"\nFile types (by first bytes; {len(types):,} distinct):")
        print(f"  {'Type':<16} {'Files':>8} {'Size':>12}")
        for name, (count, size) in sorted(types.items(), key=lambda kv: -kv[1][1])[: args.top]:
            print(f"  {name:<16} {count:>8,} {_gib(size):>12}")
    return 0


def cmd_vzip_extract(args: argparse.Namespace) -> int:
    if not (args.name or args.path or args.type or args.all):
        print("siicl: choose entries with --name, --path, --type or --all", file=sys.stderr)
        return 2
    out = paths.ensure_outside_repo(Path(args.out))
    with vzip.VzipReader.open(args.file) as r:
        if args.all:
            chosen = r.unique_data()
        else:
            chosen = [r.find(n) for n in args.name or []] + [r.find_path(p) for p in args.path or []]
            if args.type:
                wanted = set(args.type)
                chosen += [e for e in r.unique_data() if vzip.file_type(r.head(e, 64)) in wanted]
        chosen = list({e.index: e for e in chosen}.values())
        print(f"Extracting {len(chosen):,} files, {_gib(sum(e.size for e in chosen))}, to {out}", file=sys.stderr)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "index.tsv", "w") as listing:
            listing.write("file\tname\tkey\tsize\ttype\taliases\n")
            for e in chosen:
                data = r.read(e, max_size=1 << 62)
                fname = vzip.output_name(e)
                (out / fname).write_bytes(data)
                aliases = ",".join(a.name for a in r.aliases(e) if a.index != e.index)
                listing.write(f"{fname}\t{e.name}\t{e.key:08x}\t{e.size}\t{vzip.file_type(data[:64])}\t{aliases}\n")
    print(f"Wrote {out / 'index.tsv'}", file=sys.stderr)
    return 0


def _run_pass(args: argparse.Namespace, pattern: bytes | None) -> vzip.PassResult:
    t0 = time.monotonic()
    last = [0.0]

    def progress(res: vzip.PassResult, total: int) -> None:
        now = time.monotonic()
        if now - last[0] >= 2 or res.checked + len(res.failures) == total:
            last[0] = now
            print(
                f"  {res.checked + len(res.failures):,} of {total:,} files, {_gib(res.bytes)}, {now - t0:.0f} s",
                file=sys.stderr,
            )

    return vzip.check_archive(args.file, jobs=args.jobs, pattern=pattern, progress=progress)


def cmd_vzip_shaders(args: argparse.Namespace) -> int:
    stages: Counter[str] = Counter()
    ops: Counter[int] = Counter()
    checks: Counter[str] = Counter()
    counts: list[int] = []
    total_bytes = 0
    uldc_programs = 0
    problems: list[tuple[str, str]] = []
    rows = []
    programs: list[tuple[str, bytes]] = []
    with vzip.VzipReader.open(args.file) as r:
        for e in r.unique_data():
            if r.head(e, 4) != nvshader.MAGIC_BYTES:
                continue
            data = r.read(e)
            total_bytes += len(data)
            try:
                sh = nvshader.parse(data)
                insns = sh.instructions()
            except nvshader.ShaderError as err:
                problems.append((e.name, str(err)))
                continue
            stages[sh.stage] += 1
            counts.append(len(insns))
            these = [nvshader.opcode(lo) for lo, _ in insns]
            ops.update(these)
            uldc_programs += nvshader.OP_ULDC in these
            checks["control"] += sh.control_magic_ok
            checks["reserved"] += all(hi >> 62 == 0 for _, hi in insns)
            if sh.header and sh.stage == "fragment":
                checks["fragment"] += 1
                checks["kill"] += sh.header.kills_pixels == (nvshader.OP_KILL in these)
            rows.append((e.name, sh.stage, len(insns)))
            if args.nvdisasm:
                programs.append((e.name, b"".join(struct.pack("<QQ", lo, hi) for lo, hi in insns)))
    n = len(counts)
    if args.list:
        for name, stage, count in rows:
            print(f"{name:<17} {stage:<24} {count:>7,} instructions")
        print()
    print(f"Shader binaries: {n + len(problems):,} ({total_bytes / (1 << 20):.1f} MiB)")
    for stage, count in stages.most_common():
        print(f"  {stage:<24} {count:>7,}")
    if counts:
        counts.sort()
        print(f"Instructions: {sum(counts):,} (median {counts[n // 2]:,} per program, largest {counts[-1]:,})")
    print("Checks:")
    print(f"  header sizes match, and the program ends with EXIT and a BRA to itself: {n:,} of {n + len(problems):,}")
    print(f"  control section starts with 0x98761234: {checks['control']:,} of {n:,}")
    print(f"  reserved bits 126-127 are zero in every instruction: {checks['reserved']:,} of {n:,}")
    print(
        f"  KILL appears exactly when the header allows discarding pixels: {checks['kill']:,} of "
        f"{checks['fragment']:,} fragment programs"
    )
    print(f"Uniform-register loads (ULDC, Turing and later): {ops[nvshader.OP_ULDC]:,} in {uldc_programs:,} programs")
    print("Most common opcodes:")
    for op, count in ops.most_common(args.top):
        print(f"  {nvshader.opcode_name(op):<8} 0x{op:03x} {count:>10,}")
    if args.nvdisasm:
        with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as pool:
            results = list(pool.map(lambda p: nvshader.disassemble(p[1], args.nvdisasm), programs))
        failed = [(name, res.errors[0]) for (name, _), res in zip(programs, results, strict=True) if res.errors]
        print(
            f"nvdisasm --binary {args.nvdisasm}: {len(programs) - len(failed):,} of {len(programs):,} programs "
            "decode without errors"
        )
        problems += failed
    for name, err in problems[:20]:
        print(f"problem: {name}: {err}")
    return 1 if problems else 0


def cmd_vzip_disasm(args: argparse.Namespace) -> int:
    with vzip.VzipReader.open(args.file) as r:
        e = r.find_path(args.entry) if "/" in args.entry else r.find(args.entry)
        sh = nvshader.parse(r.read(e))
    code = sh.program_bytes()
    print(f"// {e.name}: {sh.stage} program, {len(code) // nvshader.INSTRUCTION_SIZE:,} instructions")
    if sh.header:
        h = sh.header
        print(
            f"// program header: type {h.sph_type}, version {h.version}, shader type {h.shader_type}, "
            f"multiple render targets {'yes' if h.mrt_enable else 'no'}, "
            f"may discard pixels {'yes' if h.kills_pixels else 'no'}"
        )
    dis = nvshader.disassemble(code, args.arch)
    print(dis.text, end="")
    for err in dis.errors:
        print(f"siicl: {err}", file=sys.stderr)
    return 1 if dis.errors else 0


def cmd_vzip_buffers(args: argparse.Namespace) -> int:
    known = _read_paths(Path(args.paths)) if args.paths else {}
    with vzip.VzipReader.open(args.file) as r:
        m = vipr.match_buffers(r)
        by_offset = {e.offset: e for e in r.unique_data()}
        records: Counter[str] = Counter()
        matched: Counter[str] = Counter()
        rows = []
        for e in r.unique_data():
            if r.head(e, 4) != vipr.MAGIC:
                continue
            data = r.read(e)
            refs = vipr.buffer_refs(data)
            if refs:
                t = vipr.gpu_buffer_type(data)
                records[t.label if t else "no GPU type"] += len(refs)
        for offset, owner in sorted(m.owners.items()):
            t = vipr.gpu_buffer_type(r.head(owner.resource, 0x20))
            matched[t.label if t else "no GPU type"] += 1
            if args.list:
                buf = by_offset[offset]
                rows.append(
                    (
                        buf.name,
                        vzip.file_type(r.head(buf, 64)),
                        buf.size,
                        owner.ref.index,
                        owner.ref.count,
                        owner.resource.name,
                        next((known[a.name] for a in r.aliases(owner.resource) if a.name in known), ""),
                    )
                )
    if args.list:
        print(f"{'Buffer':<17} {'Type':<12} {'Size':>12} Index  {'Resource':<17} Path")
        for name, kind, size, index, count, res, path in rows:
            print(f"{name:<17} {kind:<12} {size:>12,} {index:>3}/{count:<4} {res:<17} {path}")
        print()
    print(f"Images that list external buffers: {m.resources:,}")
    print(
        "Buffer records, by the GPU type the image declares, and how many match an entry stored just before the image:"
    )
    print(f"  {'Type':<24} {'Records':>9} {'Matched':>9}")
    for label, count in records.most_common():
        print(f"  {label:<24} {count:>9,} {matched[label]:>9,}")
    return 0


def cmd_vzip_verify(args: argparse.Namespace) -> int:
    res = _run_pass(args, None)
    print(f"Checked {res.checked:,} files ({_gib(res.bytes)}): {len(res.failures):,} failed")
    for name, err in res.failures[:20]:
        print(f"  {name}: {err}")
    return 1 if res.failures else 0


def cmd_vzip_grep(args: argparse.Namespace) -> int:
    pattern = bytes.fromhex(args.pattern) if args.hex else args.pattern.encode()
    res = _run_pass(args, pattern)
    with vzip.VzipReader.open(args.file) as r:
        for name, pos in sorted(res.matches):
            e = r.find(name)
            print(f"{name:<17} at {pos:#x} of {e.size:,} bytes, type {vzip.file_type(r.head(e, 64))}")
    print(f"{len(res.matches):,} of {res.checked:,} files contain the pattern", file=sys.stderr)
    return 1 if res.failures else 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="siicl", description="Research tools toward a Switch 2 compatibility layer.")
    sub = p.add_subparsers(dest="command", required=True)

    info = sub.add_parser("info", help="list the files and metadata in a NUP container")
    info.add_argument("file")
    info.add_argument("--verify", action="store_true", help="also check every file against its hash")
    info.add_argument(
        "--verify-limit",
        type=int,
        default=nup.DEFAULT_MAX_READ,
        help="skip files larger than this many bytes when verifying (default 2 GiB)",
    )
    info.set_defaults(func=cmd_info)

    scan = sub.add_parser("scan", help="report the CPU features and SVCs a game's code uses")
    scan.add_argument("file", help="a NUP container, or a folder that holds exefs files")
    scan.add_argument("--out", help="output folder (default ~/.cache/siicl/<title id>)")
    scan.add_argument("--jobs", type=int, help="parallel objdump processes (default: CPU count, at most 16)")
    scan.add_argument("--module", action="append", help="scan only this module; repeat for more")
    scan.set_defaults(func=cmd_scan)

    vz = sub.add_parser("vzip", help="read the VZIP archive in a NUP container's romfs (or a VZIP file)")
    vsub = vz.add_subparsers(dest="vzip_command", required=True)
    v_list = vsub.add_parser("list", help="list the index entries")
    v_list.add_argument("file")
    v_list.add_argument("--sort", choices=["key", "size", "offset"], default="key")
    v_list.add_argument("--limit", type=int, default=0, help="show at most this many entries")
    v_list.add_argument("--paths", help="a TSV from 'siicl vzip paths' to show known paths")
    v_list.set_defaults(func=cmd_vzip_list)
    v_stats = vsub.add_parser("stats", help="summarise the archive and count file types")
    v_stats.add_argument("file")
    v_stats.add_argument("--top", type=int, default=40, help="file types to show (default 40)")
    v_stats.set_defaults(func=cmd_vzip_stats)
    v_ext = vsub.add_parser("extract", help="write entries to a folder outside the repository")
    v_ext.add_argument("file")
    v_ext.add_argument("--out", required=True)
    v_ext.add_argument("--name", action="append", help="an entry name; repeat for more")
    v_ext.add_argument("--path", action="append", help="an asset path, such as data/x.png.import; repeat for more")
    v_ext.add_argument("--type", action="append", help="a file type as 'vzip stats' shows it; repeat for more")
    v_ext.add_argument("--all", action="store_true", help="every stored copy of data (aliases go in index.tsv)")
    v_ext.set_defaults(func=cmd_vzip_extract)
    v_paths = vsub.add_parser("paths", help="recover asset paths for index names and write them as TSV")
    v_paths.add_argument("file")
    v_paths.add_argument("--out", help="output file (default ~/.cache/siicl/<title id>/vzip-paths.tsv)")
    v_paths.set_defaults(func=cmd_vzip_paths)
    v_sh = vsub.add_parser("shaders", help="check and summarise the compiled GPU shader binaries")
    v_sh.add_argument("file")
    v_sh.add_argument("--list", action="store_true", help="also list each binary")
    v_sh.add_argument("--top", type=int, default=20, help="opcodes to show (default 20)")
    v_sh.add_argument("--nvdisasm", metavar="SM", help="also decode every program with nvdisasm, such as SM86")
    v_sh.set_defaults(func=cmd_vzip_shaders)
    v_dis = vsub.add_parser("disasm", help="disassemble one shader binary with nvdisasm")
    v_dis.add_argument("file")
    v_dis.add_argument("entry", help="an entry name, or an asset path")
    v_dis.add_argument("--arch", default="SM86", help="SM version for nvdisasm (default SM86)")
    v_dis.set_defaults(func=cmd_vzip_disasm)
    v_buf = vsub.add_parser("buffers", help="match GPU data entries to the resources that own them")
    v_buf.add_argument("file")
    v_buf.add_argument("--list", action="store_true", help="also list each matched buffer")
    v_buf.add_argument("--paths", help="a TSV from 'siicl vzip paths' to show resource paths")
    v_buf.set_defaults(func=cmd_vzip_buffers)
    v_ver = vsub.add_parser("verify", help="decompress every entry and check its CRC-32")
    v_ver.add_argument("file")
    v_ver.add_argument("--jobs", type=int, default=8, help="worker processes (default 8)")
    v_ver.set_defaults(func=cmd_vzip_verify)
    v_grep = vsub.add_parser("grep", help="list entries whose data contains a byte pattern")
    v_grep.add_argument("file")
    v_grep.add_argument("pattern")
    v_grep.add_argument("--hex", action="store_true", help="the pattern is hex, such as 424e5348")
    v_grep.add_argument("--jobs", type=int, default=8, help="worker processes (default 8)")
    v_grep.set_defaults(func=cmd_vzip_grep)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (
        nup.NupError,
        nso.NsoError,
        npdm.NpdmError,
        vzip.VzipError,
        nvshader.ShaderError,
        a64scan.ScanError,
        paths.OutputPathError,
        KeyError,
    ) as e:
        print(f"siicl: {e}", file=sys.stderr)
        return 1
