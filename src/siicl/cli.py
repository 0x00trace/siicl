"""Command-line interface: `siicl info` and `siicl scan`."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from siicl import a64scan, npdm, nso, nup, paths
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
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (nup.NupError, nso.NsoError, npdm.NpdmError, a64scan.ScanError, paths.OutputPathError) as e:
        print(f"siicl: {e}", file=sys.stderr)
        return 1
