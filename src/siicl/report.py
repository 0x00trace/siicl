"""Build the scan report from a dump's NPDM and code scans."""

from __future__ import annotations

from dataclasses import dataclass, field

from siicl.a64scan import FEATURES, CodeScan
from siicl.npdm import Npdm
from siicl.svc import SWITCH2_ONLY, svc_name


@dataclass
class ModuleScan:
    name: str
    text_size: int
    module_id: str
    scan: CodeScan


@dataclass
class Report:
    source: str
    title_id: int | None
    npdm: Npdm | None
    modules: list[ModuleScan] = field(default_factory=list)

    def combined(self) -> CodeScan:
        total = CodeScan()
        for m in self.modules:
            total.merge(m.scan)
        return total


def _feature_rows(scan: CodeScan) -> list[dict]:
    rows = []
    order = list(FEATURES)
    for fid, (count, per) in sorted(scan.features().items(), key=lambda kv: order.index(kv[0])):
        f = FEATURES[fid]
        rows.append(
            {
                "id": fid,
                "arch": f.arch,
                "title": f.title,
                "count": count,
                "mnemonics": per.most_common(6),
                "note": f.note,
            }
        )
    return rows


def _caps_dict(caps) -> dict:
    return {
        "svcs": sorted(caps.svcs),
        "priority_range": [caps.highest_priority, caps.lowest_priority],
        "core_range": [caps.min_core, caps.max_core],
        "kernel_version": list(caps.kernel_version) if caps.kernel_version else None,
        "handle_table_size": caps.handle_table_size,
        "program_type": caps.program_type,
        "misc_flags": caps.misc_flags,
        "descriptor_counts": caps.descriptor_counts,
    }


def to_dict(report: Report) -> dict:
    combined = report.combined()
    out: dict = {
        "source": report.source,
        "title_id": f"{report.title_id:016X}" if report.title_id is not None else None,
        "modules": [],
        "combined": _scan_dict(combined),
    }
    for m in report.modules:
        d = _scan_dict(m.scan)
        d.update(name=m.name, text_size=m.text_size, module_id=m.module_id)
        out["modules"].append(d)
    n = report.npdm
    if n is not None:
        out["npdm"] = {
            "name": n.name,
            "product_code": n.product_code,
            "program_id": f"{n.program_id:016X}",
            "flags": n.flags,
            "is_64bit": n.is_64bit,
            "address_space": n.address_space,
            "main_thread_priority": n.main_thread_priority,
            "main_thread_core": n.main_thread_core,
            "main_thread_stack_size": n.main_thread_stack_size,
            "system_resource_size": n.system_resource_size,
            "version": n.version,
            "fs_permissions": f"{n.fs_permissions:016X}" if n.fs_permissions is not None else None,
            "services": [{"name": s.name, "server": s.server} for s in n.services],
            "caps": _caps_dict(n.caps),
            "acid_caps": _caps_dict(n.acid_caps) if n.acid_caps else None,
        }
    return out


def _scan_dict(scan: CodeScan) -> dict:
    regs = scan.sysregs()
    return {
        "bytes": scan.size,
        "words": scan.words,
        "undefined_words": scan.undefined,
        "zero_runs_skipped": scan.zero_runs,
        "zero_words_skipped": scan.skipped_words,
        "features": _feature_rows(scan),
        "svc": {f"0x{k:02X}": v for k, v in sorted(scan.svc.items())},
        "sysregs": {"mrs": dict(regs["mrs"].most_common()), "msr": dict(regs["msr"].most_common())},
        "cache_ops": dict(scan.cache_ops().most_common()),
        "top_mnemonics": scan.mnemonics().most_common(40),
    }


def _mib(n: int) -> str:
    return f"{n / (1 << 20):.1f} MiB"


def to_markdown(report: Report) -> str:
    combined = report.combined()
    L: list[str] = ["# SIICL scan report", ""]
    L.append(f"- Source: `{report.source}`")
    if report.title_id is not None:
        L.append(f"- Title ID: `{report.title_id:016X}`")
    L.append(f"- Modules scanned: {', '.join(m.name for m in report.modules)}")
    L.append(
        f"- Code scanned: {_mib(combined.size)}, {combined.words:,} words, {combined.undefined:,} not decodable, "
        f"{combined.skipped_words:,} zero words skipped by objdump"
    )
    L.append("")

    n = report.npdm
    allowed: set[int] = n.caps.svcs if n else set()
    if n:
        c = n.caps
        L += ["## Process descriptor (NPDM)", "", "| Field | Value |", "|---|---|"]
        L.append(f"| Name | {n.name} |")
        L.append(f"| Program ID | `{n.program_id:016X}` |")
        L.append(f"| Instruction set | {'AArch64' if n.is_64bit else 'AArch32'} |")
        L.append(f"| Address space | {n.address_space} (flags 0x{n.flags:02X}) |")
        L.append(
            f"| Main thread | priority {n.main_thread_priority}, core {n.main_thread_core}, "
            f"stack {n.main_thread_stack_size:#x} |"
        )
        L.append(f"| System resource size | {n.system_resource_size:#x} |")
        if c.highest_priority is not None:
            L.append(f"| Allowed priorities | {c.highest_priority} to {c.lowest_priority} |")
            L.append(f"| Allowed cores | {c.min_core} to {c.max_core} |")
        if c.kernel_version:
            L.append(f"| Kernel version | {c.kernel_version[0]}.{c.kernel_version[1]} |")
        if c.handle_table_size is not None:
            L.append(f"| Handle table size | {c.handle_table_size} |")
        if c.program_type is not None:
            L.append(f"| Program type | {c.program_type} |")
        if c.misc_flags is not None:
            L.append(f"| Misc flags | {c.misc_flags:#x} |")
        if n.fs_permissions is not None:
            L.append(f"| FS permissions | `{n.fs_permissions:016X}` |")
        L.append("")
        clients = sorted(s.name for s in n.services if not s.server)
        servers = sorted(s.name for s in n.services if s.server)
        L.append(
            f"Services the process may connect to ({len(clients)}): " + (", ".join(f"`{s}`" for s in clients) or "none")
        )
        if servers:
            L.append("")
            L.append(f"Services the process may host ({len(servers)}): " + ", ".join(f"`{s}`" for s in servers))
        L.append("")
        if n.acid_caps and n.acid_caps.svcs != c.svcs:
            extra = sorted(n.acid_caps.svcs - c.svcs)
            L.append(
                f"The ACID allows {len(extra)} SVCs that the ACI0 does not request: "
                + ", ".join(f"0x{s:02X}" for s in extra)
            )
            L.append("")

    L += ["## System calls", ""]
    ids = sorted(set(combined.svc) | allowed)
    L += ["| SVC | Name | Allowed by NPDM | Call sites in code |", "|---|---|---|---|"]
    for i in ids:
        mark = " (Switch 2)" if i in SWITCH2_ONLY else ""
        L.append(f"| 0x{i:02X} | {svc_name(i)}{mark} | {'yes' if i in allowed else 'no'} | {combined.svc.get(i, 0)} |")
    L.append("")
    if n:
        unused = sorted(allowed - set(combined.svc))
        stray = sorted(set(combined.svc) - allowed)
        L.append(f"- Allowed but no call site found: {len(unused)}")
        L.append(
            f"- Call sites for SVCs the NPDM does not allow: {len(stray)}. The SDK links a wrapper for every SVC, "
            "including ones an application may not call, so these are usually unused stubs."
        )
        L.append("")

    L += [
        "## Architecture features (all modules)",
        "",
        "| Feature | Arm version | Words | Most common | Note |",
        "|---|---|---|---|---|",
    ]
    for row in _feature_rows(combined):
        top = ", ".join(f"{m} {c:,}" for m, c in row["mnemonics"][:4])
        L.append(f"| {row['id']}: {row['title']} | {row['arch']} | {row['count']:,} | {top} | {row['note']} |")
    L.append("")

    L += ["## Features per module", "", "| Module | Text | Words | Features above Armv8.0 |", "|---|---|---|---|"]
    for m in report.modules:
        feats = [fid for fid in m.scan.features() if FEATURES[fid].arch not in ("8.0", "-")]
        L.append(f"| {m.name} | {_mib(m.text_size)} | {m.scan.words:,} | {', '.join(feats) or 'none'} |")
    L.append("")

    regs = combined.sysregs()
    names = sorted(set(regs["mrs"]) | set(regs["msr"]), key=lambda r: -(regs["mrs"][r] + regs["msr"][r]))
    L += ["## System registers", "", "| Register | Reads (MRS) | Writes (MSR) |", "|---|---|---|"]
    for r in names:
        L.append(f"| `{r}` | {regs['mrs'][r]:,} | {regs['msr'][r]:,} |")
    L.append("")

    ops = combined.cache_ops()
    if ops:
        L += ["## Cache maintenance", "", "| Operation | Words |", "|---|---|"]
        L += [f"| `{op}` | {c:,} |" for op, c in ops.most_common()]
        L.append("")
    return "\n".join(L)
