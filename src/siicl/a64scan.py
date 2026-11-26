"""Find out which AArch64 architecture features a block of machine code uses.

The scanner disassembles the code with GNU objdump for AArch64 and reduces each
instruction to its mnemonic and the shape of its operands: register numbers and
immediate values are replaced by their kind, so `add x0, x1, #16` becomes
`("add", "x, x, #i")`. Features are recognised from these shapes.

The input is read as a flat array of 32-bit words. Data inside a text segment
is decoded too, so a feature with only a handful of hits may come from data
rather than code, and should be checked by hand.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from functools import cache
from itertools import repeat
from pathlib import Path

CHUNK_SIZE = 4 << 20  # bytes per objdump run; must be a multiple of 4
OBJDUMP_CANDIDATES = ("aarch64-linux-gnu-objdump", "aarch64-none-elf-objdump", "aarch64-elf-objdump")


class ScanError(RuntimeError):
    pass


def find_objdump() -> str:
    override = os.environ.get("SIICL_OBJDUMP")
    if override:
        return override
    for name in OBJDUMP_CANDIDATES:
        path = shutil.which(name)
        if path:
            return path
    raise ScanError("no AArch64 objdump found: install aarch64-linux-gnu-binutils or set SIICL_OBJDUMP")


# ---------------------------------------------------------------------------
# Operand shapes

_RE_TARGET = re.compile(r"(?:^|(?<=[ ,]))(?:0x)?[0-9a-f]+(?: <[^>]*>)?$")
_RE_GPR = re.compile(r"\b([wx])(?:[0-9]+|zr)\b")
_RE_SVE = re.compile(r"\b(z|p|pn)[0-9]+\b")
_RE_FPR = re.compile(r"\b([bhsdq])[0-9]+\b")
_RE_VEC = re.compile(r"\bv[0-9]+\.")
_RE_IDX = re.compile(r"\[[0-9]+\]")
_RE_IMM = re.compile(r"#-?(?:0x[0-9a-f]+|[0-9]+(?:\.[0-9]+)?(?:e[+-]?[0-9]+)?)")
_TARGET_MNEMONICS = frozenset({"b", "bl", "cbz", "cbnz", "tbz", "tbnz", "adr", "adrp"})
_LITERAL_MNEMONICS = frozenset({"ldr", "ldrsw", "prfm"})


def operand_shape(mnemonic: str, operands: str) -> str:
    """Replace register numbers, immediates and branch targets by their kind."""
    if (
        mnemonic in _TARGET_MNEMONICS
        or mnemonic.startswith(("b.", "bc."))
        or (mnemonic in _LITERAL_MNEMONICS and "[" not in operands)
    ):
        operands = _RE_TARGET.sub("A", operands)
    operands = _RE_GPR.sub(r"\1", operands)
    operands = _RE_SVE.sub(r"\1", operands)
    operands = _RE_FPR.sub(r"\1", operands)
    operands = _RE_VEC.sub("v.", operands)
    operands = _RE_IDX.sub("[i]", operands)
    return _RE_IMM.sub("#i", operands)


# ---------------------------------------------------------------------------
# Features


@dataclass(frozen=True)
class Feature:
    id: str
    arch: str
    title: str
    note: str = ""


_FEATURE_LIST = [
    # Base Armv8.0 groups, listed for context.
    Feature("A64_SIMD", "8.0", "Advanced SIMD (vector) instructions"),
    Feature(
        "A64_EXCLUSIVE",
        "8.0",
        "Exclusive load/store (LDXR/STXR family)",
        "Atomics written the Armv8.0 way. A translator must emulate the exclusive monitor.",
    ),
    Feature("A64_ACQREL", "8.0", "Load-acquire and store-release (LDAR/STLR)"),
    Feature("A64_BARRIER", "8.0", "Barriers (DMB, DSB, ISB)"),
    Feature("A64_CACHE", "8.0", "Cache maintenance (DC, IC)"),
    Feature("A64_SYSREG", "8.0", "System register access (MRS, MSR)"),
    Feature("A64_EXCEPTION", "8.0", "Exception generation (SVC, BRK, UDF and others)"),
    Feature("A64_HINT", "8.0", "Hints (NOP, YIELD, WFE and others)"),
    Feature("A64_UNDEFINED", "-", "Words objdump could not decode", "Usually data inside the text segment."),
    Feature("A64_ZERO", "-", "Single zero words (UDF #0)", "Padding between functions, not real instructions."),
    # Optional crypto and CRC in Armv8.0.
    Feature("FEAT_CRC32", "8.0", "CRC32 instructions"),
    Feature("FEAT_AES", "8.0", "AES instructions"),
    Feature("FEAT_PMULL", "8.0", "64-bit polynomial multiply (PMULL .1q)"),
    Feature("FEAT_SHA1", "8.0", "SHA-1 instructions"),
    Feature("FEAT_SHA256", "8.0", "SHA-256 instructions"),
    # Armv8.1
    Feature(
        "FEAT_LSE",
        "8.1",
        "Large System Extensions atomics (CAS, SWP, LDADD and others)",
        "x86-64 can map most of these to LOCK-prefixed instructions or CMPXCHG loops.",
    ),
    Feature("FEAT_RDM", "8.1", "Rounding doubling multiply-accumulate (SQRDMLAH, SQRDMLSH)"),
    Feature("FEAT_LOR", "8.1", "Limited ordering regions (LDLAR, STLLR)"),
    # Armv8.2
    Feature(
        "FEAT_FP16",
        "8.2",
        "Half-precision floating-point arithmetic",
        "Without AVX512-FP16, compute in single precision: it gives identical results for add, "
        "subtract, multiply, divide and square root. Fused multiply-add needs double precision.",
    ),
    Feature(
        "FEAT_DotProd",
        "8.2",
        "Int8 dot product (SDOT, UDOT)",
        "x86 VNNI multiplies unsigned by signed bytes, so SDOT and UDOT need extra steps.",
    ),
    Feature("FEAT_FHM", "8.2", "Half-precision multiply-add long (FMLAL, FMLSL)"),
    Feature("FEAT_SHA3", "8.2", "SHA-3 helpers (EOR3, BCAX, RAX1, XAR)"),
    Feature("FEAT_SHA512", "8.2", "SHA-512 instructions"),
    Feature("FEAT_SM3", "8.2", "SM3 instructions"),
    Feature("FEAT_SM4", "8.2", "SM4 instructions"),
    Feature("FEAT_DPB", "8.2", "DC CVAP (clean to point of persistence)"),
    Feature("FEAT_SVE", "8.2", "Scalable Vector Extension"),
    # Armv8.3
    Feature(
        "FEAT_PAuth_HINT",
        "8.3",
        "Pointer authentication, NOP-compatible forms (PACIASP, AUTIASP and others)",
        "These run as NOP on cores without pointer authentication. A translator can drop them.",
    ),
    Feature(
        "FEAT_PAuth",
        "8.3",
        "Pointer authentication, other forms (RETAA, BRAA, LDRAA, PACIA and others)",
        "A translator must handle these: treat PAC as no change, AUT as success, and the combined "
        "forms as plain branches and loads.",
    ),
    Feature("FEAT_JSCVT", "8.3", "JavaScript conversion (FJCVTZS)"),
    Feature("FEAT_FCMA", "8.3", "Complex-number multiply-add (FCMLA, FCADD)"),
    Feature(
        "FEAT_LRCPC",
        "8.3",
        "Load-acquire RCpc (LDAPR)",
        "On x86-64, whose memory model is stronger, a plain load is enough.",
    ),
    # Armv8.4
    Feature("FEAT_LRCPC2", "8.4", "Load-acquire RCpc with offsets (LDAPUR, STLUR)"),
    Feature("FEAT_FlagM", "8.4", "Flag manipulation (CFINV, RMIF, SETF8, SETF16)"),
    Feature("FEAT_DIT", "8.4", "Data-independent timing (MSR DIT)"),
    # Armv8.5
    Feature("FEAT_FlagM2", "8.5", "Flag format conversion (AXFLAG, XAFLAG)"),
    Feature("FEAT_FRINTTS", "8.5", "Round to 32/64-bit integer (FRINT32X and others)"),
    Feature("FEAT_SB", "8.5", "Speculation barrier (SB)"),
    Feature("FEAT_BTI", "8.5", "Branch target identification (BTI)", "Runs as NOP on cores without BTI."),
    Feature("FEAT_MTE", "8.5", "Memory tagging"),
    Feature("FEAT_RNG", "8.5", "Random number registers (RNDR, RNDRRS)"),
    Feature("FEAT_DPB2", "8.5", "DC CVADP (clean to point of deep persistence)"),
    Feature("FEAT_SSBS", "8.5", "Speculative store bypass safe (MSR SSBS)"),
    Feature("FEAT_SPECRES", "8.5", "Speculation restriction (CFP, DVP, CPP RCTX)"),
    # Armv8.6 and later
    Feature("FEAT_BF16", "8.6", "BFloat16 instructions"),
    Feature("FEAT_I8MM", "8.6", "Int8 matrix multiply (SMMLA, USDOT and others)"),
    Feature("FEAT_LS64", "8.7", "64-byte loads and stores (LD64B, ST64B)"),
    Feature("FEAT_WFxT", "8.7", "WFE/WFI with timeout (WFET, WFIT)"),
    Feature("FEAT_MOPS", "8.8", "Memory copy and set instructions (CPYP, SETP and others)"),
    Feature("FEAT_HBC", "8.8", "Hinted conditional branch (BC.cond)"),
    Feature("FEAT_CSSC", "8.9", "Common short sequence compression (scalar ABS, CNT, CTZ, MIN, MAX)"),
    Feature("FEAT_LRCPC3", "8.9", "Load-acquire RCpc, third set (LDIAPP, STILP, LDAP1, STL1)"),
    Feature("FEAT_THE", "8.9", "Translation hardening (RCW instructions)"),
    Feature("FEAT_CLRBHB", "8.9", "Clear branch history (CLRBHB)"),
    Feature("FEAT_SME", "9.2", "Scalable Matrix Extension"),
    Feature("FEAT_LSE128", "9.4", "128-bit atomics (LDCLRP, LDSETP, SWPP)"),
    Feature("FEAT_GCS", "9.4", "Guarded control stack"),
    Feature("FEAT_PAuth_LR", "9.5", "Pointer authentication using the link register (PACIASPPC and others)"),
]
FEATURES: dict[str, Feature] = {f.id: f for f in _FEATURE_LIST}

_LSE = re.compile(
    r"^(?:cas|casp|swp|ld(?:add|clr|eor|set|smax|smin|umax|umin)|st(?:add|clr|eor|set|smax|smin|umax|umin))"
    r"(?:a|al|l)?[bh]?$"
)
_LSE128 = re.compile(r"^(?:ldclrp|ldsetp|swpp)(?:a|al|l)?$")
_EXCLUSIVE = re.compile(r"^(?:ldx|ldax|stx|stlx)[rp][bh]?$")
_ACQREL = re.compile(r"^(?:ldar|stlr)[bh]?$")
_LOR = re.compile(r"^(?:ldlar|stllr)[bh]?$")
_CRC32 = re.compile(r"^crc32c?[bhwx]$")
_LRCPC = re.compile(r"^ldapr[bh]?$")
_LRCPC2 = re.compile(r"^(?:ldapur(?:[bh]|s[bhw])?|stlur[bh]?)$")
_MOPS = re.compile(r"^(?:cpyf?[pme][a-z]*|setg?[pme](?:t|n|tn)?)$")
_PAUTH_LR = re.compile(r"^(?:pac|aut|ret|pacn)\w*(?:ppc|ppcr|171615)$")
_SVE_COUNT = re.compile(r"^(?:sq|uq)?(?:cnt|inc|dec)[bhwd]$")
_SVE_TOKEN = re.compile(r"(?:^|[\s,{\[])(?:z|p|pn)(?=[.\s,}\]/]|$)")
_SME_TOKEN = re.compile(r"(?:^|[\s,{\[])za\w*")
_H_SCALAR = re.compile(r"(?:^|[\s,{\[])h(?=[\s,}\]]|$)")
_H_VECTOR = re.compile(r"v\.(?:4|8)?h\b")

_PAUTH_HINT = frozenset(
    {
        "paciasp",
        "pacibsp",
        "autiasp",
        "autibsp",
        "paciaz",
        "pacibz",
        "autiaz",
        "autibz",
        "pacia1716",
        "pacib1716",
        "autia1716",
        "autib1716",
        "xpaclri",
    }
)
_PAUTH_OTHER = frozenset(
    {
        "pacia",
        "pacib",
        "pacda",
        "pacdb",
        "autia",
        "autib",
        "autda",
        "autdb",
        "paciza",
        "pacizb",
        "pacdza",
        "pacdzb",
        "autiza",
        "autizb",
        "autdza",
        "autdzb",
        "xpaci",
        "xpacd",
        "pacga",
        "braa",
        "brab",
        "braaz",
        "brabz",
        "blraa",
        "blrab",
        "blraaz",
        "blrabz",
        "retaa",
        "retab",
        "eretaa",
        "eretab",
        "ldraa",
        "ldrab",
    }
)
_FP16_EXCLUDED = frozenset({"fcvt", "fcvtl", "fcvtl2", "fcvtn", "fcvtn2", "fmlal", "fmlal2", "fmlsl", "fmlsl2"})
_SIMPLE: dict[str, str] = {}
for _fid, _names in {
    "FEAT_AES": ("aese", "aesd", "aesmc", "aesimc"),
    "FEAT_SHA1": ("sha1c", "sha1h", "sha1m", "sha1p", "sha1su0", "sha1su1"),
    "FEAT_SHA256": ("sha256h", "sha256h2", "sha256su0", "sha256su1"),
    "FEAT_RDM": ("sqrdmlah", "sqrdmlsh"),
    "FEAT_FHM": ("fmlal", "fmlal2", "fmlsl", "fmlsl2"),
    "FEAT_SHA3": ("eor3", "bcax", "rax1", "xar"),
    "FEAT_SHA512": ("sha512h", "sha512h2", "sha512su0", "sha512su1"),
    "FEAT_SM3": ("sm3ss1", "sm3tt1a", "sm3tt1b", "sm3tt2a", "sm3tt2b", "sm3partw1", "sm3partw2"),
    "FEAT_SM4": ("sm4e", "sm4ekey"),
    "FEAT_DotProd": ("sdot", "udot"),
    "FEAT_JSCVT": ("fjcvtzs",),
    "FEAT_FCMA": ("fcmla", "fcadd"),
    "FEAT_FlagM": ("cfinv", "rmif", "setf8", "setf16"),
    "FEAT_FlagM2": ("axflag", "xaflag"),
    "FEAT_FRINTTS": ("frint32x", "frint32z", "frint64x", "frint64z"),
    "FEAT_SB": ("sb",),
    "FEAT_BTI": ("bti",),
    "FEAT_MTE": (
        "irg",
        "gmi",
        "subp",
        "subps",
        "addg",
        "subg",
        "stg",
        "stzg",
        "st2g",
        "stz2g",
        "ldg",
        "stgp",
        "ldgm",
        "stgm",
        "stzgm",
        "cmpp",
    ),
    "FEAT_SPECRES": ("cfp", "dvp", "cpp", "cosp"),
    "FEAT_BF16": ("bfdot", "bfmmla", "bfmlalb", "bfmlalt", "bfcvt", "bfcvtn", "bfcvtn2"),
    "FEAT_I8MM": ("smmla", "ummla", "usmmla", "usdot", "sudot"),
    "FEAT_LS64": ("ld64b", "st64b", "st64bv", "st64bv0"),
    "FEAT_WFxT": ("wfet", "wfit"),
    "FEAT_LRCPC3": ("ldiapp", "stilp", "ldap1", "stl1"),
    "FEAT_CLRBHB": ("clrbhb",),
    "FEAT_GCS": (
        "gcspushm",
        "gcspopm",
        "gcsss1",
        "gcsss2",
        "gcsstr",
        "gcssttr",
        "gcspushx",
        "gcspopx",
        "gcspopcx",
        "chkfeat",
    ),
    "A64_BARRIER": ("dmb", "dsb", "isb"),
    "A64_EXCEPTION": ("svc", "hvc", "smc", "brk", "hlt", "udf", "eret", "dcps1", "dcps2", "dcps3", "drps"),
    "A64_HINT": ("nop", "yield", "wfe", "wfi", "sev", "sevl", "csdb", "esb", "psb", "tsb", "dgh", "hint"),
}.items():
    for _name in _names:
        _SIMPLE[_name] = _fid
_CSSC = frozenset({"abs", "cnt", "ctz", "smax", "smin", "umax", "umin"})
_MSR_FEATURES = {"dit": "FEAT_DIT", "ssbs": "FEAT_SSBS", "tco": "FEAT_MTE"}


@cache
def classify(mnemonic: str, shape: str) -> tuple[str, ...]:
    """Return the feature IDs an instruction belongs to."""
    m, s = mnemonic, shape
    if m == ".inst":
        return ("A64_UNDEFINED",)
    if m == "udf" and s == "#0":
        return ("A64_ZERO",)
    if m in ("smstart", "smstop") or _SME_TOKEN.search(s):
        return ("FEAT_SME",)
    if _SVE_TOKEN.search(s) or m in ("rdvl", "addvl", "addpl") or _SVE_COUNT.match(m):
        return ("FEAT_SVE",)

    out: list[str] = []
    simple = _SIMPLE.get(m)
    if simple and not (m in ("cfp", "dvp", "cpp", "cosp") and "rctx" not in s):
        out.append(simple)
    if m in _PAUTH_HINT:
        out.append("FEAT_PAuth_HINT")
    elif m in _PAUTH_OTHER:
        out.append("FEAT_PAuth")
    elif _PAUTH_LR.match(m):
        out.append("FEAT_PAuth_LR")
    if _LSE.match(m):
        out.append("FEAT_LSE")
    elif _LSE128.match(m):
        out.append("FEAT_LSE128")
    elif _EXCLUSIVE.match(m) or m == "clrex":
        out.append("A64_EXCLUSIVE")
    elif _ACQREL.match(m):
        out.append("A64_ACQREL")
    elif _LOR.match(m):
        out.append("FEAT_LOR")
    elif _LRCPC.match(m):
        out.append("FEAT_LRCPC")
    elif _LRCPC2.match(m):
        out.append("FEAT_LRCPC2")
    if _CRC32.match(m):
        out.append("FEAT_CRC32")
    if m in ("pmull", "pmull2") and "v.1q" in s:
        out.append("FEAT_PMULL")
    if _MOPS.match(m):
        out.append("FEAT_MOPS")
    if m.startswith("bc."):
        out.append("FEAT_HBC")
    if m.startswith("rcw"):
        out.append("FEAT_THE")
    if m in _CSSC and s[:1] in ("x", "w"):
        out.append("FEAT_CSSC")
    if (
        (m.startswith("f") or m in ("scvtf", "ucvtf"))
        and m not in _FP16_EXCLUDED
        and (_H_SCALAR.search(s) or _H_VECTOR.search(s))
    ):
        out.append("FEAT_FP16")
    if m in ("dc", "ic"):
        if s.startswith("cvadp"):
            out.append("FEAT_DPB2")
        elif s.startswith("cvap"):
            out.append("FEAT_DPB")
        else:
            out.append("A64_CACHE")
    if m in ("mrs", "msr"):
        out.append("A64_SYSREG")
        if m == "mrs" and "rndr" in s:
            out.append("FEAT_RNG")
        elif m == "msr":
            pstate = _MSR_FEATURES.get(s.split(",", 1)[0])
            if pstate:
                out.append(pstate)
    if "v." in s:
        out.append("A64_SIMD")
    return tuple(dict.fromkeys(out))


# ---------------------------------------------------------------------------
# Scanning


@dataclass
class CodeScan:
    size: int = 0
    zero_runs: int = 0
    shapes: Counter[tuple[str, str]] = field(default_factory=Counter)
    svc: Counter[int] = field(default_factory=Counter)

    def merge(self, other: CodeScan) -> None:
        self.size += other.size
        self.zero_runs += other.zero_runs
        self.shapes.update(other.shapes)
        self.svc.update(other.svc)

    @property
    def words(self) -> int:
        return self.shapes.total()

    @property
    def skipped_words(self) -> int:
        """Zero words that objdump left out of its output ("..." lines)."""
        return self.size // 4 - self.words

    @property
    def undefined(self) -> int:
        return sum(c for (m, _), c in self.shapes.items() if m == ".inst")

    def mnemonics(self) -> Counter[str]:
        out: Counter[str] = Counter()
        for (m, _), c in self.shapes.items():
            out[m] += c
        return out

    def features(self) -> dict[str, tuple[int, Counter[str]]]:
        """Feature ID -> (instruction count, count per mnemonic)."""
        out: dict[str, tuple[int, Counter[str]]] = {}
        for (m, s), c in self.shapes.items():
            for fid in classify(m, s):
                total, per = out.get(fid, (0, Counter()))
                per[m] += c
                out[fid] = (total + c, per)
        return out

    def sysregs(self) -> dict[str, Counter[str]]:
        """'mrs' and 'msr' -> count per system register name."""
        out: dict[str, Counter[str]] = {"mrs": Counter(), "msr": Counter()}
        for (m, s), c in self.shapes.items():
            if m == "mrs":
                out["mrs"][s.split(", ", 1)[-1]] += c
            elif m == "msr":
                out["msr"][s.split(",", 1)[0]] += c
        return out

    def cache_ops(self) -> Counter[str]:
        out: Counter[str] = Counter()
        for (m, s), c in self.shapes.items():
            if m in ("dc", "ic"):
                out[f"{m} {s.split(',', 1)[0]}"] += c
        return out


def _parse_imm(text: str) -> int | None:
    try:
        return int(text.strip().lstrip("#"), 0)
    except ValueError:
        return None


def scan_objdump_lines(lines, scan: CodeScan) -> None:
    """Add objdump output lines (--no-addresses --no-show-raw-insn) to `scan`."""
    shapes = scan.shapes
    cache: dict[tuple[str, str], tuple[str, str]] = {}
    for line in lines:
        if not line.startswith("\t"):
            continue
        mnem, _, ops = line[1:].rstrip().partition("\t")
        if mnem == "...":
            scan.zero_runs += 1
            continue
        if mnem == ".inst":
            shapes[(".inst", "")] += 1
            continue
        if "//" in ops:
            ops = ops.split("//", 1)[0].rstrip()
        if mnem == "udf" and ops in ("#0", "#0x0"):
            shapes[("udf", "#0")] += 1  # a zero word; kept apart from real UDF instructions
            continue
        if mnem == "svc":
            imm = _parse_imm(ops)
            if imm is not None:
                scan.svc[imm] += 1
        key = (mnem, ops)
        hit = cache.get(key)
        if hit is None:
            hit = (mnem, operand_shape(mnem, ops))
            if len(cache) < 400_000:
                cache[key] = hit
        shapes[hit] += 1


def _scan_file(objdump: str, path: str) -> CodeScan:
    scan = CodeScan(size=os.path.getsize(path))
    cmd = [objdump, "-D", "-b", "binary", "-m", "aarch64", "--no-show-raw-insn", "--no-addresses", path]
    with tempfile.TemporaryFile() as err:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=err, text=True, bufsize=1 << 20)
        assert proc.stdout is not None
        with proc.stdout:
            scan_objdump_lines(proc.stdout, scan)
        if proc.wait() != 0:
            err.seek(0)
            raise ScanError(f"objdump failed on {path}: {err.read().decode(errors='replace').strip()}")
    return scan


def scan_code(
    code: bytes,
    *,
    jobs: int | None = None,
    workdir: str | Path | None = None,
    objdump: str | None = None,
) -> CodeScan:
    """Disassemble `code` and count instruction shapes and SVC numbers.

    The code is split into chunks that run through objdump in parallel. The
    chunks are written to a temporary folder inside `workdir` (the system
    temporary folder by default) and deleted afterwards.
    """
    if len(code) % 4:
        code = code[: len(code) - len(code) % 4]
    objdump = objdump or find_objdump()
    jobs = max(1, jobs or min(os.cpu_count() or 1, 16))
    result = CodeScan()
    if not code:
        return result
    with tempfile.TemporaryDirectory(prefix="siicl-scan-", dir=workdir) as tmp:
        paths = []
        for i in range(0, len(code), CHUNK_SIZE):
            p = Path(tmp, f"chunk{i // CHUNK_SIZE:05d}")
            p.write_bytes(code[i : i + CHUNK_SIZE])
            paths.append(str(p))
        if jobs == 1 or len(paths) == 1:
            parts = [_scan_file(objdump, p) for p in paths]
        else:
            with ProcessPoolExecutor(max_workers=min(jobs, len(paths))) as pool:
                parts = list(pool.map(_scan_file, repeat(objdump), paths))
    for part in parts:
        result.merge(part)
    return result
