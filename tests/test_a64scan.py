import shutil
import subprocess

import pytest

from siicl import a64scan
from tests.builders import AS, OBJCOPY, assemble

needs_binutils = pytest.mark.skipif(
    not (AS and OBJCOPY and shutil.which("aarch64-linux-gnu-objdump")),
    reason="aarch64-linux-gnu-binutils is not installed",
)

POSITIVE = [
    ("FEAT_LSE", "casal x0, x1, [x2]"),
    ("FEAT_LSE", "ldaddal w0, w1, [x2]"),
    ("FEAT_LSE", "stadd w0, [x1]"),
    ("FEAT_LSE", "swpb w0, w1, [x2]"),
    ("FEAT_LSE", "caspal x0, x1, x2, x3, [x4]"),
    ("FEAT_RDM", "sqrdmlah v0.4s, v1.4s, v2.4s"),
    ("FEAT_LOR", "ldlar x0, [x1]"),
    ("FEAT_CRC32", "crc32cx w0, w1, x2"),
    ("FEAT_AES", "aese v0.16b, v1.16b"),
    ("FEAT_PMULL", "pmull v0.1q, v1.1d, v2.1d"),
    ("FEAT_SHA1", "sha1c q0, s1, v2.4s"),
    ("FEAT_SHA256", "sha256h q0, q1, v2.4s"),
    ("FEAT_FP16", "fadd h0, h1, h2"),
    ("FEAT_FP16", "fmul v0.8h, v1.8h, v2.8h"),
    ("FEAT_FP16", "fcvtzs w0, h1"),
    ("FEAT_FP16", "scvtf h0, w1"),
    ("FEAT_FP16", "fmov h0, w1"),
    ("FEAT_FP16", "fmla v0.8h, v1.8h, v2.h[3]"),
    ("FEAT_DotProd", "sdot v0.4s, v1.16b, v2.16b"),
    ("FEAT_DotProd", "udot v0.2s, v1.8b, v2.4b[1]"),
    ("FEAT_FHM", "fmlal v0.2s, v1.2h, v2.2h"),
    ("FEAT_SHA3", "eor3 v0.16b, v1.16b, v2.16b, v3.16b"),
    ("FEAT_SHA512", "sha512h q0, q1, v2.2d"),
    ("FEAT_SM3", "sm3ss1 v0.4s, v1.4s, v2.4s, v3.4s"),
    ("FEAT_SM4", "sm4e v0.4s, v1.4s"),
    ("FEAT_DPB", "dc cvap, x0"),
    ("FEAT_DPB2", "dc cvadp, x0"),
    ("FEAT_SVE", "add z0.s, z1.s, z2.s"),
    ("FEAT_SVE", "ptrue p0.s"),
    ("FEAT_SVE", "addvl sp, sp, #-2"),
    ("FEAT_SVE", "cntd x0"),
    ("FEAT_PAuth_HINT", "paciasp"),
    ("FEAT_PAuth_HINT", "autibsp"),
    ("FEAT_PAuth_HINT", "xpaclri"),
    ("FEAT_PAuth", "retaa"),
    ("FEAT_PAuth", "braa x0, x1"),
    ("FEAT_PAuth", "ldraa x0, [x1]"),
    ("FEAT_PAuth", "pacia x0, x1"),
    ("FEAT_PAuth", "blraaz x0"),
    ("FEAT_PAuth", "pacga x0, x1, x2"),
    ("FEAT_JSCVT", "fjcvtzs w0, d1"),
    ("FEAT_FCMA", "fcmla v0.4s, v1.4s, v2.4s, #90"),
    ("FEAT_LRCPC", "ldapr x0, [x1]"),
    ("FEAT_LRCPC2", "ldapur x0, [x1, #8]"),
    ("FEAT_LRCPC2", "stlur w0, [x1, #-4]"),
    ("FEAT_FlagM", "cfinv"),
    ("FEAT_FlagM", "rmif x0, #3, #2"),
    ("FEAT_DIT", "msr dit, #1"),
    ("FEAT_FlagM2", "axflag"),
    ("FEAT_FRINTTS", "frint32x d0, d1"),
    ("FEAT_SB", "sb"),
    ("FEAT_BTI", "bti c"),
    ("FEAT_MTE", "irg x0, x1"),
    ("FEAT_MTE", "stg x0, [x1]"),
    ("FEAT_RNG", "mrs x0, rndr"),
    ("FEAT_SSBS", "msr ssbs, #1"),
    ("FEAT_SPECRES", "cfp rctx, x0"),
    ("FEAT_BF16", "bfdot v0.4s, v1.8h, v2.8h"),
    ("FEAT_I8MM", "smmla v0.4s, v1.16b, v2.16b"),
    ("FEAT_I8MM", "usdot v0.4s, v1.16b, v2.16b"),
    ("FEAT_LS64", "ld64b x0, [x1]"),
    ("FEAT_WFxT", "wfet x0"),
    ("FEAT_MOPS", "cpyfp [x0]!, [x1]!, x2!"),
    ("FEAT_MOPS", "setp [x0]!, x1!, x2"),
    ("FEAT_HBC", "bc.eq ."),
    ("FEAT_CSSC", "abs x0, x1"),
    ("FEAT_CSSC", "umax w0, w1, w2"),
    ("FEAT_CSSC", "ctz x0, x1"),
    ("FEAT_LRCPC3", "ldiapp x0, x1, [x2]"),
    ("FEAT_THE", "rcwcas x0, x1, [x2]"),
    ("FEAT_CLRBHB", "clrbhb"),
    ("FEAT_SME", "smstart"),
    ("FEAT_LSE128", "ldclrp x0, x1, [x2]"),
    ("FEAT_GCS", "gcspushm x0"),
    ("A64_EXCLUSIVE", "ldaxr x0, [x1]"),
    ("A64_EXCLUSIVE", "stlxr w2, x0, [x1]"),
    ("A64_ACQREL", "ldarb w0, [x1]"),
    ("A64_BARRIER", "dmb ish"),
    ("A64_CACHE", "dc zva, x0"),
    ("A64_CACHE", "ic ivau, x0"),
    ("A64_SYSREG", "mrs x0, tpidrro_el0"),
    ("A64_EXCEPTION", "svc #0x21"),
    ("A64_EXCEPTION", "brk #0"),
    ("A64_HINT", "yield"),
    ("A64_SIMD", "add v0.4s, v1.4s, v2.4s"),
]

# Armv8.0 instructions that look like a later feature but are not.
NEGATIVE = [
    ("fcvt h0, s1", "FEAT_FP16"),
    ("fcvt d0, h1", "FEAT_FP16"),
    ("fcvtl v0.4s, v1.4h", "FEAT_FP16"),
    ("fcvtn v0.4h, v1.4s", "FEAT_FP16"),
    ("add v0.8h, v1.8h, v2.8h", "FEAT_FP16"),
    ("ldr h0, [x0]", "FEAT_FP16"),
    ("fmlal v0.2s, v1.2h, v2.2h", "FEAT_FP16"),
    ("pmull v0.8h, v1.8b, v2.8b", "FEAT_PMULL"),
    ("abs v0.4s, v1.4s", "FEAT_CSSC"),
    ("cnt v0.8b, v1.8b", "FEAT_CSSC"),
    ("smax v0.4s, v1.4s, v2.4s", "FEAT_CSSC"),
    ("ldxr x0, [x1]", "FEAT_LSE"),
    ("ldar x0, [x1]", "FEAT_LRCPC"),
    ("setf8 w0", "FEAT_MOPS"),
    ("ret", "FEAT_PAuth"),
    ("ldp x0, x1, [x2]", "FEAT_SVE"),
    ("prfm pldl1keep, [x0]", "FEAT_SVE"),
    ("mov x0, xzr", "FEAT_SVE"),
    ("csel x0, x1, x2, cc", "FEAT_SVE"),
    ("ld1 {v0.16b, v1.16b}, [x0]", "FEAT_SVE"),
]


@pytest.fixture(scope="module")
def decoded(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("asm")
    code = assemble("\n".join(src for _, src in POSITIVE + [(None, n) for n, _ in NEGATIVE]), tmp)
    path = tmp / "code.bin"
    path.write_bytes(code)
    return _decode_file(path)


def _decode_file(path):
    out = subprocess.run(
        [
            a64scan.find_objdump(),
            "-D",
            "-b",
            "binary",
            "-m",
            "aarch64",
            "--no-show-raw-insn",
            "--no-addresses",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    result = []
    for line in out.splitlines():
        if line.startswith("\t"):
            scan = a64scan.CodeScan()
            a64scan.scan_objdump_lines([line], scan)
            (key,) = scan.shapes
            result.append((line.strip(), a64scan.classify(*key)))
    return result


@needs_binutils
@pytest.mark.parametrize("index", range(len(POSITIVE)), ids=[s for _, s in POSITIVE])
def test_recognises_feature(decoded, index):
    feature, _ = POSITIVE[index]
    line, features = decoded[index]
    assert feature in features, f"{line!r} classified as {features}"


@needs_binutils
@pytest.mark.parametrize("index", range(len(NEGATIVE)), ids=[s for s, _ in NEGATIVE])
def test_does_not_misclassify(decoded, index):
    _, feature = NEGATIVE[index]
    line, features = decoded[len(POSITIVE) + index]
    assert feature not in features, f"{line!r} classified as {features}"


def test_pauth_lr_names():
    for m in ("paciasppc", "autiasppc", "retaasppc", "pacnbiasppc", "autia171615"):
        assert "FEAT_PAuth_LR" in a64scan.classify(m, "")


def test_operand_shape():
    assert a64scan.operand_shape("add", "x0, x1, #0x10") == "x, x, #i"
    assert a64scan.operand_shape("ldr", "q3, [x29, #-16]!") == "q, [x, #i]!"
    assert a64scan.operand_shape("mov", "v0.s[1], w0") == "v.s[i], w"
    assert a64scan.operand_shape("add", "z0.s, p0/m, z0.s, z1.s") == "z.s, p/m, z.s, z.s"
    assert a64scan.operand_shape("mrs", "x0, s3_3_c13_c0_3") == "x, s3_3_c13_c0_3"
    assert a64scan.operand_shape("csel", "x0, x1, xzr, cc") == "x, x, x, cc"


@needs_binutils
def test_scan_code_counts_svcs_sysregs_and_zero_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(a64scan, "CHUNK_SIZE", 64)  # force several chunks and the process pool
    src = "\n".join(["svc #0x21", "svc #0x80", "mrs x0, tpidrro_el0", "msr tpidr_el0, x1", "paciasp", "retaa"] * 10)
    src += "\n.zero 128\n.inst 0xffffffff\nsvc #0x21"
    code = assemble(src, tmp_path)
    scan = a64scan.scan_code(code, jobs=2, workdir=tmp_path)
    assert scan.size == len(code)
    assert scan.svc == {0x21: 11, 0x80: 10}
    regs = scan.sysregs()
    assert regs["mrs"]["tpidrro_el0"] == 10
    assert regs["msr"]["tpidr_el0"] == 10
    assert scan.undefined == 1
    assert scan.zero_runs >= 1
    assert scan.words + scan.skipped_words == len(code) // 4

    feats = scan.features()
    assert feats["FEAT_PAuth_HINT"][0] == 10
    assert feats["FEAT_PAuth"][0] == 10
    assert list(tmp_path.glob("siicl-scan-*")) == []  # temporary chunks are removed


@needs_binutils
def test_single_zero_words_are_padding_not_udf(tmp_path):
    code = assemble("nop\n.inst 0\nnop\nudf #1", tmp_path)
    feats = a64scan.scan_code(code, jobs=1, workdir=tmp_path).features()
    assert feats["A64_ZERO"][0] == 1
    assert feats["A64_EXCEPTION"][1] == {"udf": 1}
