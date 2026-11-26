import struct

import pytest

from siicl import cli, nvshader
from tests.builders import build_vzip

CTRL = 0x000FE200


def insn(op: int, *, pred: int = 7, imm: int = 0, ctrl: int = CTRL) -> bytes:
    return struct.pack("<QQ", op | (pred << 12) | (imm << 32), ctrl << 32)


EXIT = insn(0x94D)
BRA_SELF = struct.pack("<QQ", 0xFFFFFFF000007947, 0x000FC0000383FFFF)
NOP = insn(0x918, ctrl=0x000FC000)


def sph(shader_type: int, *, pixel: bool, kills: bool = False) -> bytes:
    word = (2 if pixel else 1) | (4 << 5) | (shader_type << 10) | (1 << 14) | (int(kills) << 15)
    return struct.pack("<I", word) + bytes(nvshader.SPH_SIZE - 4)


def build_shader(body: bytes, header: bytes = b"", *, end: bool = True) -> bytes:
    code = header + body + ((EXIT + BRA_SELF) if end else b"") + NOP * 4
    control = struct.pack("<I", nvshader.CONTROL_MAGIC) + bytes(0x1C)
    return struct.pack("<IIIIQ", nvshader.MAGIC, len(code), len(control), 0, 0xABCD) + code + control


FRAGMENT = build_shader(insn(0x326) + insn(0x220) + insn(0x95B), sph(5, pixel=True, kills=True))
VERTEX = build_shader(insn(0x321) + insn(0xAB9) + insn(0x322), sph(1, pixel=False))
COMPUTE = build_shader(insn(0x919) + insn(0x223))
COMPUTE_MOV_FIRST = build_shader(insn(0x202) + insn(0x919))  # 0x7202 looks like a pixel header type


def test_parses_a_fragment_program():
    sh = nvshader.parse(FRAGMENT)
    assert sh.stage == "fragment"
    assert sh.header is not None and sh.header.kills_pixels and sh.header.mrt_enable
    assert sh.hash == 0xABCD
    assert sh.control_magic_ok
    ops = [nvshader.opcode(lo) for lo, _ in sh.instructions()]
    assert ops == [0x326, 0x220, 0x95B, 0x94D, 0x947]


def test_parses_vertex_and_compute_programs():
    assert nvshader.parse(VERTEX).stage == "vertex"
    compute = nvshader.parse(COMPUTE)
    assert compute.stage == "compute" and compute.header is None
    assert nvshader.opcode(compute.instructions()[0][0]) == 0x919
    assert nvshader.parse(COMPUTE_MOV_FIRST).stage == "compute"


def test_rejects_a_program_without_an_end():
    with pytest.raises(nvshader.ShaderError, match="EXIT"):
        nvshader.parse(build_shader(insn(0x220), end=False)).instructions()


def test_rejects_wrong_sizes_and_magic():
    with pytest.raises(nvshader.ShaderError, match="sizes"):
        nvshader.parse(FRAGMENT + b"\0")
    with pytest.raises(nvshader.ShaderError, match="magic"):
        nvshader.parse(b"VIPR" + FRAGMENT[4:])


@pytest.mark.parametrize(
    "op, name",
    [
        (0x223, "FFMA"),
        (0xA23, "FFMA"),
        (0x823, "FFMA"),
        (0x802, "MOV"),
        (0x94D, "EXIT"),
        (0xAB9, "ULDC"),
        (0x809, "0x809"),
    ],
)
def test_opcode_names(op, name):
    assert nvshader.opcode_name(op) == name


def test_cli_summarises_shaders(tmp_path, capsys):
    p = tmp_path / "a.zip"
    p.write_bytes(
        build_vzip([("f1", FRAGMENT, True), ("v1", VERTEX, True), ("c1", COMPUTE, False), ("x", b"VIPR....", False)])
    )
    assert cli.main(["vzip", "shaders", str(p), "--list"]) == 0
    out = capsys.readouterr().out
    assert "Shader binaries: 3" in out
    assert "KILL appears exactly when the header allows discarding pixels: 1 of 1 fragment programs" in out
    assert "ULDC, Turing and later): 1 in 1 programs" in out
    assert "f1 " in out and "fragment" in out


def test_cli_reports_broken_programs(tmp_path, capsys):
    p = tmp_path / "a.zip"
    p.write_bytes(build_vzip([("bad", build_shader(insn(0x220), end=False), False)]))
    assert cli.main(["vzip", "shaders", str(p)]) == 1
    assert "problem: bad" in capsys.readouterr().out


# Encodings of four simple instructions, as NVIDIA's compiler writes them.
MOV_HALF = struct.pack("<QQ", 0x3F00000000027802, 0x000FE40000000F00)  # MOV R2, 0x3f000000
REAL_EXIT = struct.pack("<QQ", 0x000000000000794D, 0x000FEA0003800000)
REAL_NOP = struct.pack("<QQ", 0x0000000000007918, 0x000FC00000000000)
needs_nvdisasm = pytest.mark.skipif(nvshader.find_nvdisasm() is None, reason="nvdisasm (cuda) is not installed")


def _real_program() -> bytes:
    code = MOV_HALF + REAL_EXIT + BRA_SELF + REAL_NOP * 2
    control = struct.pack("<I", nvshader.CONTROL_MAGIC) + bytes(0x1C)
    return struct.pack("<IIIIQ", nvshader.MAGIC, len(code), len(control), 0, 1) + code + control


def test_program_bytes_stop_at_the_end():
    sh = nvshader.parse(_real_program())
    assert sh.program_bytes() == MOV_HALF + REAL_EXIT + BRA_SELF


@needs_nvdisasm
def test_disassembles_with_nvdisasm():
    dis = nvshader.disassemble(MOV_HALF + REAL_EXIT + BRA_SELF, "SM86")
    assert dis.errors == []
    assert "MOV R2, 0x3f000000" in dis.text
    assert "EXIT" in dis.text
    assert "BRA 0x20" in dis.text


@needs_nvdisasm
def test_nvdisasm_errors_are_reported():
    assert nvshader.disassemble(b"\xff" * 16, "SM86").errors


@needs_nvdisasm
def test_cli_disasm_and_nvdisasm_check(tmp_path, capsys):
    p = tmp_path / "a.zip"
    p.write_bytes(build_vzip([("abc", _real_program(), True)]))
    assert cli.main(["vzip", "disasm", str(p), "abc"]) == 0
    out = capsys.readouterr().out
    assert "compute program, 3 instructions" in out
    assert "MOV R2, 0x3f000000" in out
    assert cli.main(["vzip", "shaders", str(p), "--nvdisasm", "SM86"]) == 0
    assert "SM86: 1 of 1 programs decode without errors" in capsys.readouterr().out
