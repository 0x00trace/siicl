"""Reader for compiled GPU shader binaries (magic 0x19866891) found in VZIP archives.

Each binary holds one NVIDIA GPU program in the 128-bit instruction encoding
that NVIDIA has used since Volta. The layout was worked out from one sample and
is described in docs/shader-format.md.
"""

from __future__ import annotations

import os
import re
import shutil
import struct
import subprocess
import tempfile
from dataclasses import dataclass, field

MAGIC = 0x19866891
MAGIC_BYTES = struct.pack("<I", MAGIC)
CONTROL_MAGIC = 0x98761234
HEADER_SIZE = 0x18
SPH_SIZE = 0x80  # NVIDIA shader program header, used by graphics stages
INSTRUCTION_SIZE = 16
SPH_VERSION = 4

STAGES = {1: "vertex", 2: "tessellation control", 3: "tessellation evaluation", 4: "geometry", 5: "fragment"}

OP_NOP, OP_EXIT, OP_BRA = 0x918, 0x94D, 0x947
OP_KILL, OP_ULDC = 0x95B, 0xAB9

# Opcode names in NVIDIA's 128-bit encoding (bits 0-11 of an instruction).
# Only opcodes this project has checked are named; others print as hex.
_FIXED_NAMES = {
    0x918: "NOP",
    0x94D: "EXIT",
    0x947: "BRA",
    0x945: "BSSY",
    0x941: "BSYNC",
    0x919: "S2R",
    0xB82: "LDC",
    0xAB9: "ULDC",
    0x308: "MUFU",
    0x326: "IPA",
    0x321: "ALD",
    0x322: "AST",
    0x95B: "KILL",
    0xB1D: "BAR",
    0x981: "LDG",
    0x986: "STG",
    0x984: "LDS",
    0x988: "STS",
}
# ALU opcodes keep their low 9 bits across operand forms (register, immediate,
# constant buffer, uniform register); bits 9-11 select the form.
_ALU_NAMES = {
    0x002: "MOV",
    0x00B: "FSETP",
    0x00C: "ISETP",
    0x010: "IADD3",
    0x012: "LOP3",
    0x019: "SHF",
    0x020: "FMUL",
    0x021: "FADD",
    0x023: "FFMA",
    0x024: "IMAD",
}


class ShaderError(ValueError):
    """The data is not a shader binary of this kind, or it is damaged."""


def opcode(word_lo: int) -> int:
    return word_lo & 0xFFF


def opcode_name(op: int) -> str:
    if op in _FIXED_NAMES:
        return _FIXED_NAMES[op]
    if 1 <= op >> 9 <= 6 and (op & 0x1FF) in _ALU_NAMES:
        return _ALU_NAMES[op & 0x1FF]
    return f"0x{op:03x}"


def is_branch_to_self(word_lo: int) -> bool:
    """BRA with a relative target of -16 bytes, the loop that ends every program."""
    return opcode(word_lo) == OP_BRA and word_lo >> 32 == 0xFFFFFFF0


@dataclass(frozen=True)
class ProgramHeader:
    """The fields of the first word of an NVIDIA shader program header (SPH)."""

    sph_type: int  # 1 vertex/tessellation/geometry, 2 pixel
    version: int
    shader_type: int
    mrt_enable: bool
    kills_pixels: bool

    @classmethod
    def from_word(cls, w: int) -> ProgramHeader:
        return cls(w & 0x1F, (w >> 5) & 0x1F, (w >> 10) & 0xF, bool(w >> 14 & 1), bool(w >> 15 & 1))

    @property
    def plausible(self) -> bool:
        return self.sph_type in (1, 2) and self.version == SPH_VERSION and self.shader_type in STAGES


@dataclass
class ShaderBinary:
    code_size: int
    control_size: int
    hash: int
    header: ProgramHeader | None
    code: bytes  # instruction words, after the program header if there is one
    control: bytes

    @property
    def stage(self) -> str:
        return STAGES.get(self.header.shader_type, "unknown") if self.header else "compute"

    @property
    def control_magic_ok(self) -> bool:
        return self.control[:4] == struct.pack("<I", CONTROL_MAGIC)

    def instructions(self) -> list[tuple[int, int]]:
        """(low, high) 64-bit halves of each instruction, up to the BRA that ends the program.

        Raises ShaderError if the program has no EXIT followed by a BRA to itself.
        """
        out = []
        seen_exit = False
        for off in range(0, len(self.code) - INSTRUCTION_SIZE + 1, INSTRUCTION_SIZE):
            lo, hi = struct.unpack_from("<QQ", self.code, off)
            out.append((lo, hi))
            if opcode(lo) == OP_EXIT:
                seen_exit = True
            elif seen_exit and is_branch_to_self(lo):
                return out
        raise ShaderError("program has no EXIT followed by a BRA to itself")

    def program_bytes(self) -> bytes:
        """The program's instructions, without the program header and the padding after it."""
        return b"".join(struct.pack("<QQ", lo, hi) for lo, hi in self.instructions())


def parse(data: bytes) -> ShaderBinary:
    if len(data) < HEADER_SIZE or data[:4] != MAGIC_BYTES:
        raise ShaderError("not a shader binary (magic is not 0x19866891)")
    _, code_size, control_size, _, digest = struct.unpack_from("<IIIIQ", data, 0)
    if HEADER_SIZE + code_size + control_size != len(data):
        raise ShaderError(f"sizes in the header ({code_size:#x} + {control_size:#x}) do not match the file")
    code = data[HEADER_SIZE : HEADER_SIZE + code_size]
    header = ProgramHeader.from_word(struct.unpack_from("<I", code, 0)[0]) if len(code) >= 4 else None
    if header is not None and header.plausible and len(code) >= SPH_SIZE:
        code = code[SPH_SIZE:]
    else:
        header = None
    return ShaderBinary(code_size, control_size, digest, header, code, data[HEADER_SIZE + code_size :])


# -- disassembly with NVIDIA's nvdisasm (CUDA toolkit) -------------------------

NVDISASM_CANDIDATES = ("nvdisasm", "/opt/cuda/bin/nvdisasm")


def find_nvdisasm() -> str | None:
    override = os.environ.get("SIICL_NVDISASM")
    if override:
        return override
    for name in NVDISASM_CANDIDATES:
        path = shutil.which(name)
        if path:
            return path
    return None


@dataclass
class Disassembly:
    text: str
    errors: list[str] = field(default_factory=list)


def disassemble(code: bytes, arch: str = "SM86", *, nvdisasm: str | None = None) -> Disassembly:
    """Disassemble raw instruction bytes with nvdisasm --binary <arch>."""
    tool = nvdisasm or find_nvdisasm()
    if tool is None:
        raise ShaderError("nvdisasm not found: install the cuda package or set SIICL_NVDISASM")
    with tempfile.NamedTemporaryFile(prefix="siicl-", suffix=".bin") as f:
        f.write(code)
        f.flush()
        proc = subprocess.run([tool, "--binary", arch, f.name], capture_output=True, text=True)
    errors = [line.strip() for line in (proc.stdout + proc.stderr).splitlines() if "error" in line.lower()]
    if proc.returncode and not errors:
        errors.append(f"nvdisasm exited with status {proc.returncode}")
    return Disassembly(proc.stdout, errors)


_LINE = re.compile(r"^\s+/\*([0-9a-f]{4,})\*/\s+(.*?)\s*;\s*$")


def disassembly_lines(text: str) -> dict[int, str]:
    """Instruction text from nvdisasm output, by instruction index (address / 16)."""
    out = {}
    for line in text.splitlines():
        m = _LINE.match(line)
        if m:
            out[int(m.group(1), 16) // INSTRUCTION_SIZE] = m.group(2)
    return out
