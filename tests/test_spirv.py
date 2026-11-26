"""Tests for the SPIR-V module builder."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from siicl import spirv

GRAMMAR = Path("/usr/include/spirv/unified1")
SPIRV_VAL = shutil.which("spirv-val")


def validate(binary: bytes, tmp_path: Path) -> subprocess.CompletedProcess:
    path = tmp_path / "module.spv"
    path.write_bytes(binary)
    return subprocess.run([SPIRV_VAL, "--target-env", "vulkan1.2", str(path)], capture_output=True, text=True)


@pytest.mark.skipif(not (GRAMMAR / "spirv.core.grammar.json").exists(), reason="spirv-headers is not installed")
def test_numbers_match_the_grammar():
    core = json.loads((GRAMMAR / "spirv.core.grammar.json").read_text())
    ops = {i["opname"]: i["opcode"] for i in core["instructions"]}
    assert {name: ops[name] for name in spirv.OPS} == spirv.OPS
    kinds = {
        k["kind"]: {e["enumerant"]: int(str(e["value"]), 0) for e in k.get("enumerants", [])}
        for k in core["operand_kinds"]
    }
    for kind, table in [
        ("Capability", spirv.CAPABILITY),
        ("StorageClass", spirv.STORAGE),
        ("Decoration", spirv.DECORATION),
        ("BuiltIn", spirv.BUILTIN),
        ("ExecutionMode", spirv.EXECUTION_MODE),
        ("Dim", spirv.DIM),
        ("ImageFormat", spirv.IMAGE_FORMAT),
        ("Scope", spirv.SCOPE),
        ("MemorySemantics", spirv.SEMANTICS),
    ]:
        assert {name: kinds[kind][name] for name in table} == table, kind
    glsl = json.loads((GRAMMAR / "extinst.glsl.std.450.grammar.json").read_text())
    gops = {i["opname"]: i["opcode"] for i in glsl["instructions"]}
    assert {name: gops[name] for name in spirv.GLSL} == spirv.GLSL


def test_shares_types_and_constants():
    m = spirv.Module()
    assert m.uint() == m.int(32)
    assert m.u32(5) == m.u32(5) != m.u32(6)
    assert m.vector(m.uint(), 3) == m.vector(m.uint(), 3)
    assert m.struct(m.uint()) != m.struct(m.uint())
    assert m.struct(m.uint(), key="a") == m.struct(m.uint(), key="a")


def test_strings_are_padded():
    assert spirv.string_words("main") == [0x6E69616D, 0]
    assert spirv.string_words("abc") == [0x00636261]


@pytest.mark.skipif(SPIRV_VAL is None, reason="spirv-val (spirv-tools) is not installed")
def test_builds_a_valid_compute_shader(tmp_path):
    m = spirv.Module()
    m.capability("Shader")
    u = m.uint()
    rt = m.runtime_array(u, 4)
    block = m.struct(rt)
    m.decorate(block, "Block")
    m.member_decorate(block, 0, "Offset", 0)
    buf = m.variable(m.pointer("StorageBuffer", block), "StorageBuffer")
    m.decorate(buf, "DescriptorSet", 0)
    m.decorate(buf, "Binding", 0)
    lid = m.variable(m.pointer("Input", m.vector(u, 3)), "Input")
    m.decorate(lid, "BuiltIn", spirv.BUILTIN["LocalInvocationId"])
    fn = m.id()
    m.code += [((5) << 16) | spirv.OPS["OpFunction"], m.void(), fn, 0, m.function_type(m.void())]
    m.label()
    x = m.val("OpCompositeExtract", u, m.val("OpLoad", m.vector(u, 3), lid), 0)
    ptr = m.val("OpAccessChain", m.pointer("StorageBuffer", u), buf, m.u32(0), x)
    m.op("OpStore", ptr, m.val("OpIMul", u, x, m.u32(3)))
    m.op("OpReturn")
    m.op("OpFunctionEnd")
    m.entry_point(fn, "main", [buf, lid])
    m.execution_mode(fn, "LocalSize", 64, 1, 1)
    result = validate(m.binary(), tmp_path)
    assert result.returncode == 0, result.stderr + result.stdout
