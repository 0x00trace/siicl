"""A small builder for SPIR-V modules, the binary shader format that Vulkan consumes.

Only the parts that siicl's translator needs are here. The numbers come from
the SPIR-V specification; tests/test_spirv.py checks them against the grammar
files of the spirv-headers package when it is installed.
"""

from __future__ import annotations

import struct

MAGIC = 0x07230203
VERSION_1_5 = 0x00010500
GENERATOR = 0  # no registered generator number

OPS = {
    "OpName": 5,
    "OpExtension": 10,
    "OpExtInstImport": 11,
    "OpExtInst": 12,
    "OpMemoryModel": 14,
    "OpEntryPoint": 15,
    "OpExecutionMode": 16,
    "OpCapability": 17,
    "OpTypeVoid": 19,
    "OpTypeBool": 20,
    "OpTypeInt": 21,
    "OpTypeFloat": 22,
    "OpTypeVector": 23,
    "OpTypeImage": 25,
    "OpTypeArray": 28,
    "OpTypeRuntimeArray": 29,
    "OpTypeStruct": 30,
    "OpTypePointer": 32,
    "OpTypeFunction": 33,
    "OpConstantTrue": 41,
    "OpConstantFalse": 42,
    "OpConstant": 43,
    "OpConstantComposite": 44,
    "OpFunction": 54,
    "OpFunctionEnd": 56,
    "OpVariable": 59,
    "OpImageTexelPointer": 60,
    "OpLoad": 61,
    "OpStore": 62,
    "OpAccessChain": 65,
    "OpArrayLength": 68,
    "OpDecorate": 71,
    "OpMemberDecorate": 72,
    "OpCompositeConstruct": 80,
    "OpCompositeExtract": 81,
    "OpImageRead": 98,
    "OpImageWrite": 99,
    "OpImageQuerySize": 104,
    "OpConvertFToU": 109,
    "OpConvertFToS": 110,
    "OpConvertSToF": 111,
    "OpConvertUToF": 112,
    "OpUConvert": 113,
    "OpSConvert": 114,
    "OpBitcast": 124,
    "OpSNegate": 126,
    "OpFNegate": 127,
    "OpIAdd": 128,
    "OpFAdd": 129,
    "OpISub": 130,
    "OpFSub": 131,
    "OpIMul": 132,
    "OpFMul": 133,
    "OpFDiv": 136,
    "OpUMulExtended": 151,
    "OpSMulExtended": 152,
    "OpIsNan": 156,
    "OpIsInf": 157,
    "OpLogicalEqual": 164,
    "OpLogicalNotEqual": 165,
    "OpLogicalOr": 166,
    "OpLogicalAnd": 167,
    "OpLogicalNot": 168,
    "OpSelect": 169,
    "OpIEqual": 170,
    "OpINotEqual": 171,
    "OpUGreaterThan": 172,
    "OpSGreaterThan": 173,
    "OpUGreaterThanEqual": 174,
    "OpSGreaterThanEqual": 175,
    "OpULessThan": 176,
    "OpSLessThan": 177,
    "OpULessThanEqual": 178,
    "OpSLessThanEqual": 179,
    "OpFOrdEqual": 180,
    "OpFUnordEqual": 181,
    "OpFOrdNotEqual": 182,
    "OpFUnordNotEqual": 183,
    "OpFOrdLessThan": 184,
    "OpFUnordLessThan": 185,
    "OpFOrdGreaterThan": 186,
    "OpFUnordGreaterThan": 187,
    "OpFOrdLessThanEqual": 188,
    "OpFUnordLessThanEqual": 189,
    "OpFOrdGreaterThanEqual": 190,
    "OpFUnordGreaterThanEqual": 191,
    "OpShiftRightLogical": 194,
    "OpShiftRightArithmetic": 195,
    "OpShiftLeftLogical": 196,
    "OpBitwiseOr": 197,
    "OpBitwiseXor": 198,
    "OpBitwiseAnd": 199,
    "OpNot": 200,
    "OpBitFieldSExtract": 202,
    "OpBitFieldUExtract": 203,
    "OpBitReverse": 204,
    "OpBitCount": 205,
    "OpControlBarrier": 224,
    "OpAtomicIAdd": 234,
    "OpAtomicSMin": 236,
    "OpAtomicUMin": 237,
    "OpAtomicSMax": 238,
    "OpAtomicUMax": 239,
    "OpAtomicAnd": 240,
    "OpAtomicOr": 241,
    "OpAtomicXor": 242,
    "OpLoopMerge": 246,
    "OpSelectionMerge": 247,
    "OpLabel": 248,
    "OpBranch": 249,
    "OpBranchConditional": 250,
    "OpSwitch": 251,
    "OpReturn": 253,
    "OpUnreachable": 255,
}

# Enumerants, by operand kind.
CAPABILITY = {"Shader": 1, "Int64": 11, "ImageQuery": 50, "Image1D": 44, "ImageBuffer": 47}
CAPABILITY |= {"GroupNonUniform": 61, "GroupNonUniformBallot": 64, "SignedZeroInfNanPreserve": 4466}
STORAGE = {"UniformConstant": 0, "Input": 1, "Workgroup": 4, "Function": 7, "Image": 11, "StorageBuffer": 12}
DECORATION = {"Block": 2, "ArrayStride": 6, "BuiltIn": 11, "NonWritable": 24, "Binding": 33, "DescriptorSet": 34}
DECORATION |= {"Offset": 35, "NoContraction": 42}
BUILTIN = {"NumWorkgroups": 24, "WorkgroupId": 26, "LocalInvocationId": 27, "SubgroupLocalInvocationId": 41}
EXECUTION_MODE = {"LocalSize": 17, "SignedZeroInfNanPreserve": 4461}
DIM = {"1D": 0, "2D": 1, "3D": 2, "Buffer": 5}
IMAGE_FORMAT = {"Rgba32ui": 30, "R32ui": 33}
SCOPE = {"Device": 1, "Workgroup": 2, "Subgroup": 3}
SEMANTICS = {"Relaxed": 0, "AcquireRelease": 0x8, "WorkgroupMemory": 0x100, "ImageMemory": 0x800}
GLSL = {"RoundEven": 2, "Trunc": 3, "Floor": 8, "Ceil": 9, "Sin": 13, "Cos": 14, "Exp2": 29, "Log2": 30}
GLSL |= {"Sqrt": 31, "InverseSqrt": 32, "UMin": 38, "SMin": 39, "UMax": 41, "SMax": 42, "SAbs": 5, "Fma": 50}
GLSL |= {"FindSMsb": 74, "FindUMsb": 75}
EXECUTION_MODEL_GLCOMPUTE = 5
ADDRESSING_LOGICAL = 0
MEMORY_MODEL_GLSL450 = 1


def string_words(s: str) -> list[int]:
    data = s.encode() + b"\0"
    data += b"\0" * (-len(data) % 4)
    return list(struct.unpack(f"<{len(data) // 4}I", data))


def _encode(opcode: int, operands: list[int]) -> list[int]:
    return [((len(operands) + 1) << 16) | opcode, *operands]


class Module:
    """Collects the sections of a module and numbers its ids. Types and constants are shared."""

    def __init__(self) -> None:
        self.bound = 1
        self.capabilities: list[int] = []
        self.imports: list[int] = []
        self.entry: list[int] = []
        self.modes: list[int] = []
        self.names: list[int] = []
        self.annotations: list[int] = []
        self.globals: list[int] = []
        self.code: list[int] = []
        self._types: dict[tuple, int] = {}
        self._consts: dict[tuple, int] = {}
        self._caps: set[str] = set()
        self._wide: set[int] = set()  # 64-bit scalar types
        self.glsl = 0

    def id(self) -> int:
        self.bound += 1
        return self.bound - 1

    # -- sections

    def capability(self, name: str) -> None:
        if name not in self._caps:
            self._caps.add(name)
            self.capabilities += _encode(OPS["OpCapability"], [CAPABILITY[name]])

    def import_glsl(self) -> int:
        if not self.glsl:
            self.glsl = self.id()
            self.imports += _encode(OPS["OpExtInstImport"], [self.glsl, *string_words("GLSL.std.450")])
        return self.glsl

    def name(self, target: int, name: str) -> None:
        self.names += _encode(OPS["OpName"], [target, *string_words(name)])

    def decorate(self, target: int, decoration: str, *literals: int) -> None:
        self.annotations += _encode(OPS["OpDecorate"], [target, DECORATION[decoration], *literals])

    def member_decorate(self, struct_id: int, member: int, decoration: str, *literals: int) -> None:
        self.annotations += _encode(OPS["OpMemberDecorate"], [struct_id, member, DECORATION[decoration], *literals])

    # -- types and constants

    def _type(self, key: tuple, opname: str, *operands: int) -> int:
        if key not in self._types:
            tid = self.id()
            self.globals += _encode(OPS[opname], [tid, *operands])
            self._types[key] = tid
        return self._types[key]

    def void(self) -> int:
        return self._type(("void",), "OpTypeVoid")

    def bool(self) -> int:
        return self._type(("bool",), "OpTypeBool")

    def int(self, width: int = 32, signed: bool = False) -> int:
        tid = self._type(("int", width, signed), "OpTypeInt", width, int(signed))
        if width == 64:
            self._wide.add(tid)
        return tid

    def uint(self) -> int:
        return self.int(32)

    def float(self, width: int = 32) -> int:
        tid = self._type(("float", width), "OpTypeFloat", width)
        if width == 64:
            self._wide.add(tid)
        return tid

    def vector(self, component: int, count: int) -> int:
        return self._type(("vec", component, count), "OpTypeVector", component, count)

    def array(self, element: int, length: int) -> int:
        return self._type(("array", element, length), "OpTypeArray", element, self.const(self.uint(), length))

    def runtime_array(self, element: int, stride: int) -> int:
        key = ("rtarray", element, stride)
        new = key not in self._types
        tid = self._type(key, "OpTypeRuntimeArray", element)
        if new:
            self.decorate(tid, "ArrayStride", stride)
        return tid

    def struct(self, *members: int, key: str = "") -> int:
        """A new struct type each time unless key names one built before."""
        k = ("struct", key, members) if key else ("struct", self.bound)
        return self._type(k, "OpTypeStruct", *members)

    def pointer(self, storage: str, pointee: int) -> int:
        return self._type(("ptr", storage, pointee), "OpTypePointer", STORAGE[storage], pointee)

    def function_type(self, ret: int, *params: int) -> int:
        return self._type(("fn", ret, params), "OpTypeFunction", ret, *params)

    def image(self, sampled: int, dim: str, *, arrayed: bool, fmt: str) -> int:
        key = ("image", sampled, dim, arrayed, fmt)
        return self._type(key, "OpTypeImage", sampled, DIM[dim], 0, int(arrayed), 0, 2, IMAGE_FORMAT[fmt])

    def const(self, type_id: int, value: int) -> int:
        """A scalar constant; value is the bit pattern (64-bit types take two words)."""
        key = ("const", type_id, value)
        if key not in self._consts:
            cid = self.id()
            words = [value & 0xFFFFFFFF] + ([(value >> 32) & 0xFFFFFFFF] if type_id in self._wide else [])
            self.globals += _encode(OPS["OpConstant"], [type_id, cid, *words])
            self._consts[key] = cid
        return self._consts[key]

    def u32(self, value: int) -> int:
        return self.const(self.uint(), value & 0xFFFFFFFF)

    def u64(self, value: int) -> int:
        return self.const(self.int(64), value & 0xFFFFFFFFFFFFFFFF)

    def f32(self, bits: int) -> int:
        return self.const(self.float(), bits & 0xFFFFFFFF)

    def boolean(self, value: bool) -> int:
        key = ("bool", value)
        if key not in self._consts:
            cid = self.id()
            self.globals += _encode(OPS["OpConstantTrue" if value else "OpConstantFalse"], [self.bool(), cid])
            self._consts[key] = cid
        return self._consts[key]

    def composite(self, type_id: int, *parts: int) -> int:
        key = ("composite", type_id, parts)
        if key not in self._consts:
            cid = self.id()
            self.globals += _encode(OPS["OpConstantComposite"], [type_id, cid, *parts])
            self._consts[key] = cid
        return self._consts[key]

    def variable(self, ptr_type: int, storage: str, *, init: int | None = None, function: bool = False) -> int:
        vid = self.id()
        ops = [ptr_type, vid, STORAGE[storage]] + ([init] if init is not None else [])
        enc = _encode(OPS["OpVariable"], ops)
        if function:
            self.code += enc
        else:
            self.globals += enc
        return vid

    # -- function code

    def op(self, opname: str, *operands: int) -> None:
        """An instruction without a result."""
        self.code += _encode(OPS[opname], list(operands))

    def val(self, opname: str, type_id: int, *operands: int) -> int:
        """An instruction with a result type; returns the result id."""
        rid = self.id()
        self.code += _encode(OPS[opname], [type_id, rid, *operands])
        return rid

    def ext(self, type_id: int, glsl_name: str, *operands: int) -> int:
        return self.val("OpExtInst", type_id, self.import_glsl(), GLSL[glsl_name], *operands)

    def label(self, lid: int | None = None) -> int:
        lid = lid or self.id()
        self.op("OpLabel", lid)
        return lid

    # -- output

    def words(self) -> list[int]:
        header = [MAGIC, VERSION_1_5, GENERATOR, self.bound, 0]
        memory_model = _encode(OPS["OpMemoryModel"], [ADDRESSING_LOGICAL, MEMORY_MODEL_GLSL450])
        return (
            header
            + self.capabilities
            + self.imports
            + memory_model
            + self.entry
            + self.modes
            + self.names
            + self.annotations
            + self.globals
            + self.code
        )

    def binary(self) -> bytes:
        w = self.words()
        return struct.pack(f"<{len(w)}I", *w)

    def entry_point(self, function: int, name: str, interface: list[int]) -> None:
        self.entry += _encode(
            OPS["OpEntryPoint"], [EXECUTION_MODEL_GLCOMPUTE, function, *string_words(name), *interface]
        )

    def execution_mode(self, function: int, mode: str, *literals: int) -> None:
        self.modes += _encode(OPS["OpExecutionMode"], [function, EXECUTION_MODE[mode], *literals])
