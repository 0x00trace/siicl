# Instruction decoder

`siicl.sass` decodes NVIDIA's 128-bit GPU instruction encoding (Volta and later) as used by SM 8.6-class Ampere GPUs. `decode(lo, hi, address)` returns an `Instruction` with its mnemonic, modifiers, predicate guard and structured operands, and `Instruction.text()` prints it as `nvdisasm --binary SM86` does. A shader translator can work from the structured form; the text form exists so the decoder can be checked against NVIDIA's disassembler.

Each opcode has a fixed list of operands, so a consumer can find an operand by its position:

- Operands that nvdisasm leaves out stay in the list with `hidden` set. Examples are a `PT` carry predicate of `IADD3`, the `PT` predicate of an unconditional `BRA`, and a full texture write mask.
- Memory and surface addresses are `addr` operands: a base register (`index`), its `scale`, an optional uniform register (`extra`) and a byte offset (`value`).
- Packed-half immediates are `himm` operands holding the 16-bit value. `BRX` and `RET` targets are `brx` and `ret` operands with the register in `index`.

## Status

- The decoder knows every opcode that the sample's 2,828 shader programs use: 162 opcodes, 1,627,782 instructions. `siicl vzip check-decoder` decodes all of them and compares each line with nvdisasm; all of them match.
- `tests/test_sass.py` builds 105 synthetic instructions from named bit fields, including field values that the sample never uses, and checks them against texts that nvdisasm printed. When nvdisasm is installed, the tests also check that it still prints those texts.
- Opcodes outside that set raise `UnknownInstruction`. Enumerated fields print `?N` for values that neither the sample nor the tests contain.

## Common fields

| Bits | Field |
|---|---|
| 0-11 | Opcode. For arithmetic opcodes, bits 0-8 name the operation and bits 9-11 the operand form. |
| 12-14, 15 | Predicate guard and its negation (7 means always) |
| 16-23 | Destination register (255 is RZ) |
| 24-31 | Source A register |
| 32-63 | Source B: register (32-39), 32-bit immediate, uniform register (32-37), or constant buffer |
| 38-53, 54-58 | Constant buffer byte offset and bank |
| 64-71 | Source C register (also the first destination of texture instructions) |
| 72, 73 | Negate and absolute value of source A |
| 62, 63 | Absolute value and negate of the operand held in bits 32-63 |
| 74, 75 | Absolute value and negate of the operand held in bits 64-71 |
| 81-83, 84-86 | Destination predicates |
| 87-89, 90 | Source predicate and its negation |
| 77-79, 80 | Second source predicate and its negation (when present) |
| 91 | Set when an operand is a uniform register |
| 105-121 | Scheduling: stall count, yield, barriers, wait mask |
| 122-125 | Operand reuse flags for sources A, B and C |

## Operand forms (bits 9-11)

| Form | Source B | Source C |
|---|---|---|
| 1 | register (bits 32-39) | register (bits 64-71) |
| 2 | register (bits 64-71) | immediate |
| 3 | register (bits 64-71) | constant buffer |
| 4 | immediate | register (bits 64-71) |
| 5 | constant buffer | register (bits 64-71) |
| 6 | uniform register | register (bits 64-71) |
| 7 | register (bits 64-71) | uniform register |

Two-source operations use B or C depending on the operation: FMUL takes its second operand from slot B (forms 1, 4, 5, 6), FADD from slot C (forms 1, 2, 3, 7).

## Selected modifier fields

| Instruction | Field |
|---|---|
| FFMA, FADD, FMUL | FTZ 80, SAT 77, rounding 78-79 (RN, RM, RP, RZ); FMUL scale 84-86 (4 = none, 5-7 = M2-M8, 1-3 = D8-D2) |
| FSETP, FSET, HSETP2 | comparison 76-79 (F, LT, EQ, LE, GT, NE, GE, NUM, NAN, LTU, EQU, LEU, GTU, NEU, GEU, T); boolean 74-75 (69-70 for HSETP2) |
| ISETP | comparison 76-78; signed 73 (clear prints .U32); boolean 74-75; EX 72 |
| MUFU | function 74-77 (COS, SIN, EX2, LG2, RCP, RSQ, RCP64H, RSQ64H, SQRT, TANH) |
| F2I, I2FP | integer size 75-76 and signed 72 (F2I) or 74 (I2FP source); NTZ 77; source size 84-85 |
| IMAD, IADD3, LEA | signed 73; X (carry in) 74; LEA shift 75-79, HI 80 |
| LOP3, PLOP3 | LUT at 72-79; PLOP3 splits it: high five bits at 72-76, low three at 64-66 |
| Packed half | swizzle: A 74-75, B 60-61, C 81-82 (0 none, 2 H0_H0, 3 H1_H1); F32 output 78 |
| Texture | dimension 61-63, SCR 60, level of detail 87-89, AOFFI 76, DC 78, NODEP 90, write mask 72-75, cache 84-86 |
| Memory | size 73-75 (U8, S8, U16, S16, 32, 64, 128); E 72; address scale 78-79; offset 40-63; semantics 76-79 |
| Branches | signed byte offset from the next instruction at 32-81 |

## Printing rules that nvdisasm follows

- Float immediates print with `%.20g`, except values of 1e10 and up, which print with `%.20e`. Infinities print as `+INF ` with a trailing space.
- `.reuse` is never printed on RZ.
- IMAD prints aliases: `IMAD.MOV` when A and B are RZ, `IMAD.IADD` for a multiplier of 1, and `IMAD.SHL` for unsigned multiplies by a power of two, except 0x10000.
- Immediates of IMAD, IADD3 and ISETP print as signed values; others print unsigned.
- A packed-half operand in a constant buffer prints as `c[0x6] [0x398]`, with a space.
- Negative memory offsets print as `+-0x20`.

## Limits

- The layouts come from the sample's instructions and the synthetic tests. Fields that neither varies may decode wrongly for other values.
- A few rules rest on very few instructions: the `.H0_NH1` half selector of HFMA2, the `.H1` source of MUFU.F16, and the surface instructions (SUST, SULD, SURED, SUATOM).
- The decoder describes instructions; [siicl.interp](interpreter.md) models what they compute.
