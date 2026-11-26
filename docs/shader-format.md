# Shader binary format

This describes the compiled GPU shader binaries found in a VZIP archive, as worked out from one sample. All integers are little-endian.

Each binary holds one GPU program in the 128-bit instruction encoding that NVIDIA has used since Volta; the sample's programs are Ampere (SM 8.6-class) code. The program header and the instruction fields below follow NVIDIA's public shader program header layout and the encoding that disassemblers show for these GPUs.

## Layout

| Offset | Size | Field |
|---|---|---|
| 0x00 | 4 | Magic 0x19866891 (bytes `91 68 86 19`) |
| 0x04 | 4 | Code size: program header, instructions and padding |
| 0x08 | 4 | Control section size |
| 0x0C | 4 | Zero |
| 0x10 | 8 | Hash (not referred to elsewhere in the sample's archive) |
| 0x18 | code size | Code |
| 0x18 + code size | control section size | Control section; starts with 0x98761234 |

## Code

Vertex and fragment programs start with a 128-byte shader program header (SPH). Compute programs have none, and their code starts with the first instruction.

The first word of the program header holds these fields:

| Bits | Field |
|---|---|
| 0-4 | Header type: 1 vertex, tessellation or geometry; 2 pixel |
| 5-9 | Version (4 in the sample) |
| 10-13 | Shader type: 1 vertex, 2 tessellation control, 3 tessellation evaluation, 4 geometry, 5 fragment |
| 14 | Multiple render targets enabled |
| 15 | The program may discard pixels |

A reader can tell the two cases apart by checking for a plausible header type, version and shader type. The first instruction of a compute program can look like a header type (for example a register `MOV`, opcode 0x202), but not like version 4 with a valid shader type.

## Instructions

Each instruction is 16 bytes, read as two 64-bit halves:

| Bits | Field |
|---|---|
| 0-11 | Opcode. For arithmetic opcodes, bits 0-8 name the operation and bits 9-11 the operand form (register, immediate, constant buffer, uniform register). |
| 12-14 | Predicate register (7 means always) |
| 15 | Predicate is negated |
| 105-125 | Scheduling information (stall count, yield, barriers, operand reuse) |
| 126-127 | Zero |

A program ends with `EXIT` (opcode 0x94D), then `BRA` (0x947) to its own address (relative target -16), then `NOP` (0x918) padding.

## How the sample was checked

`siicl vzip shaders` runs these checks over every binary:

- the sizes in the header add up to the file size
- the control section starts with 0x98761234
- the program ends with `EXIT` followed by a `BRA` to itself
- bits 126-127 are zero in every instruction
- `KILL` (0x95B) appears in a fragment program exactly when its header says it may discard pixels

It also counts `ULDC` (0xAB9), which loads a constant into a uniform register. Uniform registers were added in Turing, so programs that use `ULDC` target Turing or a later GPU.

## Decoding with nvdisasm

NVIDIA's `nvdisasm` (CUDA toolkit, pacman package `cuda`) decodes raw instruction bytes for a chosen SM version. `siicl vzip shaders --nvdisasm SM86` decodes every program this way, and `siicl vzip disasm` prints one program. Results for the sample with CUDA 13.5:

| SM version | Result |
|---|---|
| SM75 (Turing), SM80 (Ampere A100) | Rejected in about half the programs. The first rejected instruction is `I2FP` (integer-to-float conversion), which first appears in SM 8.6. |
| SM86, SM87, SM88 (Ampere) | Every program decodes, and the output is identical for all three. |
| SM89 (Ada) | Every program decodes. One conversion instruction prints under a different name. |
| SM90 (Hopper) | Every program decodes, but branch targets come out different and point outside the programs. |
| SM100, SM120 (Blackwell) | Most instructions are rejected. |

So the programs are SM 8.6-class Ampere code. The instruction bits don't show which of SM 8.6, 8.7 and 8.8 the compiler targeted.

## Control section of a compute program

A compute program has no program header, so its launch parameters are in the control section instead. `ShaderBinary.compute_info()` reads these fields:

| Offset | Size | Field |
|---|---|---|
| 0x38 | 4 | Offset of the program's constant data, from the start of the code |
| 0x3C | 4 | Size of the constant data (0 when there is none) |
| 0x4C | 4 | General registers per thread |
| 0x58 | 12 | Workgroup size in x, y and z |
| 0x64 | 4 | Shared memory per workgroup, in bytes |

How each field was checked against the sample's compute programs:

- Workgroup size: each program computes its global thread index as `SR_CTAID * n + SR_TID` with the same `n` that this field gives. A program reads `SR_TID.Y` or `SR_TID.Z` only when that dimension is larger than 1.
- Shared memory: the size is 0 exactly in the programs that have no shared-memory instructions (`LDS`, `STS`, `ATOMS`).
- Registers: in every program the count is 3 more than the highest register number the code uses.
- Constant data: the programs that read constant buffer 1 (`c[0x1][...]`) are exactly the ones with a nonzero size here, and their reads stay inside it. The data sits in the code after the last instruction's padding. This reading rests on 3 programs, and in one of them the data is all zeros.

The meaning of the other words is not known, and graphics programs have not been checked.

## Not yet known

- How the game names a shader binary's archive entry. A resource finds its binaries by index, and they are stored right before it (see [vzip-format.md](vzip-format.md)), but the entry names appear in no file.
- Most of the control section.
