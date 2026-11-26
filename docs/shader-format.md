# Shader binary format

This describes the compiled GPU shader binaries found in a VZIP archive, as worked out from one sample. All integers are little-endian.

Each binary holds one GPU program in the 128-bit instruction encoding that NVIDIA has used since Volta. The program header and the instruction fields below follow NVIDIA's public shader program header layout and the encoding that disassemblers show for these GPUs.

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

It also counts `ULDC` (0xAB9), which loads a constant into a uniform register. Uniform registers were added in Turing, so programs that use `ULDC` target Turing or a later GPU. These checks can't tell Turing, Ampere and Ada apart. NVIDIA's `nvdisasm` can: run it with `--binary SM86` (or another SM version) on the instruction bytes.

## Not yet known

- How a resource (`VIPR`) file refers to its compiled binary. The binary's hash and index key don't appear in the resource.
- The contents of the control section.
