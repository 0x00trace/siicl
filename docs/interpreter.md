# Reference interpreter

`siicl.interp` runs a compute program on the CPU. It takes the instructions that [siicl.sass](sass-decoder.md) decodes and the launch parameters from the control section ([shader-format.md](shader-format.md)), and executes every workgroup of a grid. It is the reference that translations of a program, such as SPIR-V for another GPU, are checked against. It aims for exact results, not speed.

```python
from siicl import interp, nvshader

sh = nvshader.parse(data)
program = interp.Program(sh.program_bytes(), sh.compute_info())
launch = interp.Launch(grid=(4, 4, 1), cbufs={3: b"..."}, surfaces={(0, 0x12): interp.Surface(32, 32)})
interp.run(program, launch)  # changes launch.surfaces and launch.memory in place
```

## What it models

- Registers R0-R254 and predicates P0-P6 per thread; uniform registers UR0-UR62, uniform predicates and convergence barriers B0-B15 per warp of 32 threads.
- Constant buffers by bank. Bank 1 holds the program's constant data unless the launch gives one.
- Shared memory per workgroup, with the size from the control section.
- Global memory as separate regions at 64-bit addresses (`Memory.add`).
- Surfaces (storage images) as arrays of texels of 1 to 4 words, found by the two numbers of the instruction's handle operand.

An instruction, modifier or operand that has no rule raises `Unsupported` when it runs. `Program.unsupported()` lists them without running. Today that covers texture instructions, packed half-precision arithmetic, `BRX`, `F2FP`, local memory, and a few atomic operations.

## How a warp runs

Each thread has its own program counter. A warp repeatedly picks the lowest program counter among its threads that can run, and executes that instruction for all the threads that are at it. Those threads are the active threads for warp-wide instructions such as `VOTE` and `SHFL`.

- `BSSY Bn` records the active threads in barrier register Bn. `BSYNC Bn` makes a thread wait until every thread recorded in Bn has arrived at a `BSYNC Bn` or exited. `BREAK Bn` removes a thread from Bn.
- `BAR.SYNC` makes a thread wait until every thread of the workgroup that hasn't exited waits at a barrier. Threads waiting for each other in other ways end the run with a deadlock error.
- Instructions of the uniform datapath (`UIADD3`, `ULDC`, `UMOV` and the others that write uniform registers) run once for the threads executing them together, because their registers belong to the warp.
- Warps run one after another until each is waiting or finished, and workgroups run one after another.
- A run stops with `ExecutionError` after a step limit (50 million thread-instructions by default), so a program that loops on its inputs can't run forever.
- `run(..., reverse=True)` runs workgroups, warps and the threads of each instruction in the opposite order. A result that differs between the two orders depends on the order in which threads run, for example when two threads store different values to the same texel.
- `Surface.track_writes()` records the last thread that wrote each word, and in `Surface.conflicts` the words that two threads wrote with different values. Their final value depends on the order of the two stores, which the GPU programming model leaves open.
- `Launch.mufu_results` can give `MUFU` results by instruction address and input. A check against a GPU fills it with the GPU's own results, which are approximations.

## Rules

Floating point:

- Results are computed exactly and rounded once, with the rounding mode of the instruction. `FFMA` rounds once, as fused multiply-add does.
- `FTZ` flushes subnormal inputs, and subnormal results after rounding, to zero with the same sign. `SAT` clamps the result to [0, 1], and NaN becomes 0.
- A NaN result is 0x7FFFFFFF, the NaN that NVIDIA GPUs return.
- `FMNMX` returns the other operand when one is NaN, and treats -0 as smaller than +0.
- `MUFU` computes its function exactly and rounds it to 32 bits. The hardware computes approximations, so its results can differ in the last bits. `MUFU.SIN` and `MUFU.COS` take their input in turns (`sin(2 * pi * x)`), because the compiler multiplies by 1/(2*pi) first. `MUFU` flushes subnormal inputs and results.
- `F2I` rounds with its mode (none means to nearest, ties to even), then clamps to the range of the integer type. `I2FP` and `I2F` round with their mode.

Integers:

- `IADD3` adds three operands. A negated operand adds its bitwise NOT; without `.X` each one also adds 1, so the carries come out as for a subtraction. With `.X`, the two carry-in predicates are added instead.
- `IMAD` keeps the low 32 bits of a * b + c. `IMAD.HI` gives the high 32 bits of the product.
- `LEA` computes (a << shift) + b; `LEA.HI` uses the high 32 bits of the 64-bit value c:a shifted left.
- `SHF` shifts the 64-bit value c:a and keeps the low or (with `.HI`) the high 32 bits. Without `.W` the shift amount is clamped to 32; with it, only its low 5 bits count.
- `LOP3` and `PLOP3` apply their lookup table with operand a as the most significant index bit. `PRMT` picks bytes of c:a with the selector in operand b.
- `ISETP`, `FSETP` and `FSET` combine the comparison with the predicate operand; the second destination gets the negated comparison combined the same way.

Memory and surfaces:

- Shared memory accesses must be aligned and inside the workgroup's shared memory.
- Surface coordinates are consecutive registers, one per dimension. For array surfaces, the layer is the low 16 bits of the last coordinate. Coordinates outside the surface are ignored (`IGN`) and loads there return 0.
- `SUST.P` and `SULD.P` store and load the raw register bits of the components that the mask names.
- Atomic instructions act on the lanes of a warp in lane order, and warps in order. Results that depend on that order can differ on hardware.

## Rules that are guesses

These rest on the sample's code or on a single source:

- `F2I.NTZ`: a NaN input converts to 0. Every `F2I` in the sample's compute programs has `NTZ`, so an `F2I` without it is unsupported.
- `IADD3` carry predicates: the first is set when the sum reaches 2^32, the second when it reaches 2^33. They add up to the carry, which is what `IADD3.X` needs.
- The predicate destination of `LOP3` is set when the result is nonzero, or when its predicate operand is true.
- The predicate operand of `BRA`, `EXIT`, `BREAK` and `CALL`: the instruction acts only when both its guard and this predicate are true. Two code patterns in the sample (a nested comparison and an `atan2` special case) only make sense this way.
- `CALL.REL.NOINC` acts as a branch. The sample has no `RET`, and its `CALL.REL.NOINC` instructions leave loops.
- `SUST.P` with fewer components than the surface has writes zeros to the rest.
- Surface atomics address texels, not bytes: the sample computes their coordinates the same way as for formatted stores.
- `IMAD.HI` with a nonzero addend is unsupported, because it isn't clear whether the addend joins the high or the low half.

## Limits

- The rules come from public documentation and the sample. They haven't been checked against a Switch 2.
- Instructions that read the clock, and `SHFL` from a lane that isn't active, return values that can differ from hardware.
- The interpreter is slow: about a million thread-instructions per second.
