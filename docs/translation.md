# SPIR-V translation

`siicl.translate` turns a compute program into a SPIR-V 1.5 module for Vulkan 1.2. `translate(instructions, info)` takes the decoded instructions ([sass-decoder.md](sass-decoder.md)) and the launch parameters from the control section ([shader-format.md](shader-format.md)). It returns a `Translation`: the module, the descriptor bindings the module expects, and the workgroup size. `siicl.spirv` builds the module's binary encoding, with no packages outside the standard library.
## Program structure

SPIR-V requires structured control flow, and a compiled program's branches can form any pattern. So the translation splits the program into basic blocks and runs them from a loop around a switch on a block number:

- Each case runs one basic block and stores the number of the next one. A branch stores its target, or the next block when it isn't taken. `EXIT` stores a number that ends the loop.
- The loop counts the blocks each thread runs. At the step limit (a million by default) the thread stops and sets a bit in the status buffer, so a program that loops on its inputs can't hang the GPU.
- Registers, predicates and uniform registers become function variables, initialised to zero as the interpreter does. The driver's compiler keeps them in registers.

## Resources

All descriptors are in set 0:

| Binding | Resource |
|---|---|
| 0 | Status buffer: a word of flags (bit 0: a thread reached the step limit), the number of logged `MUFU` results, then the log |
| 1 and up | One storage buffer of 32-bit words for each constant buffer bank the program reads, in bank order. Bank 1 holds the program's constant data. |
| after those | One storage image for each surface handle, in handle order: R32ui for a surface with atomic operations, Rgba32ui for the others |

Shared memory becomes a workgroup array of 32-bit words, with the size from the control section. `Translation.bindings` lists the bindings with their bank or handle, so a runner can fill them.

## Rules

Each instruction computes what the interpreter computes. Some cases need care:

- Floating-point results must not be fused or reordered. Every `OpFAdd`, `OpFSub` and `OpFMul` has the `NoContraction` decoration, and the module asks for `SignedZeroInfNanPreserve`.
- `FTZ`, `SAT`, `FMNMX` and the float comparisons of `FSETP` and `FSET` work on the bits with integer instructions, so their handling of subnormal numbers, NaN and -0 doesn't depend on the driver. `FRND` and `F2I` round a subnormal input on its bits too.
- `FMUL.RZ` and `FADD.RZ` compute the result rounded to nearest and its exact error (with a fused multiply-add for `FMUL`, and the TwoSum sequence for `FADD`), and step one unit toward zero when the error points that way.
- `F2I` compares the rounded value with the limits of the integer type before converting, because SPIR-V leaves out-of-range conversions undefined.
- `IADD3`, `LEA`, `SHF` and `PRMT` compute with 64-bit integers, so their carries and funnel shifts are exact.
- `MUFU` uses the GLSL.std.450 functions. `MUFU.SIN` and `MUFU.COS` take turns, so the translation takes away whole turns (which is exact) and multiplies by 2*pi. The GPU computes approximations, so the translation can log each input and result for the check.
- Constant buffer reads check the buffer's length, and surface instructions check the image's size, so a read outside returns 0 and a store outside is dropped, as in the interpreter.
- `BAR.SYNC` becomes `OpControlBarrier` with workgroup scope. `BSSY`, `BSYNC`, `BREAK`, `WARPSYNC`, `DEPBAR` and `NOP` have no effect: they only change which threads run together.

## Not translated yet

These raise `Unsupported`: texture instructions, global memory (`LDG`, `STG`, `ATOMG`, `RED`), warp-wide instructions (`VOTE`, `SHFL`, `MATCH`, `REDUX`), packed half-precision arithmetic, `BRX`, local memory, special registers other than thread and workgroup ids, and `FFMA` or `I2FP` with a rounding mode.

## Limits

- Threads of a warp don't wait for each other at `BSYNC`, so stores that two threads make to the same word can land in a different order than on the original GPU. The programming model leaves that order open.
- A thread reaches `BAR.SYNC` in whichever loop iteration its path leads to. Vulkan requires all threads of a workgroup to execute the same barrier; programs that reach a barrier on paths of different lengths rely on the driver.
- The RTX 5080 flushes 32-bit subnormal numbers in shader arithmetic and comparisons, and Vulkan can't ask it not to (`shaderDenormPreserveFloat32` is false). The comparisons and rounding above avoid it, but on such a GPU `FADD`, `FMUL` and `FFMA` without `.FTZ` flush too. Every such instruction in the sample's compute programs has `.FTZ`.
- The translation of `FFMA` relies on the driver fusing `Fma`, which Vulkan doesn't require.
- The loop costs speed. The aim of this first translation is correct results that can be checked, not performance.
