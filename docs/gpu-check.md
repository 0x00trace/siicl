# Checking translations on a GPU

`siicl vzip check-gpu` runs each compute program of a dump twice: as SPIR-V on a Vulkan GPU ([translation.md](translation.md)), and in the reference interpreter ([interpreter.md](interpreter.md)). It then compares every word that the program wrote to its surfaces. The two implementations of each instruction are written separately, so a difference points to a mistake in one of them, or to a GPU behaviour that neither models.

```sh
uv run siicl vzip check-gpu ~/Games/game_dumps/switch_2/game.nup
uv run siicl vzip check-gpu game.nup --entry <entry name> --seeds 16
```

It needs a Vulkan driver and the loader (pacman package `vulkan-icd-loader`). It has been run on an RTX 5080.

## How one check works

For each program and seed:

1. Inputs. Constant buffer words get values by how the program uses them: a float in a range that depends on the seed, a small integer, or, for a word that an integer comparison uses as a bound, a larger integer. The type of a word that an instruction loads into a register (`LDC`, `MOV`) comes from the first instruction that reads that register. Bank 1 holds the program's own constant data. Surfaces are filled with random words, and the grid covers the largest surface.
2. GPU run. The translation runs on the GPU and records every `MUFU` input and result.
3. Interpreter runs. The interpreter runs twice with the GPU's `MUFU` results: once in its normal order, and once with workgroups, warps and lanes reversed and the highest program counter first. `MUFU` computes approximations, so without the GPU's own results their last bits would differ and hide everything else.
4. Comparison. A word is order-dependent when the two interpreter runs disagree on it, or when two threads stored different values to it. Those words are counted but not compared, because the programming model leaves their final value open. Every other word must be equal, except that two NaNs count as equal.

The report gives, for each program, the seeds that matched, the words compared, the order-dependent words, the mismatches, and the share of the program's instructions that ran in any seed. It also says how far the GPU's `MUFU` results are from exact ones: in ulps for `RCP`, `RSQ`, `SQRT` and `EX2`, and as an absolute error for `SIN`, `COS` and `LG2`, whose accuracy NVIDIA gives that way.

## Results on the sample

With 8 seeds, on an RTX 5080:

- 13 of the 48 compute programs have rules in both the interpreter and the translator. All 13 match on every seed: no mismatches.
- The share of instructions that ran is 95-100% for 5 of the 13 programs, 43-74% for 3, and 6-27% for the other 5. Those 5 test whether a point lies inside a box before most of their work, and random inputs rarely pass that test.
- The GPU's `MUFU` results are within 2 ulps of exact for `RCP`, `SQRT` and `EX2`, and within 1e-5 for `LG2`.
- The other 35 programs are not checked yet. In 30 of them the first instruction without an interpreter rule is a texture instruction, in 2 it is `BRX`, and in 1 packed half-precision arithmetic. The last 2 use global memory, which the translator doesn't handle yet.

Two bugs were found this way. The interpreter ran uniform instructions once per thread instead of once per warp, and the translation's float comparisons depended on the GPU flushing subnormal numbers.

## What a match does not show

- Rules that the interpreter and the translation share can be wrong for both. The check compares two implementations of the same reading of each instruction, not the Switch 2's GPU.
- Code that no seed reaches is not checked. The coverage figure shows how much ran.
- `MUFU` results are replayed, so the check of the rest is exact, and the `MUFU` deviations are reported separately.
- The inputs are synthetic. Programs may behave differently with the values that the game uses.
- On GPUs that flush 32-bit subnormal numbers, `FADD`, `FMUL` and `FFMA` without `.FTZ` flush too ([translation.md](translation.md)).

## From Python

```python
from siicl import gpucheck, interp, translate, vk

program = interp.Program(shader.program_bytes(), shader.compute_info())
tr = translate.translate(program.instructions, program.info, mufu_log=True)
with vk.Gpu() as gpu:
    result = gpucheck.check(program, tr, gpu, seed=0)
```

`vk.Gpu().run(translation, launch)` runs a translation on the buffers and surfaces of an `interp.Launch` and writes the results into the launch's surfaces, as `interp.run` does.
