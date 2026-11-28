# CPU backend for real-time play

siicl runs a game's AArch64 code in Unicorn ([boot.md](boot.md)). Unicorn suits milestone 1 of the [roadmap](roadmap.md): it runs every instruction the scanner finds in game code, and two runs of the same game do the same thing. It can't give the frame rate and frame pacing a game needs when someone plays it. This document records the measurements that show this, compares the backends that could replace Unicorn, and plans the change. Stages 1 and 2 of the plan are done ([Stage 1: the C library](#stage-1-the-c-library), [Stage 2: the missing instructions](#stage-2-the-missing-instructions)); `siicl boot` still runs on Unicorn.

## Measurements of the Unicorn backend

Measured on 2026-11-27 with unicorn 2.1.4 on a Ryzen 9 9900X.

- **Where a boot spends its time.** In a profiled `siicl boot` of the sample, Unicorn's `emu_start` took 90% of the time (25.1 s of 27.9 s). siicl's own Python code took most of the rest: patching the counter reads when the process is set up took 1.2 s, and all SVC handlers together took about 0.1 s. The game's code ran at about 195 million instructions per second, on one host thread.
- **The instruction budget.** `Cpu.run` passes `count=` to `emu_start`, so that a thread's slice ends after a fixed number of instructions. Unicorn then counts every instruction it runs. On a loop of two instructions it ran 2,170 million instructions per second without `count`, and 413 million with it.
- **Parallel cores.** Two Unicorn instances, each on its own host thread, mapped the same host page with `mem_map_ptr`. Each added 1 to the same word 5,000,000 times. The two instances ran at the same time, and the word ended far below 10,000,000:

  | Instructions | Final value |
  |---|---|
  | `LDADD` | 5,038,849 |
  | `LDXR`, `ADD`, `STXR`, retried until the store succeeds | 5,117,392 |
  | `LDR`, `ADD`, `STR`, which are not atomic | 4,991,332 |

  The atomic instructions lost as many updates as the plain ones. A game's locks are built on these instructions, so Unicorn can run only one guest core at a time.
- **The cost of an SVC.** Each SVC stops Unicorn and returns to Python. With an empty handler, the stop and the restart take 5.9 µs. During the boot, siicl's handler for an IPC request (SendSyncRequest) took 62 µs on average, and its handler for a counter read 4.9 µs. At 60 frames per second a frame lasts 16.7 ms.

## What the backend needs

1. **The game's instructions.** Every feature that `siicl scan` reports for game code. Beyond Armv8.0, the sample uses LSE atomics, `LDAPR`, pointer authentication, dot product, half precision, CRC32, AES and SHA. Its pointer authentication includes the hint forms, such as `PACIBSP`, and the other forms, such as `RETAB`, which the code uses to return from functions.
2. **Parallel guest cores.** Games run threads on several cores at once. Atomic instructions and the exclusive monitor (`LDXR` and `STXR`) must work across cores.
3. **A repeatable mode.** For accuracy work, a mode in which threads switch after fixed instruction counts and time is virtual, as now, so that two boot logs can be compared line by line. Real-time play uses the host clock instead.
4. **siicl's memory.** Guest memory stays backed by host memory that siicl allocates ([boot.md](boot.md)), so the kernel, the services and the GPU code read it without copies.
5. **Fast exits.** SVCs and faults return to siicl with little cost.

## Options

### dynarmic

dynarmic is a JIT compiler from ARM to x86-64 and AArch64 code, written in C++ and licensed 0BSD. Switch 1 emulators used it as their CPU. Its A64 interface (`dynarmic/interface/A64/config.h`) covers the needs above:

- The host supplies guest memory through a page table of host pointers (`page_table`), or through one host range for the whole address space (`fastmem_pointer`). Accesses outside them go to callbacks. siicl's host memory can back the page table as it is; for the single range, siicl would map its memory into one reserved host range.
- An `ExclusiveMonitor` that all cores share implements `LDXR` and `STXR` across cores. The `MemoryWriteExclusive` callbacks compare and swap.
- `AddTicks` and `GetTicksRemaining` count instructions in the generated code, for slices of a fixed length.
- `GetCNTPCT` answers reads of the system counter, so the loader would no longer need to change counter reads into SVCs. `wall_clock_cntpct` makes the counter follow the host clock.
- `CallSVC` and `ExceptionRaised` return SVCs, undefined instructions and other exceptions to the host.

As released, it doesn't implement all of the game's instructions; siicl's fork adds most of them ([Stage 2](#stage-2-the-missing-instructions)). Version 6.7.0 (2024-03-02) is the last release, and the version Arch Linux's AUR package builds. In its A64 decoder (`src/dynarmic/frontend/A64/decoder/a64.inc`), these entries are commented out:

- LSE atomics: `CAS`, `CASP`, `LDADD`, `LDCLR`, `LDEOR`, `LDSET` and `SWP`, in all sizes and orderings
- `LDAPR`, `LDAPRB` and `LDAPRH`
- Pointer authentication other than the hints: `RETAA`, `RETAB`, `BRAA`, `BLRAA`, `LDRAA`, `PACGA`, and the register forms of `PACIA`, `AUTIA` and the others
- `STG` and the other memory tagging instructions, which the sample has in one place

It does decode the hint forms of pointer authentication (as `HINT`, which does nothing), `SDOT` and `UDOT`, and `FNEG` in half precision.

Running one instruction of each group through the C library of stage 1 (`tests/test_dynarmic.py`) confirmed the list, and found three more groups that dynarmic doesn't run:

- Half-precision arithmetic other than `FNEG`, such as `FADD` and `FMUL`. The sample uses only `FNEG`.
- `CNTVCT_EL0`, the virtual counter. dynarmic has only `CNTPCT_EL0`, which is the counter the sample reads.
- The performance monitor registers, such as `PMCR_EL0` and `PMCNTENSET_EL0`, which the sample's code also accesses.

The original repository, `github.com/merryhime/dynarmic`, no longer exists. The AUR package builds from `github.com/lioncash/dynarmic`, and several emulator projects keep their own forks. siicl keeps its own fork as well, at `github.com/0x00trace/dynarmic`.

The work, as planned before stage 2:

- `LDAPR` as `LDAR`, which is implemented. Its ordering is stronger, so the result is correct.
- The pointer authentication forms with the treatment siicl uses now: PAC instructions leave pointers unchanged, AUT instructions succeed, `RETAB` returns as `RET` does, `BRAA` branches as `BR` does, and `LDRAA` loads as `LDR` does with its scaled offset.
- The LSE atomics, in one of two ways. One is a new callback that the host implements with host atomic instructions. The other is new IR operations that the x86-64 backend emits as `LOCK`-prefixed instructions and `CMPXCHG` loops. The callback is simpler; the IR operations avoid a call for each atomic.
- `CNTVCT_EL0` with the value of `CNTPCT_EL0`, and the performance monitor registers. What a game sees when it accesses those registers on the console isn't known yet; Unicorn's result is the reference until it is.
- A small C++ library with a C interface around dynarmic, so that Python (through `ctypes`) and later code in a compiled language can call it.

### A QEMU fork

QEMU's user mode, which runs Linux programs, runs each guest thread on its own host thread and maps guest memory into the host address space. In that mode its code generator uses host atomic instructions for the guest's atomics. It implements every instruction the scanner finds, as Unicorn does. Its speed hasn't been measured for siicl.

The work: a new user-mode target for the Switch's operating system, with NSO loading, the address layout and SVCs sent to siicl's kernel. The kernel would be written in C inside QEMU, or reached through a channel to another process. QEMU counts instructions only in system mode, and doing so turns off parallel execution there, so a repeatable mode would need work of its own. QEMU is licensed GPL-2.0, and the fork would be too.

### Unicorn in parallel mode

QEMU's code generator, which Unicorn is built on, has a parallel mode that uses host atomic instructions. Unicorn doesn't turn it on. Whether a patch to Unicorn could turn it on for instances that share memory hasn't been checked. The speed would stay Unicorn's.

### A new recompiler

An A64 JIT written for siicl would give full control over accuracy and speed. It is by far the largest amount of work, and it would repeat most of what dynarmic already does.

### Summary

| Option | The game's instructions | Parallel cores | Repeatable mode | Main work |
|---|---|---|---|---|
| Unicorn (now) | All | No (measured above) | Yes, with `count` | None |
| dynarmic | After three groups are added | Yes | Yes, with ticks | Patch the decoder and the x86-64 backend; a C library around it |
| QEMU fork | All | Yes | Needs work of its own | A new user-mode target; the kernel in C or in another process |
| Unicorn in parallel mode | All | Not checked | Yes | A patch to Unicorn, not checked |
| New recompiler | To be written | To be written | To be written | All of it |

## Decision

Use dynarmic, from a fork that siicl keeps.

- It was written for this use. The repeatable mode and real-time play both come from its interface: ticks and `GetCNTPCT` for one, the shared exclusive monitor and `wall_clock_cntpct` for the other.
- The instructions it lacks form a known list, and the scanner finds them in game code.
- Unicorn stays as the reference. Each instruction added to dynarmic is tested against Unicorn on synthetic programs, in the same way that the shader translation is tested against the reference interpreter ([gpu-check.md](gpu-check.md)).

## Plan

Stages 1 to 3 keep Unicorn as the default backend and leave the kernel as it is, so they can be done alongside milestone 1. Stage 4 changes the scheduler and follows milestone 1.

| Stage | Work | Done when |
|---|---|---|
| 1. Build and C library | Done. dynarmic 6.7.0 is the git submodule `native/dynarmic`, unchanged, from `github.com/lioncash/dynarmic`; the fork is made in stage 2, with the first change to dynarmic. `native/siicl_dynarmic.cpp` is the C library: create a machine and its cores, map pages, read and write registers, run for a number of ticks, and report why a run stopped. `siicl.dynarmic` loads it with `ctypes`, as siicl loads liblz4 and libvulkan. | `tests/test_dynarmic.py` runs synthetic programs on dynarmic from Python. |
| 2. Missing instructions | Done. The fork, `github.com/0x00trace/dynarmic` (branch `siicl`), adds `LDAPR`, the pointer authentication forms, the LSE atomics and `CNTVCT_EL0`, and the submodule points at it. `tests/test_cpu_compare.py` assembles synthetic programs with `aarch64-linux-gnu-as`, runs each on Unicorn and on dynarmic, and compares the registers, the flags and the memory. | The instruction groups of the scan report run on dynarmic and give the same results as on Unicorn, except where Unicorn is wrong (see below). Three groups that no boot has needed yet are still missing. |
| 3. A backend for `siicl boot` | A dynarmic implementation of `siicl.cpu.Cpu`: `run`, `map_host`, `read`, `write`, the registers and the saved contexts. An option of `siicl boot` chooses the backend. Counter reads go through `GetCNTPCT` instead of patched SVCs. | The sample boots on dynarmic to the same point as on Unicorn, the two boot logs agree thread by thread, and the speed is measured. |
| 4. Parallel cores and real time | One host thread for each guest core, with the shared exclusive monitor, and a scheduler for each core. A real-time mode that uses the host clock, beside the repeatable mode. The language of the kernel and the services is chosen here, from the number of SVCs a frame makes after milestone 1 and the costs measured above. | The sample runs with its threads on several host threads, and two runs in the repeatable mode still give the same boot log. |

The dynarmic fork and the C library become runtime dependencies besides Unicorn. The environment section of `CLAUDE.md` is updated when they are added.

## Stage 1: the C library

Build it with CMake (pacman packages `cmake`, `boost` and `ninja`). dynarmic's other libraries come with its source, in `native/dynarmic/externals`.

```sh
git submodule update --init native/dynarmic
cmake -S native -B build/native -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build/native
```

`siicl.dynarmic` loads `build/native/libsiicl_dynarmic.so`, or the file that `$SIICL_DYNARMIC` names. Its tests skip when the library isn't built.

- A `Machine` holds the page tables and the exclusive monitor; each `Core` is one CPU. `Core.run()` returns the same `siicl.cpu.Stop` as `Cpu.run()`.
- Pages that can be both read and written go into dynarmic's page table, so the generated code reaches them directly. Accesses to other pages, such as code and read-only data, and accesses that cross a page boundary go to callbacks that check the permissions.
- A data fault stops the run at the instruction that made it, with its registers unchanged. Undefined instructions and `BRK` leave the PC at the instruction, as `siicl.cpu` does.
- dynarmic raises an exception for `YIELD`, `WFE`, `WFI`, `SEV` and `SEVL`. The C library treats them as NOPs, as Unicorn does.
- Each read of `CNTPCT_EL0` returns the core's counter and adds 1 to it. `CNTFRQ_EL0` reads 19.2 MHz.

What stage 1 found:

- **A run ends only at the end of a block of translated code.** With a budget of 100 on a loop of three instructions, runs used 100 and 102 instructions; with budgets of 7 and 1, they used 9 and 3. Sixty instructions without a branch ran to their end on a budget of 10. Two runs of the same program give the same counts, so the repeatable mode still works, but a slice can end at a different instruction than on Unicorn.
- **A core counts instructions or accepts halt requests, not both.** Between linked blocks, dynarmic checks either the budget or a halt request. A core created with `count_ticks` (the repeatable mode) stops only at its budget; a core created without it (real-time play) runs until another thread calls `Core.halt()`.
- **Counts at exceptions and faults are approximate.** A run that ends at an exception is charged its whole block, and a run that ends at a data fault is charged nothing.
- **Exclusive stores work across cores.** Two cores on two host threads each added 1 to the same word 200,000 times with `LDXR` and `STXR`, and the word ended at 400,000.
- **Read-only pages cost a call per access.** Leaving them out of dynarmic's page table keeps the permissions correct, but every access to code and read-only data goes through a callback. Stage 3 measures what this costs on a boot. dynarmic's `fastmem_pointer`, with read-only pages protected in host memory, is the faster alternative.

## Stage 2: the missing instructions

The fork is `github.com/0x00trace/dynarmic`. Its branch `siicl` is dynarmic 6.7.0 with four commits, one for each group below, so the changes can be moved onto another fork of dynarmic. All four change only the A64 frontend, so they work on both of dynarmic's backends.

- **`LDAPR`, `LDAPRB`, `LDAPRH`** are translated as `LDAR`.
- **Pointer authentication** runs as on a CPU with the keys disabled, which is what siicl does on Unicorn: the PAC and AUT instructions and `XPACI` and `XPACD` leave the pointer unchanged, `RETAA` and `RETAB` return as `RET` does, `BRAA` and `BLRAA` and their other forms branch as `BR` and `BLR` do, and `LDRAA` and `LDRAB` load as `LDR` does.
- **`PACGA`** returns a 32-bit code computed from its two inputs by multiplying and shifting, not with the architecture's QARMA cipher. A program can't read the key and can only compare one code with another, so any function of the two inputs serves. Its value differs from Unicorn's, and the comparison leaves it out.
- **The LSE atomics** (`CAS`, `CASP`, `LDADD`, `LDCLR`, `LDEOR`, `LDSET`, `LDSMAX`, `LDSMIN`, `LDUMAX`, `LDUMIN`, `SWP`, in all sizes) are each translated as an exclusive load, the operation, and an exclusive store. The store succeeds only if memory still holds the loaded value; if it fails, the instruction runs again. This uses the exclusive monitor that the cores share, so the atomics are atomic across cores, and it needed no change to the backends. Each atomic takes the monitor's lock twice. `LOCK`-prefixed host instructions would be faster; stage 3 measures whether that matters on a boot.
- **`CNTVCT_EL0`** reads the same value as `CNTPCT_EL0`.

Still missing, because no boot has needed them and the right behaviour isn't settled:

- Half-precision arithmetic other than `FNEG`.
- The performance monitor registers. A program at EL0 can access them only if the kernel allows it, and whether the console's kernel does isn't known.
- `STG` and the other memory tagging instructions.

`tests/test_dynarmic.py` checks that each of these still stops as an undefined instruction.

What the tests show:

- **Agreement with Unicorn.** `tests/test_cpu_compare.py` runs programs for the atomics, compare and swap, `LDAPR`, pointer authentication, the authenticated branches and loads, exclusive accesses, dot product, half-precision negate, CRC32, AES, SHA and polynomial multiply. Each starts from four sets of random registers, flags and memory, and both backends end with the same general registers, vector registers, flags and memory.
- **Atomics across cores.** Two cores on two host threads each ran 200,000 increments of one word, with `LDADDAL`, with a `CASAL` loop, and with a plain increment under a lock taken with `SWPAL`. In each case the word ended at 400,000.
- **Faults.** An atomic instruction on unmapped or read-only memory stops the run at the instruction with its registers unchanged, and doesn't run again.
- **Unicorn computes `LDSMAX` and `LDSMIN` wrongly at 8, 16 and 32 bits.** Unicorn 2.1.4 compares their operand as an unsigned number, so the result is wrong whenever the operand is negative at the size of the access: with 0 in memory and an operand of -1, `LDSMAX` stores -1. The 64-bit forms and the unsigned forms are right. dynarmic gives the architecture's result in every case that `test_atomic_min_and_max_follow_the_architecture` checks, so for these forms the tests compare dynarmic with the architecture's definition and not with Unicorn. The sample's code contains the 32-bit forms, so a boot on Unicorn can compute a wrong value there ([boot.md](boot.md)). `test_unicorn_signed_atomic_min_and_max` is expected to fail, and passes once a Unicorn release fixes this.

## Open questions

- Whether the fork should give dynarmic a way to end a slice at an exact instruction count, such as a limit on block length, so that boot logs from the two backends can be compared line by line.
- Whether the atomics should use `LOCK`-prefixed host instructions instead of the exclusive monitor. Stage 3 measures the boot before deciding.
- What a game sees when it accesses the performance monitor registers on the console.
- What to do about Unicorn's `LDSMAX` and `LDSMIN` while `siicl boot` still runs on Unicorn: the loader could change them into SVCs that siicl computes, as it does for counter reads.
