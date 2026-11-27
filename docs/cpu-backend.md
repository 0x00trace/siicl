# CPU backend for real-time play

siicl runs a game's AArch64 code in Unicorn ([boot.md](boot.md)). Unicorn suits milestone 1 of the [roadmap](roadmap.md): it runs every instruction the scanner finds in game code, and two runs of the same game do the same thing. It can't give the frame rate and frame pacing a game needs when someone plays it. This document records the measurements that show this, compares the backends that could replace Unicorn, and plans the change. None of it is implemented yet.

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

It doesn't implement all of the game's instructions. Version 6.7.0 (2024-03-02) is the last release, and the version Arch Linux's AUR package builds. In its A64 decoder (`src/dynarmic/frontend/A64/decoder/a64.inc`), these entries are commented out:

- LSE atomics: `CAS`, `CASP`, `LDADD`, `LDCLR`, `LDEOR`, `LDSET` and `SWP`, in all sizes and orderings
- `LDAPR`, `LDAPRB` and `LDAPRH`
- Pointer authentication other than the hints: `RETAA`, `RETAB`, `BRAA`, `BLRAA`, `LDRAA`, `PACGA`, and the register forms of `PACIA`, `AUTIA` and the others
- `STG` and the other memory tagging instructions, which the sample has in one place

It does decode the hint forms of pointer authentication (as `HINT`, which does nothing), `SDOT` and `UDOT`, and `FNEG` in half precision.

The original repository, `github.com/merryhime/dynarmic`, no longer exists. The AUR package builds from `github.com/lioncash/dynarmic`, and several emulator projects keep their own forks. siicl would keep its own fork as well.

The work:

- `LDAPR` as `LDAR`, which is implemented. Its ordering is stronger, so the result is correct.
- The pointer authentication forms with the treatment siicl uses now: PAC instructions leave pointers unchanged, AUT instructions succeed, `RETAB` returns as `RET` does, `BRAA` branches as `BR` does, and `LDRAA` loads as `LDR` does with its scaled offset.
- The LSE atomics, in one of two ways. One is a new callback that the host implements with host atomic instructions. The other is new IR operations that the x86-64 backend emits as `LOCK`-prefixed instructions and `CMPXCHG` loops. The callback is simpler; the IR operations avoid a call for each atomic.
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
| 1. Fork and build | Fork dynarmic 6.7.0 and build it with CMake. Its source includes the libraries it uses (`externals/`) except Boost, which comes from the Arch package `boost`. Write the C library: create a core, read and write registers, set the page table, run for a number of ticks, and report why a run stopped. Load it with `ctypes`, as siicl loads liblz4 and libvulkan. | A test runs a synthetic program on dynarmic from Python. |
| 2. Missing instructions | Add `LDAPR`, the pointer authentication forms and the LSE atomics to the fork. A test harness assembles synthetic programs with `aarch64-linux-gnu-as`, runs each on Unicorn and on dynarmic, and compares the registers, the flags and the memory. | Every instruction group in the scan report runs on dynarmic and gives the same results as on Unicorn. |
| 3. A backend for `siicl boot` | A dynarmic implementation of `siicl.cpu.Cpu`: `run`, `map_host`, `read`, `write`, the registers and the saved contexts. An option of `siicl boot` chooses the backend. Counter reads go through `GetCNTPCT` instead of patched SVCs. | The sample boots on dynarmic to the same point as on Unicorn, the two boot logs agree thread by thread, and the speed is measured. |
| 4. Parallel cores and real time | One host thread for each guest core, with the shared exclusive monitor, and a scheduler for each core. A real-time mode that uses the host clock, beside the repeatable mode. The language of the kernel and the services is chosen here, from the number of SVCs a frame makes after milestone 1 and the costs measured above. | The sample runs with its threads on several host threads, and two runs in the repeatable mode still give the same boot log. |

The dynarmic fork and the C library become runtime dependencies besides Unicorn. The environment section of `CLAUDE.md` is updated when they are added.

## Open questions

- Whether dynarmic ends a slice at an exact instruction count, or only between blocks of generated code. If only between blocks, threads switch at different instructions on the two backends, and boot logs can only be compared thread by thread.
- Whether LSE atomics are a callback or IR operations. Stage 2 can start with the callback and measure the boot before deciding.
- What `PACGA` returns. With the keys off its result only has to be the same for the same inputs. Matching Unicorn's value keeps the comparison in stage 2 exact; otherwise the harness leaves `PACGA` out.
- Where the fork lives, and how its changes are kept as a short series of commits that can be moved onto another maintained fork.
