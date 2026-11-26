# Booting a game

`siicl boot <file.nup>` loads a game's executables, starts its main thread and runs the game's threads until the game needs something siicl does not implement yet. It writes what happened to `boot.log` in `~/.cache/siicl/<title id>/`. `--seconds` sets how much guest time the run may use (60 seconds by default).

## The CPU

The game's code runs in Unicorn, the CPU emulator built on QEMU (`siicl.cpu`), with its `max` CPU model. That model implements every architecture feature the scanner finds in game code: LSE atomics, LDAPR, dot product, half precision, CRC32, AES and SHA.

Unicorn runs the code at EL1 with the pointer authentication keys off. PAC instructions then leave pointers unchanged, AUT instructions always succeed and RETAB returns like RET. That is the treatment the scan report asks for.

`Cpu.run()` runs until one of these stops it:

| Stop | Meaning |
|---|---|
| `svc` | An SVC instruction. The PC already points past it. |
| `fault` | A read, write or instruction fetch at an address that is unmapped or does not allow that access. |
| `undefined` | An instruction the CPU does not implement. |
| `break` | A BRK instruction. |
| `limit` | The instruction budget given to `run()` ran out. |

## The address space

Games use the 64-bit process type with a 39-bit address space, and siicl supports only that type. The system places each region at a random address. siicl uses fixed addresses, so every run places everything at the same address and two boot logs can be compared line by line. The game asks the kernel where the regions are (with GetInfo), so it does not depend on the addresses.

| Region | Start | Size | Holds |
|---|---|---|---|
| Address space | 0x8000000 | 512 GiB minus 128 MiB | Everything the process can map |
| Code | 0x8000000 | 16 GiB minus 128 MiB | The modules |
| Alias | 64 GiB | 64 GiB | Memory mapped with MapPhysicalMemory |
| Heap | 128 GiB | 32 GiB | The heap that SetHeapSize grows |
| Stack | 192 GiB | 2 GiB | Thread stacks, and memory mapped there with MapMemory |
| Kernel map | 256 GiB | 64 GiB | Pages the kernel maps for the process, such as thread-local storage |
| Remote | 384 GiB | 32 GiB | The Switch 2's remote region, which GetInfo reports; nothing is mapped there yet |

QueryMemory reports addresses past the end of the address space as one inaccessible range.

Each mapped range has a memory state, as QueryMemory reports it: `code` for .text and .rodata, `code_data` for .data and .bss, `normal` for the heap and physical memory, `stack` and `thread_local`.

Every mapping is backed by host memory that siicl allocates. Unicorn maps the guest pages straight onto it, so two guest ranges can share the same memory. MapMemory uses this to show heap pages in the stack region.

## Loading the modules

The modules are the NSO files in the exefs (`siicl.loader`). They go one after another from the start of the code region, each starting on a page boundary, in the system loader's order: rtld, main, subsdk0 to subsdk9, then sdk. In each module:

- .text is mapped read and execute
- .rodata is mapped read-only
- .data, with .bss after it, is mapped read and write; .bss starts as zeros

The loader does not relocate anything. rtld, the first module, runs first and relocates itself and the other modules.

The loader changes one kind of instruction. Each read of the system counter (`MRS Xt, CNTPCT_EL0` or `CNTVCT_EL0`) becomes `SVC #(0xF000 + t)`, which siicl answers with its virtual time (see below). Real SVC numbers are below 0x100, so these never collide with them.

## The main thread

`siicl.kernel.Process` sets up the main thread as the system does:

- Its stack is mapped in the stack region, with the size the NPDM gives. SP starts at the top.
- Its thread-local storage is a 0x200-byte slot in a page mapped in the kernel map region. TPIDRRO_EL0 holds its address. A page holds eight slots, and each new thread takes the next free one.
- PC is the first instruction of the first module, X0 is 0 and X1 is the main thread's handle.
- Its priority and core come from the NPDM.

Handles have the kernel's format: the slot index in the low 15 bits and a serial number above them, so the first handle is 0x8000.

## Threads and time

Threads take turns on the one CPU. The scheduler keeps a run queue for each core the NPDM allows, and gives the cores turns in a fixed order. On its turn, a core runs its highest-priority runnable thread until the thread makes an SVC or has run 200,000 instructions. Threads of equal priority on one core also take turns. Nothing depends on the host's timing, so two runs of the same game do the same thing and write the same log.

Time is virtual. The system counter runs at 19.2 MHz, as on the console. It advances:

- by one nanosecond for each instruction of a full 200,000-instruction turn
- by one microsecond for each SVC
- by one tick for each read of the counter, so two reads never return the same value
- to the next timeout, when every thread is waiting

A thread that is switched out between an exclusive load and its exclusive store keeps its exclusive monitor, which the console's kernel would clear. Games rarely use exclusive loads and stores (the LSE atomics replace them), so siicl does not handle this yet.

## SVCs

`siicl.svcs` implements these SVCs, following switchbrew and the open-source Mesosphere kernel:

- Memory: SetHeapSize, SetMemoryPermission, SetMemoryAttribute, MapMemory, UnmapMemory, QueryMemory, MapPhysicalMemory, UnmapPhysicalMemory
- Threads: CreateThread, StartThread, ExitThread, SleepThread, GetThreadPriority, SetThreadPriority, GetThreadCoreMask, SetThreadCoreMask, GetCurrentProcessorNumber, GetThreadId, GetProcessId, CloseHandle
- Synchronization: WaitSynchronization (on threads so far), CancelSynchronization, ArbitrateLock, ArbitrateUnlock, WaitProcessWideKeyAtomic, SignalProcessWideKey, WaitForAddress, SignalToAddress, SynchronizePreemptionState
- Other: GetSystemTick, GetInfo (including the Switch 2's remote region and page size types), Break, OutputDebugString, ExitProcess

GetInfo reports 9 GiB as the process's total memory. siicl chose this value. An SVC the NPDM does not allow ends the run as a crash, as on the console. Any other SVC, and any GetInfo type siicl does not know, ends the run as unimplemented.

## The boot log

The log has these parts, in this order:

1. The modules, with their address ranges and module IDs.
2. The memory map: each mapped range with its permissions, memory state and what it holds.
3. The main thread: handle, entry point, stack top, thread-local storage, priority and core. Then how many counter reads the loader changed.
4. One line for each SVC: the guest time in seconds, the thread number, the SVC with its arguments, and its result. For example:
   - `0.066837 t0  GetInfo(HeapRegionAddress, handle=0xffff8001, sub=0x0) -> ok, 0x2000000000`
   - `0.000001 t0  QueryMemory(address=0x0) -> ok: 0x0+0x8000000 free ---`
5. Why the run ended, with the address as a module and offset, such as `sdk+0x1a44ec`. The reasons are:
   - unimplemented: an SVC or GetInfo type siicl does not implement yet
   - crash: a fault, an undefined instruction, a BRK, or an SVC the NPDM does not allow
   - break: the game called Break (calls that only notify do not end the run)
   - exit: the game called ExitProcess
   - deadlock: every thread waits, and none has a timeout
   - time: the time limit was reached
6. For the thread that ended the run, its registers X0 to X30, SP and PC, and a backtrace: the stop address, the link register if it points into a module, then the return addresses found by following the frame pointers (X29) up the stack.
7. Every thread with its state and where it is.
8. How many times the game called each SVC.

The log describes the game's code, so it must stay outside the repository like any other output from a dump.
