# Booting a game

`siicl boot <file.nup>` loads a game's executables, starts its main thread and runs it until the game needs something siicl does not implement yet. It writes what happened to `boot.log` in `~/.cache/siicl/<title id>/`. Today it stops at the game's first SVC, because no SVCs are handled yet. Stage 2 of the [roadmap](roadmap.md) adds them.

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
| Address space | 0x8000000 | 512 GiB minus 128 MiB | Everything the process can use |
| Code | 0x8000000 | 16 GiB minus 128 MiB | The modules |
| Alias | 64 GiB | 64 GiB | Memory the game maps with MapMemory |
| Heap | 128 GiB | 32 GiB | The heap that SetHeapSize grows |
| Stack | 192 GiB | 2 GiB | Thread stacks |
| Kernel map | 256 GiB | 64 GiB | Pages the kernel maps for the process, such as thread-local storage |

Each mapped range has a memory state, as QueryMemory reports it: `code` for .text and .rodata, `code_data` for .data and .bss, `stack` and `thread_local`.

## Loading the modules

The modules are the NSO files in the exefs (`siicl.loader`). They go one after another from the start of the code region, each starting on a page boundary, in the system loader's order: rtld, main, subsdk0 to subsdk9, then sdk. In each module:

- .text is mapped read and execute
- .rodata is mapped read-only
- .data, with .bss after it, is mapped read and write; .bss starts as zeros

The loader does not relocate anything. rtld, the first module, runs first and relocates itself and the other modules.

## The main thread

`siicl.kernel.Process` sets up the main thread as the system does:

- Its stack is mapped in the stack region, with the size the NPDM gives. SP starts at the top.
- Its thread-local storage is the first 0x200-byte slot of a page mapped in the kernel map region. TPIDRRO_EL0 holds its address. A page holds eight slots.
- PC is the first instruction of the first module, X0 is 0 and X1 is the main thread's handle.
- Its priority and core come from the NPDM.

Handles have the kernel's format: the slot index in the low 15 bits and a serial number above them, so the first handle is 0x8000.

## The boot log

The log has these parts, in this order:

1. The modules, with their address ranges and module IDs.
2. The memory map: each mapped range with its permissions, memory state and what it holds.
3. The main thread: handle, entry point, stack top, thread-local storage, priority and core.
4. Why the run stopped, with the address as a module and offset, such as `rtld+0x2170`. Examples:
   - `stopped: SVC 0x06 (QueryMemory) at rtld+0x2170 (0x8002170)`
   - `stopped: unmapped read at 0x40, by the instruction at main+0x28 (0x8000028)`
5. The registers X0 to X30, SP and PC.
6. A backtrace: the stop address, the link register if it points into a module, then the return addresses found by following the frame pointers (X29) up the stack.

The log describes the game's code, so it must stay outside the repository like any other output from a dump.
