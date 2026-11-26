# Booting a game

`siicl boot <file.nup>` loads a game's executables, starts its main thread and runs the game's threads until the game needs something siicl does not implement yet. It writes what happened to `boot.log` in `~/.cache/siicl/<title id>/`.

Options:

- `--seconds`: how much guest time the run may use (60 seconds by default)
- `--stub-unknown`: answer each service command siicl does not implement with success and no data, and hand out an empty service for each service it does not know, instead of ending the run. The log lists every command answered this way. This shows how far a game gets and what it asks for next, but the game may then fail in other ways, so use it to explore, not as a result.

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
- IPC: ConnectToNamedPort (for "sm:"), SendSyncRequest, SendSyncRequestWithUserBuffer
- Events, shared and transfer memory: SignalEvent, ClearEvent, ResetSignal, MapSharedMemory, UnmapSharedMemory, CreateTransferMemory (the lent pages keep only the permission the game asks for, and are locked until the handle is closed)
- Other: GetSystemTick, GetInfo (including the Switch 2's remote region and page size types), Break, OutputDebugString, ExitProcess

MapMemory and SetMemoryPermission accept heap memory and the modules' .data and .bss. The SDK maps thread stacks from its .bss with MapMemory, and makes part of each module's .data read-only after relocating it.

GetInfo reports 9 GiB as the process's total memory. siicl chose this value. An SVC the NPDM does not allow ends the run as a crash, as on the console. Any other SVC, and any GetInfo type siicl does not know, ends the run as unimplemented.

## IPC and services

siicl answers the game's service requests itself, in Python (`siicl.services`). `siicl.ipc` reads and writes the messages: the HIPC layout, with CMIF inside for most services and TIPC for the service manager. The game connects to the "sm:" port, asks the service manager for services by name, and sends requests on the sessions it gets back. A session can become a domain, which holds several objects by ID. The control commands (converting to a domain, cloning a session, the pointer buffer size) are handled too.

The service manager does not check the NPDM's list of services. The sample uses services the list does not name, such as "bsd:s", and stops if the service manager refuses them. The list also does not name the graphics services the SDK uses.

Services so far, each with the commands the sample has used:

| Service | What siicl does |
|---|---|
| `sm:` | Hands out sessions for the services below |
| `lm` | Decodes the game's log packets into log lines, so the game's own messages appear in the boot log |
| `appletOE` | The application proxy and its interfaces: state, self, window, audio, display, library applets and application functions, plus the Switch 2 interfaces. The message queue starts with a focus-change message. The launch parameter for the preselected user gives siicl's one user account. |
| `pctl` and its variants | Parental controls: no restrictions |
| `aoc:u` | Add-on content: none |
| `ectx:aw` | Error contexts: accepted and dropped |
| `time:u`, `time:a`, `time:s` | Clocks that start at 2026-01-01 00:00:00 UTC, a UTC time zone, and the time shared memory |
| `fsp-srv` | The game's romfs, as a RomFS image built from the NUP container |
| `nifm:u` | Network requests, which fail: siicl has no network |
| `bsd:u`, `bsd:s` | Socket clients register; nothing connects |
| `acc:u0` | One user account |
| `ldr:ro` | Takes the process handle |
| `hid` | Input: the applet resource and its shared memory, which stays zeroed, so no controllers are connected. The memory takes the size of its first mapping (0x40000 bytes in the sample, as on the Switch 1). |
| Graphics supervisor | Reached with ISelfController GetGraphicsSupervisor, not through `sm:`. Its sessions include the display library's. See below. |
| `nvsrv` | The GPU driver service, handed out by the graphics supervisor. Nothing is implemented yet. In the sample its commands 0, 12 and 2 fit `nvdrv`'s Open (a device path in a buffer), Ioctl3 and Close on the Switch 1. The first ioctl is 0xC0E04705: type 'G', number 5, which on the Switch 1 is the GPU characteristics query, here with a 0xE0-byte argument. |

### The graphics supervisor

The Switch 2 SDK boots its graphics system (`nn::gs::sdk::BootGraphicsSystem`) through a graphics supervisor. The applet's ISelfController command 2000 (GetGraphicsSupervisor) hands out a session for it as a copied handle. The supervisor and the sessions it hands out speak TIPC, and their clients in the SDK write the messages by hand. No public documentation describes the commands, so siicl names them by number. This is what siicl has learned from the client code, in the order the sample sends the commands. "Handle" means one handle in the response, which the client reads before the result.

Supervisor:

| Command | Sent by | Request | Response |
|---|---|---|---|
| 4 | `nn::gsv::BootWithRemoteMemory` | The process ID, a u64 (0x8000000000000006 in the sample), zeros; a copy of the process handle | Result |
| 11 | the same | A copy of the process handle | Result |
| 101 | the same | Nothing | Result |
| 1501, 1502 | `nn::gsv::shim::detail::InitializeInterModuleImageContext` | Nothing | Result |
| 1402 | `nn::nvsrv::BootWithRemoteMemory` | Nothing | Handle: a session for the GPU driver service, `nvsrv` |
| 201 | `BootGraphicsSystem` | Nothing | Result |
| 202 | the same | Nothing | Handle: a session (see below) |
| 204 | the same | Two transfer memories (copied): 0x1000 bytes the game keeps read and write, and 0x2000 bytes it keeps read-only; arguments 1, 0x1FF and their two sizes | Two moved handles, which the client attaches as system events |
| 115 | `nn::gs::sdk::BootNativeMemoryManagement` | Memory block mode bits (2 in the sample) | The mode bits granted, as a u64 after the result. `CreateNativeMemory` tests bit 1. siicl grants what was asked. |
| 1101, 1102 | `nn::gs::sdk::BootGraphicsFrameManagement` | Nothing | Result |
| 497 | `nn::gpu::sdk::BootWithRemoteMemory` | A u64 (4 in the sample) | Handle: a session (see below) |
| 401, 402, 301 | `BootGraphicsSystem` | Nothing | Result |
| 434 | NVN's device setup | A u32 (1 in the sample) | Result |
| 305 | the same | Two u32s (0x20000 and 3 in the sample) | Handle: a session for the display library, `nn::vi2` |

The session from 202: command 1 returns a u32, which must be below 0x400; command 21 sends it back, capped at 63. siicl answers 63.

The session from 497 (its clients are in `nn::gpu::sdk`):

- Command 11 returns a range of bit positions, low then high. The client makes a 64-bit mask of the bits from low to high. siicl offers 0 to 63.
- Command 12 takes an index and returns that index's event. Index 1 is GPU address space 1's sync event: the client waits on it after each change to the address space.
- Command 21 takes an index. The client then waits on that index's event, so siicl signals it at once.
- Command 22 returns a session for the GPU memory manager (below).
- Command 33 takes an index and returns a u32 the client uses as a count. siicl answers 1.
- Command 41 comes from `nn::gpu::sdk::CreateGraphicsDevice` with a device name ("NvRmGpu"), and returns a byte the client keeps. siicl answers 0.
- Command 203 takes a name ("Gpu") and returns three u64s: whether there is a range (it must be below 5; 0 means none), then the range's lowest and highest GPU address. The SDK keeps it as the user partition of the GPU address space (`GetUserPartitionGpuAddressMin` and `Max`). NVN reserves the GPU addresses that mirror the CPU's there (0x8000000 to 0x8000000000) and then allocates more, so siicl gives 0x8000000 to 0xA000000000.
- Command 202 takes an index and returns two u64s, which the SDK keeps as the supervisor's partition (`GetGsvPartitionGpuAddressMin` and `Max`). siicl gives the rest of the game's range, 0xA000000000 to 0xC000000000.

The GPU memory manager, from the 497 session's command 22:

- Command 101 comes from `nn::gpu::sdk::CreateGpuAccessibleMemory`: the first and last address of the memory. It returns a u32 memory handle.
- Command 211 comes from `nn::gpu::sdk::NormalMapGpuMemory4K`: the address space's index, the first and last GPU address, the memory handle, the offset into it, then permissions, cacheability and flags. The client then waits on that address space's sync event. siicl records the mapping (`siicl.services.gfx.GpuMemory`, which turns GPU addresses back into CPU addresses) and signals the event. In the sample the GPU address equals the CPU address.
- Commands 213 and 102 undo 211 and 101 in the sample; siicl does not implement them yet.

The display session from 305:

- Command 3 returns a u32 below 8, and command 1 a u32 below 0x400. siicl answers 1 to both.
- Command 11 takes a u32 and returns an event, which the client polls until it is signaled. siicl hands it out already signaled.
- Command 21 returns a result.
- Command 31 takes a u32 and returns an event, which the game resets.
- Command 101 takes a display name ("default") and a u32.
- Command 263 takes a display name ("default") and returns a result.

After command 204, the client checks a header that the supervisor should have written at the start of the read-only transfer memory, made of 32-bit words: 1, a value, a size below 0x1000, then five offsets. Three of the offsets point into the read-only memory and two into the read-write memory. So the supervisor and the game talk through queues in that shared memory, signalled by the two events. siicl writes a header with values that pass the client's checks, but it does not serve the queues yet. Their values are guesses, and the sample has not used the queues so far. Closing the game's handles to the transfer memories does not give the pages back, because the supervisor still holds them, as on the console.

### The GPU driver

`nvsrv`, the GPU driver service, speaks CMIF and uses the commands of the Switch 1's `nvdrv` (`siicl.services.nv`): Open (a device path), Ioctl, Ioctl2, Ioctl3, Close, QueryEvent, SetAruid and SetGraphicsFirmwareMemoryMarginEnabled. Ioctl request numbers have the Linux layout (direction, size, type, number). siicl finds a device's handler by type and number, because argument sizes differ from the Switch 1. An unknown device or ioctl ends the run, with its request and input in the message.

`/dev/nvhost-ctrl-gpu`:

- 'G' 5, GetCharacteristics, with a 0xE0-byte argument: the buffer size and address, then a 0xD0-byte characteristics structure (0xA0 bytes on the Switch 1). Up to 0x70, the structure has the Switch 1 fields (Linux `nvgpu`'s `nvgpu_gpu_characteristics`). NVN combines the architecture and implementation into a chip ID and looks it up in its table of supported chips: 0x17E is "GA10F", the Switch 2's GPU (the table also has the Switch 1's 0x12B and 0x12E). siicl describes architecture 0x170, implementation 0xE, and the Ampere engine classes (3D 0xC797, compute 0xC7C0, channel 0xC76F, copy 0xC7B5). At 0x70 come nine signed 16-bit values that NVN compares like `nvgpu`'s interface levels (the last ioctl number of each driver interface). It refuses a value at 0x78 below 6, and turns features on above 7, 0xA, 0x12, 0x1F, 0x24, 0x26 and 0x2A. siicl gives 0x40 to all but the first, which gets 0x12: from 0x13 on, NVN maps its SMs with the VsmsMapping ioctl and checks the answer, and siicl does not know that ioctl's format yet.
- 'G' 6, GetTpcMasks: one GPC with six TPCs.
- 'G' 19, VsmsMapping on the Switch 1: siicl leaves the argument zeroed.
- QueryEvent hands out an event for each fd and event number.

`/dev/nvhost-as-gpu`, a GPU address space:

- 'A' 9, InitializeEx: the game sets the range of GPU addresses (0x8000000 to 0xC000000000 in the sample).
- 'A' 8, GetVaRegions: siicl reports small (4 KiB) pages below 16 GiB and big (128 KiB) pages above. Ioctl3 also returns the regions in its second output.

The SDK keeps its own model of each GPU address space (`nn::gpu::sdk::util`'s "virtual machine"). It answers AllocSpace and the other address-space ioctls itself, within the partitions the supervisor reported, and sends only some of them to `nvsrv`.

`/dev/nvhost-prof-gpu`, the GPU profiler: Open fails, as on a retail console.

SVC 0x80, SetGpuMemoryAttribute (an address, a size, a mask and a value): siicl records the bits for the range. switchbrew does not say what they mean.

In the sample, NVN's device setup now opens the GPU device and the address space, reserves the CPU mirror, creates and maps two areas of GPU memory, and then undoes it all and aborts. The step that fails comes after the mappings and does not show in the log yet.

With these answers, `BootGraphicsSystem` finishes in the sample, and the game goes on to initialize input (`hid`).

Two more applet answers the graphics setup needs:

- IApplicationFunctions command 210 (20.0.0+, no name on switchbrew) hands out the event that `nn::oe::GetOperationMode` checks before it uses its cached mode. The check retries on every error except a timeout, so the handle must be a real event, or the game loops for ever.
- IGraphicsScalingParametersFunctions command 10 returns a u32 that `nn::oe::GetCurrentGraphicsScale` turns into a scale: 10 is 0, 20 is 1, 30 is 2, and anything else aborts. siicl answers 10.

### The time shared memory

The Switch 2 SDK reads the clocks from a 0x1000-byte shared memory. Each value has a 32-bit counter and two copies, and the low bit of the counter says which copy is current:

| Offset | Value |
|---|---|
| 0x00 | The steady clock's base time and source ID, as on the Switch 1 (copies at 0x08 and 0x20) |
| 0x38 | The user system clock's context: offset in seconds, steady time point, source ID (copies of 0x20 bytes at 0x40 and 0x60) |
| 0x80 | The network system clock's context (copies at 0x88 and 0xA8) |
| 0xC8 | Whether the user clock corrects itself automatically (bytes at 0xCC and 0xCD) |
| 0xD0 | The steady clock's adjustment, new on the Switch 2 (copies of 0x38 bytes at 0xD8 and 0x110): base (ns), multiplier, shift, offset (ns), bound, source ID |

The SDK computes the steady clock as the time since boot plus ((time since boot − base) × multiplier >> shift) + offset. The adjustment is capped at the bound when the multiplier is not negative, and kept at least the bound when it is negative. The SDK rejects a source ID of zero. A system clock's time is its offset plus the steady clock's time, and its source ID must match the steady clock's. This layout was read from the SDK's time library in the sample.

### The romfs

A NUP container stores the romfs as separate files, but a game reads its romfs as one RomFS image. `siicl.romfs` builds the image's header and tables in memory, in the standard layout (see switchbrew's RomFS page), and reads file data from the container only when the game asks for it.

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
6. For the thread that ended the run, its registers X0 to X30, SP and PC, and a backtrace: the stop address, the link register if it points into a module, then the return addresses found by following the frame pointers (X29) up the stack. Each address the modules' dynamic symbol tables name gets the function's name (`siicl.symbols`, demangled with `c++filt` from `aarch64-linux-gnu-binutils`).
7. Every thread with its state and where it is.

   A line that repeats (the same SVC, arguments and result from the same thread) is written once, followed by a count, such as `(the line above 4 more times)`.
8. With `--stub-unknown`, the service commands and services answered without being implemented.
9. How many times the game called each SVC.

Service requests appear as SendSyncRequest lines with the service, the command and what it did, for example `SendSyncRequest(handle=0xb0012) fs IStorage Read(romfs, offset 0x0, size 0x50)`. The game's own log messages appear as `lm ILogger Log` lines.

The log describes the game's code, so it must stay outside the repository like any other output from a dump.
