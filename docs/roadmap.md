# Roadmap

## Milestone 1: the game boots and presents its first frame

### What "done" means

`siicl boot <dump.nup>` does all of the following with an unmodified game dump:

1. It loads the game's executables into one guest address space and starts the game's main thread, as the system's loader does.
2. It runs the game's AArch64 code, handles its supervisor calls (SVCs), and answers the requests the game sends to the system services it opens.
3. It writes a boot log outside the repository, to `~/.cache/siicl/<title id>/boot.log`. The log lists every thread the game starts, every SVC, every service the game opens, every service command with its result, and the text the game logs. When the game reaches something siicl does not implement, the log ends with that call, the guest address, the module and offset, and a backtrace.
4. It runs the GPU commands the game submits on a Vulkan GPU and shows the first image the game queues for display in a window. It also saves that image as a PNG file outside the repository.

The image must come from the game's own GPU commands. If the first image the game presents is only a cleared screen, a window showing that clear color meets the milestone. The first image with the game's draws and textures in it, such as a logo, is milestone 2.

### Out of scope for milestone 1

- Speed. Any speed counts.
- Sound. The audio services accept data and discard it.
- Input. siicl reports one connected controller with no buttons pressed.
- Network. The network services report no connection.
- Save data. The game starts with empty save data, kept outside the repository.
- Games other than the sample.

### Design decisions

- **CPU.** The game's code runs in Unicorn, the CPU emulator built on QEMU (Python package `unicorn`), with its `max` CPU model. A test on synthetic code ran every architecture feature in the scan report: LSE atomics, LDAPR, pointer authentication, dot product, half precision, CRC32 and AES. Unicorn is the first runtime dependency outside the standard library.
- **Threads.** Guest threads run one at a time on one host thread. siicl switches threads at SVCs and after a fixed number of instructions. Runs are then repeatable, so boot logs from two runs can be compared.
- **Kernel and services.** siicl implements the kernel and the system services itself, in Python. It runs only the game's code, never the system's own modules.
- **Speed.** Not a goal for this milestone. Code that proves too slow can move to a compiled language afterwards, chosen by measuring real boots.
- **Tests.** As elsewhere in the repository, tests build their own synthetic inputs: NSO files, AArch64 programs assembled with `aarch64-linux-gnu-as`, IPC messages and GPU command buffers.

### Stages

Each stage is a stack of pull requests. Each ends at a point that both the tests and the sample can check.

Status: stages 1 to 3 are done, and stage 4 has started. On the sample, the game sets itself up through the applet, time, file system, account and network services, mounts its romfs, starts its engine, boots its graphics system through the graphics supervisor (whose commands siicl learned from the SDK's client code), opens the GPU driver service and the default display, and starts initializing input, where the run stops. See [boot.md](boot.md).

| Stage | Work | Done when |
|---|---|---|
| 1. Loader | Place the modules (rtld first, then main, the subsdk modules and sdk) in a 39-bit address space, with the right permissions and zeroed BSS. Set up the main thread: stack, thread-local storage and entry registers. | Tests load synthetic NSO files. The sample runs its first instructions in rtld and stops at its first SVC. |
| 2. Kernel | SVC dispatch. Memory: SetHeapSize, QueryMemory, MapMemory, SetMemoryPermission, SetMemoryAttribute, MapPhysicalMemory. Threads: create, start, exit, sleep, priority and core mask. Synchronization: ArbitrateLock, ArbitrateUnlock, WaitProcessWideKeyAtomic, SignalProcessWideKey, WaitForAddress, SignalToAddress, events and WaitSynchronization. GetInfo, GetSystemTick and the counter register. | Tests run a synthetic program for each SVC. The sample relocates itself and calls ConnectToNamedPort for the service manager. |
| 3. IPC and services | Read and build IPC messages (CMIF and TIPC, domains, buffers and handles). A service base class that logs every command it does not know and returns an error that the log names. Services in the order the game opens them, starting with sm, set, appletOE, fsp-srv (the game's romfs, read from the NUP container, and empty save data), time, hid, apm, acc, pctl, nifm and lm. | Tests send synthetic messages. The sample finishes its system setup and opens the graphics services. |
| 4. Graphics services | The Switch 2's graphics supervisor, which the SDK boots the graphics system through (`nn::gsv`, reached with the applet's GetGraphicsSupervisor; its commands are undocumented). nvdrv with the nvmap, nvhost-as-gpu, nvhost-gpu and nvhost-ctrl devices: GPU address spaces, channels and sync points. SVC 0x80, SetGpuMemoryAttribute. The display service and its binder interface: connect, request, dequeue and queue buffers. | The sample submits command buffers and queues its first image for display. The boot log lists the GPU methods it submitted. |
| 5. GPU commands and the first frame | Read command buffers (GPFIFO entries and methods) and GPU virtual memory. The engines and methods the first frame uses: render targets, clears, copies, inline memory writes and the 3D engine's draw state. Translate the graphics shaders the first frame uses, building on the compute translation. Show the presented image in a window and save it as a PNG. | The window shows the first image the sample presents. |

### Risks

- The documentation the project relies on (switchbrew) covers the Switch 1 well, but Switch 2 changes such as new services, new SVCs and changed IPC commands are only partly documented. The boot log is how siicl finds them, and each one found is described in `docs/`.
- The Switch 2 boots its graphics system through a graphics supervisor whose commands no public documentation describes. siicl has to learn them from the SDK's client code in the sample, which takes longer than implementing a documented interface.
- The first frame may need graphics shaders with texture instructions, which the translator does not handle yet: 30 of the 48 compute programs in the sample stop at one. If so, stage 5 grows.
- The game may wait on threads, events or timing that one host thread does not reproduce. The scheduler stays simple at first and changes only when a boot log shows a hang.
