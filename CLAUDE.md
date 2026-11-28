# SIICL

Research tools toward a Switch 2 compatibility layer (SIICL) that runs the user's own Switch 2 game dumps on Linux. The repo holds readers for the dump formats, a scanner that measures what a game's code needs from the CPU and the OS, a GPU shader translator, and the start of the layer itself (`siicl boot`). The current goal and its stages are in `docs/roadmap.md`.

## Rule: no game content in this repository

Never commit game files or anything taken from them. This covers dumps, extracted executables and NPDM files, romfs files, icons, shaders, disassembly or decompiled code, strings, screenshots, video or audio of a game, and reports generated from a dump.

- Keep these outside the repository. The tools write to `~/.cache/siicl` by default and refuse an output folder inside the repository.
- This includes saved output of `siicl vzip disasm`. The hook refuses `.sass` and `.disasm` files, but can't recognize disassembly saved under other names.
- Tests use synthetic data that the test builds itself. Never point a test at a real dump.
- The pre-commit hook in `.githooks/` refuses such files. Enable it in each clone with `git config core.hooksPath .githooks`.
- Describe formats in `docs/` in general terms. Don't name the game a sample came from, and don't copy its file names or values.

## Commands

- Tests: `uv run --offline pytest`. The system `pytest` does not work: it can't import `unicorn` from the project's environment.
- Lint and format: `uvx ruff check` and `uvx ruff format`
- Describe a dump: `uv run siicl info <file.nup>`
- Scan a dump: `uv run siicl scan <file.nup>`
- Boot a dump until it stops, and write `boot.log`: `uv run siicl boot <file.nup>` (`--cpu dynarmic` runs it on dynarmic, about nine times faster)
- Read the game data archive: `uv run siicl vzip stats|list|paths|extract|verify|grep|shaders|disasm|buffers|check-decoder <file.nup>`
- Check the SPIR-V translation of compute programs on the GPU: `uv run siicl vzip check-gpu <file.nup>`
- Build the dynarmic backend's C library: `git submodule update --init native/dynarmic`, then `cmake -S native -B build/native -G Ninja -DCMAKE_BUILD_TYPE=Release` and `cmake --build build/native`

## Environment

- Python 3.14. The runtime uses the standard library (`compression.zstd`, and `ctypes` for the system liblz4, pacman package `lz4`) and one package from PyPI: `unicorn`, the CPU emulator that runs game code. Add no other runtime packages without asking.
- The scanner runs `aarch64-linux-gnu-objdump` (pacman package `aarch64-linux-gnu-binutils`). Tests also use `aarch64-linux-gnu-as` from the same package.
- `nvdisasm` comes from the pacman package `cuda` and is in `/opt/cuda/bin`. The shader tests that need it skip when it is missing.
- The GPU runner (`siicl.vk`) calls the system `libvulkan` through `ctypes` (pacman package `vulkan-icd-loader`). The tests that need a Vulkan GPU skip without one, and the SPIR-V tests use `spirv-val` (pacman package `spirv-tools`). `vulkan-headers`, `spirv-headers` and `vulkan-validation-layers` help when changing the runner or the translator; `vk.Gpu(validation=True)` turns the validation layer on.
- The dynarmic backend (`siicl.dynarmic`, which `siicl boot --cpu dynarmic` uses) loads `build/native/libsiicl_dynarmic.so`, or the file `$SIICL_DYNARMIC` names, through `ctypes`. It is built from `native/` with CMake (pacman packages `cmake`, `boost` and `ninja`) and the git submodule `native/dynarmic` (siicl's fork of dynarmic 6.7.0, `github.com/0x00trace/dynarmic`, branch `siicl`, licensed 0BSD). A change to dynarmic is a commit on that branch, pushed to the fork before the commit here that moves the submodule to it. The tests that need it skip when it isn't built. See `docs/cpu-backend.md`.
- Dumps live in `~/Games/game_dumps/switch_2`.
