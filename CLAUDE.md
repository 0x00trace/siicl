# SIICL

Research tools toward a Switch 2 compatibility layer (SIICL) that runs the user's own Switch 2 game dumps on Linux. Today the repo holds readers for the dump formats and a scanner that measures what a game's code needs from the CPU and the OS.

## Rule: no game content in this repository

Never commit game files or anything taken from them. This covers dumps, extracted executables and NPDM files, romfs files, icons, shaders, disassembly or decompiled code, strings, screenshots, video or audio of a game, and reports generated from a dump.

- Keep these outside the repository. The tools write to `~/.cache/siicl` by default and refuse an output folder inside the repository.
- This includes saved output of `siicl vzip disasm`. The hook refuses `.sass` and `.disasm` files, but can't recognize disassembly saved under other names.
- Tests use synthetic data that the test builds itself. Never point a test at a real dump.
- The pre-commit hook in `.githooks/` refuses such files. Enable it in each clone with `git config core.hooksPath .githooks`.
- Describe formats in `docs/` in general terms. Don't name the game a sample came from, and don't copy its file names or values.

## Commands

- Tests: `uv run --offline pytest` (the system `pytest` also works: `python -m pytest`)
- Lint and format: `uvx ruff check` and `uvx ruff format`
- Describe a dump: `uv run siicl info <file.nup>`
- Scan a dump: `uv run siicl scan <file.nup>`
- Read the game data archive: `uv run siicl vzip stats|list|paths|extract|verify|grep|shaders|disasm|buffers|check-decoder <file.nup>`

## Environment

- Python 3.14. The runtime uses only the standard library: `compression.zstd`, and `ctypes` for the system liblz4 (pacman package `lz4`).
- The scanner runs `aarch64-linux-gnu-objdump` (pacman package `aarch64-linux-gnu-binutils`). Tests also use `aarch64-linux-gnu-as` from the same package.
- `nvdisasm` comes from the pacman package `cuda` and is in `/opt/cuda/bin`. The shader tests that need it skip when it is missing.
- Dumps live in `~/Games/game_dumps/switch_2`.
