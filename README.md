# SIICL

SIICL is a research project toward a Switch 2 compatibility layer for Linux, to run your own Switch 2 game dumps. It is at the first step: measuring what a game needs before any of the layer is designed.

## What it does today

- `siicl info` reads a NUP dump container and lists its files and metadata. It checks the table of contents against its SHA-256 hash.
- `siicl scan` extracts the executables from a NUP container in memory and reports:
  - which AArch64 architecture features the code uses (for example Armv8.1 atomics, Armv8.2 half-precision arithmetic, Armv8.3 pointer authentication), with counts
  - which supervisor calls (SVCs) the code makes, and which ones the process descriptor (NPDM) allows
  - which system registers the code reads and writes
  - the services the process may connect to

Reports go to `~/.cache/siicl/<title id>/` as `report.md` and `report.json`.

## Requirements

- Arch Linux packages: `python` (3.14), `uv`, `lz4`, `aarch64-linux-gnu-binutils`
- No packages from PyPI at runtime

## Usage

```sh
uv run siicl info  ~/Games/game_dumps/switch_2/game.nup
uv run siicl scan  ~/Games/game_dumps/switch_2/game.nup
uv run siicl scan  ~/Games/game_dumps/switch_2/game.nup --module main --jobs 8
```

## No game content in this repository

Game files, anything extracted from them, and screenshots or recordings of a game must never be committed. See [CLAUDE.md](CLAUDE.md) for the full rule. After cloning, enable the pre-commit hook that enforces it:

```sh
git config core.hooksPath .githooks
```

## Formats

- [docs/nup-format.md](docs/nup-format.md): the NUP container, as worked out from a sample
