# NUP container format

This describes the NUP container as worked out from one sample dump. Fields marked "unknown" kept the same value in that sample and need more samples to explain. All integers are little-endian. "Absolute" offsets count from the start of the file.

## Layout

| Part | Where |
|---|---|
| Header | 0x0, 0x80 bytes |
| Table of contents (TOC) | header field `toc_offset`, `toc_size` bytes |
| File data | from header field `data_offset`; each file at its own absolute offset, 16-byte aligned |
| Metadata block | TOC fields `metadata_offset` and `metadata_size`; at the end of the file in the sample |

## Header

| Offset | Size | Field |
|---|---|---|
| 0x00 | 3 | Magic `NUP` |
| 0x03 | 1 | Version (1) |
| 0x04 | 2 | Header size (0x80) |
| 0x06 | 2 | Unknown (1) |
| 0x08 | 8 | `toc_offset` (0x80) |
| 0x10 | 8 | `toc_size`, padded to 16 bytes |
| 0x18 | 32 | SHA-256 of the TOC (`toc_size` bytes) |
| 0x38 | 8 | `data_offset`: first byte after the TOC |
| 0x40 | 8 | Total file size |
| 0x48 | 4 | Unknown (1) |
| 0x4C | 4 | Zero |
| 0x50 | 16 | Unknown; looks random |
| 0x60 | 32 | Zero |

## Table of contents

Offsets inside the TOC are relative to the start of the TOC, except where marked absolute.

| Offset | Size | Field |
|---|---|---|
| 0x00 | 3 | Magic `TOC` |
| 0x03 | 1 | Version (1) |
| 0x04 | 4 | Unknown (1) |
| 0x08 | 4 | Number of directory records, including the root |
| 0x0C | 4 | Zero |
| 0x10 | 8 | Number of file entries |
| 0x18 | 8 | String table offset |
| 0x20 | 8 | String table size |
| 0x28 | 8 | Zero |
| 0x30 | 8 | Title ID |
| 0x38 | 4 | Zero |
| 0x3C | 4 | Unknown (1) |
| 0x40 | 4 | Unknown (2) |
| 0x44 | 4 | Unknown (0x20000; possibly a block size) |
| 0x48 | 4 | Unknown (0x101) |
| 0x4C | 4 | Zero |
| 0x50 | 8 | `metadata_offset` (absolute) |
| 0x58 | 8 | `metadata_size` |
| 0x60 | 16 | Zero |
| 0x70 | 0x18 each | Directory records |
| after them | 0x50 each | File entries |
| after them | 4 each | One u32 per node, counting up from 1 (unknown use) |
| string table offset | string table size | Names, not NUL-terminated |

### Directory record (0x18 bytes)

| Offset | Size | Field |
|---|---|---|
| 0x00 | 4 | Name offset in the string table |
| 0x04 | 4 | Name length (0 for the root) |
| 0x08 | 4 | Index of the first child node |
| 0x0C | 4 | Number of child nodes |
| 0x10 | 8 | Zero |

The first record is the root. Child nodes are numbered with the directories first: node `i` is directory record `i + 1` when `i` is less than the number of non-root directories, and file entry `i - (number of directories - 1)` otherwise.

The sample has two directories, `exefs` (the executables and their NPDM, as on Switch 1) and `romfs` (the game's data).

### File entry (0x50 bytes)

| Offset | Size | Field |
|---|---|---|
| 0x00 | 4 | Name offset in the string table |
| 0x04 | 4 | Name length |
| 0x08 | 8 | Size after decompression |
| 0x10 | 8 | Absolute offset of the stored data |
| 0x18 | 8 | Stored size |
| 0x20 | 32 | SHA-256 of the data after decompression |
| 0x40 | 4 | Compression: 0 none, 1 zstd (one frame) |
| 0x44 | 12 | Zero |

## Metadata block

A sequence of records with no padding between them:

| Offset | Size | Field |
|---|---|---|
| 0x0 | 4 | Kind |
| 0x4 | 4 | Size of the data |
| 0x8 | size | Data |

Kinds seen: 1 (a copy of the NPDM), 3 (a JPEG image, probably the icon), 5 and 8 (short text, probably the title name).

## Contents of the sample

- The data is decrypted. The executables are standard `NSO0` files and the process descriptor is a standard `META` NPDM, so Switch 1 tools for those formats apply.
- The romfs holds one large file stored without compression: a VZIP archive, described in [vzip-format.md](vzip-format.md).
