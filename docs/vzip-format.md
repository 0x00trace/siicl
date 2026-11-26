# VZIP archive format

This describes the VZIP archive as worked out from one sample. All integers are little-endian. Offsets count from the start of the archive.

A VZIP archive is a ZIP64 file with an index in front of its first entry. Standard ZIP tools can list the entries but can't extract the compressed ones: every entry is marked as stored (method 0), while compressed data uses the archive's own LZ4 framing.

## Layout

| Part | Where |
|---|---|
| Index header | 0x0, 12 bytes |
| Index records | 0xC, 40 bytes each |
| String table | right after the records |
| Zero padding | up to the first ZIP local header |
| ZIP entries | local header, data, data descriptor, for each stored copy of data |
| ZIP central directory, ZIP64 end records, end record | at the end |

## Index header

| Offset | Size | Field |
|---|---|---|
| 0x0 | 4 | Magic `VZIP` |
| 0x4 | 4 | Number of records |
| 0x8 | 4 | String table size |

## Index record (40 bytes)

| Offset | Size | Field |
|---|---|---|
| 0x00 | 4 | Name offset in the string table |
| 0x04 | 4 | Unknown (0x65FFFFFF in every record of the sample) |
| 0x08 | 8 | Data offset, 16-byte aligned |
| 0x10 | 8 | Stored size |
| 0x18 | 8 | Size after decompression |
| 0x20 | 4 | Compression: 0 none, 1 linked LZ4 blocks |
| 0x24 | 4 | Key |

Records are sorted by key in ascending order, so a reader can find a key by binary search. Keys can repeat; the sample has four repeated keys.

## Names and keys

- The string table holds the names, each ending with a NUL byte.
- A name is the FNV-1a 64-bit hash of the asset's path, written in lowercase hexadecimal without leading zeros. The path is hashed without its leading `/`, for example `data/dir/file.png.import`.
- A key is the FNV-1a 32-bit hash of the name's text.
- Several records can point at the same data. These aliases share one stored copy, and only one of their names appears in the ZIP central directory.
- The archive does not store paths. Resource files name other assets by path, so hashing those path strings recovers many names (`siicl vzip paths`).

## ZIP part

- Local headers use method 0 and flag 0x0008 (sizes in a data descriptor). Their size fields are 0xFFFFFFFF, with the sizes in a ZIP64 extra field (tag 0x0001).
- Extra field 0xA11E holds a 16-bit alignment (16) followed by zero bytes, so that each entry's data starts on a 16-byte boundary.
- The CRC-32 in the central directory is computed over the decompressed data.
- A data descriptor follows the data: `PK\x07\x08`, CRC-32, then the stored size and the decompressed size as 8-byte values.
- The end record's counts and central directory offset are set to their maximum values, and the ZIP64 end records give the real ones.

## Compressed data

Compressed data is a sequence of blocks. Each block is a 4-byte length followed by a raw LZ4 block of that length. Each block decodes to 64 KiB, except the last.

The blocks are linked, as in LZ4's streaming mode: a block can copy from the 64 KiB of output that comes before it. So the blocks must be decoded in order into one buffer. The first block doesn't depend on any other, so the start of a file can be read by decoding the first block alone.

## Contents of the sample

- Resource files start with the magic `VIPR`. They hold a resource type string and the paths of the asset they were made from. Texture and shader resources have type strings that end in `nvn-gen2`.
- The other files have no common header. They include data that the resource files refer to, WAV audio, audio banks and JSON text.
