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
- A name is the FNV-1a 64-bit hash of the asset's path, written in lowercase hexadecimal without leading zeros. The path is hashed without its leading `/`, and the letters A-Z are lowercased first, as the game's lookup does. For example, `/Data/Dir/File.png.import` and `data/dir/file.png.import` give the same name.
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

## Resources and their GPU data

A resource (`VIPR`) file holds one image in the engine's binary format. When the image has GPU data, such as compiled shaders, that data is not inside the file: each piece is a separate archive entry, which the image calls an external buffer.

- The 32-bit word at offset 0x18 of the resource declares the GPU buffer type: bits 26-29 are the graphics backend (12 is NVN gen2) and bits 22-24 the kind (1 buffer, 2 texture).
- The image lists each external buffer in a record of 32 bytes: the buffer count (a 32-bit value followed by the tag 0xBFFF0000), the buffer's index (a 32-bit value followed by the tag 0xCFFF0000), 8 zero bytes, and the buffer's size as a 64-bit value.
- For compiled shaders, the buffers are stored as entries directly before their resource, in archive order. In the sample this pairs all 2,828 shader binaries with their resources: the sizes in the records match the preceding entries for 980 of the 990 resources, and the other 10 list buffers whose identical data was stored only once, before another resource. `siicl vzip buffers` does this matching.
- Texture and mesh data don't follow this layout. None of the texture records match the entries before their resource.
- The archive names of the buffer entries are not stored in any file in the archive, and none of the tested ways to build them from the resource's path, index or GUIDs gives them. The game computes them in code that this project has not traced yet.

## Contents of the sample

- Resource files start with the magic `VIPR`. They hold a resource type string and the paths of the asset they were made from. Texture and shader resources have type strings that end in `nvn-gen2`.
- The other files have no common header. They include data that the resource files refer to, WAV audio, audio banks and JSON text.
