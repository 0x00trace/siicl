"""LZ4 block compression through the system liblz4 (Arch package `lz4`).

NSO segments are raw LZ4 blocks, not LZ4 frames, so the `lz4` command cannot
read them. ctypes keeps the project free of PyPI dependencies.
"""

from __future__ import annotations

import ctypes
import ctypes.util
from collections.abc import Iterable

_INT_MAX = 0x7FFFFFFF
_lib: ctypes.CDLL | None = None


def _load() -> ctypes.CDLL:
    global _lib
    if _lib is None:
        lib = ctypes.CDLL(ctypes.util.find_library("lz4") or "liblz4.so.1")
        lib.LZ4_decompress_safe.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
        lib.LZ4_decompress_safe.restype = ctypes.c_int
        lib.LZ4_compressBound.argtypes = [ctypes.c_int]
        lib.LZ4_compressBound.restype = ctypes.c_int
        lib.LZ4_compress_default.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
        lib.LZ4_compress_default.restype = ctypes.c_int
        lib.LZ4_decompress_safe_usingDict.argtypes = [
            ctypes.c_char_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        lib.LZ4_decompress_safe_usingDict.restype = ctypes.c_int
        lib.LZ4_decompress_safe_partial.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        lib.LZ4_decompress_safe_partial.restype = ctypes.c_int
        lib.LZ4_createStream.restype = ctypes.c_void_p
        lib.LZ4_freeStream.argtypes = [ctypes.c_void_p]
        lib.LZ4_compress_fast_continue.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        lib.LZ4_compress_fast_continue.restype = ctypes.c_int
        _lib = lib
    return _lib


def decompress_block(src: bytes, size: int) -> bytes:
    """Decompress one raw LZ4 block that must decode to exactly `size` bytes."""
    if size > _INT_MAX or len(src) > _INT_MAX:
        raise ValueError("LZ4 block is larger than liblz4 supports")
    dst = ctypes.create_string_buffer(size)
    n = _load().LZ4_decompress_safe(src, dst, len(src), size)
    if n != size:
        raise ValueError(f"LZ4 block decoded to {n} bytes, expected {size}")
    return dst.raw


def compress_block(src: bytes) -> bytes:
    """Compress `src` into one raw LZ4 block."""
    lib = _load()
    bound = lib.LZ4_compressBound(len(src))
    dst = ctypes.create_string_buffer(bound)
    n = lib.LZ4_compress_default(src, dst, len(src), bound)
    if n <= 0:
        raise ValueError("LZ4 compression failed")
    return dst.raw[:n]


WINDOW = 0x10000  # LZ4 matches reach back at most 64 KiB


def decompress_linked(blocks: Iterable[bytes], size: int, block_size: int = 0x10000) -> bytes:
    """Decompress LZ4 blocks written in linked mode into one buffer of `size` bytes.

    Each block decodes to at most `block_size` bytes and may copy from up to
    64 KiB of the output before it, so the blocks must be decoded in order.
    """
    if size > _INT_MAX:
        raise ValueError("output is larger than liblz4 supports")
    lib = _load()
    out = ctypes.create_string_buffer(size)
    base = ctypes.addressof(out)
    pos = 0
    for block in blocks:
        if pos >= size:
            raise ValueError("more LZ4 blocks than the output size allows")
        window = min(pos, WINDOW)
        got = lib.LZ4_decompress_safe_usingDict(
            block, base + pos, len(block), min(block_size, size - pos), base + pos - window, window
        )
        if got <= 0:
            raise ValueError(f"LZ4 block at output offset {pos} is damaged")
        pos += got
    if pos != size:
        raise ValueError(f"LZ4 blocks decoded to {pos} bytes, expected {size}")
    return out.raw


def decompress_head(block: bytes, n: int) -> bytes:
    """Decode only the first `n` bytes of an LZ4 block."""
    dst = ctypes.create_string_buffer(n)
    got = _load().LZ4_decompress_safe_partial(block, dst, len(block), n, n)
    if got < 0:
        raise ValueError("LZ4 block is damaged")
    return dst.raw[:got]


def compress_linked(data: bytes, block_size: int = 0x10000) -> list[bytes]:
    """Compress `data` into LZ4 blocks in linked mode (the inverse of decompress_linked)."""
    lib = _load()
    src = ctypes.create_string_buffer(data, len(data))  # stays in place, so earlier blocks stay usable as history
    base = ctypes.addressof(src)
    bound = lib.LZ4_compressBound(block_size)
    dst = ctypes.create_string_buffer(bound)
    stream = lib.LZ4_createStream()
    try:
        blocks = []
        for pos in range(0, len(data), block_size):
            n = min(block_size, len(data) - pos)
            got = lib.LZ4_compress_fast_continue(stream, base + pos, dst, n, bound, 1)
            if got <= 0:
                raise ValueError("LZ4 compression failed")
            blocks.append(dst.raw[:got])
        return blocks
    finally:
        lib.LZ4_freeStream(stream)
