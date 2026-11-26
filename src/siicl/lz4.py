"""LZ4 block compression through the system liblz4 (Arch package `lz4`).

NSO segments are raw LZ4 blocks, not LZ4 frames, so the `lz4` command cannot
read them. ctypes keeps the project free of PyPI dependencies.
"""

from __future__ import annotations

import ctypes
import ctypes.util

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
