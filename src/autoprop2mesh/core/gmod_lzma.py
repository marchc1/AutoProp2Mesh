"""LZMA in the layout produced by GMod's util.Compress (Bootil).

Layout: 5 bytes of LZMA properties, an 8 byte little-endian uncompressed size,
then the raw LZMA1 stream. This is the classic ".lzma" (LZMA_Alone) layout,
except that Bootil always writes the real size and its decoder relies on it.
"""

import lzma
import struct

# Matches the props GMod itself writes (lc=3 lp=0 pb=2, 64 KiB dictionary).
_FILTERS = [{"id": lzma.FILTER_LZMA1, "preset": 6, "dict_size": 1 << 16}]


def compress(data: bytes) -> bytes:
    packed = lzma.compress(data, format=lzma.FORMAT_ALONE, filters=_FILTERS)
    # liblzma writes an "unknown size" marker (all 0xFF); Bootil's Extract()
    # sizes its output buffer from this field, so it must be the real size.
    return packed[:5] + struct.pack("<Q", len(data)) + packed[13:]


def decompress(data: bytes) -> bytes:
    if len(data) <= 13:
        raise ValueError("data too short to be GMod LZMA")
    size = struct.unpack_from("<Q", data, 5)[0]
    # GMod's decoder stops at `size` and ignores anything after it, so a
    # stream may carry an end marker despite a known size (ours do). liblzma
    # rejects that combination, so decode as "size unknown" and truncate.
    patched = data[:5] + b"\xff" * 8 + data[13:]
    dec = lzma.LZMADecompressor(format=lzma.FORMAT_ALONE)
    out = dec.decompress(patched, max_length=size)
    if len(out) < size:
        raise ValueError("LZMA stream ended early")
    return out[:size]
