"""Valve Texture Format decoding to RGBA float32 (numpy).

Only the first frame / face / slice is decoded, at the largest mip level that
fits within `max_size`.
"""

import struct

import numpy as np

FLAG_ENVMAP = 0x4000

# format id -> (name, bytes per pixel or None for block compressed)
FORMATS = {
    0: ("RGBA8888", 4), 1: ("ABGR8888", 4), 2: ("RGB888", 3), 3: ("BGR888", 3),
    4: ("RGB565", 2), 5: ("I8", 1), 6: ("IA88", 2), 7: ("P8", 1), 8: ("A8", 1),
    9: ("RGB888_BLUESCREEN", 3), 10: ("BGR888_BLUESCREEN", 3), 11: ("ARGB8888", 4),
    12: ("BGRA8888", 4), 13: ("DXT1", None), 14: ("DXT3", None), 15: ("DXT5", None),
    16: ("BGRX8888", 4), 17: ("BGR565", 2), 18: ("BGRX5551", 2), 19: ("BGRA4444", 2),
    20: ("DXT1_ONEBITALPHA", None), 21: ("BGRA5551", 2), 22: ("UV88", 2),
    23: ("UVWQ8888", 4), 24: ("RGBA16161616F", 8), 25: ("RGBA16161616", 8),
    26: ("UVLX8888", 4),
}


class VTFError(Exception):
    pass


def _image_size(fmt, w, h):
    name, bpp = FORMATS[fmt]
    if bpp is None:
        block = 8 if name in ("DXT1", "DXT1_ONEBITALPHA") else 16
        return max(1, (w + 3) // 4) * max(1, (h + 3) // 4) * block
    return w * h * bpp


class TextureInfo:
    def __init__(self):
        self.width = self.height = 0
        self.flags = 0
        self.format = 0
        self.has_alpha = False


def _rgb565(v):
    r = ((v >> 11) & 31).astype(np.float32) / 31.0
    g = ((v >> 5) & 63).astype(np.float32) / 63.0
    b = (v & 31).astype(np.float32) / 31.0
    return r, g, b


def _decode_color_block(blocks, four_color_only):
    """blocks: (n, 8) uint8 DXT colour blocks -> (n, 16, 4) float32."""
    c0 = blocks[:, 0].astype(np.uint16) | (blocks[:, 1].astype(np.uint16) << 8)
    c1 = blocks[:, 2].astype(np.uint16) | (blocks[:, 3].astype(np.uint16) << 8)
    bits = (blocks[:, 4].astype(np.uint32) | (blocks[:, 5].astype(np.uint32) << 8)
            | (blocks[:, 6].astype(np.uint32) << 16) | (blocks[:, 7].astype(np.uint32) << 24))
    r0, g0, b0 = _rgb565(c0)
    r1, g1, b1 = _rgb565(c1)
    col0 = np.stack([r0, g0, b0, np.ones_like(r0)], -1)
    col1 = np.stack([r1, g1, b1, np.ones_like(r1)], -1)
    four = (c0 > c1) if not four_color_only else np.ones(len(c0), dtype=bool)
    four = four[:, None]
    col2 = np.where(four, (2 * col0 + col1) / 3, (col0 + col1) / 2)
    col3 = np.where(four, (col0 + 2 * col1) / 3, np.zeros_like(col0))
    palette = np.stack([col0, col1, col2, col3], 1)  # (n, 4, 4)
    shifts = np.arange(16, dtype=np.uint32) * 2
    idx = (bits[:, None] >> shifts) & 3  # (n, 16)
    return np.take_along_axis(palette, idx[:, :, None].astype(np.int64), 1)


def _decode_dxt(data, w, h, kind):
    bw, bh = max(1, (w + 3) // 4), max(1, (h + 3) // 4)
    block = 8 if kind == "DXT1" else 16
    raw = np.frombuffer(data, dtype=np.uint8, count=bw * bh * block).reshape(-1, block)
    if kind == "DXT1":
        px = _decode_color_block(raw, False)
    else:
        px = _decode_color_block(raw[:, 8:16], True)
        if kind == "DXT3":
            a = raw[:, 0:8]
            lo = (a & 0x0F).astype(np.float32) / 15.0
            hi = (a >> 4).astype(np.float32) / 15.0
            px[:, :, 3] = np.stack([lo, hi], -1).reshape(-1, 16)
        else:  # DXT5
            a0 = raw[:, 0].astype(np.float32) / 255.0
            a1 = raw[:, 1].astype(np.float32) / 255.0
            bits = np.zeros(len(raw), dtype=np.uint64)
            for i in range(6):
                bits |= raw[:, 2 + i].astype(np.uint64) << np.uint64(8 * i)
            idx = (bits[:, None] >> (np.arange(16, dtype=np.uint64) * np.uint64(3))) & np.uint64(7)
            idx = idx.astype(np.int64)
            eight = (a0 > a1)[:, None]
            k = np.arange(8, dtype=np.float32)[None, :]
            # Entries 2..7 interpolate: ((8-k)*a0 + (k-1)*a1) / 7
            pal8 = (a0[:, None] * (8 - k) + a1[:, None] * (k - 1)) / 7.0
            pal8[:, 0], pal8[:, 1] = a0, a1
            # Entries 2..5 interpolate: ((6-k)*a0 + (k-1)*a1) / 5; 6 = 0, 7 = 1
            k6 = np.clip(k, 0, 6)
            pal6 = (a0[:, None] * (6 - k6) + a1[:, None] * (k6 - 1)) / 5.0
            pal6[:, 0], pal6[:, 1] = a0, a1
            pal6[:, 6], pal6[:, 7] = 0.0, 1.0
            pal = np.where(eight, pal8, pal6)
            px[:, :, 3] = np.take_along_axis(pal, idx, 1)
    img = px.reshape(bh, bw, 4, 4, 4).transpose(0, 2, 1, 3, 4).reshape(bh * 4, bw * 4, 4)
    return img[:h, :w]


def _decode_uncompressed(data, w, h, name):
    if name in ("RGBA16161616F",):
        v = np.frombuffer(data, dtype=np.float16, count=w * h * 4).reshape(h, w, 4).astype(np.float32)
        return np.clip(v, 0.0, 1.0)
    if name == "RGBA16161616":
        return np.frombuffer(data, dtype=np.uint16, count=w * h * 4).reshape(h, w, 4).astype(np.float32) / 65535.0
    if name in ("RGB565", "BGR565", "BGRX5551", "BGRA5551", "BGRA4444"):
        v = np.frombuffer(data, dtype="<u2", count=w * h).reshape(h, w)
        if name == "RGB565":
            b, g, r = _rgb565(v)
            a = np.ones_like(r)
        elif name == "BGR565":
            r, g, b = _rgb565(v)
            a = np.ones_like(r)
        elif name == "BGRA4444":
            b = (v & 15) / 15.0
            g = ((v >> 4) & 15) / 15.0
            r = ((v >> 8) & 15) / 15.0
            a = ((v >> 12) & 15) / 15.0
        else:
            b = (v & 31) / 31.0
            g = ((v >> 5) & 31) / 31.0
            r = ((v >> 10) & 31) / 31.0
            a = ((v >> 15) & 1).astype(np.float32) if name == "BGRA5551" else np.ones(v.shape)
        return np.stack([r, g, b, a], -1).astype(np.float32)

    bpp = FORMATS_BY_NAME[name]
    px = np.frombuffer(data, dtype=np.uint8, count=w * h * bpp).reshape(h, w, bpp).astype(np.float32) / 255.0
    one = np.ones((h, w), dtype=np.float32)
    channels = {
        "RGBA8888": lambda: (px[..., 0], px[..., 1], px[..., 2], px[..., 3]),
        "ABGR8888": lambda: (px[..., 3], px[..., 2], px[..., 1], px[..., 0]),
        "RGB888": lambda: (px[..., 0], px[..., 1], px[..., 2], one),
        "BGR888": lambda: (px[..., 2], px[..., 1], px[..., 0], one),
        "RGB888_BLUESCREEN": lambda: (px[..., 0], px[..., 1], px[..., 2], one),
        "BGR888_BLUESCREEN": lambda: (px[..., 2], px[..., 1], px[..., 0], one),
        "ARGB8888": lambda: (px[..., 1], px[..., 2], px[..., 3], px[..., 0]),
        "BGRA8888": lambda: (px[..., 2], px[..., 1], px[..., 0], px[..., 3]),
        "BGRX8888": lambda: (px[..., 2], px[..., 1], px[..., 0], one),
        "UVWQ8888": lambda: (px[..., 0], px[..., 1], px[..., 2], px[..., 3]),
        "UVLX8888": lambda: (px[..., 0], px[..., 1], px[..., 2], px[..., 3]),
        "I8": lambda: (px[..., 0], px[..., 0], px[..., 0], one),
        "P8": lambda: (px[..., 0], px[..., 0], px[..., 0], one),
        "A8": lambda: (one, one, one, px[..., 0]),
        "IA88": lambda: (px[..., 0], px[..., 0], px[..., 0], px[..., 1]),
        "UV88": lambda: (px[..., 0], px[..., 1], one * 0.0, one),
    }
    fn = channels.get(name)
    if fn is None:
        raise VTFError("unsupported format %s" % name)
    return np.stack(fn(), -1)


FORMATS_BY_NAME = {name: bpp for name, bpp in FORMATS.values()}


def decode(data, max_size=1024):
    """Returns (TextureInfo, rgba) where rgba is (h, w, 4) float32, top row first."""
    if data[:4] != b"VTF\0":
        raise VTFError("not a VTF file")
    major, minor, header_size = struct.unpack_from("<3I", data, 4)
    width, height, flags, frames, first_frame = struct.unpack_from("<HHIHH", data, 16)
    hi_fmt = struct.unpack_from("<i", data, 52)[0]
    mip_count = data[56]
    lo_fmt = struct.unpack_from("<i", data, 57)[0]
    lo_w, lo_h = data[61], data[62]
    depth = struct.unpack_from("<H", data, 63)[0] if (major, minor) >= (7, 2) else 1
    depth = max(depth, 1)

    if hi_fmt not in FORMATS:
        raise VTFError("unknown image format %d" % hi_fmt)

    faces = 1
    if flags & FLAG_ENVMAP:
        faces = 7 if (minor < 5 and first_frame != 0xFFFF) else 6
    frames = max(frames, 1)
    mip_count = max(mip_count, 1)

    if (major, minor) >= (7, 3):
        num_resources = struct.unpack_from("<I", data, 68)[0]
        image_offset = None
        for i in range(num_resources):
            tag, _flags, value = struct.unpack_from("<3sBI", data, 80 + i * 8)
            if tag == b"\x30\x00\x00":
                image_offset = value
        if image_offset is None:
            raise VTFError("VTF has no high resolution image resource")
    else:
        image_offset = header_size
        if lo_fmt in FORMATS and lo_w and lo_h:
            image_offset += _image_size(lo_fmt, lo_w, lo_h)

    # Pick the largest mip within max_size.
    mip = 0
    while mip < mip_count - 1 and max(width >> mip, height >> mip) > max_size:
        mip += 1

    # Mips are stored smallest first; each holds frames * faces * slices.
    offset = image_offset
    for m in range(mip_count - 1, mip, -1):
        mw, mh = max(1, width >> m), max(1, height >> m)
        md = max(1, depth >> m)
        offset += _image_size(hi_fmt, mw, mh) * frames * faces * md

    w, h = max(1, width >> mip), max(1, height >> mip)
    size = _image_size(hi_fmt, w, h)
    chunk = data[offset:offset + size]
    if len(chunk) < size:
        raise VTFError("VTF image data is truncated")

    name = FORMATS[hi_fmt][0]
    if name.startswith("DXT"):
        rgba = _decode_dxt(chunk, w, h, "DXT1" if name.startswith("DXT1") else name)
    else:
        rgba = _decode_uncompressed(chunk, w, h, name)

    info = TextureInfo()
    info.width, info.height, info.flags, info.format = w, h, flags, hi_fmt
    info.full_width, info.full_height = width, height
    info.has_alpha = bool(flags & (0x1000 | 0x2000)) or name in ("DXT3", "DXT5", "DXT1_ONEBITALPHA")
    return info, np.ascontiguousarray(rgba, dtype=np.float32)
