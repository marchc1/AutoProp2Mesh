"""AdvDupe2 file codec, revision 5 (lua/advdupe2/sh_codec.lua).

A dupe file is:  "AD2F" <rev byte> "\\n" <info block> "\\n" <util.Compress(body)>
The info block is `key \\1 value \\1 ...` terminated by `\\2`, and the body is a
tagged binary serialization of a Lua table.

Lua values map to Python as:
    table  -> dict (or list, which encodes as a Lua array starting at 1)
    number -> int / float      boolean -> bool      string -> str / bytes
    Vector -> Vector           Angle   -> Angle
"""

import struct

from . import gmod_lzma

REVISION = 5
INFO_CHECK = "\r\n\t\n"


class Vector(tuple):
    __slots__ = ()

    def __new__(cls, x=0.0, y=0.0, z=0.0):
        return super().__new__(cls, (float(x), float(y), float(z)))

    def __repr__(self):
        return "Vector(%g, %g, %g)" % self


class Angle(tuple):
    """Source QAngle: (pitch, yaw, roll) in degrees."""
    __slots__ = ()

    def __new__(cls, p=0.0, y=0.0, r=0.0):
        return super().__new__(cls, (float(p), float(y), float(r)))

    def __repr__(self):
        return "Angle(%g, %g, %g)" % self


# --------------------------------------------------------------------------
# Encoding
# --------------------------------------------------------------------------

def _is_lua_array(d: dict) -> bool:
    n = len(d)
    if n == 0:
        return True
    for i in range(1, n + 1):
        if i not in d:
            return False
    return True


def _write(out: bytearray, obj):
    # bool must be tested before int (bool is an int subclass).
    if isinstance(obj, bool):
        out.append(253 if obj else 252)
    elif isinstance(obj, Vector):
        out.append(250)
        out += struct.pack("<3d", *obj)
    elif isinstance(obj, Angle):
        out.append(249)
        out += struct.pack("<3d", *obj)
    elif isinstance(obj, (int, float)):
        out.append(251)
        out += struct.pack("<d", float(obj))
    elif isinstance(obj, (str, bytes, bytearray)):
        data = obj.encode("utf-8") if isinstance(obj, str) else bytes(obj)
        n = len(data)
        if n < 246:
            out.append(n)
        else:
            out.append(248)
            out += struct.pack("<I", n)
        out += data
    elif isinstance(obj, (list, tuple)):
        out.append(254)
        for v in obj:
            _write(out, v)
        out.append(246)
    elif isinstance(obj, dict):
        if _is_lua_array(obj):
            out.append(254)
            for i in range(1, len(obj) + 1):
                _write(out, obj[i])
        else:
            out.append(255)
            for k, v in obj.items():
                if v is None:
                    continue
                _write(out, k)
                _write(out, v)
        out.append(246)
    else:
        raise TypeError("cannot serialize %r for AdvDupe2" % type(obj))


def serialize(obj) -> bytes:
    out = bytearray()
    _write(out, obj)
    return bytes(out)


def _make_info(info: dict) -> bytes:
    parts = []
    for k, v in info.items():
        k = str(k)
        v = str(v)
        if any(c in k + v for c in "\1\2"):
            raise ValueError("info fields may not contain \\1 or \\2")
        parts.append(k + "\1" + v + "\1")
    return ("".join(parts) + "\2").encode("utf-8")


def encode(dupe: dict, info: dict) -> bytes:
    body = gmod_lzma.compress(serialize(dupe))
    info = dict(info)
    info["check"] = INFO_CHECK
    info["size"] = len(body)
    return b"AD2F" + bytes([REVISION]) + b"\n" + _make_info(info) + b"\n" + body


# --------------------------------------------------------------------------
# Decoding (used for tests and for inspecting existing dupes)
# --------------------------------------------------------------------------

class TableKey:
    """Wraps a decoded table that was used as a table key (identity hashed)."""
    __slots__ = ("table",)

    def __init__(self, table):
        self.table = table


class _Reader:
    _END = object()

    def __init__(self, data: bytes):
        self.data = data
        self.pos = 0
        self.tables = []

    def _byte(self):
        b = self.data[self.pos]
        self.pos += 1
        return b

    def _unpack(self, fmt, size):
        v = struct.unpack_from(fmt, self.data, self.pos)
        self.pos += size
        return v

    def read(self):
        t = self._byte()
        if t == 255:
            tbl = {}
            self.tables.append(tbl)
            while True:
                k = self.read()
                if k is self._END:
                    return tbl
                if isinstance(k, dict):  # Lua allows tables as keys
                    k = TableKey(k)
                tbl[k] = self.read()
        if t == 254:
            tbl = {}
            self.tables.append(tbl)
            i = 1
            while True:
                v = self.read()
                if v is self._END:
                    return tbl
                tbl[i] = v
                i += 1
        if t == 253:
            return True
        if t == 252:
            return False
        if t == 251:
            v = self._unpack("<d", 8)[0]
            return int(v) if v.is_integer() else v
        if t == 250:
            return Vector(*self._unpack("<3d", 24))
        if t == 249:
            return Angle(*self._unpack("<3d", 24))
        if t == 248:
            n = self._unpack("<I", 4)[0]
            s = self.data[self.pos:self.pos + n]
            self.pos += n
            return s
        if t == 247:
            return self.tables[self._unpack("<H", 2)[0] - 1]
        if t == 246:
            return self._END
        s = self.data[self.pos:self.pos + t]
        self.pos += t
        return s


def _bytes_to_str(obj):
    """Convert decoded byte strings to str where they are valid UTF-8."""
    if isinstance(obj, bytes):
        try:
            return obj.decode("utf-8")
        except UnicodeDecodeError:
            return obj
    if isinstance(obj, dict):
        return {(k if isinstance(k, TableKey) else _bytes_to_str(k)): _bytes_to_str(v)
                for k, v in obj.items()}
    return obj


def decode(filedata: bytes, text_strings=True):
    """Returns (dupe_table, info_dict). Only revisions 4/5 tags are handled."""
    if filedata[:4] != b"AD2F":
        raise ValueError("not an AdvDupe2 file")
    rev = filedata[4]
    if rev != REVISION:
        raise ValueError("unsupported AdvDupe2 revision %d" % rev)
    rest = filedata[6:]
    end = rest.index(b"\2")
    info = {}
    fields = rest[:end].split(b"\1")
    for i in range(0, len(fields) - 1, 2):
        info[fields[i].decode("utf-8", "replace")] = fields[i + 1].decode("utf-8", "replace")
    body = gmod_lzma.decompress(rest[end + 2:])
    tbl = _Reader(body).read()
    return (_bytes_to_str(tbl) if text_strings else tbl), info
