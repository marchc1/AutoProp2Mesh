"""Readers for Valve VPK packs and Garry's Mod GMA addons.

Both expose:
    entries: dict[lowercase_path] -> location tuple
    read(path) -> bytes | None
"""

import lzma
import struct
import threading
from collections import OrderedDict

NUL = bytes(1)

VPK_SIGNATURE = 0x55AA1234


class VPK:
    """A *_dir.vpk pack. `entries` may be passed in from a cached index."""

    def __init__(self, dir_path, entries=None):
        self.path = dir_path
        self._archives = {}
        self._lock = threading.Lock()
        base = dir_path[:-len("_dir.vpk")] if dir_path.lower().endswith("_dir.vpk") else dir_path[:-4]
        self._archive_base = base
        with open(dir_path, "rb") as f:
            sig, version, tree_size = struct.unpack("<III", f.read(12))
            if sig != VPK_SIGNATURE:
                raise ValueError("%s is not a VPK" % dir_path)
            header_size = 12 if version == 1 else 28
            self._data_start = header_size + tree_size
            if entries is None:
                f.seek(header_size)
                entries = self._parse_tree(f.read(tree_size))
        self.entries = entries

    @staticmethod
    def _parse_tree(tree):
        entries = {}
        pos = 0

        def cstr():
            nonlocal pos
            end = tree.index(b"\0", pos)
            s = tree[pos:end].decode("utf-8", "replace")
            pos = end + 1
            return s

        unpack = struct.unpack_from
        while True:
            ext = cstr()
            if not ext:
                break
            while True:
                directory = cstr()
                if not directory:
                    break
                while True:
                    name = cstr()
                    if not name:
                        break
                    _crc, preload, archive, offset, length, _term = unpack("<IHHIIH", tree, pos)
                    pos += 18
                    preload_data = tree[pos:pos + preload] if preload else b""
                    pos += preload
                    full = name if ext == " " else name + "." + ext
                    if directory != " ":
                        full = directory + "/" + full
                    entries[full.lower()] = (archive, offset, length, preload_data)
        return entries

    def read(self, path):
        entry = self.entries.get(path)
        if entry is None:
            return None
        archive, offset, length, preload = entry
        if length == 0:
            return preload
        if archive == 0x7FFF:
            file_path, offset = self.path, offset + self._data_start
        else:
            file_path = "%s_%03d.vpk" % (self._archive_base, archive)
        with self._lock:
            fh = self._archives.get(file_path)
            if fh is None:
                fh = self._archives[file_path] = open(file_path, "rb")
            fh.seek(offset)
            return preload + fh.read(length)

    def close(self):
        for fh in self._archives.values():
            fh.close()
        self._archives.clear()


class _Truncated(Exception):
    pass


def parse_gma_index(buf):
    """Parses a GMA header + file table from the start of `buf`.
    Returns {path: (offset, size)} with offsets into the uncompressed GMA.
    Raises _Truncated when `buf` ends before the file table does."""
    n = len(buf)
    if n < 5:
        raise _Truncated()
    if buf[:4] != b"GMAD":
        raise ValueError("not a GMA")
    version = buf[4]
    pos = 5 + 16  # steamid, timestamp

    def cstr():
        nonlocal pos
        end = buf.find(NUL, pos)
        if end < 0:
            raise _Truncated()
        s = buf[pos:end]
        pos = end + 1
        return s

    if version > 1:
        while cstr():
            pass
    cstr()  # title
    cstr()  # description
    cstr()  # author
    pos += 4  # addon version
    files = []
    unpack = struct.unpack_from
    while True:
        if pos + 4 > n:
            raise _Truncated()
        (num,) = unpack("<I", buf, pos)
        pos += 4
        if num == 0:
            break
        name = cstr().decode("utf-8", "replace").lower().replace("\\", "/")
        if pos + 12 > n:
            raise _Truncated()
        size, _crc = unpack("<qI", buf, pos)
        pos += 12
        files.append((name, size))
    entries = {}
    offset = pos
    for name, size in files:
        entries[name] = (offset, size)
        offset += size
    return entries


class GMA:
    """A .gma addon. `entries` maps path -> (offset, size)."""

    def __init__(self, path, entries=None):
        self.path = path
        self._fh = None
        self._lock = threading.Lock()
        self.entries = entries if entries is not None else self.read_index(path)

    @staticmethod
    def read_index(path):
        chunk = 1 << 20
        with open(path, "rb") as f:
            buf = f.read(chunk)
            while True:
                try:
                    return parse_gma_index(buf)
                except _Truncated:
                    more = f.read(len(buf))
                    if not more:
                        raise ValueError("truncated GMA")
                    buf += more

    def read(self, path):
        entry = self.entries.get(path)
        if entry is None:
            return None
        offset, size = entry
        with self._lock:
            if self._fh is None:
                self._fh = open(self.path, "rb")
            self._fh.seek(offset)
            return self._fh.read(size)

    def close(self):
        if self._fh:
            self._fh.close()
            self._fh = None


class _BinStream:
    """A resumable decompression of one legacy .bin."""
    __slots__ = ("dec", "buf", "done")

    def __init__(self, path):
        with open(path, "rb") as f:
            data = f.read()
        self.dec = lzma.LZMADecompressor(format=lzma.FORMAT_ALONE)
        self.buf = bytearray(self.dec.decompress(data, max_length=0))
        self.done = False

    def extend_to(self, length):
        while len(self.buf) < length and not self.done:
            if self.dec.eof:
                self.done = True
                break
            chunk = self.dec.decompress(b"", max_length=max(length - len(self.buf), 1 << 20))
            if not chunk and self.dec.needs_input:
                self.done = True
                break
            self.buf += chunk


class LegacyBin:
    """A legacy workshop item: an LZMA-compressed GMA (*.bin).

    Nothing is extracted to disk. Indexing decompresses only the head of the
    stream. Reads continue a per-file decompression from wherever the last
    read stopped, and recently used streams stay in memory (bounded)."""

    _streams = OrderedDict()  # path -> _BinStream (shared across instances)
    CACHE_LIMIT = 256 * 1024 * 1024
    _lock = threading.Lock()

    def __init__(self, path, entries=None):
        self.path = path
        self.entries = entries if entries is not None else self.read_index(path)

    @classmethod
    def read_index(cls, path):
        stream = _BinStream(path)
        want = 512 * 1024
        while True:
            stream.extend_to(want)
            try:
                return parse_gma_index(bytes(stream.buf))
            except _Truncated:
                if stream.done:
                    raise ValueError("truncated GMA")
                want *= 4

    def read(self, path):
        entry = self.entries.get(path)
        if entry is None:
            return None
        offset, size = entry
        end = offset + size
        cls = LegacyBin
        with cls._lock:
            stream = cls._streams.get(self.path)
            if stream is None:
                stream = cls._streams[self.path] = _BinStream(self.path)
            cls._streams.move_to_end(self.path)
            stream.extend_to(end)
            data = bytes(stream.buf[offset:end])
            total = sum(len(s.buf) for s in cls._streams.values())
            while total > cls.CACHE_LIMIT and len(cls._streams) > 1:
                _, old = cls._streams.popitem(last=False)
                total -= len(old.buf)
        return data if len(data) == size else None

    def close(self):
        with LegacyBin._lock:
            LegacyBin._streams.pop(self.path, None)
