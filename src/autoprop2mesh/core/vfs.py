"""A read-only virtual filesystem that mirrors Garry's Mod's GAME search path.

Mount order (highest priority first), following garrysmod/gameinfo.txt and
the addon/depot systems:

    1. legacy addons          garrysmod/addons/<folder>/  (+ user extra paths)
    2. garrysmod/             loose files
    3. workshop addons        steamapps/workshop/content/4000/*/*.gma|*.bin
                              garrysmod/addons/*.gma
    4. garrysmod_dir.vpk
    5. sourceengine/hl2_*.vpk
    6. mounted games          cfg/mountdepots.txt, cfg/mount.cfg
    7. sourceengine/, garrysmod/download/, fallbacks_dir.vpk

Workshop enable/disable state lives in Steam, so every installed addon is
mounted.
"""

import os
import pickle
import time

from . import archives, keyvalues, steam

# folder -> (appid, sub directories to mount), from GameDepotSystem.cpp
MOUNTABLE_GAMES = {
    "hl2": (220, ["hl2", "episodic", "ep2", "lostcoast"]),
    "cstrike": (240, None), "tf": (440, None), "dod": (300, None),
    "hl2mp": (320, None), "hl1": (280, None), "hl1mp": (360, None),
    "zeno_clash": (22208, None), "portal": (400, None), "diprip": (17530, None),
    "zps": (17500, None), "pvkii": (17570, None), "dystopia": (17580, None),
    "insurgency": (17700, None), "ageofchivalry": (17510, None),
    "left4dead2": (550, None), "left4dead": (500, None), "portal2": (620, None),
    "swarm": (630, None), "nucleardawn": (17710, None), "dinodday": (70000, None),
    "csgo": (730, None), "berimbau": (225600, None), "infra": (251110, None),
    "fof": (265630, None), "thestanleyparable": (221910, None),
    "gstringv2": (1224600, None), "treason": (1786950, None), "bms": (362890, None),
}

_CACHE_VERSION = 2


class DirSource:
    """Loose files. Directory listings are cached (lower-case name -> real
    name), so lookups for files this folder lacks cost no disk access."""
    kind = "dir"

    def __init__(self, root):
        self.path = root
        self._listings = {}

    def _listing(self, real_dir):
        hit = self._listings.get(real_dir)
        if hit is None:
            try:
                hit = {n.lower(): n for n in os.listdir(real_dir)}
            except OSError:
                hit = {}
            self._listings[real_dir] = hit
        return hit

    def _resolve(self, path):
        real = self.path
        for part in path.split("/"):
            name = self._listing(real).get(part)
            if name is None:
                return None
            real = os.path.join(real, name)
        return real

    def read(self, path):
        full = self._resolve(path)
        if full and os.path.isfile(full):
            with open(full, "rb") as f:
                return f.read()
        return None

    def exists(self, path):
        full = self._resolve(path)
        return bool(full) and os.path.isfile(full)

    def invalidate(self):
        self._listings.clear()

    def list(self, prefix, ext):
        top = os.path.join(self.path, prefix)
        out = []
        if not os.path.isdir(top):
            return out
        base_len = len(self.path.rstrip("\\/")) + 1
        for dirpath, _dirs, files in os.walk(top):
            for fn in files:
                if fn.lower().endswith(ext):
                    out.append(os.path.join(dirpath, fn)[base_len:].replace("\\", "/").lower())
        return out


class ArchiveSource:
    def __init__(self, archive, kind):
        self.archive = archive
        self.kind = kind
        self.path = archive.path

    def read(self, path):
        return self.archive.read(path)

    def exists(self, path):
        return path in self.archive.entries

    def list(self, prefix, ext):
        return [p for p in self.archive.entries if p.startswith(prefix) and p.endswith(ext)]


class MountConfig:
    def __init__(self, gmod_dir=None, extra_paths=(), mount_workshop=True, mount_games=True):
        self.gmod_dir = gmod_dir
        self.extra_paths = [p for p in extra_paths if p]
        self.mount_workshop = mount_workshop
        self.mount_games = mount_games


def find_content_roots(path, depth=3):
    """A folder is a content root when it holds models/ or materials/.
    Otherwise its children are searched (e.g. legacy/<author>/<addon>)."""
    if not os.path.isdir(path):
        return []
    try:
        names = {n.lower() for n in os.listdir(path)}
    except OSError:
        return []
    if "models" in names or "materials" in names:
        return [path]
    if depth <= 0:
        return []
    out = []
    for n in sorted(os.listdir(path)):
        child = os.path.join(path, n)
        if os.path.isdir(child):
            out.extend(find_content_roots(child, depth - 1))
    return out


def _dir_vpks(directory):
    try:
        names = sorted(os.listdir(directory))
    except OSError:
        return []
    return [os.path.join(directory, n) for n in names if n.lower().endswith("_dir.vpk")]


class FileSystem:
    def __init__(self, config: MountConfig, cache_dir=None, log=print):
        self.config = config
        self.cache_dir = cache_dir
        self.log = log
        self.sources = []
        self.errors = []
        self._list_cache = {}
        self._index_cache = {}
        self._index_dirty = False
        self._build()

    # -- building -----------------------------------------------------------

    def _load_index_cache(self):
        if not self.cache_dir:
            return
        p = os.path.join(self.cache_dir, "archive_index.pickle")
        try:
            with open(p, "rb") as f:
                data = pickle.load(f)
            if data.get("version") == _CACHE_VERSION:
                self._index_cache = data["entries"]
        except Exception:
            self._index_cache = {}

    def _save_index_cache(self):
        if not self.cache_dir or not self._index_dirty:
            return
        os.makedirs(self.cache_dir, exist_ok=True)
        p = os.path.join(self.cache_dir, "archive_index.pickle")
        try:
            with open(p + ".tmp", "wb") as f:
                pickle.dump({"version": _CACHE_VERSION, "entries": self._index_cache}, f,
                            protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(p + ".tmp", p)
        except OSError as e:
            self.log("AutoProp2Mesh: could not write index cache: %s" % e)

    def _cached_index(self, path, loader):
        st = os.stat(path)
        key = os.path.normcase(path)
        hit = self._index_cache.get(key)
        if hit and hit[0] == st.st_size and hit[1] == st.st_mtime:
            return hit[2]
        entries = loader()
        self._index_cache[key] = (st.st_size, st.st_mtime, entries)
        self._index_dirty = True
        return entries

    def add_dir(self, path):
        if os.path.isdir(path):
            self.sources.append(DirSource(path))

    def add_vpk(self, path):
        try:
            entries = self._cached_index(path, lambda: archives.VPK(path).entries)
            self.sources.append(ArchiveSource(archives.VPK(path, entries), "vpk"))
        except Exception as e:
            self.errors.append("%s: %s" % (path, e))

    def add_gma(self, path):
        try:
            entries = self._cached_index(path, lambda: archives.GMA.read_index(path))
            self.sources.append(ArchiveSource(archives.GMA(path, entries), "gma"))
        except Exception as e:
            self.errors.append("%s: %s" % (path, e))

    def add_legacy_bin(self, path):
        # Some legacy items (saves, dupes) are not addons; they index as
        # empty and are skipped.
        def load():
            try:
                return archives.LegacyBin.read_index(path)
            except ValueError:
                return {}
        try:
            entries = self._cached_index(path, load)
            if entries:
                self.sources.append(ArchiveSource(archives.LegacyBin(path, entries), "bin"))
        except Exception as e:
            self.errors.append("%s: %s" % (path, e))

    def _build(self):
        t0 = time.perf_counter()
        self._load_index_cache()
        cfg = self.config
        gmod = cfg.gmod_dir
        gm = os.path.join(gmod, "garrysmod") if gmod else None
        se = os.path.join(gmod, "sourceengine") if gmod else None
        self.phase_times = []
        last = [time.perf_counter()]

        def mark(name):
            now = time.perf_counter()
            self.phase_times.append((name, now - last[0]))
            last[0] = now

        mark("index cache")
        # 1. legacy addons
        if gm:
            addons = os.path.join(gm, "addons")
            if os.path.isdir(addons):
                for n in sorted(os.listdir(addons)):
                    p = os.path.join(addons, n)
                    if os.path.isdir(p):
                        self.add_dir(p)
        for extra in cfg.extra_paths:
            for root in find_content_roots(extra):
                self.add_dir(root)

        mark("legacy addons")
        # 2. loose garrysmod/
        if gm:
            self.add_dir(gm)

        # 3. workshop + floating gma addons
        if gm and cfg.mount_workshop:
            for ws in self._workshop_dirs():
                for item in sorted(os.listdir(ws)):
                    item_dir = os.path.join(ws, item)
                    if not os.path.isdir(item_dir):
                        continue
                    for fn in sorted(os.listdir(item_dir)):
                        low = fn.lower()
                        if low.endswith(".gma"):
                            self.add_gma(os.path.join(item_dir, fn))
                        elif low.endswith(".bin"):
                            self.add_legacy_bin(os.path.join(item_dir, fn))
            addons = os.path.join(gm, "addons")
            if os.path.isdir(addons):
                for fn in sorted(os.listdir(addons)):
                    if fn.lower().endswith(".gma"):
                        self.add_gma(os.path.join(addons, fn))

        mark("workshop addons")
        # 4. + 5. base game packs
        if gm:
            for p in _dir_vpks(gm):
                if "fallbacks" not in os.path.basename(p).lower():
                    self.add_vpk(p)
        if se:
            for name in ("hl2_textures", "hl2_sound_vo_english", "hl2_sound_misc", "hl2_misc"):
                p = os.path.join(se, name + "_dir.vpk")
                if os.path.isfile(p):
                    self.add_vpk(p)

        mark("base game packs")
        # 6. mounted games
        if gm and cfg.mount_games:
            self._mount_games(gm, se)

        mark("mounted games")
        # 7. tail paths
        if se:
            self.add_dir(se)
        if gm:
            self.add_dir(os.path.join(gm, "download"))
            p = os.path.join(gm, "fallbacks_dir.vpk")
            if os.path.isfile(p):
                self.add_vpk(p)

        self._save_index_cache()
        mark("tail + save cache")
        self.build_time = time.perf_counter() - t0
        self.log("AutoProp2Mesh: mounted %d content sources in %.2fs (%d errors)"
                 % (len(self.sources), self.build_time, len(self.errors)))
        self.log("  " + ", ".join("%s %.2fs" % pt for pt in self.phase_times))

    def _workshop_dirs(self):
        out = []
        gmod = os.path.normpath(self.config.gmod_dir)
        # .../steamapps/common/GarrysMod -> .../steamapps/workshop/content/4000
        steamapps = os.path.dirname(os.path.dirname(gmod))
        p = os.path.join(steamapps, "workshop", "content", str(steam.GMOD_APPID))
        if os.path.isdir(p):
            out.append(p)
        return out

    def _mount_game_dir(self, d):
        for p in _dir_vpks(d):
            self.add_vpk(p)
        self.add_dir(d)

    def _mount_games(self, gm, se):
        mounted = []
        depots = os.path.join(gm, "cfg", "mountdepots.txt")
        if os.path.isfile(depots):
            root = keyvalues.parse_file(depots).get_node("gamedepotsystem")
            if root:
                mounted = [k.lower() for k, v in root if isinstance(v, str) and v.strip() not in ("0", "")]
        for folder in mounted:
            game = MOUNTABLE_GAMES.get(folder)
            if not game:
                continue
            appid, subdirs = game
            install, _lib = steam.find_app(appid)
            if install:
                for sub in (subdirs or [folder]):
                    self._mount_game_dir(os.path.join(install, sub))
            elif se:
                fallback = os.path.join(se, "content_%s_dir.vpk" % folder)
                if os.path.isfile(fallback):
                    self.add_vpk(fallback)

        mountcfg = os.path.join(gm, "cfg", "mount.cfg")
        if os.path.isfile(mountcfg):
            root = keyvalues.parse_file(mountcfg).get_node("mountcfg")
            if root:
                for _k, v in root:
                    if isinstance(v, str) and os.path.isdir(v):
                        self._mount_game_dir(v)

    # -- queries ------------------------------------------------------------

    @staticmethod
    def normalize(path):
        return path.replace("\\", "/").lstrip("/").lower()

    def read(self, path):
        path = self.normalize(path)
        for src in self.sources:
            data = src.read(path)
            if data is not None:
                return data
        return None

    def exists(self, path):
        path = self.normalize(path)
        return any(src.exists(path) for src in self.sources)

    def locate(self, path):
        path = self.normalize(path)
        for src in self.sources:
            if src.exists(path):
                return src.path
        return None

    def list_files(self, prefix, ext):
        """All unique paths under `prefix` ending in `ext` (lower case)."""
        key = (prefix, ext)
        hit = self._list_cache.get(key)
        if hit is None:
            found = set()
            for src in self.sources:
                found.update(src.list(prefix, ext))
            hit = self._list_cache[key] = sorted(found)
        return hit

    def close(self):
        for src in self.sources:
            if isinstance(src, ArchiveSource):
                src.archive.close()
