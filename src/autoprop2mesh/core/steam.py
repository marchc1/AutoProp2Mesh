"""Locating Steam, its library folders and installed apps.
"""

import os
import sys

from . import keyvalues

GMOD_APPID = 4000


def find_steam():
    if sys.platform == "win32":
        import winreg
        candidates = (
            (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Valve\Steam", "InstallPath"),
            (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\Valve\Steam", "InstallPath"),
            (winreg.HKEY_CURRENT_USER, r"SOFTWARE\Valve\Steam", "SteamPath"),
        )
        for hive, key, value in candidates:
            try:
                with winreg.OpenKey(hive, key) as k:
                    path = winreg.QueryValueEx(k, value)[0]
                    if path and os.path.isdir(path):
                        return os.path.normpath(path)
            except OSError:
                pass
        return None

    home = os.path.expanduser("~")
    if sys.platform == "darwin":
        paths = [os.path.join(home, "Library", "Application Support", "Steam")]
    else:
        paths = [
            os.path.join(home, ".local", "share", "Steam"),
            os.path.join(home, ".steam", "steam"),
            os.path.join(home, ".var", "app", "com.valvesoftware.Steam", ".local", "share", "Steam"),
        ]
    for p in paths:
        if os.path.isdir(p):
            return p
    return None


def library_folders(steam_path=None):
    """Yields (library_path, set_of_appid_strings_or_None)."""
    steam_path = steam_path or find_steam()
    if not steam_path:
        return
    vdf = os.path.join(steam_path, "steamapps", "libraryfolders.vdf")
    seen = set()
    if os.path.isfile(vdf):
        root = keyvalues.parse_file(vdf).get_node("libraryfolders")
        if root:
            for _, entry in root:
                if not isinstance(entry, keyvalues.KVNode):
                    continue
                path = entry.get("path")
                if not path:
                    continue
                apps = entry.get_node("apps")
                key = os.path.normcase(os.path.normpath(path))
                if key in seen:
                    continue
                seen.add(key)
                yield path, ({k for k, _ in apps} if apps else None)
    key = os.path.normcase(os.path.normpath(steam_path))
    if key not in seen:
        yield steam_path, None


def find_app(appid, steam_path=None):
    """Returns (install_dir, library_path) or (None, None)."""
    appid = str(appid)
    for lib, apps in library_folders(steam_path):
        manifest = os.path.join(lib, "steamapps", "appmanifest_%s.acf" % appid)
        # `apps` is only a cache; the manifest file is authoritative.
        if not os.path.isfile(manifest):
            continue
        state = keyvalues.parse_file(manifest).get_node("AppState")
        installdir = state.get("installdir") if state else None
        if installdir:
            path = os.path.join(lib, "steamapps", "common", installdir)
            if os.path.isdir(path):
                return path, lib
    return None, None


def find_gmod():
    """Returns (gmod_install_dir, library_path) or (None, None)."""
    return find_app(GMOD_APPID)
