"""Builds the AutoProp2Mesh extension zip (no Blender needed).

Release versions look like 2026.10.01.42 (commit date, then the commit count).
Blender requires a semantic version in the manifest, which cannot have four
parts or leading zeros, so the manifest gets 2026.1001.42 instead: same date
and count, and it still sorts newer for every later build.

    python tools/build_extension.py                    # version from git
    python tools/build_extension.py --version 2026.10.01.42 --out build
"""

import argparse
import os
import re
import subprocess
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE = os.path.join(ROOT, "src", "autoprop2mesh")
EXCLUDE_DIRS = {"__pycache__", ".git", ".vscode", ".idea"}
EXCLUDE_EXT = {".pyc", ".pyo", ".blend1"}

RELEASE_RE = re.compile(r"^(\d{4})\.(\d{2})\.(\d{2})\.(\d+)$")
SEMVER_RE = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$")


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def release_version_from_git():
    """<commit date YYYY.MM.DD (UTC)>.<number of commits>"""
    env = dict(os.environ, TZ="UTC")
    date = subprocess.check_output(
        ["git", "show", "-s", "--format=%cd", "--date=format-local:%Y.%m.%d", "HEAD"],
        cwd=ROOT, text=True, env=env).strip()
    return "%s.%s" % (date, git("rev-list", "--count", "HEAD"))


def manifest_version(release):
    m = RELEASE_RE.match(release)
    if not m:
        raise SystemExit("version must look like YYYY.MM.DD.N, got %r" % release)
    year, month, day, count = m.groups()
    version = "%d.%d%02d.%d" % (int(year), int(month), int(day), int(count))
    assert SEMVER_RE.match(version), version
    return version


def patched_manifest(version):
    path = os.path.join(SOURCE, "blender_manifest.toml")
    with open(path, encoding="utf-8") as f:
        text = f.read()
    text, n = re.subn(r'(?m)^version\s*=\s*".*"$', 'version = "%s"' % version, text)
    if n != 1:
        raise SystemExit("could not find the version line in blender_manifest.toml")
    try:
        import tomllib  # Python 3.11+
        data = tomllib.loads(text)
        for key in ("schema_version", "id", "version", "name", "type", "blender_version_min"):
            if key not in data:
                raise SystemExit("manifest is missing %r" % key)
    except ImportError:
        pass
    return text


def build(release, out_dir):
    version = manifest_version(release)
    os.makedirs(out_dir, exist_ok=True)
    zip_path = os.path.join(out_dir, "autoprop2mesh-%s.zip" % release)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for dirpath, dirnames, filenames in os.walk(SOURCE):
            dirnames[:] = sorted(d for d in dirnames if d not in EXCLUDE_DIRS)
            for fn in sorted(filenames):
                if os.path.splitext(fn)[1] in EXCLUDE_EXT:
                    continue
                full = os.path.join(dirpath, fn)
                arc = os.path.relpath(full, SOURCE).replace(os.sep, "/")
                if arc == "blender_manifest.toml":
                    z.writestr(arc, patched_manifest(version))
                else:
                    z.write(full, arc)
    return zip_path, version


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", help="release version YYYY.MM.DD.N (default: from git)")
    ap.add_argument("--out", default=os.path.join(ROOT, "build"), help="output folder (default: build/)")
    args = ap.parse_args()
    release = args.version or release_version_from_git()
    zip_path, version = build(release, args.out)
    print("release version: %s" % release)
    print("manifest version: %s" % version)
    print("built: %s" % zip_path)
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as f:
            f.write("release=%s\nmanifest=%s\nzip=%s\n" % (release, version, zip_path))


if __name__ == "__main__":
    sys.exit(main())
