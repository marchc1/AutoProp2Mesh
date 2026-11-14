"""Prop2Mesh data: controller part lists (compressed JSON with embedded OBJ
meshes) and sent_prop2mesh dupe entities.

Format notes (verified against Prop2Mesh's cl_meshlab.lua / init.lua and real
dupes):
  * A controller's part list is util.TableToJSON(partlist) compressed with
    util.Compress; its key / `crc` is util.CRC of the *compressed* bytes.
  * OBJ parts are { objd = <crc of obj text>, objn = <name>, pos, ang, ... }
    with the OBJ text stored in partlist.custom[objd].
  * P2M's OBJ reader only understands `v x y z` and `f a b c` (positive,
    1-based indices, plain decimals). UVs are box-projected and normals are
    computed from the faces, so neither is written.
  * P2M computes face normals as cross(v3 - v1, v2 - v1): clockwise winding
    is front facing, as everywhere in Source. Blender is counter-clockwise,
    so faces are written reversed.
  * Meshes are rendered unindexed (3 vertices per triangle) and a mesh is
    limited to 64k vertices, so one part may hold at most 21333 triangles.
"""

import json
import zlib

import numpy as np

from . import gmod_lzma
from .ad2 import Angle, Vector

MAX_PART_TRIANGLES = 21333
DEFAULT_MATERIAL = "hunter/myplastic"
DEFAULT_MODEL = "models/p2m/cube.mdl"
MAX_DUPE_BYTES = 32_000_000  # AdvDupe2.MaxDupeSize


class PartGeometry:
    """One exported object. Positions are entity-local Source units."""

    def __init__(self, name, positions, triangles, vsmooth=None, vinside=False):
        self.name = name
        self.positions = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
        self.triangles = np.asarray(triangles, dtype=np.int64).reshape(-1, 3)
        self.vsmooth = vsmooth      # None = flat, else smoothing angle (degrees)
        self.vinside = vinside      # also render back faces


class ModelPartSpec:
    """A model part: P2M builds it from the model itself (`prop` parts), so
    only the model path and placement are exported, never vertices.

    pos/ang/scale are relative to the entity. Clips are (normal, distance)
    pairs in the part's scaled local space; P2M keeps dot(v, n) >= d."""

    def __init__(self, name, model, pos, ang, scale=(1, 1, 1), clips=(), bodygroup=0,
                 flat=False, vinside=False):
        self.name = name
        self.model = model
        self.pos = Vector(*pos)
        self.ang = Angle(*ang)
        self.scale = Vector(*scale)
        self.clips = [(Vector(*n), float(d)) for n, d in clips]
        self.bodygroup = int(bodygroup)
        self.flat = flat
        self.vinside = vinside


class MeshPartSpec:
    """An OBJ part read from a dupe. positions are in the part's own space
    (part scale applied), triangles are counter-clockwise."""

    def __init__(self, name, positions, triangles, pos=(0, 0, 0), ang=(0, 0, 0), vsmooth=None, vinside=False):
        self.name = name
        self.positions = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
        self.triangles = np.asarray(triangles, dtype=np.int64).reshape(-1, 3)
        self.pos = Vector(*pos)
        self.ang = Angle(*ang)
        self.vsmooth = vsmooth
        self.vinside = vinside


class ControllerSpec:
    # scale, clips and link* are only filled in by the importer; export bakes them into the parts
    def __init__(self, color=(255, 255, 255, 255), material=DEFAULT_MATERIAL,
                 uvs=0, bump=False, name="", parts=(), scale=(1, 1, 1), clips=(),
                 linkpos=(0, 0, 0), linkang=(0, 0, 0)):
        self.color = tuple(int(max(0, min(255, round(c)))) for c in color)
        self.material = material or DEFAULT_MATERIAL
        self.uvs = int(max(0, min(512, uvs)))
        self.bump = bool(bump)
        self.name = name or ""
        self.parts = list(parts)
        self.scale = Vector(*scale)
        self.clips = [(Vector(*n), float(d)) for n, d in clips]
        self.linkpos = Vector(*linkpos)
        self.linkang = Angle(*linkang)


class EntitySpec:
    def __init__(self, key, model, pos, ang, parent_key=None, controllers=()):
        self.key = key              # any hashable; mapped to dupe indices
        self.model = model or DEFAULT_MODEL
        self.pos = Vector(*pos)     # world (dupe-origin relative) Source units
        self.ang = Angle(*ang)
        self.parent_key = parent_key
        self.controllers = list(controllers)


# --------------------------------------------------------------------------
# OBJ / part lists
# --------------------------------------------------------------------------

def _fmt(x, decimals=4):
    """Plain decimal (P2M's OBJ reader cannot parse exponents)."""
    s = "%.*f" % (decimals, x)
    s = s.rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def obj_text(positions, triangles):
    """OBJ text for P2M from CCW triangles (written clockwise)."""
    used, inverse = np.unique(triangles.reshape(-1), return_inverse=True)
    faces = inverse.reshape(-1, 3) + 1
    verts = positions[used]
    lines = ["v %s %s %s\n" % (_fmt(x), _fmt(y), _fmt(z)) for x, y, z in verts.tolist()]
    lines += ["f %d %d %d\n" % (a, c, b) for a, b, c in faces.tolist()]
    return "".join(lines)


def split_part(part):
    """Splits a part into chunks of at most MAX_PART_TRIANGLES triangles."""
    tris = part.triangles
    if len(tris) <= MAX_PART_TRIANGLES:
        return [part]
    out = []
    for i, start in enumerate(range(0, len(tris), MAX_PART_TRIANGLES)):
        out.append(PartGeometry("%s.%d" % (part.name, i + 1), part.positions,
                                tris[start:start + MAX_PART_TRIANGLES], part.vsmooth, part.vinside))
    return out


def _ascii(s):
    return "".join(c if 32 <= ord(c) < 127 else "_" for c in s)[:64] or "part"


def _gmod_json(value):
    """JSON with GMod's Vector/Angle string forms."""
    def conv(v):
        if isinstance(v, Vector):
            return "[%s %s %s]" % tuple(_fmt(c, 6) for c in v)
        if isinstance(v, Angle):
            return "{%s %s %s}" % tuple(_fmt(c, 6) for c in v)
        if isinstance(v, dict):
            return {str(k): conv(x) for k, x in v.items()}
        if isinstance(v, list):
            return [conv(x) for x in v]
        return v
    return json.dumps(conv(value), separators=(",", ":"))


def build_partlist(parts):
    """Returns (compressed_blob, crc_string, stats) or (None, "!none", stats)
    when there is no geometry."""
    stats = {"parts": 0, "triangles": 0, "vertices": 0}
    partlist = {}
    custom = {}
    index = 0
    for part in parts:
        if isinstance(part, ModelPartSpec):
            entry = {"prop": part.model, "pos": part.pos, "ang": part.ang}
            if any(abs(c - 1.0) > 1e-6 for c in part.scale):
                entry["scale"] = part.scale
            if part.clips:
                entry["clips"] = [{"n": n, "d": d} for n, d in part.clips]
            if part.bodygroup:
                entry["bodygroup"] = part.bodygroup
            if part.flat:
                entry["vsmooth"] = 1
            if part.vinside:
                entry["vinside"] = 1
            index += 1
            partlist[index] = entry
            stats["parts"] += 1
            stats["models"] = stats.get("models", 0) + 1
            continue
        if len(part.triangles) == 0:
            continue
        for chunk in split_part(part):
            text = obj_text(chunk.positions, chunk.triangles)
            objd = str(zlib.crc32(text.encode("ascii")))
            custom[objd] = text
            entry = {
                "objd": objd,
                "objn": _ascii(chunk.name),
                "pos": Vector(0, 0, 0),
                "ang": Angle(0, 0, 0),
            }
            if chunk.vsmooth:
                entry["vsmooth"] = float(chunk.vsmooth)
            if chunk.vinside:
                entry["vinside"] = 1
            index += 1
            partlist[index] = entry
            stats["parts"] += 1
            stats["triangles"] += len(chunk.triangles)
            stats["vertices"] += text.count("v ")
    if not partlist:
        return None, "!none", stats
    if custom:
        # Mixed table: array part 1..n plus `custom`. GMod's util.JSONToTable
        # turns numeric object keys back into numbers (P2M relies on this
        # too: it looks custom data up with tonumber(objd)).
        partlist["custom"] = custom
        body = partlist
    else:
        # Pure array, exactly as util.TableToJSON writes it.
        body = [partlist[i] for i in range(1, index + 1)]
    blob = gmod_lzma.compress(_gmod_json(body).encode("utf-8"))
    return blob, str(zlib.crc32(blob)), stats


# --------------------------------------------------------------------------
# Dupe
# --------------------------------------------------------------------------

def _controller_table(spec, crc):
    r, g, b, a = spec.color
    tbl = {
        "crc": crc,
        "uvs": spec.uvs,
        "bump": spec.bump,
        "col": {"r": r, "g": g, "b": b, "a": a},
        "mat": spec.material,
        "scale": Vector(1, 1, 1),
        "clips": {},
    }
    if spec.name:
        tbl["name"] = spec.name
    return tbl


def build_dupe(entities, export_names=True):
    """Returns (dupe_table, report). Entity keys are remapped to 1..n with
    parents before children."""
    order = []
    by_key = {e.key: e for e in entities}
    placed = set()

    def place(e, chain=()):
        if e.key in placed:
            return
        if e.key in chain:
            raise ValueError("parenting loop between P2M entities")
        parent = by_key.get(e.parent_key)
        if parent is not None:
            place(parent, chain + (e.key,))
        placed.add(e.key)
        order.append(e)

    for e in entities:
        place(e)
    index_of = {e.key: i + 1 for i, e in enumerate(order)}

    report = {"entities": len(order), "controllers": 0, "parts": 0, "triangles": 0, "models": 0, "warnings": []}
    ents = {}
    for e in order:
        controllers = {}
        partlists = {}
        for spec in e.controllers:
            blob, crc, stats = build_partlist(spec.parts)
            if blob is not None:
                partlists[crc] = blob
            if not export_names:
                spec = ControllerSpec(spec.color, spec.material, spec.uvs, spec.bump, "", ())
            controllers[len(controllers) + 1] = _controller_table(spec, crc)
            report["controllers"] += 1
            report["parts"] += stats["parts"]
            report["triangles"] += stats["triangles"]
            report["models"] = report.get("models", 0) + stats.get("models", 0)
        ent = {
            "Class": "sent_prop2mesh",
            "Model": e.model,
            "PhysicsObjects": {0: {"Pos": e.pos, "Angle": e.ang, "Frozen": True}},
            "BuildDupeInfo": {"IsNPC": False, "IsVehicle": False},
        }
        if controllers:
            ent["EntityMods"] = {"prop2mesh": {1: controllers, 2: partlists}}
        if e.parent_key is not None and e.parent_key in index_of:
            ent["BuildDupeInfo"]["DupeParentID"] = index_of[e.parent_key]
        ents[index_of[e.key]] = ent

    head = 1
    dupe = {
        "Entities": ents,
        "Constraints": {},
        "HeadEnt": {"Index": head, "Pos": Vector(0, 0, 0), "Z": 0},
        "Description": "Exported by AutoProp2Mesh",
    }
    return dupe, report


# --------------------------------------------------------------------------
# Reading dupes
# --------------------------------------------------------------------------

ENTITY_CLASSES = ("sent_prop2mesh", "sent_prop2mesh_legacy")


def _number(v, default=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _triple(v, default=(0.0, 0.0, 0.0)):
    """A Vector/Angle from the dupe table or from GMod JSON ("[x y z]", "{p y r}")."""
    if isinstance(v, (Vector, Angle)):
        return tuple(v)
    if isinstance(v, bytes):
        v = v.decode("utf-8", "replace")
    if isinstance(v, str):
        parts = v.strip().strip("[]{}").split()
        if len(parts) == 3:
            try:
                return tuple(float(p) for p in parts)
            except ValueError:
                pass
        return default
    if isinstance(v, dict):
        keys = (1, 2, 3) if 1 in v else ("x", "y", "z") if "x" in v else ("p", "y", "r")
        try:
            return tuple(float(v[k]) for k in keys)
        except (KeyError, TypeError, ValueError):
            return default
    if isinstance(v, (list, tuple)) and len(v) == 3:
        try:
            return tuple(float(c) for c in v)
        except (TypeError, ValueError):
            return default
    return default


def _custom_key(k):
    # P2M looks custom meshes up with tonumber(objd)
    try:
        return str(int(float(k)))
    except (TypeError, ValueError):
        return str(k)


def _lua_array(t):
    """The 1..n part of a table, the way Lua's # sees it."""
    if isinstance(t, list):
        return list(t)
    out = []
    while True:
        v = t.get(str(len(out) + 1))
        if v is None:
            v = t.get(len(out) + 1)
        if v is None:
            return out
        out.append(v)


def decode_partlist(blob):
    """(list of part dicts, {objd: obj text}) from a compressed part list."""
    if isinstance(blob, str):
        blob = blob.encode("utf-8")   # ad2.decode turns blobs that happen to be valid UTF-8 into str
    table = json.loads(gmod_lzma.decompress(blob).decode("utf-8", "replace"))
    if isinstance(table, list):
        return table, {}
    if not isinstance(table, dict):
        raise ValueError("part list is not a table")
    custom = table.get("custom")
    if isinstance(custom, list):
        custom = {str(i + 1): v for i, v in enumerate(custom)}
    elif isinstance(custom, dict):
        custom = {_custom_key(k): v for k, v in custom.items()}
    else:
        custom = {}
    return _lua_array(table), custom


def parse_obj(text, invert=False):
    """Positions and counter-clockwise triangles from OBJ text, read like P2M
    reads it: only `v` and `f`, faces as fans, clockwise is the front."""
    verts = []
    tris = []
    for line in text.split("\n"):
        tokens = line.split()
        if not tokens:
            continue
        head = tokens[0]
        if head == "v" and len(tokens) >= 4:
            try:
                verts.append((float(tokens[1]), float(tokens[2]), float(tokens[3])))
            except ValueError:
                verts.append((0.0, 0.0, 0.0))
        elif head == "f" and len(tokens) >= 4:
            face = []
            n = len(verts)
            for tok in tokens[1:]:
                try:
                    i = int(tok.split("/")[0])
                except ValueError:
                    face = None
                    break
                i = i - 1 if i > 0 else n + i
                if not 0 <= i < n:
                    face = None
                    break
                face.append(i)
            if not face:
                continue
            a = face[0]
            for b, c in zip(face[1:-1], face[2:]):
                tris.append((a, b, c) if invert else (a, c, b))
    return np.array(verts, dtype=np.float64).reshape(-1, 3), np.array(tris, dtype=np.int64).reshape(-1, 3)


def _clips(entries):
    out = []
    if not isinstance(entries, (list, dict)):
        return out
    for c in _lua_array(entries):
        if not isinstance(c, dict):
            continue
        n = np.array(_triple(c.get("n")))
        length = np.linalg.norm(n)
        if length < 1e-12:
            continue
        out.append((tuple(n / length), _number(c.get("d")) / length))
    return out


def _count(notes, key):
    notes[key] = notes.get(key, 0) + 1


def read_parts(blob, notes):
    """Mesh and model parts of one controller. Anything skipped is counted in notes."""
    parts, custom = decode_partlist(blob)
    out = []
    for part in parts:
        if not isinstance(part, dict):
            continue
        pos = _triple(part.get("pos"))
        ang = _triple(part.get("ang"))
        scale = _triple(part.get("scale"), (1.0, 1.0, 1.0))
        vinside = bool(_number(part.get("vinside")))
        model = part.get("prop") or part.get("holo")
        if model:
            model = str(model).replace("\\", "/").lower()
            if part.get("submodels"):
                _count(notes, "submodels")
            name = model.rsplit("/", 1)[-1]
            out.append(ModelPartSpec(
                name[:-4] if name.endswith(".mdl") else name, model, pos, ang, scale,
                _clips(part.get("clips")) + _clips(part.get("iclips")),
                int(_number(part.get("bodygroup"))), _number(part.get("vsmooth")) == 1, vinside))
        elif part.get("objd") is not None:
            text = custom.get(_custom_key(part.get("objd")))
            if not isinstance(text, str):
                _count(notes, "missing_obj")
                continue
            positions, triangles = parse_obj(text, bool(_number(part.get("vinvert"))))
            if len(triangles) == 0:
                _count(notes, "empty_obj")
                continue
            name = str(part.get("objn") or "mesh")
            if name.lower().endswith(".obj"):
                name = name[:-4]
            smooth = _number(part.get("vsmooth"))
            out.append(MeshPartSpec(name, positions * scale, triangles, pos, ang, smooth or None, vinside))
        elif part.get("primitive"):
            _count(notes, "primitive")
        else:
            _count(notes, "unknown")
    return out


def _read_controller(c, partlists, notes):
    col = c.get("col") if isinstance(c.get("col"), dict) else {}
    clips = []
    if isinstance(c.get("clips"), dict):
        for clip in c["clips"].values():
            if isinstance(clip, dict) and all(i in clip for i in (1, 2, 3, 4)):
                clips.append(((_number(clip[1]), _number(clip[2]), _number(clip[3])), _number(clip[4])))
    if clips:
        _count(notes, "controller_clips")
    parts = []
    blob = partlists.get(str(c.get("crc")))
    if blob is not None:
        try:
            parts = read_parts(blob, notes)
        except Exception:
            _count(notes, "bad_partlist")
    return ControllerSpec(
        color=[_number(col.get(k), 255) for k in "rgba"],
        material=str(c.get("mat") or DEFAULT_MATERIAL),
        uvs=min(abs(int(np.floor(_number(c.get("uvs"))))), 512),
        bump=bool(c.get("bump")), name=str(c.get("name") or ""), parts=parts,
        scale=_triple(c.get("scale"), (1.0, 1.0, 1.0)), clips=clips,
        linkpos=_triple(c.get("linkpos")), linkang=_triple(c.get("linkang")))


def read_entity(dent, notes):
    """EntitySpec for a P2M entity from core.dupe, keyed by its dupe index."""
    mods = dent.table.get("EntityMods")
    data = mods.get("prop2mesh") if isinstance(mods, dict) else None
    controllers = []
    if isinstance(data, dict):
        lists = data.get(2) if isinstance(data.get(2), dict) else {}
        lists = {str(k): v for k, v in lists.items()}
        ctrls = data.get(1)
        for c in (_lua_array(ctrls) if isinstance(ctrls, dict) else ()):
            if isinstance(c, dict):
                controllers.append(_read_controller(c, lists, notes))
    return EntitySpec(dent.index, dent.model, dent.pos, dent.ang, dent.parent, controllers)


_NOTE_TEXT = {
    "primitive": "%d primitive part(s) skipped (they need the Primitive addon)",
    "submodels": "%d model part(s) hide submodels in game; Blender shows (and exports) the whole model",
    "missing_obj": "%d mesh part(s) skipped: their OBJ data is missing from the dupe",
    "empty_obj": "%d mesh part(s) skipped: no faces",
    "unknown": "%d part(s) of an unknown kind skipped",
    "bad_partlist": "%d controller part list(s) could not be read",
    "controller_clips": "%d controller(s) use controller clip planes, which are not imported",
}


def note_warnings(notes):
    return [_NOTE_TEXT[k] % v for k, v in notes.items() if v and k in _NOTE_TEXT]
