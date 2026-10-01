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


class ControllerSpec:
    def __init__(self, color=(255, 255, 255, 255), material=DEFAULT_MATERIAL,
                 uvs=0, bump=False, name="", parts=()):
        self.color = tuple(int(max(0, min(255, round(c)))) for c in color)
        self.material = material or DEFAULT_MATERIAL
        self.uvs = int(max(0, min(512, uvs)))
        self.bump = bool(bump)
        self.name = name or ""
        self.parts = list(parts)


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
