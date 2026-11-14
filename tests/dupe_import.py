"""AdvDupe2 import test, run headless with a real Garry's Mod install:

    blender -b --factory-startup --python tests/dupe_import.py
    blender -b --factory-startup --python tests/dupe_import.py -- path/to/dupe.txt

Round trip: builds a scene, exports it, imports it back and exports again;
both dupes must describe the same Prop2Mesh content. With a dupe path it
also imports that dupe, re-exports it, and checks every part ends up where
Prop2Mesh would draw it.
"""
import math
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import bpy
import numpy as np
from mathutils import Euler, Matrix, Vector

failures = []


def check(cond, msg):
    print(("  PASS " if cond else "  FAIL ") + msg)
    if not cond:
        failures.append(msg)


import autoprop2mesh as addon
from autoprop2mesh import content, entities, export, importer, modelparts, sync
from autoprop2mesh.core import ad2, dupe, p2m

addon.register()
print("Blender", bpy.app.version_string)
content.get_fs()
print(content.status_text())
ctx = bpy.context
S = tempfile.mkdtemp(prefix="p2m_import_")


def clear():
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o)
    for c in list(ctx.scene.collection.children):
        if c.name != "Collection":
            bpy.data.collections.remove(c)


def read(path):
    with open(path, "rb") as f:
        table, _ = ad2.decode(f.read())
    notes = {}
    ents = [p2m.read_entity(d, notes) for d in dupe.read_entities(table) if d.cls in p2m.ENTITY_CLASSES]
    return ents, notes


def part_matrix(pos, ang, scale=(1, 1, 1)):
    return importer.source_matrix(pos, ang, 1.0, scale)


def mesh_points(spec, frame):
    """Entity-space vertex positions of a mesh part."""
    m = np.array(frame @ part_matrix(spec.pos, spec.ang))
    return spec.positions @ m[:3, :3].T + m[:3, 3]


def clip_plane(n, d, m, s):
    """Entity-space (normal, distance) of a P2M clip. Clips are in the scaled model space."""
    unscaled = m @ Matrix.Diagonal((1 / s[0], 1 / s[1], 1 / s[2], 1))
    n = Vector(n)
    wn = (unscaled.to_3x3().inverted().transposed() @ n).normalized()
    return wn, wn.dot(unscaled @ (n * d))


def controller_frame(cs):
    return part_matrix(cs.linkpos, cs.linkang, cs.scale)


def close(a, b, tol):
    return np.allclose(np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64), atol=tol)


def triangles(parts, frame):
    """Entity-space triangles (m, 3, 3) of mesh parts. Zero area ones are left out, import drops them."""
    rows = [mesh_points(p, frame)[p.triangles] for p in parts]
    pts = np.concatenate(rows) if rows else np.zeros((0, 3, 3))
    double_area = np.linalg.norm(np.cross(pts[:, 1] - pts[:, 0], pts[:, 2] - pts[:, 0]), axis=1)
    return pts[double_area > 1e-8]


def thickness(t):
    double_area = np.linalg.norm(np.cross(t[..., 1, :] - t[..., 0, :], t[..., 2, :] - t[..., 0, :]), axis=-1)
    longest = np.linalg.norm(t - np.roll(t, 1, axis=-2), axis=-1).max(axis=-1)
    return double_area / np.maximum(longest, 1e-12)


def _canonical(pts, grid):
    """Rotates each triangle to start at its smallest corner. Keeps the winding."""
    g = np.round(pts / grid).astype(np.int64)

    def less(a, b):
        return (a[:, 0] < b[:, 0]) | ((a[:, 0] == b[:, 0]) & ((a[:, 1] < b[:, 1]) | ((a[:, 1] == b[:, 1]) & (a[:, 2] < b[:, 2]))))
    start = np.where(less(g[:, 1], g[:, 0]), 1, 0)
    start = np.where(less(g[:, 2], g[np.arange(len(g)), start]), 2, start)
    idx = (start[:, None] + np.arange(3)) % 3
    return np.take_along_axis(g, idx[:, :, None], 1).reshape(-1, 9), np.take_along_axis(pts, idx[:, :, None], 1)


def unmatched(a, b, grid=0.02, tol=1e-3, min_thickness=0.01):
    """(triangles of a missing from b, triangles of b missing from a), winding included.
    Slivers thinner than min_thickness are not counted, OBJ's 4 decimals can collapse them."""
    ka, pa = _canonical(a, grid)
    kb, pb = _canonical(b, grid)
    buckets = {}
    for j, k in enumerate(map(bytes, kb)):
        buckets.setdefault(k, []).append(j)
    taken = np.zeros(len(pb), dtype=bool)
    left_a = []
    for i, k in enumerate(map(bytes, ka)):
        for j in buckets.get(k, ()):
            if not taken[j] and np.abs(pb[j] - pa[i]).max() <= tol:
                taken[j] = True
                break
        else:
            left_a.append(i)
    rest_a, rest_b = pa[left_a], pb[~taken]
    used = np.zeros(len(rest_b), dtype=bool)
    lost = np.zeros(len(rest_a), dtype=bool)
    # grid rounding can split equal triangles, match the rest by distance
    if len(rest_a) and len(rest_b) and len(rest_a) * len(rest_b) <= 4e6:
        rolls = np.stack([np.roll(rest_b, r, axis=1) for r in range(3)])   # (3, n, 3, 3)
        dist = np.stack([np.abs(rolls - t).reshape(3, len(rest_b), 9).max(axis=2).min(axis=0) for t in rest_a])
        lost[:] = True
        # closest pairs first
        for flat in np.argsort(dist, axis=None):
            i, j = divmod(int(flat), len(rest_b))
            if dist[i, j] > tol:
                break
            if lost[i] and not used[j]:
                lost[i] = False
                used[j] = True
    else:
        lost[:] = True
    visible_a = thickness(rest_a) > min_thickness if len(rest_a) else np.zeros(0, dtype=bool)
    visible_b = thickness(rest_b) > min_thickness if len(rest_b) else np.zeros(0, dtype=bool)
    return int((lost & visible_a).sum()), int((~used & visible_b).sum())


def compare(before, after, label, tol=2e-3):
    """Every part of `before` must come back in `after` at the same entity-space place."""
    check(len(before) == len(after), "%s: %d entities -> %d" % (label, len(before), len(after)))
    after_sorted = sorted(after, key=lambda e: tuple(np.round(e.pos, 2)))
    before_sorted = sorted(before, key=lambda e: tuple(np.round(e.pos, 2)))
    bad = []
    for eb, ea in zip(before_sorted, after_sorted):
        if not close(eb.pos, ea.pos, tol) or eb.model.lower() != ea.model.lower():
            bad.append("entity %s moved or changed model" % eb.key)
            continue
        if len(eb.controllers) != len(ea.controllers):
            bad.append("entity %s: %d controllers -> %d" % (eb.key, len(eb.controllers), len(ea.controllers)))
            continue
        for i, (cb, ca) in enumerate(zip(eb.controllers, ea.controllers)):
            where = "entity %s controller %d" % (eb.key, i)
            # export strips controller names
            sb = (cb.color, cb.material, cb.bump, cb.name.strip())
            sa = (ca.color, ca.material, ca.bump, ca.name)
            if sb != sa:
                bad.append(where + " settings differ: %r -> %r" % (sb, sa))
            frame = controller_frame(cb)
            mb = [p for p in cb.parts if isinstance(p, p2m.ModelPartSpec)]
            ma = [p for p in ca.parts if isinstance(p, p2m.ModelPartSpec)]
            if len(mb) != len(ma):
                bad.append(where + " model parts %d -> %d" % (len(mb), len(ma)))
            # export writes parts in Blender's child order, so pair each with its closest match
            unused = list(ma)
            for pb in mb:
                want = frame @ part_matrix(pb.pos, pb.ang, pb.scale)
                same = [p for p in unused if p.model == pb.model]
                if not same:
                    bad.append(where + " model part %s missing" % pb.model)
                    continue
                pa = min(same, key=lambda p: np.abs(np.array(part_matrix(p.pos, p.ang, p.scale)) - np.array(want)).max())
                unused.remove(pa)
                got = part_matrix(pa.pos, pa.ang, pa.scale)
                if not close(want, got, tol * 10):
                    bad.append(where + " model part %s misplaced" % pb.model)
                    continue
                if (pb.bodygroup, pb.flat, pb.vinside) != (pa.bodygroup, pa.flat, pa.vinside):
                    bad.append(where + " model part %s flags %r -> %r" % (
                        pb.model, (pb.bodygroup, pb.flat, pb.vinside), (pa.bodygroup, pa.flat, pa.vinside)))
                if len(pb.clips) != len(pa.clips):
                    bad.append(where + " model part %s clips %d -> %d" % (pb.model, len(pb.clips), len(pa.clips)))
                    continue
                for (nb, db), (na, da) in zip(pb.clips, pa.clips):
                    wb = clip_plane(nb, db, want, pb.scale)
                    wa = clip_plane(na, da, got, pa.scale)
                    if (wb[0] - wa[0]).length > 1e-3 or abs(wb[1] - wa[1]) > tol * 10:
                        bad.append(where + " model part %s clip moved" % pb.model)
            tb = triangles([p for p in cb.parts if isinstance(p, p2m.MeshPartSpec)], frame)
            ta = triangles([p for p in ca.parts if isinstance(p, p2m.MeshPartSpec)], Matrix.Identity(4))
            lost, gained = unmatched(tb, ta)
            if lost or gained:
                bad.append(where + " mesh: %d of %d triangles lost or flipped, %d new" % (lost, len(tb), gained))
    for b in bad[:20]:
        print("    ", b)
    check(not bad, "%s: every part comes back in place (%d problems)" % (label, len(bad)))


def export_all(path):
    sync.full_sync(ctx.scene)
    specs, _warnings = export.gather(ctx)
    return export.write_dupe(path, specs)


# --- round trip ---------------------------------------------------------
print("\n== round trip")
clear()
A = entities.create_entity(ctx, location=(10, 5, 2))
A.rotation_euler = Euler((math.radians(10), math.radians(-25), math.radians(40)), "XYZ")
A.p2m.model = "models/props_c17/oildrum001.mdl"
c0 = A.p2m.controllers[0]
c0.name, c0.color, c0.material = "Body", (1.0, 0.5, 0.25, 1.0), "phoenix_storms/metalset_1-2"
c1 = entities.add_controller(A, "Glass")
c1.color, c1.uvs, c1.bump = (0.2, 0.6, 1.0, 0.5), 24, True

bpy.ops.mesh.primitive_cube_add(size=12, location=(14, 5, 8))
cube = ctx.active_object
bpy.ops.mesh.primitive_uv_sphere_add(radius=3, location=(10, 9, 2))
sphere = ctx.active_object
sphere.scale = (-1, 1, 1)
sphere.p2m.smooth_mode, sphere.p2m.smooth_angle = "SMOOTH", 45
sphere.p2m.render_inside = True
for o, c in ((cube, c0), (sphere, c1)):
    o.parent = A
    o.matrix_parent_inverse = A.matrix_world.inverted()
    o.p2m.controller_uid = c.uid

mp = modelparts.create_model_part(ctx, model="models/props_c17/oildrum001.mdl")
mp.parent = A
mp.location = (3, -4, 6)
mp.rotation_euler = Euler((0.3, 0.2, -0.4), "XYZ")
mp.scale = (1.5, 0.75, 1.0)
mp.p2m.controller_uid = c0.uid
mp.p2m.model_flat = True
plane = modelparts.add_clip_plane(ctx, mp)
plane.location = (0, 0, 20)
plane.rotation_euler = (0.4, 0, 0)

B = entities.create_entity(ctx, location=(-30, 0, 10), model="models/hunter/blocks/cube025x025x025.mdl")
B.parent = A
B.matrix_parent_inverse = A.matrix_world.inverted()
bpy.ops.mesh.primitive_cylinder_add(radius=2, depth=6, location=(-30, 0, 16))
cyl = ctx.active_object
cyl.parent = B
cyl.matrix_parent_inverse = B.matrix_world.inverted()
cyl.p2m.controller_uid = B.p2m.controllers[0].uid
ctx.view_layer.update()

first = os.path.join(S, "first.txt")
export_all(first)
ents_first, _ = read(first)

clear()
ic, objects = importer.import_dupe(ctx, first)
check(ic.stats["entities"] == 2 and ic.stats["controllers"] == 3, "imported 2 entities, 3 controllers (%r)" % dict(ic.stats))
check(ic.stats["model_parts"] == 1 and ic.stats["mesh_parts"] == 3, "imported 1 model part, 3 mesh parts")
ents_by_model = {o.p2m.model: o for o in objects.values()}
child = ents_by_model.get("models/hunter/blocks/cube025x025x025.mdl")
check(child is not None and child.parent is ents_by_model.get("models/props_c17/oildrum001.mdl"), "nested entity is parented")
imp_mp = next((o for o in bpy.data.objects if o.p2m.is_model_part), None)
check(imp_mp is not None and len(modelparts.clip_plane_objects(imp_mp)) == 1, "model part has its clip plane")
check(imp_mp is not None and imp_mp.p2m.model_flat, "model part keeps flat shading")
glass = next((c for o in objects.values() for c in o.p2m.controllers if c.name == "Glass"), None)
check(glass is not None and glass.uvs == 24 and glass.bump and abs(glass.color[3] - 0.5) < 0.01, "controller settings restored")
smooth = [o for o in bpy.data.objects if o.p2m.smooth_mode == "SMOOTH"]
check(len(smooth) == 1 and abs(smooth[0].p2m.smooth_angle - 45) < 1e-3 and smooth[0].p2m.render_inside,
      "smooth angle and render inside restored")
second = os.path.join(S, "second.txt")
export_all(second)
ents_second, _ = read(second)
compare(ents_first, ents_second, "round trip")

# --- body groups --------------------------------------------------------
print("\n== body groups")
clear()
E = entities.create_entity(ctx)
mp = modelparts.create_model_part(ctx, model="models/props_c17/oildrum001.mdl", bodygroup=0)
check(mp.p2m.pending_bodygroup == -1, "pending mask consumed once the model loads")
mp2 = modelparts.create_model_part(ctx, model="models/this/does_not_exist.mdl", bodygroup=5)
check(mp2.p2m.bodygroup_mask() == 5, "mask kept when the model has no body groups (%d)" % mp2.p2m.bodygroup_mask())

# --- synthetic dupe: controller scale and link offset, OBJ quirks ---------
print("\n== controller frames")
clear()
obj_text = "# comment\nv 0 0 0\nv 10 0 0\nv 10 10 0\nv 0 10 0\nvt 0 0\nvn 0 0 1\nf 1/1/1 2/1/1 3/1/1 4/1/1\nv 5 -5 3\nf -1 -4 -5\n"
partlist = {
    1: {"objd": "123", "objn": "quad.obj", "pos": ad2.Vector(1, 2, 3), "ang": ad2.Angle(0, 90, 0), "scale": ad2.Vector(2, 1, 1), "vinvert": 1},
    2: {"prop": "models/props_c17/oildrum001.mdl", "pos": ad2.Vector(5, 0, 0), "ang": ad2.Angle(10, 20, 30),
        "clips": [{"n": ad2.Vector(0, 0, 1), "d": 20.0}], "iclips": [{"n": ad2.Vector(1, 0, 0), "d": -4.0}]},
    3: {"primitive": {"construct": "cube"}, "pos": ad2.Vector(), "ang": ad2.Angle()},
    "custom": {"123": obj_text},
}
blob = p2m.gmod_lzma.compress(p2m._gmod_json(partlist).encode())
crc = "999"
table = {"Entities": {1: {
    "Class": "sent_prop2mesh", "Model": "models/p2m/cube.mdl",
    "PhysicsObjects": {0: {"Pos": ad2.Vector(0, 0, 50), "Angle": ad2.Angle(0, 45, 0)}},
    "EntityMods": {"prop2mesh": {1: {1: {"crc": crc, "uvs": 8, "bump": False, "col": {"r": 255, "g": 0, "b": 0, "a": 255},
                                         "mat": "hunter/myplastic", "scale": ad2.Vector(3, 3, 3), "clips": {},
                                         "linkpos": ad2.Vector(0, 0, 10), "linkang": ad2.Angle(0, 0, 90)}},
                                 2: {crc: blob}}},
}}, "HeadEnt": {"Index": 1, "Pos": ad2.Vector(), "Z": 0}}
synth = os.path.join(S, "synth.txt")
with open(synth, "wb") as f:
    f.write(ad2.encode(table, {"name": "test"}))
ents_synth, notes = read(synth)
quad = ents_synth[0].controllers[0].parts[0]
check(len(quad.triangles) == 3, "OBJ fan + negative indices read (%d triangles)" % len(quad.triangles))
check(notes.get("primitive") == 1, "primitive part counted as skipped")
check(len(ents_synth[0].controllers[0].parts[1].clips) == 2, "iclips merged into clips")
ic, objects = importer.import_dupe(ctx, synth)
check(any("primitive" in w for w in ic.warnings), "primitive skip is reported")
ent = next(iter(objects.values()))
check(ent.p2m.controllers[0].uvs == 24, "UV scale follows the uniform controller scale (8 x 3 = %d)" % ent.p2m.controllers[0].uvs)
synth2 = os.path.join(S, "synth2.txt")
export_all(synth2)
ents_synth2, _ = read(synth2)
compare(ents_synth, ents_synth2, "controller frame")

# --- a real dupe --------------------------------------------------------
args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
for path in args:
    print("\n== %s" % path)
    clear()
    before, notes = read(path)
    ic, objects = importer.import_dupe(ctx, path)
    print("   ", dict(ic.stats), "skipped", dict(ic.skipped))
    for w in ic.warnings:
        print("    warning:", w)
    check(ic.stats["entities"] == len(before), "imported all %d P2M entities" % len(before))
    out = os.path.join(S, "real.txt")
    export_all(out)
    after, _ = read(out)
    compare(before, after, os.path.basename(path))

print("\n%d failure(s)" % len(failures))
for f in failures:
    print("  -", f)
addon.unregister()
sys.exit(1 if failures else 0)
