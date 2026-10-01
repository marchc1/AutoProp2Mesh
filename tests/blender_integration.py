"""End-to-end test, run headless with a real Garry's Mod install:

    blender -b --factory-startup --python tests/blender_integration.py

Builds a scene (nested entities; mesh, text, curve, mirrored and indirect
parts), checks material/UV-preview syncing, exports, and verifies the dupe:
Source angle math reproduces Blender's world transforms, P2M UVs, winding.
"""
import math
import os
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import bpy
import numpy as np
from mathutils import Euler, Matrix, Vector

import tempfile
S = tempfile.mkdtemp(prefix="p2m_test_")
failures = []


def check(cond, msg):
    print(("  PASS " if cond else "  FAIL ") + msg)
    if not cond:
        failures.append(msg)


import autoprop2mesh as addon
from autoprop2mesh import browser, content, entities, export, materials, modelparts, props, sync, uvpreview
from autoprop2mesh.core import ad2, clipping, gmod_lzma, p2m, studiomdl

addon.register()
print("Blender", bpy.app.version_string)
fs = content.get_fs()
print(content.status_text())
check(fs is not None, "content filesystem mounted")

for o in list(bpy.data.objects):
    bpy.data.objects.remove(o)
ctx = bpy.context
scene = ctx.scene

# --- entity A with a model, rotated -------------------------------------
A = entities.create_entity(ctx, location=(10, 5, 2))
A.rotation_euler = Euler((math.radians(10), math.radians(-25), math.radians(40)), "XYZ")
check(A.data.name.startswith("P2M Model: models/p2m/cube.mdl"), "entity shows models/p2m/cube.mdl (%s)" % A.data.name)
check(A.p2m.model_status == "", "model status clean")
c0 = A.p2m.controllers[0]
c0.name = "Body"
c0.color = (1.0, 0.5, 0.25, 1.0)
c0.material = "phoenix_storms/metalset_1-2"
c1 = entities.add_controller(A, "Glass")
c1.color = (0.2, 0.6, 1.0, 0.5)
c1.uvs = 24

A.p2m.model = "models/this/does_not_exist.mdl"
check("error.mdl" in A.p2m.model_status, "missing model falls back to error.mdl: %s" % A.p2m.model_status)
A.p2m.model = "models/props_c17/oildrum001.mdl"
check(A.p2m.model_status == "" and "oildrum001" in A.data.name, "model switched to oildrum001")
check(len(A.data.materials) and A.data.materials[0].name.startswith("SRC: "), "model has Source material")

# --- parts ------------------------------------------------------------
bpy.ops.mesh.primitive_cube_add(size=12, location=(14, 5, 8))
cube = ctx.active_object
bpy.ops.object.text_add(location=(10, 0, 12))
text = ctx.active_object
text.data.body = "P2M"
bpy.ops.curve.primitive_bezier_circle_add(radius=4, location=(6, 5, 2))
curve = ctx.active_object
curve.data.bevel_depth = 0.5
bpy.ops.mesh.primitive_uv_sphere_add(radius=3, location=(10, 9, 2))
sphere = ctx.active_object
sphere.scale = (-1, 1, 1)  # mirrored
sphere.p2m.smooth_mode = "SMOOTH"
sphere.p2m.smooth_angle = 45
bpy.ops.mesh.primitive_cylinder_add(radius=1, depth=4, location=(10, 9, 8))
nested = ctx.active_object  # will be a child of the sphere (indirect)

for o in ctx.selected_objects:
    o.select_set(False)
for o in (cube, text, curve):
    o.select_set(True)
A.select_set(True)
ctx.view_layer.objects.active = A
r = bpy.ops.p2m.attach(entity=A.name, uid=c0.uid)
check(r == {"FINISHED"}, "attach operator")
check(cube.parent == A and text.parent == A, "parts parented to entity")
check((cube.matrix_world.translation - Vector((14, 5, 8))).length < 1e-4, "attach keeps world transform")

sphere.parent = A
sphere.matrix_parent_inverse = A.matrix_world.inverted()
sphere.p2m.controller_uid = c1.uid
nested.parent = sphere
nested.matrix_parent_inverse = sphere.matrix_world.inverted()
sync.full_sync(scene)

check(all(s.link == "OBJECT" and s.material == c0.blender_material for s in cube.material_slots), "cube gets controller material (object-linked)")
check(text.material_slots and text.material_slots[0].material == c0.blender_material, "text object gets controller material")
check(nested.material_slots and nested.material_slots[0].material == c1.blender_material, "indirect child inherits controller from parent")
check(cube.modifiers.get(uvpreview.MODIFIER_NAME) is not None, "UV preview modifier added")
mat = c1.blender_material
check(abs(mat.diffuse_color[3] - 0.5) < 1e-6, "viewport color carries alpha")
check(mat.node_tree.nodes.active.name == "p2m_solid_preview", "solid preview image node is active")
rm = getattr(mat, "surface_render_method", None)
check(rm in (None, "BLENDED"), "translucent controller uses blended render method (%s)" % rm)

# --- UV preview matches P2M's box mapping ------------------------------
dg = ctx.evaluated_depsgraph_get()
ev = cube.evaluated_get(dg)
me = ev.to_mesh()
attr = me.attributes.get("p2m_uv")
check(attr is not None and attr.domain == "CORNER", "p2m_uv corner attribute exists")
if attr is not None:
    uv = np.empty(len(me.loops) * 2, dtype=np.float32)
    attr.data.foreach_get("vector", uv)
    uv = uv.reshape(-1, 2)
    inv = A.matrix_world.inverted() @ cube.matrix_world
    us = props.unit_scale(scene)
    k = 1.0 / 48.0
    worst = 0.0
    for poly in me.polygons:
        n = (inv.to_3x3() @ poly.normal)
        ax, ay, az = abs(n.x), abs(n.y), abs(n.z)
        for li in poly.loop_indices:
            p = (inv @ me.vertices[me.loops[li].vertex_index].co) * us
            sg = lambda c: -1 if c < 0 else 1
            if ax > ay and ax > az:
                u, v = p.z * sg(n.x) * k, p.y * k
            elif ay > az:
                u, v = p.x * k, p.z * sg(n.y) * k
            else:
                u, v = p.x * -sg(n.z) * k, p.y * k
            worst = max(worst, abs(uv[li, 0] - u), abs(uv[li, 1] - (1 - v)))
    check(worst < 1e-3, "GN UVs match P2M getBoxUV (max error %.2e)" % worst)
    check(me.attributes.get("UVMap") is not None, "active UV map overwritten for solid view")
ev.to_mesh_clear()

# --- entity B parented to A ---------------------------------------------
B = entities.create_entity(ctx, location=(30, -10, 5))
B.rotation_euler = Euler((0, 0, math.radians(90)), "XYZ")
B.parent = A
B.matrix_parent_inverse = A.matrix_world.inverted()
bpy.ops.mesh.primitive_monkey_add(size=5, location=(30, -10, 10))
monkey = ctx.active_object
monkey.parent = B
monkey.matrix_parent_inverse = B.matrix_world.inverted()
monkey.p2m.controller_uid = B.p2m.controllers[0].uid
sync.full_sync(scene)
check(monkey.material_slots[0].material == B.p2m.controllers[0].blender_material, "nested entity owns its own parts")
check(sync.resolve(monkey)[0] is B, "part resolves to nearest entity")

# --- unparenting removes the material -------------------------------------
mw = curve.matrix_world.copy()
curve.parent = None
curve.matrix_world = mw
sync.full_sync(scene)
check(not curve.modifiers.get(uvpreview.MODIFIER_NAME) and all(s.link == "DATA" for s in curve.material_slots),
      "unparented part loses controller material and modifier")
curve.parent = A
curve.matrix_parent_inverse = A.matrix_world.inverted()
sync.full_sync(scene)

# --- duplicate entity gets its own materials ------------------------------
for o in ctx.selected_objects:
    o.select_set(False)
B.select_set(True)
ctx.view_layer.objects.active = B
bpy.ops.object.duplicate()
B2 = ctx.active_object
sync.full_sync(scene)
check(B2.p2m.controllers[0].blender_material != B.p2m.controllers[0].blender_material, "duplicated entity gets separate controller materials")
bpy.data.objects.remove(B2)

# --- model part (prop part) with body group, scale and clip planes -------
MODEL = "models/items/ammocrate_ar2.mdl"
mp = modelparts.create_model_part(ctx, location=(-20, 10, 15))
mp.p2m.model = MODEL
ammo = [bg for bg in mp.p2m.bodygroups if "\n" in bg.options]
check(mp.p2m.is_model_part and len(ammo) == 1 and ammo[0].name == "ammo", "model part loads body groups (%s)" % [bg.name for bg in mp.p2m.bodygroups])
full_tris = len(mp.data.polygons)
ammo[0].choice = "1"
BG_CHOICES = mp.p2m.bodygroup_choices()
check(len(mp.data.polygons) != full_tris, "body group switch changes preview (%d -> %d tris)" % (full_tris, len(mp.data.polygons)))
mp.rotation_euler = Euler((math.radians(20), math.radians(35), math.radians(-60)), "XYZ")
mp.scale = (0.6, 1.3, 0.9)
mp.p2m.model_flat = True
ctx.view_layer.update()  # operators/rebuilds read matrix_world
for o in ctx.selected_objects:
    o.select_set(False)
mp.select_set(True)
A.select_set(True)
ctx.view_layer.objects.active = A
bpy.ops.p2m.attach(entity=A.name, uid=c0.uid)
bg_tris = len(mp.data.polygons)
plane = modelparts.add_clip_plane(ctx, mp)
plane.location = (2, 3, 4)
plane.rotation_euler = Euler((math.radians(30), math.radians(-15), 0), "XYZ")
plane2 = modelparts.add_clip_plane(ctx, mp)
plane2.location = (0, 0, -6)
plane2.rotation_euler = Euler((math.radians(170), 0, math.radians(40)), "XYZ")
ctx.view_layer.update()
modelparts.rebuild(mp, force=True)
sync.full_sync(scene)
check(0 < len(mp.data.polygons) and len(mp.data.polygons) != bg_tris, "clip planes cut the preview (%d -> %d tris)" % (bg_tris, len(mp.data.polygons)))
check(all(s.material == c0.blender_material for s in mp.material_slots), "model part gets controller material")
mod = mp.modifiers.get(uvpreview.MODIFIER_NAME)
ids = uvpreview._input_ids(mod.node_group)
check(uvpreview._get_input(mod, ids["Box UV"]) is False, "uvs=0: model part keeps model UVs in preview")
check(not sync.is_geometry(plane) and sync.resolve(plane)[1] is None or plane.p2m.is_clip_plane, "clip planes are not parts")

# --- export -------------------------------------------------------------
specs, warnings = export.gather(ctx)
print("  warnings:", warnings)
out = os.path.join(S, "blender_export.txt")
report = export.write_dupe(out, specs)
print("  report:", report)
check(report["entities"] == 2 and report["controllers"] == 3, "exported 2 entities / 3 controllers")

dupe, info = ad2.decode(open(out, "rb").read(), text_strings=False)
ents = dupe[b"Entities"]
by_model = {e[b"Model"]: (i, e) for i, e in ents.items()}
ia, ea = by_model[b"models/props_c17/oildrum001.mdl"]
ib, eb = by_model[b"models/p2m/cube.mdl"]
check(eb[b"BuildDupeInfo"].get(b"DupeParentID") == ia, "entity B parented to A in dupe")
check(ea[b"BuildDupeInfo"].get(b"DupeParentID") is None, "entity A is a root")
ctrls = ea[b"EntityMods"][b"prop2mesh"][1]
check(ctrls[1][b"name"] == b"Body" and ctrls[2][b"name"] == b"Glass", "controller names exported")
check(ctrls[1][b"col"] == {b"r": 255, b"g": 128, b"b": 64, b"a": 255}, "controller color %r" % ctrls[1][b"col"])
check(ctrls[2][b"col"][b"a"] == 128 and ctrls[2][b"uvs"] == 24, "controller alpha / uvs")
check(ctrls[1][b"mat"] == b"phoenix_storms/metalset_1-2", "controller material")


def source_matrix(pos, ang):
    """Source AngleMatrix + translation (mathlib.cpp)."""
    p, y, r = (math.radians(a) for a in ang)
    sp, cp, sy, cy, sr, cr = math.sin(p), math.cos(p), math.sin(y), math.cos(y), math.sin(r), math.cos(r)
    m = np.array([
        [cp * cy, sr * sp * cy + cr * -sy, cr * sp * cy + -sr * -sy, pos[0]],
        [cp * sy, sr * sp * sy + cr * cy, cr * sp * sy + -sr * cy, pos[1]],
        [-sp, sr * cp, cr * cp, pos[2]],
        [0, 0, 0, 1]])
    return m


phys = ea[b"PhysicsObjects"][0]
M = source_matrix(phys[b"Pos"], phys[b"Angle"])
check(np.allclose(M, np.array(A.matrix_world), atol=1e-4), "Source AngleMatrix(exported angle) == Blender matrix_world")

# Part vertices: entity-local OBJ -> world via Source math == Blender world
import json
crc = ctrls[1][b"crc"].decode()
partlist = json.loads(gmod_lzma.decompress(ea[b"EntityMods"][b"prop2mesh"][2][crc.encode()]))
names = [partlist[k].get("objn") for k in partlist if k != "custom"]
check(set(names) >= {p2m._ascii(o.name) for o in (cube, text, curve)}, "controller 0 parts: %s" % names)
cube_obj = next(partlist["custom"][partlist[k]["objd"]] for k in partlist if k != "custom" and partlist[k].get("objn") == cube.name)
verts = np.array([[float(x) for x in l.split()[1:]] for l in cube_obj.splitlines() if l.startswith("v ")])
world = (M @ np.c_[verts, np.ones(len(verts))].T).T[:, :3]
expect = np.array([cube.matrix_world @ v.co for v in cube.data.vertices])
d = max(np.min(np.linalg.norm(expect - w, axis=1)) for w in world)
check(d < 1e-3 and len(verts) == 8, "cube vertices land at Blender world positions via Source math (err %.1e)" % d)

crc1 = ctrls[2][b"crc"].decode()
pl1 = json.loads(gmod_lzma.decompress(ea[b"EntityMods"][b"prop2mesh"][2][crc1.encode()]))
sph = next(pl1[k] for k in pl1 if k != "custom" and pl1[k].get("objn") == sphere.name)
check(abs(sph.get("vsmooth", 0) - 45) < 1e-6, "smoothing angle exported (vsmooth=%s)" % sph.get("vsmooth"))
check(any(pl1[k].get("objn") == nested.name for k in pl1 if k != "custom"), "indirect child exported under inherited controller")
# mirrored sphere: outward normals in P2M convention (cross(v3-v1, v2-v1))
sobj = pl1["custom"][sph["objd"]]
vs = np.array([[float(x) for x in l.split()[1:]] for l in sobj.splitlines() if l.startswith("v ")])
fs_ = np.array([[int(x) - 1 for x in l.split()[1:]] for l in sobj.splitlines() if l.startswith("f ")])
v1, v2, v3 = vs[fs_[:, 0]], vs[fs_[:, 1]], vs[fs_[:, 2]]
n = np.cross(v3 - v1, v2 - v1)
centre = vs.mean(0)
outward = ((n * ((v1 + v2 + v3) / 3 - centre)).sum(1) > 0).mean()
check(outward > 0.99, "mirrored part keeps outward normals in P2M (%.1f%%)" % (outward * 100))

# Model part: replay Prop2Mesh's getVertsFromMDL on the exported values and
# compare with Blender's (clipped) preview mesh in world space.
props_parts = [partlist[k] for k in partlist if k != "custom" and "prop" in partlist[k]]
check(len(props_parts) == 1 and props_parts[0]["prop"] == MODEL, "model part exported as a prop entry")
if props_parts:
    pp = props_parts[0]
    check("objd" not in pp and pp.get("bodygroup") == mp.p2m.bodygroup_mask() > 0 and pp.get("vsmooth") == 1 and len(pp.get("clips", [])) == 2,
          "prop entry: bodygroup %s, vsmooth %s, %d clips" % (pp.get("bodygroup"), pp.get("vsmooth"), len(pp.get("clips", []))))
    vec = lambda s_: np.array([float(x) for x in s_.strip("[]{}").split()])
    data = studiomdl.load(content.read, MODEL, BG_CHOICES)
    soup = data.positions[data.triangles].astype(np.float64) * vec(pp["scale"])
    for clip in pp["clips"]:
        soup = clipping.clip(soup, vec(clip["n"]), clip["d"])
    R = source_matrix((0, 0, 0), vec(pp["ang"]))[:3, :3]
    local = soup.reshape(-1, 3) @ R.T + vec(pp["pos"])
    world_p2m = (M @ np.c_[local, np.ones(len(local))].T).T[:, :3]
    world_blender = np.array([mp.matrix_world @ v.co for v in mp.data.vertices])
    d1 = max(np.min(np.linalg.norm(world_blender - w, axis=1)) for w in world_p2m)
    d2 = max(np.min(np.linalg.norm(world_p2m - w, axis=1)) for w in world_blender)
    check(d1 < 1e-3 and d2 < 1e-3, "P2M's clipped/scaled/rotated model == Blender preview (err %.1e / %.1e)" % (d1, d2))

open(os.path.join(S, "blender_body.bin"), "wb").write(gmod_lzma.decompress(open(out, "rb").read()[open(out, "rb").read().index(b"\2") + 2:]))

# --- model browser preview ------------------------------------------------
try:
    browser.render_preview("models/props_c17/oildrum001.mdl")
    pv = browser._pcoll.get("models/props_c17/oildrum001.mdl")
    check(pv is not None and tuple(pv.image_size) == (browser.PREVIEW_SIZE, browser.PREVIEW_SIZE), "browser preview icon created")
except Exception:
    traceback.print_exc()
    check(False, "browser preview")

bpy.ops.wm.save_as_mainfile(filepath=os.path.join(S, "p2m_test.blend"))
addon.unregister()
print("\nRESULT:", "ALL PASSED" if not failures else "%d FAILED" % len(failures))
