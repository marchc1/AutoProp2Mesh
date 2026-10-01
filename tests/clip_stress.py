"""Clip plane stress test (needs a GMod install). A crash = segfault/exit code;
success prints "STRESS OK". Covers planes exactly through vertices / coplanar
with faces (which used to crash Blender's custom normals), planes clipping
everything or nothing, random multi-plane setups, flat and smooth shading.

    blender -b --factory-startup --python tests/clip_stress.py
"""
import sys, os, random
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import bpy
from mathutils import Vector, Euler
import autoprop2mesh as addon
from autoprop2mesh import modelparts, content
from autoprop2mesh.core import clipping, studiomdl
import numpy as np
addon.register()
fs = content.get_fs()
ctx = bpy.context
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o)
log = sys.stdout
models = ["models/hunter/blocks/cube025x025x025.mdl", "models/props_c17/oildrum001.mdl", "models/hunter/plates/plate1x1.mdl",
          "models/props_junk/wood_crate001a.mdl", "models/p2m/cube.mdl", "models/props_borealis/bluebarrel001.mdl"]
for m in models:
    try:
        data = studiomdl.load(content.read, m)
    except Exception as e:
        log.write("skip %s %s\n" % (m, e)); continue
    zmin, zmax = data.positions[:, 2].min(), data.positions[:, 2].max()
    # count degenerate triangles produced by a plane through the origin / exact bbox planes
    for z in (0.0, float(zmin), float(zmax)):
        soup = data.positions[data.triangles].astype(np.float64)
        out = clipping.clip(soup, (0, 0, 1), z)
        area = np.linalg.norm(np.cross(out[:, 1] - out[:, 0], out[:, 2] - out[:, 0]), axis=1)
        log.write("%s plane z=%.3f -> %d tris, %d zero-area\n" % (m, z, len(out), int((area < 1e-9).sum())))
    mp = modelparts.create_model_part(ctx, location=(0, 0, 0))
    mp.p2m.model = m
    for z in (0.0, float(zmin), float(zmax), float(zmax) + 5):
        plane = modelparts.add_clip_plane(ctx, mp)
        plane.location = (0, 0, z)
        ctx.view_layer.update()
        log.write("rebuild %s z=%.3f ... " % (m, z))
        modelparts.rebuild(mp, force=True)
        log.write("ok, %d polys\n" % len(mp.data.polygons))
        bpy.data.objects.remove(plane)
log.write("SURVIVED\n")
# --- stress: planes through every distinct vertex height on each axis, random planes, multi-plane, flat/smooth
rng = random.Random(3)
count = 0
for m in models:
    try:
        data = studiomdl.load(content.read, m)
    except Exception:
        continue
    mp = modelparts.create_model_part(ctx, location=(0, 0, 0))
    mp.p2m.model = m
    cases = []
    for axis, rot in ((2, (0, 0, 0)), (0, (0, 1.5707963, 0)), (1, (-1.5707963, 0, 0))):
        levels = sorted(set(np.round(data.positions[:, axis], 4).tolist()))
        picks = levels if len(levels) <= 12 else rng.sample(levels, 12)
        for lv in picks:
            loc = [0, 0, 0]; loc[axis] = lv
            cases.append([(tuple(loc), rot)])
            cases.append([(tuple(loc), (rot[0] + 3.14159265, rot[1], rot[2]))])  # flipped
    for _ in range(25):
        cases.append([((rng.uniform(-30, 30), rng.uniform(-30, 30), rng.uniform(-30, 30)),
                       (rng.uniform(-3, 3), rng.uniform(-3, 3), rng.uniform(-3, 3))) for _ in range(rng.randint(1, 4))])
    cases.append([((0, 0, 1e6), (0, 0, 0))])      # clips everything
    cases.append([((0, 0, -1e6), (0, 0, 0))])     # clips nothing
    for flat in (False, True):
        mp.p2m.model_flat = flat
        for planes in cases:
            objs = []
            for loc, rot in planes:
                p = modelparts.add_clip_plane(ctx, mp)
                p.location = loc
                p.rotation_euler = rot
                objs.append(p)
            ctx.view_layer.update()
            modelparts.rebuild(mp, force=True)
            mp.data.validate(verbose=False)
            for p in objs:
                bpy.data.objects.remove(p)
            count += 1
log.write("STRESS OK: %d clip configurations rebuilt without crashing\n" % count)
