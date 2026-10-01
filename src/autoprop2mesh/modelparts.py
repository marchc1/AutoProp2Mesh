"""Model parts: Prop2Mesh "prop" parts. In Blender they show the model (with
the chosen body groups, cut by their clip planes); on export only the model
path, placement, scale, clips and body group mask are written.

Clip planes are child objects of the part. A plane keeps the side its local
+Z axis points to (the arrow). Prop2Mesh applies clips to the scaled model,
before rotation, keeping dot(v, n) >= d.
"""

from collections import OrderedDict

import bmesh
import bpy
import numpy as np
from mathutils import Vector

from . import content, props
from .core import clipping, studiomdl, vfs

DEFAULT_MODEL = "models/hunter/blocks/cube025x025x025.mdl"
MESH_PREFIX = "P2M Part: "

_models = OrderedDict()   # (path, choices) -> ModelData
_MODEL_CACHE_SIZE = 64
_signatures = {}          # object pointer -> last built signature
_queued = set()
_building = set()         # objects being rebuilt (ignore their prop updates)


def _load(path, choices):
    key = (path, tuple(choices))
    data = _models.get(key)
    if data is not None:
        _models.move_to_end(key)
        return data
    data = studiomdl.load(content.read, path, choices)
    _models[key] = data
    while len(_models) > _MODEL_CACHE_SIZE:
        _models.popitem(last=False)
    return data


def clear_cache():
    _models.clear()
    _signatures.clear()


# --------------------------------------------------------------------------
# Clip planes
# --------------------------------------------------------------------------

def clip_plane_objects(part):
    return [c for c in part.children if c.p2m.is_clip_plane]


def plane_in_part_space(part, plane_obj):
    """(unit normal, distance) of a clip plane in the part's local space."""
    m = part.matrix_world.inverted() @ plane_obj.matrix_world
    normal = (m.to_3x3().inverted().transposed() @ Vector((0.0, 0.0, 1.0)))
    if normal.length < 1e-12:
        return None
    normal.normalize()
    point = m.translation
    return normal, normal.dot(point)


def local_planes(part):
    out = []
    for c in clip_plane_objects(part):
        pl = plane_in_part_space(part, c)
        if pl is not None:
            out.append(pl)
    return out


def export_clips(local, scale, unit_scale):
    """Converts part-space planes (Blender units) to Prop2Mesh clips: planes
    in the scaled model space (Source units). With x = s * v * u,
    dot(v, m) >= dl  <=>  dot(x, m / s) >= u * dl."""
    out = []
    for m, dl in local:
        ms = Vector((m.x / scale.x, m.y / scale.y, m.z / scale.z))
        length = ms.length
        if length < 1e-12:
            continue
        n = ms / length
        out.append(((n.x, n.y, n.z), unit_scale * dl / length))
    return out


def _plane_mesh(size):
    me = bpy.data.meshes.new("P2M Clip Plane")
    bm = bmesh.new()
    h = size / 2
    corners = [bm.verts.new(v) for v in ((-h, -h, 0), (h, -h, 0), (h, h, 0), (-h, h, 0))]
    for i in range(4):
        bm.edges.new((corners[i], corners[(i + 1) % 4]))
    # Arrow pointing to the kept side (+Z).
    base = bm.verts.new((0, 0, 0))
    tip = bm.verts.new((0, 0, size * 0.35))
    bm.edges.new((base, tip))
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        bm.edges.new((tip, bm.verts.new((dx * size * 0.06, dy * size * 0.06, size * 0.27))))
    bm.edges.new((corners[0], corners[2]))
    bm.edges.new((corners[1], corners[3]))
    bm.to_mesh(me)
    bm.free()
    return me


def add_clip_plane(context, part):
    size = max(max(part.dimensions) / max(max(part.scale), 1e-6), 1e-3) * 0.75 or 1.0
    me = _plane_mesh(size)
    plane = bpy.data.objects.new("Clip Plane", me)
    for coll in part.users_collection:
        coll.objects.link(plane)
        break
    else:
        context.collection.objects.link(plane)
    plane.p2m.is_clip_plane = True
    plane.parent = part
    plane.matrix_parent_inverse.identity()
    plane.display_type = "WIRE"
    plane.show_in_front = True
    plane.hide_render = True
    rebuild(part)
    return plane


# --------------------------------------------------------------------------
# Preview mesh
# --------------------------------------------------------------------------

def _signature(obj):
    planes = tuple((tuple(round(c, 5) for c in n), round(d, 5)) for n, d in local_planes(obj))
    return (vfs.FileSystem.normalize(obj.p2m.model or ""), tuple(obj.p2m.bodygroup_choices()),
            obj.p2m.model_flat, planes, round(props.unit_scale(), 6))


def _sync_bodygroups(obj, data):
    p = obj.p2m
    want = [(name, base, "\n".join(models)) for name, base, models in data.bodyparts]
    have = [(bg.name, bg.base, bg.options) for bg in p.bodygroups]
    if want == have:
        return False
    choices = p.bodygroup_choices()
    p.bodygroups.clear()
    for i, (name, base, options) in enumerate(want):
        bg = p.bodygroups.add()
        bg.name, bg.base, bg.options = name, base, options
        c = choices[i] if i < len(choices) and have and have[i][2] == options else 0
        bg.choice = str(c)
    return True


def _fill_mesh(me, soup, mat_index, flat):
    """soup columns: pos(3) normal(3) uv(2)."""
    me.clear_geometry()
    if len(soup) == 0:
        return
    # Weld identical positions so smoothing/editing behave.
    rounded = np.round(soup[:, :, 0:3], 5)
    unique, inverse = np.unique(rounded.reshape(-1, 3), axis=0, return_inverse=True)
    tri_verts = inverse.reshape(-1, 3)
    # A face may not use a vertex twice (welding can collapse thin or cut
    # triangles); Blender's normal code crashes on such faces.
    ok = (tri_verts[:, 0] != tri_verts[:, 1]) & (tri_verts[:, 1] != tri_verts[:, 2]) & (tri_verts[:, 0] != tri_verts[:, 2])
    soup, mat_index, tri_verts = soup[ok], mat_index[ok], tri_verts[ok]
    m = len(soup)
    if m == 0:
        return
    used, inverse = np.unique(tri_verts.reshape(-1), return_inverse=True)
    unique = unique[used]
    inverse = inverse.reshape(-1)
    corners = soup.reshape(-1, soup.shape[2])
    me.vertices.add(len(unique))
    me.vertices.foreach_set("co", unique.astype(np.float32).reshape(-1))
    me.loops.add(m * 3)
    me.loops.foreach_set("vertex_index", inverse.astype(np.int32))
    me.polygons.add(m)
    me.polygons.foreach_set("loop_start", np.arange(0, m * 3, 3, dtype=np.int32))
    me.polygons.foreach_set("material_index", mat_index.astype(np.int32))
    uv = me.uv_layers.get("UVMap") or me.uv_layers.new(name="UVMap")
    uv.data.foreach_set("uv", corners[:, 6:8].astype(np.float32).reshape(-1))
    me.update(calc_edges=True)
    # Belt and braces: never hand Blender's normal code invalid topology.
    if me.validate(clean_customdata=False) or len(me.loops) != m * 3:
        return
    if not flat:
        me.polygons.foreach_set("use_smooth", np.ones(m, dtype=bool))
        normals = corners[:, 3:6]
        lengths = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = np.where(lengths > 1e-12, normals / np.maximum(lengths, 1e-12), (0.0, 0.0, 1.0))
        if np.isfinite(normals).all():
            me.normals_split_custom_set(normals.tolist())


def rebuild(obj, refresh_bodygroups=False, force=False):
    """(Re)builds a model part's preview mesh when its inputs changed."""
    if not obj.p2m.is_model_part or obj.as_pointer() in _building:
        return
    _building.add(obj.as_pointer())
    try:
        _rebuild(obj, refresh_bodygroups, force)
    finally:
        _building.discard(obj.as_pointer())


def _rebuild(obj, refresh_bodygroups, force):
    sig = _signature(obj)
    if not force and not refresh_bodygroups and _signatures.get(obj.as_pointer()) == sig:
        return
    path = sig[0]
    status = ""
    data = None
    if content.get_fs() is None:
        status = content.status_text()
    else:
        try:
            data = _load(path, obj.p2m.bodygroup_choices() if not refresh_bodygroups else [])
            if _sync_bodygroups(obj, data) or refresh_bodygroups:
                data = _load(path, obj.p2m.bodygroup_choices())
        except Exception as e:
            status = "%s (showing error.mdl)" % e
            try:
                data = _load(content.ERROR_MODEL, [])
            except Exception as e2:
                status = "%s; error.mdl unavailable: %s" % (e, e2)
    obj.p2m.model_status = status

    if data is None:
        old = obj.data
        obj.data = content.warning_mesh(props.unit_scale())
        if old is not None and old.users == 0 and old.name.startswith(MESH_PREFIX):
            bpy.data.meshes.remove(old)
        _signatures[obj.as_pointer()] = _signature(obj)
        return

    me = obj.data
    if me is None or me.users > 1 or not me.name.startswith(MESH_PREFIX):
        me = bpy.data.meshes.new(MESH_PREFIX + obj.name)
        obj.data = me

    us = props.unit_scale()
    tris = data.triangles
    soup = np.concatenate([
        data.positions[tris].astype(np.float64) / us,
        data.normals[tris].astype(np.float64),
        data.uvs[tris].astype(np.float64),
        np.repeat(data.tri_material[:, None, None], 3, axis=1).astype(np.float64),
    ], axis=2) if len(tris) else np.zeros((0, 3, 9))
    soup = clipping.clip_all(soup, [((n.x, n.y, n.z), d) for n, d in local_planes(obj)])
    mat_index = soup[:, 0, 8].round().astype(np.int32) if len(soup) else np.zeros(0, np.int32)

    _fill_mesh(me, soup[:, :, :8], mat_index, obj.p2m.model_flat)
    mats = [content.source_material(c) for c in data.materials]
    if [m for m in me.materials] != mats:
        me.materials.clear()
        for m in mats:
            me.materials.append(m)
    me.update()
    _signatures[obj.as_pointer()] = _signature(obj)
    # Material slot count may have changed: let sync re-bind the controller.
    from . import sync
    sync.request_full_sync()


def queue(obj):
    if not _queued:
        bpy.app.timers.register(_flush, first_interval=0.0)
    _queued.add(obj.name)


def _flush():
    names = list(_queued)
    _queued.clear()
    for name in names:
        obj = bpy.data.objects.get(name)
        if obj is not None and obj.p2m.is_model_part:
            try:
                rebuild(obj)
            except Exception as e:
                print("AutoProp2Mesh: model part rebuild failed:", e)
    return None


def check_all():
    """Rebuilds parts whose planes changed without a depsgraph update (e.g. a
    clip plane was deleted)."""
    for obj in bpy.data.objects:
        if obj.p2m.is_model_part and obj.library is None:
            if _signatures.get(obj.as_pointer()) != _signature(obj):
                rebuild(obj)


def create_model_part(context, location=None, model=DEFAULT_MODEL):
    me = bpy.data.meshes.new(MESH_PREFIX + "Model Part")
    obj = bpy.data.objects.new("P2M Model Part", me)
    context.collection.objects.link(obj)
    if location is not None:
        obj.location = location
    obj.p2m.is_model_part = True
    obj.p2m.model = model
    if not obj.p2m.bodygroups and not len(obj.data.polygons):
        rebuild(obj, refresh_bodygroups=True)
    return obj
