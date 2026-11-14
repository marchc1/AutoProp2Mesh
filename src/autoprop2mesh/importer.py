"""AdvDupe2 import. Prop2Mesh entities come in as entities with their
controllers, mesh parts and model parts; other classes are skipped for now.

Coordinates are the inverse of export.py: the dupe's paste origin lands on
Blender's world origin.
"""

import math
import os
from collections import Counter

import bpy
import numpy as np
from bpy.props import BoolProperty, StringProperty
from bpy_extras.io_utils import ImportHelper
from mathutils import Euler, Matrix, Vector

from . import content, entities, modelparts, prefs, props, sync
from .core import ad2, dupe, p2m

DEFAULT_UV_SCALE = 48


def source_matrix(pos, ang, unit_scale, scale=(1.0, 1.0, 1.0)):
    """Source position + QAngle (pitch, yaw, roll) -> Blender matrix."""
    p, y, r = ang
    rot = Euler((math.radians(r), math.radians(p), math.radians(y)), "XYZ").to_matrix().to_4x4()
    return Matrix.Translation(Vector(pos) / unit_scale) @ rot @ Matrix.Diagonal((*scale, 1.0))


class ImportContext:
    def __init__(self, context, collection, unit_scale):
        self.context = context
        self.collection = collection
        self.unit_scale = unit_scale
        self.notes = {}
        self.warnings = []
        self.stats = Counter()
        self.clipped_parts = []


# --------------------------------------------------------------------------
# Prop2Mesh
# --------------------------------------------------------------------------

def _mesh_object(name, positions, triangles, collection):
    # weld matching positions, P2M smooths across them anyway
    unique, inverse = np.unique(np.round(positions, 5), axis=0, return_inverse=True)
    tris = inverse.reshape(-1)[triangles]
    ok = (tris[:, 0] != tris[:, 1]) & (tris[:, 1] != tris[:, 2]) & (tris[:, 0] != tris[:, 2])
    tris = tris[ok]
    # Blender drops faces on the same vertices as another face (double sided panels), give those their own copies
    _, first, counts = np.unique(np.sort(tris, axis=1), axis=0, return_index=True, return_counts=True)
    if (counts > 1).any():
        repeat = np.ones(len(tris), dtype=bool)
        repeat[first] = False
        extra = tris[repeat]
        tris[repeat] = len(unique) + np.arange(extra.size).reshape(-1, 3)
        unique = np.concatenate([unique, unique[extra.reshape(-1)]])
    me = bpy.data.meshes.new(name)
    me.vertices.add(len(unique))
    me.vertices.foreach_set("co", unique.astype(np.float32).reshape(-1))
    me.loops.add(len(tris) * 3)
    me.loops.foreach_set("vertex_index", tris.astype(np.int32).reshape(-1))
    me.polygons.add(len(tris))
    me.polygons.foreach_set("loop_start", np.arange(0, len(tris) * 3, 3, dtype=np.int32))
    me.update(calc_edges=True)
    me.validate(clean_customdata=False)
    obj = bpy.data.objects.new(name, me)
    collection.objects.link(obj)
    return obj


def _smooth(obj, angle):
    me = obj.data
    if not len(me.polygons):
        return
    me.polygons.foreach_set("use_smooth", np.ones(len(me.polygons), dtype=bool))
    if hasattr(me, "set_sharp_from_angle"):  # Blender 4.1+
        me.set_sharp_from_angle(angle=math.radians(min(angle, 180.0)))
    obj.p2m.smooth_mode = "SMOOTH"
    obj.p2m.smooth_angle = min(angle, 180.0)


def _model_part(ic, spec, parent, uid):
    us = ic.unit_scale
    obj = modelparts.create_model_part(ic.context, model=spec.model, name=spec.name, collection=ic.collection,
                                       bodygroup=spec.bodygroup)
    obj.p2m.model_holo = spec.holo
    obj.p2m.model_flat = spec.flat
    obj.p2m.render_inside = spec.vinside
    obj.parent = parent
    obj.matrix_basis = source_matrix(spec.pos, spec.ang, us, spec.scale)
    obj.p2m.controller_uid = uid
    if modelparts.fix_rule(obj) is not None:
        ic.clipped_parts.append(obj)   # turned models preview by their scale
    s = Vector(spec.scale)
    for n, d in spec.clips:
        # P2M clips live in the scaled model space (Source units), planes in the unscaled part space
        n = Vector(n)
        local_n = Vector((n.x * s.x, n.y * s.y, n.z * s.z))
        point = Vector(c * d / ((sc if abs(sc) > 1e-9 else 1e-9) * us) for c, sc in zip(n, s))
        if local_n.length > 1e-12:
            modelparts.add_clip_plane(ic.context, obj, local_n, point, rebuild_part=False)
    if spec.clips and obj not in ic.clipped_parts:
        ic.clipped_parts.append(obj)
    if obj.p2m.model_status:
        ic.stats["model_errors"] += 1
    ic.stats["model_parts"] += 1
    return obj


def _mesh_part(ic, spec, parent, uid):
    obj = _mesh_object(spec.name, spec.positions / ic.unit_scale, spec.triangles, ic.collection)
    obj.parent = parent
    obj.matrix_basis = source_matrix(spec.pos, spec.ang, ic.unit_scale)
    if spec.vsmooth:
        _smooth(obj, spec.vsmooth)
    obj.p2m.render_inside = spec.vinside
    obj.p2m.controller_uid = uid
    ic.stats["mesh_parts"] += 1
    ic.stats["triangles"] += len(spec.triangles)
    return obj


def _uv_scale(ic, cs):
    """P2M box-projects before the controller scale; the scale gets baked into
    the parts, so the UV scale follows it where it can."""
    s = tuple(cs.scale)
    if all(abs(c - 1.0) < 1e-6 for c in s):
        return cs.uvs
    has_mesh = any(isinstance(p, p2m.MeshPartSpec) for p in cs.parts)
    has_model = any(isinstance(p, p2m.ModelPartSpec) for p in cs.parts)
    if max(s) - min(s) > 1e-6:
        if has_mesh or cs.uvs:
            ic.notes["uv_nonuniform"] = ic.notes.get("uv_nonuniform", 0) + 1
        return cs.uvs
    if cs.uvs:
        uvs = cs.uvs * abs(s[0])
    elif has_mesh and not has_model:
        uvs = DEFAULT_UV_SCALE * abs(s[0])
    else:
        if has_mesh:
            ic.notes["uv_mixed"] = ic.notes.get("uv_mixed", 0) + 1
        return cs.uvs
    if not 1 <= round(uvs) <= 512:
        ic.notes["uv_clamped"] = ic.notes.get("uv_clamped", 0) + 1
    return int(max(1, min(512, round(uvs))))


def build_p2m(ic, de):
    spec = p2m.read_entity(de, ic.notes)
    model = spec.model.replace("\\", "/").lower()
    ent = entities.create_entity(ic.context, model=model, name="P2M %d" % de.index, collection=ic.collection)
    ent.matrix_world = source_matrix(spec.pos, spec.ang, ic.unit_scale)
    if ent.p2m.model_status:
        ic.stats["model_errors"] += 1
    for i, cs in enumerate(spec.controllers):
        ctrl = ent.p2m.controllers[0] if i == 0 else entities.add_controller(ent)
        ctrl.name = cs.name
        ctrl.color = [c / 255.0 for c in cs.color]
        ctrl.material = cs.material
        ctrl.uvs = _uv_scale(ic, cs)
        ctrl.bump = cs.bump
        frame = ent
        if any(tuple(cs.linkpos)) or any(tuple(cs.linkang)) or any(abs(c - 1.0) > 1e-6 for c in cs.scale):
            # empty holding the controller's link offset and scale
            frame = bpy.data.objects.new("%s #%d" % (ctrl.name or "Controller", i), None)
            frame.empty_display_type = "ARROWS"
            frame.empty_display_size = 4.0 / ic.unit_scale
            ic.collection.objects.link(frame)
            frame.parent = ent
            frame.matrix_basis = source_matrix(cs.linkpos, cs.linkang, ic.unit_scale, cs.scale)
            frame.p2m.controller_uid = ctrl.uid
        for part in cs.parts:
            if isinstance(part, p2m.ModelPartSpec):
                _model_part(ic, part, frame, ctrl.uid)
            else:
                _mesh_part(ic, part, frame, ctrl.uid)
        ic.stats["controllers"] += 1
    ic.stats["entities"] += 1
    return ent


# Entity class -> builder(ImportContext, DupeEntity) -> Blender object. Add new classes here.
BUILDERS = {cls: build_p2m for cls in p2m.ENTITY_CLASSES}

_EXTRA_NOTES = {
    "uv_nonuniform": "%d controller(s) are scaled non-uniformly; their textures will stretch differently after export",
    "uv_mixed": "%d scaled controller(s) mix mesh and model parts with UV scale 0; their mesh textures will scale differently after export",
    "uv_clamped": "%d scaled controller(s) needed a UV scale outside 1-512; it was clamped",
}


def _ancestors(index, by_index):
    """Dupe parent chain of an entity, or None if it loops."""
    chain = []
    p = by_index[index].parent
    while p is not None and p in by_index:
        if p == index or p in chain:
            return None
        chain.append(p)
        p = by_index[p].parent
    return chain


def _parent_all(objects, ents):
    """Parents each imported object to its nearest imported dupe ancestor, keeping world transforms."""
    by_index = {de.index: de for de in ents}
    world = {i: o.matrix_world.copy() for i, o in objects.items()}
    for index, obj in objects.items():
        parent = next((p for p in _ancestors(index, by_index) or () if p in objects), None)
        if parent is None:
            continue
        obj.parent = objects[parent]
        obj.matrix_parent_inverse = Matrix.Identity(4)
        obj.matrix_basis = world[parent].inverted() @ world[index]


def import_dupe(context, filepath, new_collection=True):
    """Returns (ImportContext, created objects by dupe index)."""
    with open(filepath, "rb") as f:
        table, _info = ad2.decode(f.read())
    ents = dupe.read_entities(table)

    if new_collection:
        collection = bpy.data.collections.new(bpy.path.display_name_from_filepath(filepath) or "AdvDupe2")
        context.scene.collection.children.link(collection)
    else:
        collection = context.collection or context.scene.collection
    ic = ImportContext(context, collection, props.unit_scale(context.scene))
    content.get_fs()  # wait for the content index, model parts need it

    objects = {}
    skipped = Counter()
    for de in ents:
        builder = BUILDERS.get(de.cls)
        if builder is None:
            skipped[de.cls or "?"] += 1
            continue
        objects[de.index] = builder(ic, de)
    _parent_all(objects, ents)

    # clip planes are read through matrix_world, which needs the parenting evaluated
    context.view_layer.update()
    for obj in ic.clipped_parts:
        modelparts.rebuild(obj, force=True)
    sync.full_sync(context.scene)

    ic.warnings += p2m.note_warnings(ic.notes)
    ic.warnings += [_EXTRA_NOTES[k] % v for k, v in ic.notes.items() if k in _EXTRA_NOTES]
    if ic.stats["model_errors"]:
        ic.warnings.append("%d model(s) could not be loaded and show error.mdl (missing content?)" % ic.stats["model_errors"])
    ic.skipped = skipped
    return ic, objects


class P2M_OT_import_advdupe2(bpy.types.Operator, ImportHelper):
    """Import the Prop2Mesh entities from an AdvDupe2 file"""
    bl_idname = "import_scene.p2m_advdupe2"
    bl_label = "Import AdvDupe2 (Prop2Mesh)"
    bl_options = {"REGISTER", "UNDO", "PRESET"}

    filename_ext = ".txt"
    filter_glob: StringProperty(default="*.txt", options={"HIDDEN"})
    new_collection: BoolProperty(
        name="New Collection", default=True,
        description="Put the imported objects in a new collection named after the dupe")

    def invoke(self, context, event):
        directory = prefs.default_dupe_dir()
        if directory and os.path.isdir(directory) and not self.filepath:
            self.filepath = directory + os.sep
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        if context.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        try:
            ic, objects = import_dupe(context, self.filepath, self.new_collection)
        except Exception as e:
            self.report({"ERROR"}, "Import failed: %s" % e)
            return {"CANCELLED"}
        if not objects:
            self.report({"WARNING"}, "No Prop2Mesh entities in %s" % os.path.basename(self.filepath))
            return {"CANCELLED"}

        for o in context.selected_objects:
            o.select_set(False)
        for obj in objects.values():
            obj.select_set(True)
        context.view_layer.objects.active = next(iter(objects.values()))
        content.show_textures_in_solid_view(context)

        for w in ic.warnings:
            self.report({"WARNING"}, w)
        s = ic.stats
        msg = "Imported %d entities, %d controllers, %d model parts, %d mesh parts (%d triangles)" % (
            s["entities"], s["controllers"], s["model_parts"], s["mesh_parts"], s["triangles"])
        if ic.skipped:
            msg += "; skipped %d other entities (%s)" % (
                sum(ic.skipped.values()), ", ".join("%s x%d" % kv for kv in ic.skipped.most_common(4)))
        self.report({"INFO"}, msg)
        return {"FINISHED"}

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.prop(self, "new_collection")
        layout.prop(context.scene.p2m, "unit_scale", text="Units per BU")


def menu_func_import(self, context):
    self.layout.operator(P2M_OT_import_advdupe2.bl_idname, text="AdvDupe2 Prop2Mesh (.txt)")


classes = (P2M_OT_import_advdupe2,)
