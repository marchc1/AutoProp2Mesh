"""AdvDupe2 export.

Coordinates: Blender and Source are both right-handed and Z-up, so positions
map straight across (scaled by the scene's Source-units-per-Blender-unit).
Entity forward is +X in both. A Source QAngle (pitch, yaw, roll) is the
rotation Rz(yaw) * Ry(pitch) * Rx(roll), which is Blender's XYZ Euler with
x = roll, y = pitch, z = yaw.

The dupe origin (the point AdvDupe2 pastes at) is Blender's world origin.
"""

import math
import os

import bpy
import numpy as np
from bpy.props import BoolProperty, StringProperty
from bpy_extras.io_utils import ExportHelper
from mathutils import Matrix

from . import entities, modelparts, prefs, props, sync, uvpreview
from .core import ad2, p2m


def matrix_to_source(matrix, unit_scale):
    """World matrix -> (Source position, Source angle, scale-free matrix)."""
    loc, rot, _scale = matrix.decompose()
    e = rot.to_euler("XYZ")
    ang = (math.degrees(e.y), math.degrees(e.z), math.degrees(e.x))
    pos = (loc.x * unit_scale, loc.y * unit_scale, loc.z * unit_scale)
    frame = Matrix.Translation(loc) @ rot.to_matrix().to_4x4()
    return pos, ang, frame


def _smooth_angle(obj, mesh):
    mode = obj.p2m.smooth_mode
    if mode == "FLAT":
        return None
    if mode == "SMOOTH":
        return obj.p2m.smooth_angle or None
    n = len(mesh.polygons)
    if n == 0:
        return None
    flags = np.zeros(n, dtype=bool)
    mesh.polygons.foreach_get("use_smooth", flags)
    if not flags.any():
        return None
    # Blender 4.1+ "Smooth by Angle" modifier: use its angle.
    for mod in obj.modifiers:
        if mod.type == "NODES" and mod.node_group and mod.node_group.name.startswith("Smooth by Angle") \
                and mod.show_viewport:
            for item in mod.node_group.interface.items_tree:
                if getattr(item, "in_out", None) == "INPUT" and item.name == "Angle":
                    value = uvpreview._get_input(mod, item.identifier)
                    if isinstance(value, (int, float)):
                        return math.degrees(float(value))
    return 180.0


def part_geometry(obj, depsgraph, entity_frame_inv, unit_scale):
    ob_eval = obj.evaluated_get(depsgraph)
    try:
        mesh = ob_eval.to_mesh()
    except RuntimeError:
        return None
    if mesh is None:
        return None
    try:
        mesh.calc_loop_triangles()
        nv = len(mesh.vertices)
        nt = len(mesh.loop_triangles)
        if nv == 0 or nt == 0:
            return None
        co = np.empty(nv * 3, dtype=np.float32)
        mesh.vertices.foreach_get("co", co)
        tris = np.empty(nt * 3, dtype=np.int32)
        mesh.loop_triangles.foreach_get("vertices", tris)
        vsmooth = _smooth_angle(obj, mesh)
    finally:
        ob_eval.to_mesh_clear()

    m = np.array(entity_frame_inv @ ob_eval.matrix_world, dtype=np.float64)
    pos = co.reshape(-1, 3).astype(np.float64) @ m[:3, :3].T + m[:3, 3]
    pos *= unit_scale
    tris = tris.reshape(-1, 3)
    if np.linalg.det(m[:3, :3]) < 0:  # mirrored: winding flips
        tris = tris[:, ::-1]
    return p2m.PartGeometry(obj.name, pos, tris, vsmooth, obj.p2m.render_inside)


def model_part_spec(obj, entity_frame_inv, unit_scale, warnings):
    """A model part: only the model path and placement are exported."""
    m = entity_frame_inv @ obj.matrix_world
    loc, rot, scale = m.decompose()
    rebuilt = Matrix.LocRotScale(loc, rot, scale)
    if any(abs(a - b) > 1e-4 for ra, rb in zip(m, rebuilt) for a, b in zip(ra, rb)):
        warnings.append("%s is skewed by a non-uniformly scaled parent; Prop2Mesh cannot represent that" % obj.name)
    e = rot.to_euler("XYZ")
    clips = modelparts.export_clips(modelparts.local_planes(obj), scale, unit_scale)
    if obj.p2m.model_status:
        warnings.append("%s: %s" % (obj.name, obj.p2m.model_status))
    model = obj.p2m.model.strip().replace("\\", "/").lower()
    if model and not model.endswith(".mdl"):
        model += ".mdl"
    return p2m.ModelPartSpec(
        obj.name, model,
        (loc.x * unit_scale, loc.y * unit_scale, loc.z * unit_scale),
        (math.degrees(e.y), math.degrees(e.z), math.degrees(e.x)),
        tuple(scale), clips, obj.p2m.bodygroup_mask(), obj.p2m.model_flat, obj.p2m.render_inside)


def _visible(obj, context):
    try:
        return obj.visible_get(view_layer=context.view_layer)
    except Exception:
        return not obj.hide_viewport


def gather(context, selection_only=False, include_hidden=False):
    """Returns (list of EntitySpec, warnings)."""
    scene = context.scene
    us = props.unit_scale(scene)
    warnings = []
    all_entities = [o for o in scene.objects if o.p2m.is_entity]
    if selection_only:
        chosen = set()
        for o in context.selected_objects:
            if o.p2m.is_entity:
                chosen.add(o)
            else:
                ent = entities.owning_entity(o)
                if ent is not None:
                    chosen.add(ent)
        export_entities = [o for o in all_entities if o in chosen]
    else:
        export_entities = all_entities
    if not export_entities:
        return [], ["No Prop2Mesh entities to export"]
    exported = set(export_entities)

    depsgraph = context.evaluated_depsgraph_get()
    specs = []
    unassigned = 0
    for ent in export_entities:
        pos, ang, frame = matrix_to_source(ent.matrix_world, us)
        scale = ent.matrix_world.to_scale()
        if any(abs(s - 1.0) > 1e-4 for s in scale):
            warnings.append("%s is scaled; entity scale is ignored (parts keep their world size)" % ent.name)
        frame_inv = frame.inverted()

        parent_key = None
        p = ent.parent
        while p is not None:
            if p in exported:
                parent_key = p.name
                break
            p = p.parent

        by_uid = {c.uid: [] for c in ent.p2m.controllers}
        for obj, ctrl in entities.entity_parts(ent, include_unassigned=True):
            if ctrl is None:
                unassigned += 1
                continue
            if not include_hidden and not _visible(obj, context):
                continue
            if obj.p2m.is_model_part:
                by_uid[ctrl.uid].append(model_part_spec(obj, frame_inv, us, warnings))
                continue
            geo = part_geometry(obj, depsgraph, frame_inv, us)
            if geo is not None:
                by_uid[ctrl.uid].append(geo)

        controllers = []
        for c in ent.p2m.controllers:
            controllers.append(p2m.ControllerSpec(
                color=[v * 255.0 for v in c.color], material=c.material.strip(),
                uvs=c.uvs, bump=c.bump, name=c.name.strip(), parts=by_uid[c.uid]))
        specs.append(p2m.EntitySpec(ent.name, ent.p2m.model.strip(), pos, ang, parent_key, controllers))
    if unassigned:
        warnings.append("%d object(s) below entities have no controller and were skipped" % unassigned)
    return specs, warnings


def write_dupe(filepath, specs, export_names=True, author="AutoProp2Mesh"):
    dupe, report = p2m.build_dupe(specs, export_names=export_names)
    data = ad2.encode(dupe, {"name": author})
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with open(filepath, "wb") as f:
        f.write(data)
    report["bytes"] = len(data)
    return report


class P2M_OT_export_advdupe2(bpy.types.Operator, ExportHelper):
    """Export Prop2Mesh entities as an AdvDupe2 file"""
    bl_idname = "export_scene.p2m_advdupe2"
    bl_label = "Export AdvDupe2 (Prop2Mesh)"
    bl_options = {"PRESET"}

    filename_ext = ".txt"
    filter_glob: StringProperty(default="*.txt", options={"HIDDEN"})
    selection_only: BoolProperty(
        name="Selected Entities Only", default=False,
        description="Export only selected entities (or entities owning selected parts)",
    )
    include_hidden: BoolProperty(name="Include Hidden Parts", default=False)
    export_names: BoolProperty(name="Export Controller Names", default=True)

    def invoke(self, context, event):
        directory = prefs.default_dupe_dir()
        name = bpy.path.display_name_from_filepath(bpy.data.filepath) or "p2m_export"
        if directory:
            self.filepath = os.path.join(directory, name + self.filename_ext)
        else:
            self.filepath = name + self.filename_ext
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        sync.full_sync(context.scene)
        specs, warnings = gather(context, self.selection_only, self.include_hidden)
        if not specs:
            self.report({"ERROR"}, warnings[0] if warnings else "Nothing to export")
            return {"CANCELLED"}
        try:
            report = write_dupe(self.filepath, specs, self.export_names)
        except Exception as e:
            self.report({"ERROR"}, "Export failed: %s" % e)
            return {"CANCELLED"}
        for w in warnings + report["warnings"]:
            self.report({"WARNING"}, w)
        if report["bytes"] > p2m.MAX_DUPE_BYTES:
            self.report({"WARNING"}, "Dupe is larger than AdvDupe2's 32 MB limit")
        self.report({"INFO"}, "Exported %d entities, %d controllers, %d parts (%d model parts, %d mesh triangles), %.1f KB to %s" % (
            report["entities"], report["controllers"], report["parts"], report["models"], report["triangles"],
            report["bytes"] / 1024.0, os.path.basename(self.filepath)))
        return {"FINISHED"}

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.prop(self, "selection_only")
        layout.prop(self, "include_hidden")
        layout.prop(self, "export_names")
        layout.prop(context.scene.p2m, "unit_scale", text="Units per BU")


def menu_func_export(self, context):
    self.layout.operator(P2M_OT_export_advdupe2.bl_idname, text="AdvDupe2 Prop2Mesh (.txt)")


classes = (P2M_OT_export_advdupe2,)
