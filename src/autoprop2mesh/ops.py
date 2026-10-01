import bpy
from bpy.props import EnumProperty, IntProperty, StringProperty

from . import browser, content, entities, materialbrowser, materials, modelparts, prefs, props, sync


def _entity_from(op, context):
    name = getattr(op, "entity", "")
    if name:
        obj = bpy.data.objects.get(name)
        return obj if obj is not None and obj.p2m.is_entity else None
    return entities.context_entity(context)


def _model_target(op, context):
    """An entity or model part: by name, else the active object."""
    name = getattr(op, "entity", "")
    obj = bpy.data.objects.get(name) if name else context.active_object
    if obj is not None and (obj.p2m.is_entity or obj.p2m.is_model_part):
        return obj
    return None


class P2M_OT_add_entity(bpy.types.Operator):
    """Add a Prop2Mesh entity (sent_prop2mesh) at the 3D cursor"""
    bl_idname = "object.p2m_add_entity"
    bl_label = "Prop2Mesh Entity"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        if context.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        for o in context.selected_objects:
            o.select_set(False)
        obj = entities.create_entity(context, location=context.scene.cursor.location.copy())
        content.show_textures_in_solid_view(context)
        obj.select_set(True)
        context.view_layer.objects.active = obj
        if obj.p2m.model_status:
            self.report({"WARNING"}, obj.p2m.model_status)
        return {"FINISHED"}


class P2M_OT_add_model_part(bpy.types.Operator):
    """Add a model part: a Prop2Mesh part built from a model (exported as the
    model path and placement, not as vertices). Attach it to a controller
    with Ctrl+P like any mesh"""
    bl_idname = "object.p2m_add_model_part"
    bl_label = "Prop2Mesh Model Part"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        if context.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        for o in context.selected_objects:
            o.select_set(False)
        obj = modelparts.create_model_part(context, location=context.scene.cursor.location.copy())
        content.show_textures_in_solid_view(context)
        obj.select_set(True)
        context.view_layer.objects.active = obj
        if obj.p2m.model_status:
            self.report({"WARNING"}, obj.p2m.model_status)
        return {"FINISHED"}


class P2M_OT_add_clip_plane(bpy.types.Operator):
    """Add a clipping plane to the model part. Move/rotate it to cut the
    model; the arrow points to the side that is kept"""
    bl_idname = "p2m.add_clip_plane"
    bl_label = "Add Clip Plane"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        obj = context.active_object
        if obj is not None and obj.p2m.is_clip_plane:
            obj = obj.parent
        if obj is None or not obj.p2m.is_model_part:
            self.report({"ERROR"}, "Select a model part")
            return {"CANCELLED"}
        plane = modelparts.add_clip_plane(context, obj)
        for o in context.selected_objects:
            o.select_set(False)
        plane.select_set(True)
        context.view_layer.objects.active = plane
        return {"FINISHED"}


class P2M_OT_flip_clip_plane(bpy.types.Operator):
    """Flip the clip plane so it keeps the other side"""
    bl_idname = "p2m.flip_clip_plane"
    bl_label = "Flip Clip Plane"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        from mathutils import Matrix
        flipped = False
        for obj in context.selected_objects:
            if obj.p2m.is_clip_plane:
                obj.matrix_basis = obj.matrix_basis @ Matrix.Rotation(3.141592653589793, 4, "X")
                flipped = True
                if obj.parent is not None:
                    modelparts.rebuild(obj.parent)
        return {"FINISHED"} if flipped else {"CANCELLED"}


class P2M_OT_controller_add(bpy.types.Operator):
    """Add a controller to the entity"""
    bl_idname = "p2m.controller_add"
    bl_label = "Add Controller"
    bl_options = {"REGISTER", "UNDO"}

    entity: StringProperty(options={"HIDDEN"})

    def execute(self, context):
        ent = _entity_from(self, context)
        if ent is None:
            return {"CANCELLED"}
        entities.add_controller(ent)
        return {"FINISHED"}


class P2M_OT_controller_remove(bpy.types.Operator):
    """Remove the active controller. Its parts become unassigned"""
    bl_idname = "p2m.controller_remove"
    bl_label = "Remove Controller"
    bl_options = {"REGISTER", "UNDO"}

    entity: StringProperty(options={"HIDDEN"})

    def execute(self, context):
        ent = _entity_from(self, context)
        if ent is None:
            return {"CANCELLED"}
        p = ent.p2m
        i = p.active_controller
        if not 0 <= i < len(p.controllers):
            return {"CANCELLED"}
        mat = p.controllers[i].blender_material
        p.controllers.remove(i)
        p.active_controller = min(i, len(p.controllers) - 1)
        sync.full_sync(context.scene)
        if mat is not None and mat.users == 0:
            solid = bpy.data.images.get("P2M Solid: " + mat.name)
            bpy.data.materials.remove(mat)
            if solid is not None and solid.users == 0:
                bpy.data.images.remove(solid)
        return {"FINISHED"}


class P2M_OT_controller_move(bpy.types.Operator):
    """Move the active controller up or down (changes its export index)"""
    bl_idname = "p2m.controller_move"
    bl_label = "Move Controller"
    bl_options = {"REGISTER", "UNDO"}

    entity: StringProperty(options={"HIDDEN"})
    direction: EnumProperty(items=(("UP", "Up", ""), ("DOWN", "Down", "")))

    def execute(self, context):
        ent = _entity_from(self, context)
        if ent is None:
            return {"CANCELLED"}
        p = ent.p2m
        i = p.active_controller
        j = i - 1 if self.direction == "UP" else i + 1
        if not (0 <= i < len(p.controllers) and 0 <= j < len(p.controllers)):
            return {"CANCELLED"}
        p.controllers.move(i, j)
        p.active_controller = j
        return {"FINISHED"}


class P2M_OT_attach(bpy.types.Operator):
    """Parent the selected objects to the entity (keeping their transforms)
    and attach them to the controller"""
    bl_idname = "p2m.attach"
    bl_label = "Attach to Controller"
    bl_options = {"REGISTER", "UNDO"}

    entity: StringProperty(options={"HIDDEN"})
    uid: IntProperty(default=-1, options={"HIDDEN"})

    def execute(self, context):
        ent = _entity_from(self, context)
        if ent is None:
            self.report({"ERROR"}, "No Prop2Mesh entity")
            return {"CANCELLED"}
        uid = self.uid
        if uid < 0 and 0 <= ent.p2m.active_controller < len(ent.p2m.controllers):
            uid = ent.p2m.controllers[ent.p2m.active_controller].uid
        if props.controller_by_uid(ent, uid) is None:
            self.report({"ERROR"}, "Entity has no such controller")
            return {"CANCELLED"}
        count = 0
        for obj in context.selected_objects:
            if obj == ent or obj.p2m.is_entity or obj.p2m.is_clip_plane:
                continue
            if entities.owning_entity(obj) is not ent:
                mw = obj.matrix_world.copy()
                obj.parent = ent
                obj.matrix_parent_inverse = ent.matrix_world.inverted()
                obj.matrix_world = mw
            obj.p2m.controller_uid = uid
            count += 1
        sync.full_sync(context.scene)
        content.show_textures_in_solid_view(context)
        if not count:
            self.report({"WARNING"}, "Select the objects to attach (and the entity last)")
            return {"CANCELLED"}
        return {"FINISHED"}


class P2M_OT_set_part_controller(bpy.types.Operator):
    """Assign the selected parts to a controller of their entity"""
    bl_idname = "p2m.set_part_controller"
    bl_label = "Set Controller"
    bl_options = {"REGISTER", "UNDO"}

    uid: IntProperty(default=-1)

    def execute(self, context):
        for obj in context.selected_objects or [context.active_object]:
            if obj is not None and not obj.p2m.is_entity:
                obj.p2m.controller_uid = self.uid
        sync.full_sync(context.scene)
        return {"FINISHED"}


class P2M_OT_select_parts(bpy.types.Operator):
    """Select every object attached to the controller"""
    bl_idname = "p2m.select_parts"
    bl_label = "Select Parts"
    bl_options = {"REGISTER", "UNDO"}

    entity: StringProperty(options={"HIDDEN"})
    uid: IntProperty(default=-1, options={"HIDDEN"})

    def execute(self, context):
        ent = _entity_from(self, context)
        if ent is None:
            return {"CANCELLED"}
        uid = self.uid
        if uid < 0 and 0 <= ent.p2m.active_controller < len(ent.p2m.controllers):
            uid = ent.p2m.controllers[ent.p2m.active_controller].uid
        for o in context.selected_objects:
            o.select_set(False)
        for obj, ctrl in entities.entity_parts(ent):
            if ctrl.uid == uid:
                try:
                    obj.select_set(True)
                except RuntimeError:
                    pass
        return {"FINISHED"}


class P2M_OT_reload_model(bpy.types.Operator):
    """Reload the model from the game files"""
    bl_idname = "p2m.reload_model"
    bl_label = "Reload Model"
    bl_options = {"REGISTER", "UNDO"}

    entity: StringProperty(options={"HIDDEN"})

    def execute(self, context):
        obj = _model_target(self, context)
        if obj is None:
            return {"CANCELLED"}
        if obj.p2m.is_entity:
            entities.apply_model(obj, reload=True)
        else:
            modelparts.clear_cache()
            modelparts.rebuild(obj, refresh_bodygroups=True, force=True)
        if obj.p2m.model_status:
            self.report({"WARNING"}, obj.p2m.model_status)
        return {"FINISHED"}


_search_cache = {}


def _search_items(kind):
    hit = _search_cache.get(kind)
    if hit is None:
        fs = content.get_fs()
        if fs is None:
            return [("", content.status_text(), "")]
        if kind == "materials":
            files = fs.list_files("materials/", ".vmt")
            hit = [(p[10:-4], p[10:-4], "") for p in files]
        else:
            files = fs.list_files("models/", ".mdl")
            hit = [(p, p, "") for p in files]
        _search_cache[kind] = hit
    return hit


def clear_search_cache():
    _search_cache.clear()


class P2M_OT_search_material(bpy.types.Operator):
    """Search all mounted materials"""
    bl_idname = "p2m.search_material"
    bl_label = "Search Material"
    bl_options = {"REGISTER", "UNDO"}
    bl_property = "material"

    material: EnumProperty(items=lambda self, context: _search_items("materials"))
    entity: StringProperty(options={"HIDDEN"})
    uid: IntProperty(default=-1, options={"HIDDEN"})

    def invoke(self, context, event):
        if content.get_fs() is None:
            self.report({"ERROR"}, content.status_text())
            return {"CANCELLED"}
        context.window_manager.invoke_search_popup(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        ent = _entity_from(self, context)
        ctrl = props.controller_by_uid(ent, self.uid)
        if ctrl is None or not self.material:
            return {"CANCELLED"}
        ctrl.material = self.material
        return {"FINISHED"}


class P2M_OT_search_model(bpy.types.Operator):
    """Search all mounted models by path"""
    bl_idname = "p2m.search_model"
    bl_label = "Search Model"
    bl_options = {"REGISTER", "UNDO"}
    bl_property = "model"

    model: EnumProperty(items=lambda self, context: _search_items("models"))
    entity: StringProperty(options={"HIDDEN"})

    def invoke(self, context, event):
        if content.get_fs() is None:
            self.report({"ERROR"}, content.status_text())
            return {"CANCELLED"}
        context.window_manager.invoke_search_popup(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        obj = _model_target(self, context)
        if obj is None or not self.model:
            return {"CANCELLED"}
        obj.p2m.model = self.model
        return {"FINISHED"}


class P2M_OT_rebuild_content(bpy.types.Operator):
    """Re-scan Garry's Mod content (after installing addons or changing paths)"""
    bl_idname = "p2m.rebuild_content"
    bl_label = "Rescan Content"

    def execute(self, context):
        prefs._detected.clear()
        clear_search_cache()
        browser.invalidate()
        materialbrowser.invalidate()
        materials.clear_cache()
        content.start_build(force=True)
        self.report({"INFO"}, "Rescanning Garry's Mod content in the background")
        return {"FINISHED"}


class P2M_OT_refresh_materials(bpy.types.Operator):
    """Reload every controller's material and texture"""
    bl_idname = "p2m.refresh_materials"
    bl_label = "Refresh Materials"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        materials.refresh_all()
        return {"FINISHED"}


classes = (
    P2M_OT_add_entity, P2M_OT_add_model_part, P2M_OT_add_clip_plane, P2M_OT_flip_clip_plane,
    P2M_OT_controller_add, P2M_OT_controller_remove, P2M_OT_controller_move,
    P2M_OT_attach, P2M_OT_set_part_controller, P2M_OT_select_parts, P2M_OT_reload_model,
    P2M_OT_search_material, P2M_OT_search_model, P2M_OT_rebuild_content, P2M_OT_refresh_materials,
)
