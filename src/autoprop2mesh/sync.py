"""Keeps parts in sync with their controllers.

A part is any geometry object (mesh, curve, text, ...) below a P2M entity. Its
controller is its own `controller_uid`, or else the nearest one set on an
ancestor between it and the entity. Bound parts get the controller material
in object-linked slots (their own mesh materials are untouched) plus the UV
preview modifier; unbinding (unparenting, detaching, deleting the controller)
restores them.
"""

import bpy

from . import materials, modelparts, props, uvpreview

GEOMETRY_TYPES = {"MESH", "CURVE", "SURFACE", "META", "FONT"}

_busy = False
_full_pending = False


def is_geometry(obj):
    return obj.type in GEOMETRY_TYPES and not obj.p2m.is_clip_plane


def resolve(obj):
    """Returns (entity or None, controller or None) for a part."""
    uid = obj.p2m.controller_uid
    p = obj.parent
    while p is not None:
        if p.p2m.is_entity:
            return p, props.controller_by_uid(p, uid)
        if uid < 0:
            uid = p.p2m.controller_uid
        p = p.parent
    return None, None


def _bind(obj, entity, ctrl, scene):
    mat = ctrl.blender_material
    if mat is None:
        mat = materials.update_controller(entity, ctrl)
    if len(obj.material_slots) == 0:
        data = obj.data
        if data is not None and hasattr(data, "materials"):
            data.materials.append(None)
            obj.p2m.added_slot = True
    for slot in obj.material_slots:
        if slot.link != "OBJECT":
            slot.link = "OBJECT"
        if slot.material is not mat:
            slot.material = mat
    box_uv = not obj.p2m.is_model_part or ctrl.uvs > 0
    uvpreview.ensure(obj, entity, props.unit_scale(scene), ctrl.uvs, box_uv)
    key = "%s|%d" % (entity.name, ctrl.uid)
    if obj.p2m.bound_key != key:
        obj.p2m.bound_key = key


def _unbind(obj):
    for slot in obj.material_slots:
        if slot.link == "OBJECT":
            slot.material = None
            slot.link = "DATA"
    if obj.p2m.added_slot:
        data = obj.data
        if data is not None and hasattr(data, "materials") and len(data.materials) == 1 and data.materials[0] is None:
            data.materials.clear()
        obj.p2m.added_slot = False
    uvpreview.remove(obj)
    obj.p2m.bound_key = ""


def sync_object(obj, scene=None):
    if obj.p2m.is_entity or obj.type not in GEOMETRY_TYPES:
        return
    if obj.p2m.is_clip_plane:
        if obj.p2m.bound_key:
            _unbind(obj)
        return
    entity, ctrl = resolve(obj)
    if entity is not None and ctrl is not None:
        _bind(obj, entity, ctrl, scene or bpy.context.scene)
    elif obj.p2m.bound_key or obj.modifiers.get(uvpreview.MODIFIER_NAME):
        _unbind(obj)


def _fix_shared_materials():
    """Duplicated entities share their source's controller materials; give
    each controller a material of its own. The entity recorded as a
    material's owner keeps it; any other user gets a copy."""
    users = {}
    for obj in bpy.data.objects:
        if obj.p2m.is_entity:
            for ctrl in obj.p2m.controllers:
                if ctrl.blender_material is not None:
                    users.setdefault(ctrl.blender_material, []).append((obj, ctrl))
    for mat, refs in users.items():
        owners = [r for r in refs if r[0] == mat.p2m_owner]
        keep = owners[0] if owners else refs[0]
        if mat.p2m_owner != keep[0]:
            mat.p2m_owner = keep[0]
        for obj, ctrl in refs:
            if (obj, ctrl) == keep:
                continue
            new = mat.copy()
            new.name = "P2M %s #%d" % (obj.name, ctrl.uid)
            new.p2m_owner = obj
            ctrl.blender_material = new
            materials.update_controller(obj, ctrl)


def full_sync(scene=None):
    global _busy
    if _busy:
        return
    _busy = True
    try:
        scene = scene or bpy.context.scene
        _fix_shared_materials()
        modelparts.check_all()
        for obj in bpy.data.objects:
            if obj.library is None:
                sync_object(obj, scene)
    finally:
        _busy = False


def request_full_sync(immediate=False):
    global _full_pending
    if immediate:
        full_sync()
        return
    if not _full_pending:
        _full_pending = True
        bpy.app.timers.register(_deferred_full_sync, first_interval=0.0)


def _deferred_full_sync():
    global _full_pending
    _full_pending = False
    try:
        full_sync()
    except Exception as e:
        print("AutoProp2Mesh sync error:", e)
    return None


@bpy.app.handlers.persistent
def on_depsgraph_update(scene, depsgraph):
    global _busy
    if _busy:
        return
    touched = []
    entity_touched = False
    for update in depsgraph.updates:
        idb = update.id
        if isinstance(idb, bpy.types.Object) and update.is_updated_transform:
            obj = idb.original
            if obj.p2m.is_clip_plane:
                if obj.parent is not None and obj.parent.p2m.is_model_part:
                    modelparts.queue(obj.parent)
                continue
            touched.append(obj)
            entity_touched |= obj.p2m.is_entity
    if not touched:
        return
    if entity_touched:
        # Entities being duplicated/deleted can affect many objects.
        request_full_sync()
        return
    _busy = True
    try:
        for obj in touched:
            sync_object(obj, scene)
            for child in obj.children_recursive:
                sync_object(child, scene)
    except Exception as e:
        print("AutoProp2Mesh sync error:", e)
    finally:
        _busy = False


def _safety_timer():
    # Catches changes that do not produce depsgraph updates (e.g. undo).
    try:
        if bpy.context.scene is not None:
            full_sync()
    except Exception as e:
        print("AutoProp2Mesh sync error:", e)
    return 1.5


@bpy.app.handlers.persistent
def on_load_post(*_args):
    materials.regenerate_solid_previews()
    request_full_sync()


@bpy.app.handlers.persistent
def on_undo_redo(*_args):
    request_full_sync()


def register():
    bpy.app.handlers.depsgraph_update_post.append(on_depsgraph_update)
    bpy.app.handlers.load_post.append(on_load_post)
    bpy.app.handlers.undo_post.append(on_undo_redo)
    bpy.app.handlers.redo_post.append(on_undo_redo)
    bpy.app.timers.register(_safety_timer, first_interval=1.5, persistent=True)


def unregister():
    for lst, fn in ((bpy.app.handlers.depsgraph_update_post, on_depsgraph_update),
                    (bpy.app.handlers.load_post, on_load_post),
                    (bpy.app.handlers.undo_post, on_undo_redo),
                    (bpy.app.handlers.redo_post, on_undo_redo)):
        if fn in lst:
            lst.remove(fn)
    if bpy.app.timers.is_registered(_safety_timer):
        bpy.app.timers.unregister(_safety_timer)
