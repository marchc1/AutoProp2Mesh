import bpy

from . import content, props
from .core.p2m import DEFAULT_MODEL


def is_entity(obj):
    return obj is not None and obj.p2m.is_entity


def owning_entity(obj):
    """The nearest P2M entity among obj's ancestors (not obj itself)."""
    p = obj.parent if obj else None
    while p is not None:
        if p.p2m.is_entity:
            return p
        p = p.parent
    return None


def context_entity(context):
    """The entity the UI should show: the active object if it is one,
    otherwise the entity owning the active object."""
    obj = context.active_object
    if obj is None:
        return None
    return obj if is_entity(obj) else owning_entity(obj)


def apply_model(obj, reload=False):
    me, status = content.model_mesh(obj.p2m.model, props.unit_scale(), reload=reload)
    old = obj.data
    if old is not me:
        obj.data = me
        if old is not None and old.users == 0 and old.name.startswith("P2M Entity"):
            bpy.data.meshes.remove(old)
    obj.p2m.model_status = status


def refresh_all_models(scene=None):
    for obj in bpy.data.objects:
        if obj.p2m.is_entity and obj.type == "MESH":
            apply_model(obj)


def add_controller(entity, name=""):
    p = entity.p2m
    ctrl = p.controllers.add()
    ctrl.uid = p.next_uid
    p.next_uid += 1
    ctrl.name = name
    p.active_controller = len(p.controllers) - 1
    from . import materials
    materials.update_controller(entity, ctrl)
    return ctrl


def create_entity(context, location=None, model=DEFAULT_MODEL, name="P2M Entity"):
    me = bpy.data.meshes.new("P2M Entity")
    obj = bpy.data.objects.new(name, me)
    context.collection.objects.link(obj)
    if location is not None:
        obj.location = location
    obj.p2m.is_entity = True
    obj.p2m.model = model  # triggers apply_model
    if obj.p2m.model == model and obj.data is me:
        apply_model(obj)
    obj.show_name = True
    add_controller(obj)
    return obj


def entity_parts(entity, include_unassigned=False):
    """Objects whose owning entity is `entity`, with their resolved controller."""
    from . import sync
    out = []
    for obj in entity.children_recursive:
        if obj.p2m.is_entity or not sync.is_geometry(obj):
            continue
        ent, ctrl = sync.resolve(obj)
        if ent is entity and (ctrl is not None or include_unassigned):
            out.append((obj, ctrl))
    return out
