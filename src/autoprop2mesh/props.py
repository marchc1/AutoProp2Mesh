import bpy
from bpy.props import (BoolProperty, CollectionProperty, EnumProperty, FloatProperty,
                       FloatVectorProperty, IntProperty, PointerProperty, StringProperty)

from .core.p2m import DEFAULT_MATERIAL, DEFAULT_MODEL


def _controller_changed(self, context):
    from . import materials
    entity = self.id_data
    materials.update_controller(entity, self)


def _controller_uvs_changed(self, context):
    from . import materials, sync
    materials.update_controller(self.id_data, self)
    sync.request_full_sync()


def _model_changed(self, context):
    from . import entities, modelparts
    obj = self.id_data
    if self.is_entity:
        entities.apply_model(obj)
    elif self.is_model_part:
        modelparts.rebuild(obj, refresh_bodygroups=True)


def _model_part_changed(self, context):
    from . import modelparts
    obj = self.id_data
    if obj.p2m.is_model_part:
        modelparts.rebuild(obj)


_bodygroup_items = {}


def _bodygroup_choices(self, context):
    names = self.options.split("\n") if self.options else [""]
    key = tuple(names)
    items = _bodygroup_items.get(key)
    if items is None:
        items = []
        for i, n in enumerate(names):
            label = n[:-4] if n.lower().endswith(".smd") else n
            items.append((str(i), "%d: %s" % (i, label or "(blank)"), "", i))
        _bodygroup_items[key] = items
    return items


def _part_changed(self, context):
    from . import sync
    sync.request_full_sync(immediate=True)


def _unit_scale_changed(self, context):
    from . import entities, sync
    entities.refresh_all_models(context.scene)
    sync.request_full_sync(immediate=True)


class P2M_Controller(bpy.types.PropertyGroup):
    uid: IntProperty(default=-1)
    name: StringProperty(
        name="Name", update=_controller_changed,
        description="Controller name (shown in Blender and exported as the controller name)",
    )
    color: FloatVectorProperty(
        name="Color", subtype="COLOR_GAMMA", size=4, min=0.0, max=1.0,
        default=(1.0, 1.0, 1.0, 1.0), update=_controller_changed,
        description="Controller color and alpha (alpha below 1 renders translucent)",
    )
    material: StringProperty(
        name="Material", default=DEFAULT_MATERIAL, update=_controller_changed,
        description="Source material path, e.g. hunter/myplastic",
    )
    uvs: IntProperty(
        name="UV Scale", default=0, min=0, max=512, update=_controller_uvs_changed,
        description="Prop2Mesh texture scale in Source units per texture repeat (0 = default, 48)",
    )
    bump: BoolProperty(
        name="Bump", default=False, update=_controller_changed,
        description="Generate tangents so bump/normal mapped materials render correctly",
    )
    blender_material: PointerProperty(type=bpy.types.Material)


class P2M_Bodygroup(bpy.types.PropertyGroup):
    name: StringProperty()
    base: IntProperty(default=1)
    options: StringProperty()  # submodel names, newline separated
    choice: EnumProperty(name="Body Group", items=_bodygroup_choices, update=_model_part_changed)


class P2M_ObjectProps(bpy.types.PropertyGroup):
    # -- entity -----------------------------------------------------------
    is_entity: BoolProperty(default=False)
    model: StringProperty(
        name="Model", default=DEFAULT_MODEL, update=_model_changed,
        description="Model of the sent_prop2mesh entity itself",
    )
    model_status: StringProperty()
    controllers: CollectionProperty(type=P2M_Controller)
    active_controller: IntProperty(name="Active Controller", default=0)
    next_uid: IntProperty(default=0)

    # -- model part (a Prop2Mesh "prop" part: exported as model + placement)
    is_model_part: BoolProperty(default=False)
    bodygroups: CollectionProperty(type=P2M_Bodygroup)
    model_flat: BoolProperty(
        name="Flat Shading", default=False, update=_model_part_changed,
        description="Use flat shading instead of the model's normals (Prop2Mesh vsmooth)",
    )

    # -- clip plane (child of a model part) ------------------------------
    is_clip_plane: BoolProperty(default=False)

    # -- part -------------------------------------------------------------
    controller_uid: IntProperty(
        default=-1, update=_part_changed,
        description="Controller this object belongs to (-1 = inherit from parent objects)",
    )
    smooth_mode: EnumProperty(
        name="Shading",
        items=(
            ("AUTO", "Auto", "Flat when all faces are flat shaded, otherwise smooth (using a Smooth by Angle modifier's angle if there is one)"),
            ("FLAT", "Flat", "Flat shading"),
            ("SMOOTH", "Smooth", "Smooth normals between faces meeting below the angle"),
        ),
        default="AUTO",
    )
    smooth_angle: FloatProperty(
        name="Smooth Angle", default=60.0, min=0.0, max=180.0, subtype="NONE",
        description="Prop2Mesh smoothing angle in degrees",
    )
    render_inside: BoolProperty(
        name="Render Inside", default=False,
        description="Also render back faces (Prop2Mesh vinside)",
    )

    def bodygroup_choices(self):
        out = []
        for bg in self.bodygroups:
            try:
                out.append(int(bg.choice))
            except (TypeError, ValueError):
                out.append(0)
        return out

    def bodygroup_mask(self):
        return sum(c * bg.base for c, bg in zip(self.bodygroup_choices(), self.bodygroups))

    # -- bookkeeping (sync) ----------------------------------------------
    bound_key: StringProperty()
    added_slot: BoolProperty(default=False)


class P2M_SceneProps(bpy.types.PropertyGroup):
    unit_scale: FloatProperty(
        name="Source Units per Blender Unit", default=1.0, min=1e-4, max=1e4,
        update=_unit_scale_changed,
        description="1 Blender unit = this many Source (Hammer) units",
    )


classes = (P2M_Controller, P2M_Bodygroup, P2M_ObjectProps, P2M_SceneProps)


def register():
    bpy.types.Object.p2m = PointerProperty(type=P2M_ObjectProps)
    bpy.types.Scene.p2m = PointerProperty(type=P2M_SceneProps)
    bpy.types.Material.p2m_owner = PointerProperty(type=bpy.types.Object)


def unregister():
    del bpy.types.Material.p2m_owner
    del bpy.types.Scene.p2m
    del bpy.types.Object.p2m


def unit_scale(scene=None):
    scene = scene or bpy.context.scene
    return scene.p2m.unit_scale if scene else 1.0


def controller_by_uid(entity, uid):
    if entity is None or uid < 0:
        return None
    for c in entity.p2m.controllers:
        if c.uid == uid:
            return c
    return None


def controller_label(entity, ctrl):
    if ctrl.name.strip():
        return '"%s"' % ctrl.name
    for i, c in enumerate(entity.p2m.controllers):
        if c == ctrl:
            return "#%d" % i
    return "#?"
