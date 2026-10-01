"""Hammer-style model browser: a filterable list of every mounted model with
a rendered preview of the highlighted one."""

import os

import bpy
import bpy.utils.previews
from bpy.props import CollectionProperty, IntProperty, StringProperty

from . import content
from .core import raster, studiomdl, vmt, vtf

MAX_RESULTS = 3000
PREVIEW_SIZE = 320

_pcoll = None
_info = {}  # model path -> dict


class P2M_BrowserItem(bpy.types.PropertyGroup):
    path: StringProperty()


def _refill(self, context):
    state = context.window_manager.p2m_browser
    state.items.clear()
    fs = content.get_fs()
    if fs is None:
        return
    tokens = [t for t in state.filter.lower().replace("\\", "/").split() if t]
    count = 0
    for p in fs.list_files("models/", ".mdl"):
        if all(t in p for t in tokens):
            state.items.add().path = p
            count += 1
            if count >= MAX_RESULTS:
                break
    state.truncated = count >= MAX_RESULTS
    state.index = -1


def _selected(self, context):
    state = context.window_manager.p2m_browser
    if 0 <= state.index < len(state.items):
        render_preview(state.items[state.index].path)


class P2M_BrowserState(bpy.types.PropertyGroup):
    filter: StringProperty(name="Filter", update=_refill, options={"TEXTEDIT_UPDATE"},
                           description="Space separated words that must all appear in the model path")
    items: CollectionProperty(type=P2M_BrowserItem)
    index: IntProperty(default=-1, update=_selected)
    truncated: bpy.props.BoolProperty()
    target: StringProperty()


def _textures(model):
    out = []
    for candidates in model.materials:
        tex = None
        for c in candidates:
            mat = vmt.load(content.read, c)
            if mat is None:
                continue
            if mat.base_texture:
                data = content.read(mat.base_texture)
                if data:
                    try:
                        tex = vtf.decode(data, 256)[1]
                    except Exception:
                        tex = None
            break
        out.append(tex)
    return out


def render_preview(path):
    if _pcoll is None or path in _pcoll:
        return
    info = {"error": None, "tris": 0, "materials": [], "source": None}
    _info[path] = info
    fs = content.get_fs()
    info["source"] = fs.locate(path) if fs else None
    try:
        model = studiomdl.load(content.read, path)
        info["tris"] = len(model.triangles)
        info["materials"] = [c[0] if c else "?" for c in model.materials]
        rgba = raster.render(model, _textures(model), size=PREVIEW_SIZE)
    except Exception as e:
        info["error"] = str(e)
        return
    pv = _pcoll.new(path)
    pv.image_size = (PREVIEW_SIZE, PREVIEW_SIZE)
    pv.image_pixels_float.foreach_set(rgba[::-1].reshape(-1))


class P2M_UL_models(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        layout.label(text=item.path[len("models/"):], icon="MESH_DATA")


class P2M_OT_model_browser(bpy.types.Operator):
    """Browse all mounted models with a preview (like Hammer's model browser)"""
    bl_idname = "p2m.model_browser"
    bl_label = "Model Browser"
    bl_options = {"REGISTER", "UNDO"}

    entity: StringProperty(options={"HIDDEN"})

    def invoke(self, context, event):
        if content.get_fs() is None:
            self.report({"ERROR"}, content.status_text())
            return {"CANCELLED"}
        state = context.window_manager.p2m_browser
        obj = bpy.data.objects.get(self.entity) if self.entity else context.active_object
        if obj is None or not (obj.p2m.is_entity or obj.p2m.is_model_part):
            self.report({"ERROR"}, "Select a Prop2Mesh entity or model part")
            return {"CANCELLED"}
        state.target = obj.name
        if not state.filter:
            # Start out showing the current model's folder.
            state.filter = os.path.dirname(obj.p2m.model.lower())[len("models/"):]
        else:
            _refill(None, context)
        for i, item in enumerate(state.items):
            if item.path == obj.p2m.model.lower():
                state.index = i
                break
        return context.window_manager.invoke_props_dialog(self, width=900)

    def draw(self, context):
        state = context.window_manager.p2m_browser
        split = self.layout.split(factor=0.55)
        left = split.column()
        left.prop(state, "filter", text="", icon="VIEWZOOM")
        left.template_list("P2M_UL_models", "", state, "items", state, "index", rows=22)
        left.label(text="%d models%s" % (len(state.items), " (refine the filter to see more)" if state.truncated else ""))

        right = split.column()
        path = state.items[state.index].path if 0 <= state.index < len(state.items) else None
        if path is None:
            right.label(text="Select a model to preview")
            return
        if _pcoll is not None and path in _pcoll:
            right.template_icon(icon_value=_pcoll[path].icon_id, scale=16.0)
        info = _info.get(path, {})
        col = right.column(align=True)
        col.label(text=path)
        if info.get("error"):
            col.label(text=info["error"], icon="ERROR")
        else:
            col.label(text="%d triangles" % info.get("tris", 0))
            for m in info.get("materials", [])[:4]:
                col.label(text=m, icon="MATERIAL")
        if info.get("source"):
            col.label(text="From: %s" % os.path.basename(info["source"].rstrip("\\/")), icon="FILE_FOLDER")

    def execute(self, context):
        state = context.window_manager.p2m_browser
        obj = bpy.data.objects.get(state.target)
        if obj is None or not (0 <= state.index < len(state.items)):
            return {"CANCELLED"}
        obj.p2m.model = state.items[state.index].path
        return {"FINISHED"}


classes = (P2M_BrowserItem, P2M_BrowserState, P2M_UL_models, P2M_OT_model_browser)


def register():
    global _pcoll
    _pcoll = bpy.utils.previews.new()
    bpy.types.WindowManager.p2m_browser = bpy.props.PointerProperty(type=P2M_BrowserState)


def unregister():
    global _pcoll
    del bpy.types.WindowManager.p2m_browser
    if _pcoll is not None:
        bpy.utils.previews.remove(_pcoll)
    _pcoll = None
    _info.clear()
