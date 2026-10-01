"""Bridges the Source content (virtual filesystem, models, textures) into
Blender data: model meshes, Source materials and images."""

import os
import tempfile
import threading

import bmesh
import bpy
import numpy as np

from . import prefs
from .core import studiomdl, vfs, vmt, vtf

ERROR_MODEL = "models/error.mdl"
MODEL_MESH_PREFIX = "P2M Model: "
SRC_MAT_PREFIX = "SRC: "
SRC_IMG_PREFIX = "SRC: "

# --------------------------------------------------------------------------
# Virtual filesystem (built on a background thread)
# --------------------------------------------------------------------------

_fs = None
_fs_error = None
_fs_thread = None
_fs_lock = threading.Lock()


def _cache_dir():
    try:
        return bpy.utils.extension_path_user(__package__, path="cache", create=True)
    except Exception:
        return os.path.join(tempfile.gettempdir(), "autoprop2mesh_cache")


def _make_config():
    p = prefs.get_prefs()
    extra = [bpy.path.abspath(sp.path) for sp in p.extra_paths if sp.path] if p else []
    return vfs.MountConfig(
        gmod_dir=prefs.resolved_gmod_dir(p),
        extra_paths=extra,
        mount_workshop=p.mount_workshop if p else True,
        mount_games=p.mount_games if p else True,
    )


def start_build(force=False):
    """Starts (re)building the filesystem in the background."""
    global _fs, _fs_error, _fs_thread
    with _fs_lock:
        if _fs_thread and _fs_thread.is_alive():
            return
        if _fs is not None and not force:
            return
        config = _make_config()   # read prefs on the main thread
        cache = _cache_dir()
        old = _fs
        _fs = None
        _fs_error = None

        def work():
            global _fs, _fs_error
            try:
                if not config.gmod_dir:
                    raise RuntimeError("Garry's Mod install not found (set it in the add-on preferences)")
                fs = vfs.FileSystem(config, cache_dir=cache)
                _fs = fs
            except Exception as e:  # reported in the UI
                _fs_error = str(e)
            if old:
                old.close()

        _fs_thread = threading.Thread(target=work, name="AutoProp2Mesh content index", daemon=True)
        _fs_thread.start()


def get_fs(wait=True):
    if _fs is None:
        start_build()
        if wait and _fs_thread:
            _fs_thread.join()
    return _fs


def is_ready():
    return _fs is not None


def status_text():
    if _fs is not None:
        return "Content: %d sources mounted (%.1fs)" % (len(_fs.sources), _fs.build_time)
    if _fs_error:
        return "Content: %s" % _fs_error
    if _fs_thread and _fs_thread.is_alive():
        return "Content: indexing..."
    return "Content: not loaded"


def shutdown():
    global _fs
    if _fs:
        _fs.close()
    _fs = None


def read(path):
    fs = get_fs()
    return fs.read(path) if fs else None


# --------------------------------------------------------------------------
# Images and Source materials
# --------------------------------------------------------------------------

def _texture_size():
    p = prefs.get_prefs()
    return p.texture_size if p else 1024


def image_from_rgba(name, rgba, pack=True):
    """rgba: (h, w, 4) float32, top row first."""
    h, w = rgba.shape[:2]
    img = bpy.data.images.new(name, w, h, alpha=True)
    img.pixels.foreach_set(np.ascontiguousarray(rgba[::-1]).reshape(-1))
    img.update()
    if pack:
        try:
            img.pack()
        except Exception:
            pass
    return img


def error_image():
    """GMod's purple/black missing texture checker."""
    name = SRC_IMG_PREFIX + "__error__"
    img = bpy.data.images.get(name)
    if img is None:
        n = 8
        cells = (np.add.outer(np.arange(64) // n, np.arange(64) // n) % 2).astype(bool)
        rgba = np.zeros((64, 64, 4), dtype=np.float32)
        rgba[..., 3] = 1
        rgba[cells] = (1.0, 0.0, 1.0, 1.0)
        img = image_from_rgba(name, rgba)
    return img


def texture_image(vtf_path):
    """Loads (or reuses) a VTF as a packed Blender image. None if missing."""
    name = SRC_IMG_PREFIX + vtf_path
    img = bpy.data.images.get(name)
    if img is not None:
        return img
    data = read(vtf_path)
    if data is None:
        return None
    try:
        _info, rgba = vtf.decode(data, _texture_size())
    except Exception as e:
        print("AutoProp2Mesh: cannot decode %s: %s" % (vtf_path, e))
        return None
    img = image_from_rgba(name, rgba)
    img["p2m_vtf"] = vtf_path
    return img


class MaterialInfo:
    """What the previews need to know about a VMT."""

    def __init__(self, vmt_path):
        self.vmt_path = vmt_path
        self.found = False
        self.image = None
        self.translucent = False


def material_info(vmt_path):
    info = MaterialInfo(vmt_path)
    mat = vmt.load(read, vmt_path) if vmt_path else None
    if mat is None:
        info.image = error_image()
        return info
    info.found = True
    info.translucent = mat.translucent
    tex = mat.base_texture
    if tex:
        info.image = texture_image(tex) or error_image()
    return info


def average_color(img):
    """Linear RGBA average of an image (its pixels are sRGB encoded)."""
    w, h = img.size
    if not w or not h:
        return (0.8, 0.8, 0.8, 1.0)
    px = np.empty(w * h * 4, dtype=np.float32)
    img.pixels.foreach_get(px)
    r, g, b, _a = px.reshape(-1, 4).mean(0)

    def lin(c):
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    return (lin(r), lin(g), lin(b), 1.0)


def show_textures_in_solid_view(context):
    """Solid view only draws textures with Color: Texture. Switch viewports
    still on Blender's default (Material) so P2M textures show up."""
    screen = getattr(context, "screen", None)
    if screen is None:
        return
    for area in screen.areas:
        if area.type != "VIEW_3D":
            continue
        shading = area.spaces.active.shading
        if shading.color_type == "MATERIAL":
            shading.color_type = "TEXTURE"


def _link(nt, a, b):
    nt.links.new(a, b)


def _principled_alpha(bsdf):
    return bsdf.inputs.get("Alpha")


def set_blend(mat, blended):
    if hasattr(mat, "surface_render_method"):
        mat.surface_render_method = "BLENDED" if blended else "DITHERED"
    if hasattr(mat, "blend_method"):
        try:
            mat.blend_method = "BLEND" if blended else "OPAQUE"
        except Exception:
            pass


def enable_nodes(mat):
    # Material.use_nodes is deprecated (always on) in Blender 5.x.
    if getattr(mat, "node_tree", None) is None or not getattr(mat, "use_nodes", True):
        try:
            mat.use_nodes = True
        except Exception:
            pass


def source_material(candidates):
    """Material for an MDL texture: the first VMT candidate that exists."""
    fs = get_fs()
    vmt_path = None
    if fs:
        for c in candidates:
            if fs.exists(c):
                vmt_path = c
                break
    key = vmt_path or (candidates[0] if candidates else "__missing__")
    name = SRC_MAT_PREFIX + key
    mat = bpy.data.materials.get(name)
    if mat is not None:
        return mat
    info = material_info(vmt_path)
    mat = bpy.data.materials.new(name)
    enable_nodes(mat)
    nt = mat.node_tree
    bsdf = nt.nodes.get("Principled BSDF")
    if bsdf is None:
        nt.nodes.clear()
        out = nt.nodes.new("ShaderNodeOutputMaterial")
        bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
        _link(nt, bsdf.outputs[0], out.inputs[0])
    bsdf.inputs["Roughness"].default_value = 0.8
    if info.image is not None:
        tex = nt.nodes.new("ShaderNodeTexImage")
        tex.image = info.image
        tex.location = (-400, 200)
        _link(nt, tex.outputs["Color"], bsdf.inputs["Base Color"])
        # Solid view with Color: Material draws this flat colour.
        mat.diffuse_color = average_color(info.image)
        if info.translucent and _principled_alpha(bsdf):
            _link(nt, tex.outputs["Alpha"], _principled_alpha(bsdf))
            set_blend(mat, True)
        nt.nodes.active = tex
    mat["p2m_vmt"] = key
    return mat


# --------------------------------------------------------------------------
# Model meshes
# --------------------------------------------------------------------------

def _mesh_name(model_path, unit_scale):
    if abs(unit_scale - 1.0) < 1e-9:
        return MODEL_MESH_PREFIX + model_path
    return "%s%s @%g" % (MODEL_MESH_PREFIX, model_path, unit_scale)


def _mesh_from_model(name, data, unit_scale):
    me = bpy.data.meshes.new(name)
    pos = data.positions.astype(np.float64) / unit_scale
    tris = data.triangles
    me.vertices.add(len(pos))
    me.vertices.foreach_set("co", pos.astype(np.float32).reshape(-1))
    n = len(tris)
    me.loops.add(n * 3)
    me.loops.foreach_set("vertex_index", tris.reshape(-1).astype(np.int32))
    me.polygons.add(n)
    me.polygons.foreach_set("loop_start", np.arange(0, n * 3, 3, dtype=np.int32))
    me.polygons.foreach_set("material_index", data.tri_material.astype(np.int32))
    uv_layer = me.uv_layers.new(name="UVMap")
    uv_layer.data.foreach_set("uv", data.uvs[tris.reshape(-1)].reshape(-1))
    me.update(calc_edges=True)
    me.validate(clean_customdata=False)
    try:
        me.polygons.foreach_set("use_smooth", np.ones(n, dtype=bool))
        me.normals_split_custom_set_from_vertices(data.normals.tolist())
    except Exception:
        pass
    for candidates in data.materials:
        me.materials.append(source_material(candidates))
    return me


def warning_mesh(unit_scale=1.0):
    """A '!' built from cubes: shown when neither the model nor error.mdl load."""
    name = MODEL_MESH_PREFIX + "__warning__"
    if abs(unit_scale - 1.0) > 1e-9:
        name += " @%g" % unit_scale
    me = bpy.data.meshes.get(name)
    if me is not None:
        return me
    me = bpy.data.meshes.new(name)
    bm = bmesh.new()
    s = 1.0 / unit_scale
    # (center, half size) in Source units: stem and dot of the '!'
    for (cx, cy, cz), (hx, hy, hz) in (((0, 0, 14), (2, 2, 8)), ((0, 0, 1.5), (2, 2, 1.5))):
        res = bmesh.ops.create_cube(bm, size=1.0)
        for v in res["verts"]:
            v.co.x = (cx + v.co.x * 2 * hx) * s
            v.co.y = (cy + v.co.y * 2 * hy) * s
            v.co.z = (cz + v.co.z * 2 * hz) * s
    bm.to_mesh(me)
    bm.free()
    mat = bpy.data.materials.get("P2M Missing Model")
    if mat is None:
        mat = bpy.data.materials.new("P2M Missing Model")
        mat.diffuse_color = (1.0, 0.75, 0.0, 1.0)
        enable_nodes(mat)
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            bsdf.inputs["Base Color"].default_value = (1.0, 0.75, 0.0, 1.0)
            em = bsdf.inputs.get("Emission Color") or bsdf.inputs.get("Emission")
            if em:
                em.default_value = (1.0, 0.5, 0.0, 1.0)
            strength = bsdf.inputs.get("Emission Strength")
            if strength:
                strength.default_value = 1.0
    me.materials.append(mat)
    return me


def model_mesh(model_path, unit_scale=1.0, reload=False):
    """Returns (mesh, status) where status is "" on success, or explains the
    fallback that is shown instead."""
    model_path = vfs.FileSystem.normalize(model_path or "")
    if model_path and not model_path.endswith(".mdl"):
        model_path += ".mdl"

    def load(path):
        name = _mesh_name(path, unit_scale)
        me = bpy.data.meshes.get(name)
        if me is not None and not reload:
            return me, None
        try:
            data = studiomdl.load(read, path)
        except Exception as e:
            return None, str(e)
        new = _mesh_from_model(name + ".tmp" if me else name, data, unit_scale)
        if me is not None:
            me.user_remap(new)
            bpy.data.meshes.remove(me)
            new.name = name
        new["p2m_model"] = path
        return new, None

    if get_fs() is None:
        return warning_mesh(unit_scale), status_text()
    if model_path:
        me, err = load(model_path)
        if me:
            return me, ""
    else:
        err = "no model set"
    me, err2 = load(ERROR_MODEL)
    if me:
        return me, "%s (showing error.mdl)" % err
    return warning_mesh(unit_scale), "%s; error.mdl unavailable: %s" % (err, err2)
