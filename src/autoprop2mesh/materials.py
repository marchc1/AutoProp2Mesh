"""Controller materials.

Each controller owns one Blender material shared by all of its parts:

  Attribute "p2m_uv" -> Image (VMT $basetexture) -> x controller color -> BSDF
  alpha = controller alpha (x texture alpha when the VMT is translucent)

Solid view cannot multiply a texture by a color, so the material also holds
an unconnected image node with a pre-tinted copy of the texture, set as the
active node: Solid > Texture shows texture x color, Solid > Material shows
the flat controller color (material viewport display color).
"""

import bpy
import numpy as np

from . import content
from .core import vmt

TINT_MAX = 256
_material_cache = {}  # vmt path -> (image name or None, translucent)


def srgb_to_linear(c):
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _material_lookup(material):
    path = vmt.material_path(material or "")
    hit = _material_cache.get(path)
    if hit is not None:
        img = bpy.data.images.get(hit[0]) if hit[0] else None
        if hit[0] is None or img is not None:
            return img, hit[1]
    info = content.material_info(path)
    _material_cache[path] = (info.image.name if info.image else None, info.translucent)
    return info.image, info.translucent


def clear_cache():
    _material_cache.clear()


def _node(nt, idname, name, location):
    n = nt.nodes.get(name)
    if n is None or n.bl_idname != idname:
        if n is not None:
            nt.nodes.remove(n)
        n = nt.nodes.new(idname)
        n.name = name
        n.label = name.replace("p2m_", "").replace("_", " ").title()
    n.location = location
    return n


def _build_tree(mat):
    content.enable_nodes(mat)
    nt = mat.node_tree
    for n in list(nt.nodes):
        if not n.name.startswith("p2m_"):
            nt.nodes.remove(n)
    out = _node(nt, "ShaderNodeOutputMaterial", "p2m_output", (600, 0))
    bsdf = _node(nt, "ShaderNodeBsdfPrincipled", "p2m_bsdf", (300, 0))
    uv = _node(nt, "ShaderNodeAttribute", "p2m_uv", (-700, 0))
    uv.attribute_name = "p2m_uv"
    tex = _node(nt, "ShaderNodeTexImage", "p2m_texture", (-450, 0))
    tint = _node(nt, "ShaderNodeVectorMath", "p2m_tint", (-100, 100))
    tint.operation = "MULTIPLY"
    alpha = _node(nt, "ShaderNodeMath", "p2m_alpha", (-100, -150))
    alpha.operation = "MULTIPLY"
    solid = _node(nt, "ShaderNodeTexImage", "p2m_solid_preview", (-450, -350))
    solid.label = "Solid View Preview (texture x color)"

    nt.links.clear()
    nt.links.new(uv.outputs["Vector"], tex.inputs["Vector"])
    nt.links.new(uv.outputs["Vector"], solid.inputs["Vector"])
    nt.links.new(tex.outputs["Color"], tint.inputs[0])
    nt.links.new(tint.outputs["Vector"], bsdf.inputs["Base Color"])
    nt.links.new(alpha.outputs[0], bsdf.inputs["Alpha"])
    nt.links.new(bsdf.outputs[0], out.inputs["Surface"])
    bsdf.inputs["Roughness"].default_value = 0.7
    return nt


def _tinted_image(mat, base, rgba):
    """Writes base x color into the material's solid preview image."""
    name = "P2M Solid: " + mat.name
    if base is not None and base.size[0] > 0:
        w, h = base.size
        px = np.empty(w * h * 4, dtype=np.float32)
        base.pixels.foreach_get(px)
        px = px.reshape(h, w, 4)
        step = max(1, max(w, h) // TINT_MAX)
        px = px[::step, ::step].copy()
    else:
        px = np.ones((4, 4, 4), dtype=np.float32)
    px[..., 0:3] *= np.asarray(rgba[:3], dtype=np.float32)
    h, w = px.shape[:2]
    img = bpy.data.images.get(name)
    if img is None or tuple(img.size) != (w, h):
        if img is not None:
            bpy.data.images.remove(img)
        img = bpy.data.images.new(name, w, h, alpha=True)
    img.pixels.foreach_set(px.reshape(-1))
    img.update()
    return img


def update_controller(entity, ctrl):
    """Creates/updates the controller's Blender material."""
    mat = ctrl.blender_material
    if mat is None:
        mat = bpy.data.materials.new("P2M %s #%d" % (entity.name, ctrl.uid))
        ctrl.blender_material = mat
        mat.p2m_owner = entity
    nt = mat.node_tree if mat.node_tree and mat.node_tree.nodes.get("p2m_bsdf") else None
    if nt is None:
        nt = _build_tree(mat)

    base, translucent = _material_lookup(ctrl.material)
    tex = nt.nodes["p2m_texture"]
    tex.image = base
    r, g, b, a = ctrl.color
    lin = (srgb_to_linear(r), srgb_to_linear(g), srgb_to_linear(b))
    nt.nodes["p2m_tint"].inputs[1].default_value = lin
    alpha = nt.nodes["p2m_alpha"]
    alpha.inputs[1].default_value = a
    alpha_link = [l for l in nt.links if l.to_node == alpha and l.to_socket == alpha.inputs[0]]
    if translucent and not alpha_link:
        nt.links.new(tex.outputs["Alpha"], alpha.inputs[0])
    elif not translucent:
        for l in alpha_link:
            nt.links.remove(l)
        alpha.inputs[0].default_value = 1.0
    content.set_blend(mat, a < 0.999 or translucent)

    mat.diffuse_color = (lin[0], lin[1], lin[2], a)
    solid = nt.nodes["p2m_solid_preview"]
    solid.image = _tinted_image(mat, base, (r, g, b, a))
    nt.nodes.active = solid
    mat["p2m_controller_uid"] = ctrl.uid
    return mat


def refresh_all():
    clear_cache()
    for obj in bpy.data.objects:
        if obj.p2m.is_entity:
            for ctrl in obj.p2m.controllers:
                update_controller(obj, ctrl)


def regenerate_solid_previews():
    """Tinted preview images are not packed; rebuild them after file load."""
    for obj in bpy.data.objects:
        if obj.p2m.is_entity:
            for ctrl in obj.p2m.controllers:
                mat = ctrl.blender_material
                if mat and mat.node_tree and mat.node_tree.nodes.get("p2m_solid_preview"):
                    base = mat.node_tree.nodes["p2m_texture"].image
                    mat.node_tree.nodes["p2m_solid_preview"].image = _tinted_image(mat, base, tuple(ctrl.color))
