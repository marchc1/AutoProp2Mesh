import os

import bpy
from bpy.props import BoolProperty, CollectionProperty, IntProperty, StringProperty

from .core import steam


class P2M_SearchPath(bpy.types.PropertyGroup):
    path: StringProperty(
        name="Path", subtype="DIR_PATH",
        description="A content folder (has models/ or materials/), or a folder of such folders",
    )


class P2M_AddonPreferences(bpy.types.AddonPreferences):
    bl_idname = __package__

    gmod_dir: StringProperty(
        name="Garry's Mod Folder", subtype="DIR_PATH",
        description="The GarrysMod install folder (the one containing garrysmod/). Leave empty to auto-detect",
    )
    extra_paths: CollectionProperty(type=P2M_SearchPath)
    extra_paths_index: IntProperty()
    mount_workshop: BoolProperty(name="Mount Workshop Addons", default=True)
    mount_games: BoolProperty(
        name="Mount Games", default=True,
        description="Mount the games enabled in Garry's Mod (cfg/mountdepots.txt and cfg/mount.cfg)",
    )
    texture_size: IntProperty(
        name="Max Texture Size", default=1024, min=64, max=4096,
        description="Largest texture mip level loaded for previews",
    )

    def draw(self, context):
        from . import content
        layout = self.layout
        col = layout.column()
        col.prop(self, "gmod_dir")
        detected = resolved_gmod_dir(self)
        col.label(text="Using: %s" % (detected or "not found"),
                  icon="CHECKMARK" if detected else "ERROR")

        box = layout.box()
        box.label(text="Extra Content Folders")
        row = box.row()
        row.template_list("UI_UL_list", "p2m_paths", self, "extra_paths", self, "extra_paths_index", rows=3)
        sub = row.column(align=True)
        sub.operator("p2m.search_path_add", icon="ADD", text="")
        sub.operator("p2m.search_path_remove", icon="REMOVE", text="")
        if 0 <= self.extra_paths_index < len(self.extra_paths):
            box.prop(self.extra_paths[self.extra_paths_index], "path", text="")

        row = layout.row()
        row.prop(self, "mount_workshop")
        row.prop(self, "mount_games")
        layout.prop(self, "texture_size")

        row = layout.row()
        row.label(text=content.status_text())
        row.operator("p2m.rebuild_content", icon="FILE_REFRESH")


class P2M_OT_search_path_add(bpy.types.Operator):
    bl_idname = "p2m.search_path_add"
    bl_label = "Add Content Folder"
    bl_options = {"INTERNAL"}

    def execute(self, context):
        p = get_prefs()
        p.extra_paths.add()
        p.extra_paths_index = len(p.extra_paths) - 1
        return {"FINISHED"}


class P2M_OT_search_path_remove(bpy.types.Operator):
    bl_idname = "p2m.search_path_remove"
    bl_label = "Remove Content Folder"
    bl_options = {"INTERNAL"}

    def execute(self, context):
        p = get_prefs()
        if 0 <= p.extra_paths_index < len(p.extra_paths):
            p.extra_paths.remove(p.extra_paths_index)
            p.extra_paths_index = max(0, p.extra_paths_index - 1)
        return {"FINISHED"}


def get_prefs():
    addon = bpy.context.preferences.addons.get(__package__)
    return addon.preferences if addon else None


_detected = []


def resolved_gmod_dir(prefs=None):
    prefs = prefs or get_prefs()
    if prefs and prefs.gmod_dir:
        p = bpy.path.abspath(prefs.gmod_dir).rstrip("\\/")
        if os.path.isdir(os.path.join(p, "garrysmod")):
            return p
        # Accept the garrysmod/ folder itself, too.
        if os.path.basename(p).lower() == "garrysmod" and os.path.isdir(p):
            return os.path.dirname(p)
    if not _detected:
        try:
            _detected.append(steam.find_gmod()[0])
        except Exception:
            _detected.append(None)
    return _detected[0]


def default_dupe_dir():
    gmod = resolved_gmod_dir()
    if not gmod:
        return None
    return os.path.join(gmod, "garrysmod", "data", "advdupe2")


classes = (P2M_SearchPath, P2M_AddonPreferences, P2M_OT_search_path_add, P2M_OT_search_path_remove)
