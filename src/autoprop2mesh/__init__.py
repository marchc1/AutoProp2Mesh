"""AutoProp2Mesh: build Prop2Mesh (sent_prop2mesh) entities in Blender and
export them as AdvDupe2 dupes for Garry's Mod."""

try:
    import bpy
except ImportError:  # imported outside Blender (e.g. testing autoprop2mesh.core)
    bpy = None

if bpy is not None:
    from . import browser, content, export, materialbrowser, modelparts, ops, prefs, props, sync, ui

    _class_groups = (prefs.classes, props.classes, browser.classes, materialbrowser.classes, ops.classes,
                     export.classes, ui.classes)


def _start_content():
    content.start_build()
    # The add-on may be enabled with a file already open: rebuild the
    # (unpacked) solid-view preview images that load_post would have made.
    try:
        sync.on_load_post()
    except Exception as e:
        print("AutoProp2Mesh:", e)
    return None


def register():
    for group in _class_groups:
        for cls in group:
            bpy.utils.register_class(cls)
    props.register()
    browser.register()
    ui.register()
    sync.register()
    # Index the game content in the background as soon as Blender is idle.
    bpy.app.timers.register(_start_content, first_interval=0.5)


def unregister():
    if bpy.app.timers.is_registered(_start_content):
        bpy.app.timers.unregister(_start_content)
    sync.unregister()
    ui.unregister()
    browser.unregister()
    props.unregister()
    for group in reversed(_class_groups):
        for cls in reversed(group):
            bpy.utils.unregister_class(cls)
    content.shutdown()
