"""Hammer-style material browser (modelled on Hammer's texture browser): a
scrollable grid of material textures with their names on blue bars, and a
bottom bar with Size, Filter, the selection, and Opaque / Translucent /
Only-used filters. Drawn over the 3D viewport (see browserbase / canvas)."""

from collections import OrderedDict

import bpy
import numpy as np
from bpy.props import IntProperty, StringProperty

from . import browserbase, canvas, content, props
from .core import vmt, vtf

SIZES = (64, 128, 256)

_names = None                    # all material names ("hunter/myplastic")
_flags = {}                      # name -> True (translucent) / False / None (no VMT)
_images = canvas.TextureCache(limit=800)
_meta = OrderedDict()            # name -> (full_w, full_h, image_w, image_h)
_state = {"filter": "", "size": 128, "opaque": True, "translucent": True, "used": False}


def _all_names():
    global _names
    if _names is None:
        fs = content.get_fs()
        files = fs.list_files("materials/", ".vmt") if fs else []
        _names = [p[len("materials/"):-len(".vmt")] for p in files]
    return _names


def _vmt(name):
    return vmt.load(content.read, vmt.material_path(name))


def _translucent(name):
    if name not in _flags:
        try:
            mat = _vmt(name)
        except Exception:
            mat = None
        _flags[name] = None if mat is None else mat.translucent
    return _flags[name]


def thumbnail(name, size):
    """(rgba, (full_w, full_h)) for a material's base texture."""
    try:
        mat = _vmt(name)
    except Exception:
        mat = None
    if mat is not None:
        _flags[name] = mat.translucent
    tex = mat.base_texture if mat else None
    data = content.read(tex) if tex else None
    if data is None:
        return canvas.checker(64), (0, 0)
    try:
        info, rgba = vtf.decode(data, size)
    except Exception:
        return canvas.checker(64), (0, 0)
    rgba = rgba.copy()
    if not (mat and mat.translucent):
        rgba[..., 3] = 1.0  # alpha often holds a specular mask
    return rgba, (info.full_width, info.full_height)


def invalidate():
    global _names
    _names = None
    _flags.clear()
    _meta.clear()
    _images.clear()


class P2M_OT_material_browser(browserbase.BrowserModal, bpy.types.Operator):
    """Browse all mounted materials with previews (like Hammer's texture browser)"""
    bl_idname = "p2m.material_browser"
    bl_label = "Material Browser"
    bl_options = {"REGISTER", "UNDO"}

    entity: StringProperty(options={"HIDDEN"})
    uid: IntProperty(default=-1, options={"HIDDEN"})

    # -- lifecycle ----------------------------------------------------------
    def setup(self, context):
        if content.get_fs() is None:
            self.report({"ERROR"}, content.status_text())
            return False
        ent = bpy.data.objects.get(self.entity)
        ctrl = props.controller_by_uid(ent, self.uid) if ent else None
        if ctrl is None:
            self.report({"ERROR"}, "No controller to set the material on")
            return False
        self.target = (ent.name, ctrl.uid)
        self.selected = vmt.material_path(ctrl.material)[len("materials/"):-len(".vmt")]
        self.scroll = 0.0
        self.wanted = []
        self._scroll_to_selected = True
        self._refresh()
        return True

    def accept(self, context):
        ent = bpy.data.objects.get(self.target[0])
        ctrl = props.controller_by_uid(ent, self.target[1]) if ent else None
        if ctrl is None or not self.selected:
            return False
        ctrl.material = self.selected
        return True

    # -- data ---------------------------------------------------------------
    def _refresh(self):
        tokens = [t for t in _state["filter"].lower().replace("\\", "/").split() if t]
        if _state["used"]:
            used = set()
            for obj in bpy.data.objects:
                if obj.p2m.is_entity:
                    for c in obj.p2m.controllers:
                        used.add(vmt.material_path(c.material)[len("materials/"):-len(".vmt")])
            names = sorted(used)
        else:
            names = _all_names()
        self.candidates = [n for n in names if all(t in n for t in tokens)] if tokens else list(names)
        self.scanning = not (_state["opaque"] and _state["translucent"])
        if self.scanning:
            self.items = []
            self.scan_pos = 0
        else:
            self.items = self.candidates
        self.scroll = 0.0

    def _passes(self, name):
        t = _translucent(name)
        if t is None:
            return False
        return _state["translucent"] if t else _state["opaque"]

    def filter_text(self):
        return _state["filter"]

    def on_text(self, text):
        _state["filter"] = text
        self._refresh()

    def on_action(self, action, double):
        kind = action[0]
        if kind == "select":
            self.selected = action[1]
            if double:
                return "ACCEPT"
        elif kind == "size":
            i = SIZES.index(_state["size"]) if _state["size"] in SIZES else 1
            _state["size"] = SIZES[(i + 1) % len(SIZES)]
            self._scroll_to_selected = True
        elif kind in ("opaque", "translucent", "used"):
            _state[kind] = not _state[kind]
            self._refresh()
        return None

    def work(self):
        if self.scanning and self.scan_pos < len(self.candidates):
            # Classify a slice of materials per call (VMT reads are quick).
            end = min(len(self.candidates), self.scan_pos + 40)
            for name in self.candidates[self.scan_pos:end]:
                if self._passes(name):
                    self.items.append(name)
            self.scan_pos = end
            return True
        size = _state["size"]
        while self.wanted:
            name = self.wanted.pop(0)
            key = "%d:%s" % (size, name)
            if not _images.has(key):
                rgba, full = thumbnail(name, size)
                _images.put(key, rgba)
                _meta[name] = (full[0], full[1])
                return True
        return False

    def scroll_at(self, mx, my, delta):
        s = canvas.ui_scale()
        self.scroll = max(0.0, self.scroll + delta * (_state["size"] + 20) * s * 0.5)

    # -- drawing ------------------------------------------------------------
    def draw_ui(self, cv, W, H):
        s = cv.s
        hover = self.hover
        size = _state["size"]
        bar_h = 62 * s
        gx, gy, gw, gh = 2 * s, 2 * s, W - 14 * s, H - bar_h - 4 * s
        tile = size * s
        label_h = 14 * s
        gap = 3 * s
        cols = max(1, int((gw + gap) // (tile + gap)))
        cell_h = tile + label_h + gap
        items = self.items
        rows_total = (len(items) + cols - 1) // cols
        max_scroll = max(0.0, rows_total * cell_h - gh)
        if self._scroll_to_selected and self.selected in items:
            r = items.index(self.selected) // cols
            self.scroll = max(0.0, r * cell_h - gh / 2 + cell_h / 2)
            self._scroll_to_selected = False
        self.scroll = min(self.scroll, max_scroll)
        first_row = int(self.scroll // cell_h)
        last_row = int((self.scroll + gh) // cell_h) + 1

        wanted = []
        cv.rect(gx, gy, gw, gh, (0, 0, 0, 1))
        cv.clip(gx, gy, gw, gh)
        for r in range(first_row, min(rows_total, last_row + 1)):
            for c in range(cols):
                i = r * cols + c
                if i >= len(items):
                    break
                name = items[i]
                x = gx + c * (tile + gap)
                y = gy + r * cell_h - self.scroll
                key = "%d:%s" % (size, name)
                tex = _images.get(key)
                if tex is not None:
                    rgba = _images.pixels[key]
                    ih, iw = rgba.shape[:2]
                    k = tile / max(iw, ih)
                    # Hammer draws non-square textures from the top-left.
                    cv.image(tex, x, y, iw * k, ih * k)
                else:
                    wanted.append(name)
                    cv.text("...", x, y + tile / 2 - 6 * s, 12, canvas.DIM, max_w=tile, align="CENTER")
                if hover == ("select", name):
                    cv.rect(x, y, tile, tile, canvas.HOVER)
                selected = name == self.selected
                cv.rect(x, y + tile, tile, label_h, (0.0, 0.0, 1.0, 1.0) if not selected else (0.85, 0.15, 0.15, 1))
                cv.text(name, x + 2 * s, y + tile + 1 * s, 9, (1, 1, 1, 1), max_w=tile - 4 * s)
                if selected:
                    cv.frame(x, y, tile, tile + label_h, canvas.SELECT, max(1, int(2 * s)))
                cv.hot(x, y, tile, tile + label_h, ("select", name))
        cv.unclip()
        for r in (last_row + 1,):
            for c in range(cols):
                i = r * cols + c
                if 0 <= i < len(items) and not _images.has("%d:%s" % (size, items[i])):
                    wanted.append(items[i])
        self.wanted = wanted
        if max_scroll > 0:
            sb_h = max(20 * s, gh * gh / (gh + max_scroll))
            sb_y = gy + (gh - sb_h) * (self.scroll / max_scroll)
            cv.rect(W - 10 * s, sb_y, 6 * s, sb_h, canvas.BORDER)

        # ---- bottom bar ----------------------------------------------------
        by = H - bar_h
        cv.rect(0, by, W, bar_h, canvas.PANEL)
        cv.rect(0, by, W, 1, canvas.BORDER)
        row1 = by + 8 * s
        row2 = by + 34 * s
        x = 10 * s
        cv.text("Size:", x, row1 + 4 * s, 11)
        cv.button("%dx%d" % (size, size), x + 36 * s, row1, 80 * s, 20 * s, ("size",), hover=hover == ("size",))
        x += 130 * s
        cv.text("Filter:", x, row1 + 4 * s, 11)
        cv.field(_state["filter"], x + 42 * s, row1, 240 * s, 20 * s, ("focus", "filter"),
                 focused=self.focus == "filter", placeholder="just start typing")
        x += 300 * s
        cv.text(self.selected or "(no selection)", x, row1 + 4 * s, 11, canvas.TEXT, max_w=320 * s)
        meta = _meta.get(self.selected)
        if meta and meta[0]:
            cv.text("%dx%d" % meta, x, row2 + 4 * s, 11, canvas.DIM)
        cx = x + 340 * s
        cv.checkbox("Opaque", _state["opaque"], cx, row1 + 4 * s, ("opaque",))
        cv.checkbox("Translucent", _state["translucent"], cx, row2 + 4 * s, ("translucent",))
        cv.checkbox("Only used materials", _state["used"], 10 * s, row2 + 4 * s, ("used",))
        status = "%d materials" % len(items)
        if self.scanning and self.scan_pos < len(self.candidates):
            status += "  (checking %d%%...)" % (100 * self.scan_pos // max(1, len(self.candidates)))
        cv.text(status, 170 * s, row2 + 4 * s, 11, canvas.DIM)
        bw = 90 * s
        cv.button("Apply", W - 2 * bw - 20 * s, row1, bw, 22 * s, ("ok",), active=True, hover=hover == ("ok",))
        cv.button("Cancel", W - bw - 10 * s, row1, bw, 22 * s, ("cancel",), hover=hover == ("cancel",))
        cv.text("Double-click or Enter to apply, Esc to cancel", W - 2 * bw - 20 * s, row2 + 4 * s, 10, canvas.DIM,
                max_w=2 * bw + 10 * s, align="RIGHT")


classes = (P2M_OT_material_browser,)
