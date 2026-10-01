"""Hammer-style model browser: folder tree on the left, a scrollable grid of
rendered thumbnails, and Info / Materials / Body Groups tabs below. Drawn
over the 3D viewport (see browserbase / canvas)."""

import os
from collections import OrderedDict

import bpy
from bpy.props import StringProperty

from . import browserbase, canvas, content
from .core import raster, studiomdl, vmt, vtf

PREVIEW_SIZE = 256          # Info tab preview
THUMB_SIZE = 128            # grid thumbnails

_info = {}                  # model path -> dict
_textures = OrderedDict()   # vtf path -> rgba (small), LRU
_tree = None                # folder -> {"dirs": [...], "files": [...]}
_recursive = {}             # folder -> all files below it
_expanded = {"models"}
_images = canvas.TextureCache()   # "thumb:"/"big:" + path -> GPU texture
_state = {"filter": "", "subfolders": False, "folder": "models", "tab": "INFO"}


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

def _build_tree():
    global _tree
    fs = content.get_fs()
    tree = {"models": {"dirs": set(), "files": []}}
    if fs is not None:
        for p in fs.list_files("models/", ".mdl"):
            folder = p.rpartition("/")[0]
            tree.setdefault(folder, {"dirs": set(), "files": []})["files"].append(p)
            child = folder
            while "/" in child:
                parent = child.rpartition("/")[0]
                tree.setdefault(parent, {"dirs": set(), "files": []})["dirs"].add(child)
                child = parent
    for node in tree.values():
        node["dirs"] = sorted(node["dirs"])
    _tree = tree
    _recursive.clear()


def _files_below(folder):
    hit = _recursive.get(folder)
    if hit is None:
        node = _tree.get(folder, {"dirs": [], "files": []})
        hit = list(node["files"])
        for d in node["dirs"]:
            hit.extend(_files_below(d))
        _recursive[folder] = hit
    return hit


def _texture(vtf_path):
    if vtf_path in _textures:
        _textures.move_to_end(vtf_path)
        return _textures[vtf_path]
    data = content.read(vtf_path)
    try:
        tex = vtf.decode(data, THUMB_SIZE)[1] if data else None
    except Exception:
        tex = None
    _textures[vtf_path] = tex
    while len(_textures) > 256:
        _textures.popitem(last=False)
    return tex


def _model_textures(model):
    out = []
    for candidates in model.materials:
        tex = None
        for c in candidates:
            mat = vmt.load(content.read, c)
            if mat is None:
                continue
            if mat.base_texture:
                tex = _texture(mat.base_texture)
            break
        out.append(tex)
    return out


def _load_info(path):
    info = _info.get(path)
    if info is not None:
        return info, None
    info = {"error": None, "tris": 0, "verts": 0, "materials": [], "source": None, "bodyparts": []}
    _info[path] = info
    fs = content.get_fs()
    info["source"] = fs.locate(path) if fs else None
    try:
        model = studiomdl.load(content.read, path)
    except Exception as e:
        info["error"] = str(e)
        return info, None
    info["tris"] = len(model.triangles)
    info["verts"] = len(model.positions)
    info["materials"] = [c[0] if c else "?" for c in model.materials]
    info["bodyparts"] = model.bodyparts
    return info, model


def _render(path, size, supersample):
    _info_entry, model = _load_info(path)
    if model is None:
        try:
            model = studiomdl.load(content.read, path)
        except Exception:
            return canvas.checker(size)
    return raster.render(model, _model_textures(model), size=size, supersample=supersample)


def thumbnail_rgba(path):
    return _render(path, THUMB_SIZE, 1)


def preview_rgba(path):
    """Large preview for the Info tab: (PREVIEW_SIZE, PREVIEW_SIZE, 4)."""
    return _render(path, PREVIEW_SIZE, 2)


def invalidate():
    """Content changed (rescan): rebuild everything next time."""
    global _tree
    _tree = None
    _recursive.clear()
    _info.clear()
    _textures.clear()
    _images.clear()


# --------------------------------------------------------------------------
# Operator
# --------------------------------------------------------------------------

class P2M_OT_model_browser(browserbase.BrowserModal, bpy.types.Operator):
    """Browse all mounted models with thumbnails (like Hammer's model browser)"""
    bl_idname = "p2m.model_browser"
    bl_label = "Model Browser"
    bl_options = {"REGISTER", "UNDO"}

    entity: StringProperty(options={"HIDDEN"})

    # -- lifecycle ----------------------------------------------------------
    def setup(self, context):
        if content.get_fs() is None:
            self.report({"ERROR"}, content.status_text())
            return False
        obj = bpy.data.objects.get(self.entity) if self.entity else context.active_object
        if obj is None or not (obj.p2m.is_entity or obj.p2m.is_model_part):
            self.report({"ERROR"}, "Select a Prop2Mesh entity or model part")
            return False
        self.target = obj.name
        if _tree is None:
            _build_tree()
        current = obj.p2m.model.lower().replace("\\", "/")
        if current and not current.endswith(".mdl"):
            current += ".mdl"
        folder = current.rpartition("/")[0] if current.startswith("models/") else _state["folder"]
        while folder and folder not in _tree:
            folder = folder.rpartition("/")[0]
        folder = folder or "models"
        f = folder
        while f:
            _expanded.add(f)
            f = f.rpartition("/")[0]
        _state["folder"] = folder
        self.selected = current if current and content.get_fs().exists(current) else ""
        self.tree_scroll = 0
        self.grid_scroll = 0.0
        self.wanted = []
        self._scroll_to_selected = bool(self.selected)
        self._scroll_tree_to_folder = True
        self._refresh()
        return True

    def accept(self, context):
        obj = bpy.data.objects.get(self.target)
        if obj is None or not self.selected:
            return False
        obj.p2m.model = self.selected
        return True

    # -- data ---------------------------------------------------------------
    def _refresh(self):
        folder = _state["folder"]
        tokens = [t for t in _state["filter"].lower().replace("\\", "/").split() if t]
        files = _files_below(folder) if (tokens or _state["subfolders"]) else _tree.get(folder, {"files": []})["files"]
        if tokens:
            files = [p for p in files if all(t in p for t in tokens)]
        self.grid = sorted(files)
        self.grid_scroll = 0.0

    def _tree_rows(self):
        rows = []

        def add(folder, depth):
            node = _tree.get(folder)
            if node is None:
                return
            rows.append((folder, depth, bool(node["dirs"])))
            if folder in _expanded:
                for d in node["dirs"]:
                    add(d, depth + 1)
        add("models", 0)
        return rows

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
        elif kind == "folder":
            _state["folder"] = action[1]
            self._refresh()
            if double:
                self._toggle(action[1])
        elif kind == "toggle":
            self._toggle(action[1])
        elif kind == "subfolders":
            _state["subfolders"] = not _state["subfolders"]
            self._refresh()
        elif kind == "tab":
            _state["tab"] = action[1]
        return None

    def _toggle(self, folder):
        if folder in _expanded:
            _expanded.discard(folder)
        else:
            _expanded.add(folder)

    def work(self):
        if self.selected and not _images.has("big:" + self.selected):
            _images.put("big:" + self.selected, preview_rgba(self.selected))
            return True
        while self.wanted:
            path = self.wanted.pop(0)
            key = "thumb:" + path
            if not _images.has(key):
                try:
                    _images.put(key, thumbnail_rgba(path))
                except Exception as e:
                    print("AutoProp2Mesh thumbnail failed:", path, e)
                    _images.put(key, canvas.checker(THUMB_SIZE))
                return True
        return False

    # -- layout -------------------------------------------------------------
    def _metrics(self, W, H, s):
        pad = 8 * s
        left_w = min(320 * s, W * 0.3)
        info_h = 190 * s
        grid = (left_w + pad, pad + 24 * s, W - left_w - 2 * pad, H - info_h - 3 * pad - 24 * s)
        tree = (pad, pad + 84 * s, left_w - 2 * pad, H - 84 * s - 110 * s - pad)
        return pad, left_w, info_h, grid, tree

    def scroll_at(self, mx, my, delta):
        s = canvas.ui_scale()
        W, H = self.region.width, self.region.height - self.top_inset()
        _pad, left_w, _info_h, grid, tree = self._metrics(W, H, s)
        if mx < left_w:
            self.tree_scroll = max(0, self.tree_scroll + int(delta * 3))
        elif my < grid[1] + grid[3]:
            self.grid_scroll = max(0.0, self.grid_scroll + delta * (THUMB_SIZE + 22) * s * 0.5)

    # -- drawing ------------------------------------------------------------
    def draw_ui(self, cv, W, H):
        s = cv.s
        pad, left_w, info_h, (gx, gy, gw, gh), (tx, ty, tw, th) = self._metrics(W, H, s)
        hover = self.hover

        # ---- left: filter, options, tree, full path, buttons --------------
        cv.rect(0, 0, left_w, H, canvas.PANEL)
        cv.text("Model Browser", pad, pad, 13)
        cv.field(_state["filter"], pad, pad + 24 * s, left_w - 2 * pad, 22 * s, ("focus", "filter"),
                 focused=self.focus == "filter", placeholder="Filter (just start typing)")
        cv.checkbox("Check subfolders for files", _state["subfolders"], pad, pad + 56 * s, ("subfolders",))

        cv.rect(tx, ty, tw, th, canvas.FIELD)
        cv.frame(tx, ty, tw, th, canvas.BORDER)
        rows = self._tree_rows()
        row_h = 18 * s
        visible = max(1, int(th // row_h))
        if self._scroll_tree_to_folder:
            idx = next((i for i, r in enumerate(rows) if r[0] == _state["folder"]), 0)
            self.tree_scroll = max(0, idx - visible // 2)
            self._scroll_tree_to_folder = False
        self.tree_scroll = max(0, min(self.tree_scroll, max(0, len(rows) - visible)))
        cv.clip(tx, ty, tw, th)
        for i, (folder, depth, has_dirs) in enumerate(rows[self.tree_scroll:self.tree_scroll + visible + 1]):
            y = ty + i * row_h
            x = tx + 4 * s + depth * 14 * s
            if folder == _state["folder"]:
                cv.rect(tx, y, tw, row_h, canvas.ACCENT)
            elif hover == ("folder", folder):
                cv.rect(tx, y, tw, row_h, canvas.HOVER)
            cv.hot(tx, y, tw, row_h, ("folder", folder))
            if has_dirs:
                b = 9 * s
                bx, by = x, y + (row_h - b) / 2
                cv.rect(bx, by, b, b, canvas.FIELD)
                cv.frame(bx, by, b, b, canvas.DIM)
                cv.rect(bx + 2 * s, by + b / 2 - 0.5, b - 4 * s, 1, canvas.TEXT)
                if folder not in _expanded:
                    cv.rect(bx + b / 2 - 0.5, by + 2 * s, 1, b - 4 * s, canvas.TEXT)
                cv.hot(bx - 2 * s, y, b + 4 * s, row_h, ("toggle", folder))
            name = "All Models" if folder == "models" else folder.rpartition("/")[2]
            cv.text(name, x + 14 * s, y + 3 * s, 11, canvas.TEXT, max_w=tw - (x - tx) - 60 * s)
            cv.text(str(len(_files_below(folder))), tx, y + 3 * s, 10, canvas.DIM, max_w=tw - 6 * s, align="RIGHT")
        cv.unclip()

        by = ty + th + 8 * s
        cv.text("Full path:", pad, by, 11, canvas.DIM)
        cv.field(self.selected, pad, by + 16 * s, left_w - 2 * pad, 22 * s, ("noop",))
        bw = (left_w - 3 * pad) / 2
        cv.button("OK", pad, H - pad - 26 * s, bw, 26 * s, ("ok",), active=True, hover=hover == ("ok",))
        cv.button("Cancel", 2 * pad + bw, H - pad - 26 * s, bw, 26 * s, ("cancel",), hover=hover == ("cancel",))

        # ---- right: thumbnail grid -----------------------------------------
        cv.text("%d models" % len(self.grid), gx, pad + 4 * s, 11, canvas.DIM)
        cv.text("Click to select, double-click or Enter to use, Esc to cancel, wheel to scroll",
                gx, pad + 4 * s, 10, canvas.DIM, max_w=gw, align="RIGHT")
        tile = THUMB_SIZE * s
        label_h = 16 * s
        gap = 6 * s
        cols = max(1, int((gw + gap) // (tile + gap)))
        cell_h = tile + label_h + gap
        rows_total = (len(self.grid) + cols - 1) // cols
        max_scroll = max(0.0, rows_total * cell_h - gh)
        if self._scroll_to_selected and self.selected in self.grid:
            r = self.grid.index(self.selected) // cols
            self.grid_scroll = max(0.0, r * cell_h - gh / 2 + cell_h / 2)
            self._scroll_to_selected = False
        self.grid_scroll = min(self.grid_scroll, max_scroll)
        first_row = int(self.grid_scroll // cell_h)
        last_row = int((self.grid_scroll + gh) // cell_h) + 1
        wanted = []
        cv.clip(gx, gy, gw, gh)
        for r in range(first_row, min(rows_total, last_row + 1)):
            for c in range(cols):
                i = r * cols + c
                if i >= len(self.grid):
                    break
                path = self.grid[i]
                x = gx + c * (tile + gap)
                y = gy + r * cell_h - self.grid_scroll
                selected = path == self.selected
                tex = _images.get("thumb:" + path)
                cv.rect(x, y, tile, tile, (0.2, 0.2, 0.2, 1))
                if tex is not None:
                    cv.image(tex, x, y, tile, tile)
                else:
                    wanted.append(path)
                    cv.text("...", x, y + tile / 2 - 6 * s, 12, canvas.DIM, max_w=tile, align="CENTER")
                if hover == ("select", path):
                    cv.rect(x, y, tile, tile + label_h, canvas.HOVER)
                cv.rect(x, y + tile, tile, label_h, canvas.ACCENT if selected else (0.22, 0.22, 0.22, 1))
                cv.text(os.path.basename(path), x + 2 * s, y + tile + 2 * s, 10, canvas.TEXT, max_w=tile - 4 * s,
                        align="CENTER")
                if selected:
                    cv.frame(x - 2 * s, y - 2 * s, tile + 4 * s, tile + label_h + 4 * s, canvas.SELECT, max(1, int(2 * s)))
                cv.hot(x, y, tile, tile + label_h, ("select", path))
        cv.unclip()
        # Thumbnails just above/below the view are rendered next.
        for r in (last_row + 1, first_row - 1):
            for c in range(cols):
                i = r * cols + c
                if 0 <= i < len(self.grid) and not _images.has("thumb:" + self.grid[i]):
                    wanted.append(self.grid[i])
        self.wanted = wanted
        if max_scroll > 0:
            bar_h = max(20 * s, gh * gh / (gh + max_scroll))
            bar_y = gy + (gh - bar_h) * (self.grid_scroll / max_scroll)
            cv.rect(gx + gw + 2 * s, bar_y, 4 * s, bar_h, canvas.BORDER)

        # ---- bottom: tabs ----------------------------------------------------
        iy = H - info_h - pad
        tab_w = 110 * s
        for k, (key, label) in enumerate((("INFO", "Info"), ("MATERIALS", "Materials"), ("BODYGROUPS", "Body Groups"))):
            cv.button(label, gx + k * (tab_w + 2 * s), iy, tab_w, 22 * s, ("tab", key),
                      active=_state["tab"] == key, hover=hover == ("tab", key))
        body_y = iy + 24 * s
        body_h = info_h - 24 * s
        cv.rect(gx, body_y, gw, body_h, canvas.PANEL)
        cv.frame(gx, body_y, gw, body_h, canvas.BORDER)
        if not self.selected:
            cv.text("Select a model above", gx + pad, body_y + pad, 11, canvas.DIM)
            return
        info, _model = _load_info(self.selected)
        lx = gx + pad
        ly = body_y + pad
        if _state["tab"] == "INFO":
            ps = body_h - 2 * pad
            tex = _images.get("big:" + self.selected)
            cv.rect(lx, ly, ps, ps, (0.2, 0.2, 0.2, 1))
            if tex is not None:
                cv.image(tex, lx, ly, ps, ps)
            lx += ps + 12 * s
            lines = [(self.selected, canvas.TEXT)]
            if info.get("error"):
                lines.append((info["error"], (1, 0.5, 0.5, 1)))
            else:
                groups = [bp for bp in info["bodyparts"] if len(bp[2]) > 1]
                lines += [("%d triangles, %d vertices" % (info["tris"], info["verts"]), canvas.TEXT),
                          ("%d materials, %d body groups" % (len(info["materials"]), len(groups)), canvas.TEXT)]
            if info.get("source"):
                lines.append(("From: %s" % info["source"], canvas.DIM))
            for k, (line, col) in enumerate(lines):
                cv.text(line, lx, ly + k * 18 * s, 11, col, max_w=gx + gw - lx - pad)
        elif _state["tab"] == "MATERIALS":
            for k, m in enumerate(info["materials"][:8] or ["No materials"]):
                cv.text(m, lx, ly + k * 18 * s, 11, canvas.TEXT, max_w=gw - 2 * pad)
        else:
            groups = [bp for bp in info["bodyparts"] if len(bp[2]) > 1]
            if not groups:
                cv.text("No body groups", lx, ly, 11, canvas.DIM)
            line = 0
            for name, _base, models in groups:
                opts = ", ".join("%d: %s" % (i, (m[:-4] if m.lower().endswith(".smd") else m) or "(blank)")
                                 for i, m in enumerate(models))
                cv.text("%s  -  %s" % (name, opts), lx, ly + line * 18 * s, 11, canvas.TEXT, max_w=gw - 2 * pad)
                line += 1


classes = (P2M_OT_model_browser,)


def register():
    pass


def unregister():
    invalidate()
