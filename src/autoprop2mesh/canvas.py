"""A tiny immediate-mode UI drawn with the gpu module, for the Hammer-style
browsers. Blender's layout system cannot make a large image clickable, so the
browsers draw themselves over the 3D viewport from a modal operator.

Coordinates passed to the drawing helpers are top-down (y = 0 at the top of
the region); they are converted to Blender's bottom-up region space here.
Every frame, widgets register "hotspots" (rect + action) that the modal
operator hit-tests on the next mouse event.
"""

import blf
import bpy
import gpu
import numpy as np
from gpu_extras.batch import batch_for_shader

FONT = 0

# Colours (sRGB-ish, straight alpha), close to Hammer's browsers
BG = (0.10, 0.10, 0.10, 0.98)
PANEL = (0.16, 0.16, 0.16, 1.0)
FIELD = (0.07, 0.07, 0.07, 1.0)
BORDER = (0.30, 0.30, 0.30, 1.0)
TEXT = (0.90, 0.90, 0.90, 1.0)
DIM = (0.60, 0.60, 0.60, 1.0)
ACCENT = (0.0, 0.0, 1.0, 1.0)        # Hammer's selection blue
LABEL_BAR = (0.0, 0.0, 0.85, 1.0)
HOVER = (1.0, 1.0, 1.0, 0.10)
SELECT = (1.0, 0.95, 0.4, 1.0)


def ui_scale():
    try:
        return bpy.context.preferences.system.ui_scale
    except Exception:
        return 1.0


class Canvas:
    def __init__(self, region_height, top_offset=0, scroll=None):
        self.h = region_height - top_offset  # drawing space starts below the offset
        self.s = ui_scale()
        self.hotspots = []
        self.scroll = scroll if scroll is not None else {}
        self.scrollers = {}     # key -> scroll region info (for the modal)
        self.fields = {}        # key -> (text origin x, font size) of text fields
        self.hover = None
        self.active_bar = None
        self._clip = None
        self._color = gpu.shader.from_builtin("UNIFORM_COLOR")
        self._image = gpu.shader.from_builtin("IMAGE")

    # -- coordinates -------------------------------------------------------
    def _y(self, y, height):
        return self.h - y - height

    # -- primitives ----------------------------------------------------------
    def rect(self, x, y, w, h, color):
        if w <= 0 or h <= 0:
            return
        y0 = self._y(y, h)
        verts = ((x, y0), (x + w, y0), (x + w, y0 + h), (x, y0 + h))
        batch = batch_for_shader(self._color, "TRIS", {"pos": verts}, indices=((0, 1, 2), (0, 2, 3)))
        gpu.state.blend_set("ALPHA")
        self._color.bind()
        self._color.uniform_float("color", color)
        batch.draw(self._color)

    def frame(self, x, y, w, h, color, t=1):
        self.rect(x, y, w, t, color)
        self.rect(x, y + h - t, w, t, color)
        self.rect(x, y, t, h, color)
        self.rect(x + w - t, y, t, h, color)

    def image(self, tex, x, y, w, h):
        y0 = self._y(y, h)
        verts = ((x, y0), (x + w, y0), (x + w, y0 + h), (x, y0 + h))
        uvs = ((0, 0), (1, 0), (1, 1), (0, 1))
        batch = batch_for_shader(self._image, "TRIS", {"pos": verts, "texCoord": uvs},
                                 indices=((0, 1, 2), (0, 2, 3)))
        gpu.state.blend_set("ALPHA")
        self._image.bind()
        self._image.uniform_sampler("image", tex)
        batch.draw(self._image)

    def text(self, s, x, y, size=11, color=TEXT, max_w=None, align="LEFT"):
        """y is the top of the text line."""
        px = size * self.s
        blf.size(FONT, px)
        if max_w is not None:
            s = self.ellipsize(s, max_w, size)
        w = blf.dimensions(FONT, s)[0]
        if align == "CENTER" and max_w is not None:
            x = x + (max_w - w) / 2
        elif align == "RIGHT" and max_w is not None:
            x = x + max_w - w
        blf.color(FONT, *color)
        blf.position(FONT, x, self.h - y - px * 0.95, 0)
        blf.draw(FONT, s)
        return w

    def text_width(self, s, size=11):
        blf.size(FONT, size * self.s)
        return blf.dimensions(FONT, s)[0]

    def ellipsize(self, s, max_w, size=11):
        blf.size(FONT, size * self.s)
        if blf.dimensions(FONT, s)[0] <= max_w:
            return s
        while s and blf.dimensions(FONT, s + "...")[0] > max_w:
            s = s[:-1]
        return s + "..."

    def clip(self, x, y, w, h):
        self._clip = (x, y, w, h)
        gpu.state.scissor_test_set(True)
        gpu.state.scissor_set(int(x), int(self._y(y, h)), max(0, int(w)), max(0, int(h)))

    def unclip(self):
        self._clip = None
        gpu.state.scissor_test_set(False)

    # -- interaction --------------------------------------------------------
    def hot(self, x, y, w, h, action):
        """Registers a clickable rect, cut to the current clip rect so things
        scrolled out of view cannot be clicked."""
        if self._clip is not None:
            cx, cy, cw, ch = self._clip
            x0, y0 = max(x, cx), max(y, cy)
            x1, y1 = min(x + w, cx + cw), min(y + h, cy + ch)
            if x1 <= x0 or y1 <= y0:
                return
            x, y, w, h = x0, y0, x1 - x0, y1 - y0
        self.hotspots.append((x, y, w, h, action))

    @staticmethod
    def hit(hotspots, mx, my_top):
        for x, y, w, h, action in reversed(hotspots):
            if x <= mx < x + w and y <= my_top < y + h:
                return action
        return None

    # -- widgets ------------------------------------------------------------
    def button(self, label, x, y, w, h, action, active=False, hover=False):
        self.rect(x, y, w, h, (0.28, 0.28, 0.28, 1) if not active else (0.25, 0.35, 0.6, 1))
        if hover:
            self.rect(x, y, w, h, HOVER)
        self.frame(x, y, w, h, BORDER)
        self.text(label, x, y + (h - 11 * self.s) / 2 - 1, 11, TEXT, max_w=w, align="CENTER")
        self.hot(x, y, w, h, action)

    def checkbox(self, label, value, x, y, action):
        b = 12 * self.s
        self.rect(x, y, b, b, FIELD)
        self.frame(x, y, b, b, BORDER)
        if value:
            self.rect(x + 3 * self.s, y + 3 * self.s, b - 6 * self.s, b - 6 * self.s, TEXT)
        w = self.text(label, x + b + 5 * self.s, y - 1, 11, TEXT)
        self.hot(x, y, b + 5 * self.s + w, b, action)
        return b + 5 * self.s + w

    def field(self, edit, key, x, y, w, h, focused=False, placeholder="", size=11):
        """An editable single-line text field (state in a browserbase.TextEdit)."""
        s = self.s
        self.rect(x, y, w, h, FIELD)
        self.frame(x, y, w, h, (0.45, 0.55, 0.9, 1) if focused else BORDER)
        pad = 5 * s
        inner_x, inner_w = x + pad, w - 2 * pad
        text = edit.text
        ty = y + (h - size * s) / 2 - 1
        if not text and not focused:
            self.text(placeholder, inner_x, ty, size, DIM, max_w=inner_w)
        else:
            # Keep the cursor in view by scrolling the text horizontally.
            cursor_x = self.text_width(text[:edit.cursor], size)
            if cursor_x - edit.view > inner_w - 2:
                edit.view = cursor_x - inner_w + 2
            elif cursor_x - edit.view < 0:
                edit.view = cursor_x
            edit.view = max(0.0, min(edit.view, max(0.0, self.text_width(text, size) - inner_w + 2)))
            origin = inner_x - edit.view
            prev_clip = self._clip
            self.clip(inner_x, y, inner_w, h)
            sel = edit.selection()
            if sel and focused:
                sx0 = origin + self.text_width(text[:sel[0]], size)
                sx1 = origin + self.text_width(text[:sel[1]], size)
                self.rect(sx0, y + 3 * s, sx1 - sx0, h - 6 * s, (0.25, 0.4, 0.8, 1))
            self.text(text, origin, ty, size, TEXT)
            if focused:
                self.rect(origin + cursor_x, y + 3 * s, max(1, int(s)), h - 6 * s, TEXT)
            if prev_clip:
                self.clip(*prev_clip)
            else:
                self.unclip()
        self.fields[key] = (inner_x - edit.view, size)
        self.hot(x, y, w, h, ("focus", key))

    @staticmethod
    def index_at(text, x, size):
        """Character index nearest to x pixels into `text`."""
        blf.size(FONT, size * ui_scale())
        best, best_d = 0, abs(x)
        for i in range(1, len(text) + 1):
            d = abs(blf.dimensions(FONT, text[:i])[0] - x)
            if d < best_d:
                best, best_d = i, d
        return best

    def readout(self, value, x, y, w, h, size=11):
        """A read-only, non-interactive text box."""
        self.rect(x, y, w, h, FIELD)
        self.frame(x, y, w, h, BORDER)
        self.text(value, x + 5 * self.s, y + (h - size * self.s) / 2 - 1, size, TEXT, max_w=w - 10 * self.s)

    # -- scrolling ------------------------------------------------------------
    SCROLLBAR = 12

    def scrollbar_width(self):
        return self.SCROLLBAR * self.s

    def scroll_region(self, key, x, y, w, h, content_h, wheel_step):
        """Declares a scrollable area; draws its scrollbar along the right
        edge (inside the rect) and returns the clamped scroll offset."""
        max_scroll = max(0.0, content_h - h)
        off = min(max(0.0, self.scroll.get(key, 0.0)), max_scroll)
        self.scroll[key] = off
        info = {"rect": (x, y, w, h), "max": max_scroll, "step": wheel_step}
        sbw = self.scrollbar_width()
        track = (x + w - sbw, y, sbw, h)
        self.rect(*track, (0.12, 0.12, 0.12, 1))
        if max_scroll > 0:
            thumb_h = max(24 * self.s, h * h / content_h)
            thumb_y = y + (h - thumb_h) * (off / max_scroll)
            info["track"] = track
            info["thumb"] = (thumb_y, thumb_h)
            active = self.active_bar == key
            hovered = self.hover == ("scrollbar", key)
            color = (0.62, 0.62, 0.62, 1) if active else (0.48, 0.48, 0.48, 1) if hovered else (0.36, 0.36, 0.36, 1)
            self.rect(track[0] + 2 * self.s, thumb_y + 1, sbw - 4 * self.s, thumb_h - 2, color)
            self.hot(*track, ("scrollbar", key))
        self.scrollers[key] = info
        return off

    def autoscroll_marker(self, x, y):
        """The anchor shown while middle-mouse autoscrolling."""
        s = self.s
        r = 9 * s
        self.rect(x - r, y - r, 2 * r, 2 * r, (0.1, 0.1, 0.1, 0.85))
        self.frame(x - r, y - r, 2 * r, 2 * r, TEXT)
        self.rect(x - 1, y - 1, 2, 2, TEXT)
        for dy in (-1, 1):
            for k in range(3):
                w = (5 - 2 * k) * s
                self.rect(x - w / 2, y + dy * (r - (2 + k) * s) - 0.5, w, 1, TEXT)


class TextureCache:
    """numpy RGBA (top row first) -> GPUTexture, created lazily while drawing
    (GPU resources need the draw context)."""

    def __init__(self, limit=600):
        self.limit = limit
        self.pixels = {}
        self.textures = {}
        self.order = []

    def put(self, key, rgba):
        self.pixels[key] = rgba
        self.textures.pop(key, None)

    def has(self, key):
        return key in self.pixels

    def get(self, key):
        tex = self.textures.get(key)
        if tex is None:
            rgba = self.pixels.get(key)
            if rgba is None:
                return None
            h, w = rgba.shape[:2]
            flat = np.ascontiguousarray(rgba[::-1], dtype=np.float32).reshape(-1)
            buf = gpu.types.Buffer("FLOAT", flat.size, flat)
            tex = gpu.types.GPUTexture((w, h), format="RGBA16F", data=buf)
            self.textures[key] = tex
            self.order.append(key)
            if len(self.order) > self.limit:
                old = self.order.pop(0)
                self.textures.pop(old, None)
        return tex

    def clear(self):
        self.pixels.clear()
        self.textures.clear()
        self.order.clear()


def checker(n=64, a=(0.5, 0.0, 0.5, 1.0), b=(0.0, 0.0, 0.0, 1.0), cell=8):
    cells = (np.add.outer(np.arange(n) // cell, np.arange(n) // cell) % 2).astype(bool)
    img = np.empty((n, n, 4), dtype=np.float32)
    img[:] = b
    img[cells] = a
    return img
