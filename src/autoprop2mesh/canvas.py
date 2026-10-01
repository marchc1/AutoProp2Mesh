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
    def __init__(self, region_height, top_offset=0):
        self.h = region_height - top_offset  # drawing space starts below the offset
        self.s = ui_scale()
        self.hotspots = []
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
        gpu.state.scissor_test_set(True)
        gpu.state.scissor_set(int(x), int(self._y(y, h)), max(0, int(w)), max(0, int(h)))

    def unclip(self):
        gpu.state.scissor_test_set(False)

    # -- interaction --------------------------------------------------------
    def hot(self, x, y, w, h, action):
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

    def field(self, value, x, y, w, h, action, focused=False, placeholder=""):
        self.rect(x, y, w, h, FIELD)
        self.frame(x, y, w, h, (0.45, 0.55, 0.9, 1) if focused else BORDER)
        shown = value if value or focused else placeholder
        tw = self.text(shown, x + 5 * self.s, y + (h - 11 * self.s) / 2 - 1, 11,
                       TEXT if value else DIM, max_w=w - 10 * self.s)
        if focused:
            self.rect(x + 6 * self.s + tw, y + 3 * self.s, 1, h - 6 * self.s, TEXT)
        self.hot(x, y, w, h, action)


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
