"""Shared modal machinery for the Hammer-style browsers: draws a full overlay
over the 3D viewport, routes mouse/keyboard/wheel input to the browser, and
runs background work (thumbnail rendering) on a timer so results stream in.

Scrolling (wheel, draggable scrollbars, middle-mouse autoscroll) and text
editing (cursor, selection, clipboard) are handled here for every browser."""

import time

import bpy

from . import canvas

TICK = 1.0 / 60.0
WORK_BUDGET = 0.02  # seconds of background work per timer tick

# Middle-mouse autoscroll: pixels per second for each pixel of distance from
# the anchor beyond the dead zone, plus a quadratic term so it speeds up.
AUTOSCROLL_DEADZONE = 6
AUTOSCROLL_LINEAR = 4.0
AUTOSCROLL_QUADRATIC = 0.08


def _is_word(ch):
    return ch.isalnum() or ch == "_"


class TextEdit:
    """A single-line text field's state: text, cursor and selection."""

    def __init__(self, text="", readonly=False):
        self.text = text
        self.cursor = len(text)
        self.anchor = None      # selection start (None = no selection)
        self.view = 0.0         # horizontal scroll, maintained while drawing
        self.readonly = readonly

    def selection(self):
        if self.anchor is None or self.anchor == self.cursor:
            return None
        return (min(self.anchor, self.cursor), max(self.anchor, self.cursor))

    def set_text(self, text):
        self.text = text
        self.cursor = len(text)
        self.anchor = None

    def select_all(self):
        self.anchor = 0
        self.cursor = len(self.text)

    def selected_text(self):
        sel = self.selection()
        return self.text[sel[0]:sel[1]] if sel else ""

    def _delete_selection(self):
        sel = self.selection()
        if not sel:
            return False
        self.text = self.text[:sel[0]] + self.text[sel[1]:]
        self.cursor = sel[0]
        self.anchor = None
        return True

    def insert(self, s):
        if self.readonly:
            return
        s = "".join(ch for ch in s if ch.isprintable())
        self._delete_selection()
        self.text = self.text[:self.cursor] + s + self.text[self.cursor:]
        self.cursor += len(s)
        self.anchor = None

    def _word_left(self, i):
        while i > 0 and not _is_word(self.text[i - 1]):
            i -= 1
        while i > 0 and _is_word(self.text[i - 1]):
            i -= 1
        return i

    def _word_right(self, i):
        n = len(self.text)
        while i < n and not _is_word(self.text[i]):
            i += 1
        while i < n and _is_word(self.text[i]):
            i += 1
        return i

    def backspace(self, word=False):
        if self.readonly or self._delete_selection():
            return
        start = self._word_left(self.cursor) if word else max(0, self.cursor - 1)
        self.text = self.text[:start] + self.text[self.cursor:]
        self.cursor = start

    def delete(self, word=False):
        if self.readonly or self._delete_selection():
            return
        end = self._word_right(self.cursor) if word else min(len(self.text), self.cursor + 1)
        self.text = self.text[:self.cursor] + self.text[end:]

    def move(self, to, extend):
        if extend:
            if self.anchor is None:
                self.anchor = self.cursor
        else:
            sel = self.selection()
            self.anchor = None
            if sel and to in ("left", "right"):
                # Collapsing a selection moves to its edge, like most editors.
                self.cursor = sel[0] if to == "left" else sel[1]
                return
        if to == "left":
            self.cursor = max(0, self.cursor - 1)
        elif to == "right":
            self.cursor = min(len(self.text), self.cursor + 1)
        elif to == "word_left":
            self.cursor = self._word_left(self.cursor)
        elif to == "word_right":
            self.cursor = self._word_right(self.cursor)
        elif to == "home":
            self.cursor = 0
        elif to == "end":
            self.cursor = len(self.text)


class BrowserModal:
    """Mixin for a modal operator. Subclasses implement:
        setup(context) -> bool          draw_ui(cv, w, h)
        on_action(action, double)       accept(context) -> bool
        work() -> bool (did something)  filter_text() / on_text(text)
    Scroll regions are declared while drawing with cv.scroll_region(); their
    offsets live in self.scroll[key]."""

    def invoke(self, context, event):
        area = context.area if context.area and context.area.type == "VIEW_3D" else None
        if area is None:
            for a in context.window.screen.areas:
                if a.type == "VIEW_3D":
                    area = a
                    break
        if area is None:
            self.report({"ERROR"}, "Needs a 3D viewport")
            return {"CANCELLED"}
        self.area = area
        self.region = next(r for r in area.regions if r.type == "WINDOW")
        self.scroll = {}
        self.scrollers = {}
        self.fields = {}
        self.edits = {}
        if not self.setup(context):
            return {"CANCELLED"}
        self.edits.setdefault("filter", TextEdit(self.filter_text()))
        self._saved = None  # viewport UI is hidden on the first timer tick
        self.hotspots = []
        self.hover = None
        self.mouse = (0, 0)
        self.focus = None
        self._last_click = (0.0, None)
        self._drag_bar = None       # (key, grab offset within the thumb)
        self._drag_text = None      # field key while drag-selecting
        self._auto = None           # middle-mouse autoscroll state
        self._last_tick = time.perf_counter()
        self._draw_handle = bpy.types.SpaceView3D.draw_handler_add(self._draw, (), "WINDOW", "POST_PIXEL")
        self._timer = context.window_manager.event_timer_add(TICK, window=context.window)
        context.window_manager.modal_handler_add(self)
        area.tag_redraw()
        return {"RUNNING_MODAL"}

    def _hide_viewport_ui(self):
        """Hide the viewport's own UI so nothing draws over the browser.
        (Changing region visibility from invoke() does not reliably relayout
        the area, so this runs from the first timer tick.)"""
        space = self.area.spaces.active
        self._saved = {}
        # The tool settings strip cannot be hidden while a modal operator
        # runs, so it is left visible and the browser starts below it.
        for attr in ("show_region_ui", "show_region_toolbar", "show_region_header", "show_gizmo"):
            if hasattr(space, attr):
                self._saved[attr] = getattr(space, attr)
                setattr(space, attr, False)
        self.area.tag_redraw()

    def top_inset(self):
        """Height of header regions drawn over the top of the window region."""
        win = self.region
        top = win.y + win.height
        lowest = top
        for r in self.area.regions:
            if r.type in {"HEADER", "TOOL_HEADER"} and r.height > 1 and win.y <= r.y < top:
                lowest = min(lowest, r.y)
        return top - lowest

    # -- drawing -----------------------------------------------------------
    def _draw(self):
        if bpy.context.area != self.area:
            return
        region = self.region
        inset = self.top_inset()
        cv = canvas.Canvas(region.height, inset, scroll=self.scroll)
        cv.active_bar = self._drag_bar[0] if self._drag_bar else None
        cv.hover = self.hover
        cv.rect(0, -inset, region.width, region.height, canvas.BG)
        try:
            self.draw_ui(cv, region.width, region.height - inset)
        except Exception as e:  # never leave the user stuck in a broken overlay
            cv.unclip()
            cv.text("Browser error: %s" % e, 20, 20, 12, (1, 0.4, 0.4, 1))
            print("AutoProp2Mesh browser draw error:", repr(e))
        cv.unclip()
        if self._auto is not None:
            cv.autoscroll_marker(self._auto["x"], self._auto["y"])
        self.hotspots = cv.hotspots
        self.scrollers = cv.scrollers
        self.fields = cv.fields

    # -- scrolling ---------------------------------------------------------
    def _scroller_at(self, mx, my):
        for key, info in self.scrollers.items():
            x, y, w, h = info["rect"]
            if x <= mx < x + w and y <= my < y + h:
                return key
        return None

    def _scroll_by(self, key, pixels):
        info = self.scrollers.get(key)
        if info is None:
            return
        self.scroll[key] = min(max(0.0, self.scroll.get(key, 0.0) + pixels), info["max"])

    def _drag_scrollbar(self, my):
        key, grab = self._drag_bar
        info = self.scrollers.get(key)
        if not info or info["max"] <= 0:
            return
        _tx, ty, _tw, th = info["track"]
        thumb_h = info["thumb"][1]
        span = max(1.0, th - thumb_h)
        frac = min(1.0, max(0.0, (my - grab - ty) / span))
        self.scroll[key] = frac * info["max"]

    def _autoscroll_tick(self, dt):
        a = self._auto
        d = self.mouse[1] - a["y"]
        dead = AUTOSCROLL_DEADZONE * canvas.ui_scale()
        if abs(d) <= dead:
            return
        dist = abs(d) - dead
        speed = AUTOSCROLL_LINEAR * dist + AUTOSCROLL_QUADRATIC * dist * dist
        self._scroll_by(a["key"], speed * dt * (1 if d > 0 else -1))

    # -- text ----------------------------------------------------------------
    def _text_changed(self, key, before):
        edit = self.edits[key]
        if key == "filter" and edit.text != before:
            self.on_text(edit.text)

    def _field_index(self, key, mx):
        geo = self.fields.get(key)
        edit = self.edits.get(key)
        if geo is None or edit is None:
            return 0
        origin, size = geo
        return canvas.Canvas.index_at(edit.text, mx - origin, size)

    def _handle_text_key(self, context, event, key):
        """Returns True when the key was used by the text field."""
        edit = self.edits[key]
        before = edit.text
        t = event.type
        ctrl = event.ctrl or event.oskey
        if ctrl and t == "A":
            edit.select_all()
        elif ctrl and t == "C":
            if edit.selected_text():
                context.window_manager.clipboard = edit.selected_text()
        elif ctrl and t == "X":
            if edit.selected_text() and not edit.readonly:
                context.window_manager.clipboard = edit.selected_text()
                edit.backspace()
        elif ctrl and t == "V":
            edit.insert(context.window_manager.clipboard.replace("\r", " ").replace("\n", " "))
        elif t == "BACK_SPACE":
            edit.backspace(word=ctrl)
        elif t == "DEL":
            edit.delete(word=ctrl)
        elif t == "LEFT_ARROW":
            edit.move("word_left" if ctrl else "left", event.shift)
        elif t == "RIGHT_ARROW":
            edit.move("word_right" if ctrl else "right", event.shift)
        elif t == "HOME":
            edit.move("home", event.shift)
        elif t == "END":
            edit.move("end", event.shift)
        elif event.unicode and event.unicode.isprintable() and not ctrl and not event.alt:
            edit.insert(event.unicode)
        else:
            return False
        self._text_changed(key, before)
        return True

    # -- events ------------------------------------------------------------
    def modal(self, context, event):
        if self.area is None or not self.area.regions:
            return self._finish(context, cancelled=True)
        self.area.tag_redraw()
        if event.type == "TIMER":
            now = time.perf_counter()
            dt, self._last_tick = now - self._last_tick, now
            if self._saved is None:
                self._hide_viewport_ui()
            if self._auto is not None:
                self._autoscroll_tick(min(dt, 0.1))
            deadline = time.perf_counter() + WORK_BUDGET
            while time.perf_counter() < deadline:
                if not self.work():
                    break
            return {"RUNNING_MODAL"}

        region = self.region
        mx = event.mouse_x - region.x
        my = region.height - (event.mouse_y - region.y) - self.top_inset()
        self.mouse = (mx, my)

        if event.type == "MOUSEMOVE":
            if self._drag_bar is not None:
                self._drag_scrollbar(my)
            elif self._drag_text is not None:
                edit = self.edits[self._drag_text]
                if edit.anchor is None:
                    edit.anchor = edit.cursor
                edit.cursor = self._field_index(self._drag_text, mx)
            self.hover = canvas.Canvas.hit(self.hotspots, mx, my)
            return {"RUNNING_MODAL"}

        # Middle mouse: hold and move away from the press point to scroll.
        # A quick click without moving keeps autoscroll on until the next click.
        if event.type == "MIDDLEMOUSE":
            if event.value == "PRESS":
                if self._auto is not None and self._auto["sticky"]:
                    self._auto = None
                    return {"RUNNING_MODAL"}
                key = self._scroller_at(mx, my)
                if key is not None:
                    self._auto = {"key": key, "x": mx, "y": my, "t": time.perf_counter(), "sticky": False}
            elif event.value == "RELEASE" and self._auto is not None and not self._auto["sticky"]:
                quick = time.perf_counter() - self._auto["t"] < 0.3
                still = abs(my - self._auto["y"]) <= AUTOSCROLL_DEADZONE * canvas.ui_scale()
                if quick and still:
                    self._auto["sticky"] = True
                else:
                    self._auto = None
            return {"RUNNING_MODAL"}
        if self._auto is not None and self._auto["sticky"] and event.value == "PRESS" and \
                event.type in {"LEFTMOUSE", "RIGHTMOUSE", "ESC"}:
            self._auto = None
            return {"RUNNING_MODAL"}

        if event.type in {"WHEELUPMOUSE", "WHEELDOWNMOUSE"}:
            key = self._scroller_at(mx, my)
            if key is not None:
                step = self.scrollers[key]["step"]
                self._scroll_by(key, -step if event.type == "WHEELUPMOUSE" else step)
            return {"RUNNING_MODAL"}
        if event.type == "TRACKPADPAN":
            key = self._scroller_at(mx, my)
            if key is not None:
                self._scroll_by(key, event.mouse_prev_y - event.mouse_y)
            return {"RUNNING_MODAL"}

        if event.type == "LEFTMOUSE" and event.value == "RELEASE":
            self._drag_bar = None
            self._drag_text = None
            return {"RUNNING_MODAL"}

        if event.type == "LEFTMOUSE" and event.value in {"PRESS", "DOUBLE_CLICK"}:
            action = canvas.Canvas.hit(self.hotspots, mx, my)
            now = time.perf_counter()
            double = event.value == "DOUBLE_CLICK" or (
                action is not None and self._last_click[1] == action and now - self._last_click[0] < 0.4)
            self._last_click = (now, action)
            if action is None:
                self.focus = None
                return {"RUNNING_MODAL"}
            if action[0] == "scrollbar":
                key = action[1]
                info = self.scrollers.get(key)
                if info and info["max"] > 0:
                    thumb_y, thumb_h = info["thumb"]
                    if thumb_y <= my < thumb_y + thumb_h:
                        grab = my - thumb_y
                    else:
                        grab = thumb_h / 2  # clicking the track jumps there
                    self._drag_bar = (key, grab)
                    self._drag_scrollbar(my)
                return {"RUNNING_MODAL"}
            if action == ("ok",):
                return self._finish(context, cancelled=not self.accept(context))
            if action == ("cancel",):
                return self._finish(context, cancelled=True)
            if action[0] == "focus":
                key = action[1]
                self.focus = key
                edit = self.edits[key]
                if double:
                    edit.select_all()
                else:
                    edit.anchor = None
                    edit.cursor = self._field_index(key, mx)
                    self._drag_text = key
                return {"RUNNING_MODAL"}
            self.focus = None
            if self.on_action(action, double) == "ACCEPT":
                return self._finish(context, cancelled=not self.accept(context))
            return {"RUNNING_MODAL"}

        if event.value != "PRESS":
            return {"RUNNING_MODAL"}
        if event.type == "ESC" or event.type == "RIGHTMOUSE":
            if self.focus and event.type == "ESC":
                self.focus = None
                return {"RUNNING_MODAL"}
            return self._finish(context, cancelled=True)
        if event.type in {"RET", "NUMPAD_ENTER"}:
            return self._finish(context, cancelled=not self.accept(context))

        if self.focus in self.edits:
            if self._handle_text_key(context, event, self.focus):
                return {"RUNNING_MODAL"}
        else:
            # Typing (or pasting) goes to the filter, like Hammer, even
            # without clicking it first.
            ctrl = event.ctrl or event.oskey
            typing = event.unicode and event.unicode.isprintable() and not ctrl and not event.alt
            if typing or event.type == "BACK_SPACE" or (ctrl and event.type in {"V", "A"}):
                self.focus = "filter"
                self.edits["filter"].move("end", False)
                self._handle_text_key(context, event, "filter")
                return {"RUNNING_MODAL"}
        if event.type in {"PAGE_UP", "PAGE_DOWN"}:
            key = self._scroller_at(mx, my)
            if key is not None:
                info = self.scrollers[key]
                self._scroll_by(key, (-1 if event.type == "PAGE_UP" else 1) * info["rect"][3] * 0.9)
        return {"RUNNING_MODAL"}

    def _finish(self, context, cancelled):
        if getattr(self, "_draw_handle", None) is not None:
            bpy.types.SpaceView3D.draw_handler_remove(self._draw_handle, "WINDOW")
            self._draw_handle = None
        if getattr(self, "_timer", None) is not None:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None
        try:
            space = self.area.spaces.active
            for attr, value in (self._saved or {}).items():
                setattr(space, attr, value)
            self.area.tag_redraw()
        except (ReferenceError, AttributeError):
            pass
        self.cleanup()
        return {"CANCELLED"} if cancelled else {"FINISHED"}

    # -- defaults ------------------------------------------------------------
    def cleanup(self):
        pass

    def work(self):
        return False

    def filter_text(self):
        return ""

    def on_text(self, text):
        pass
