"""Shared modal machinery for the Hammer-style browsers: draws a full overlay
over the 3D viewport, routes mouse/keyboard/wheel input to the browser, and
runs background work (thumbnail rendering) on a timer so results stream in."""

import time

import bpy

from . import canvas

TICK = 1.0 / 30.0
WORK_BUDGET = 0.03  # seconds of background work per timer tick


class BrowserModal:
    """Mixin for a modal operator. Subclasses implement:
        setup(context) -> bool        draw_ui(cv, w, h)
        on_action(action, double)     on_text(text)  (filter typing)
        work() -> bool (did something)  accept(context) -> bool
    and may override scroll(rect_key, delta)."""

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
        if not self.setup(context):
            return {"CANCELLED"}
        self._saved = None  # viewport UI is hidden on the first timer tick
        self.hotspots = []
        self.hover = None
        self.mouse = (0, 0)
        self.focus = None
        self._last_click = (0.0, None)
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
        cv = canvas.Canvas(region.height, inset)
        cv.rect(0, -inset, region.width, region.height, canvas.BG)
        try:
            self.draw_ui(cv, region.width, region.height - inset)
        except Exception as e:  # never leave the user stuck in a broken overlay
            cv.text("Browser error: %s" % e, 20, 20, 12, (1, 0.4, 0.4, 1))
            print("AutoProp2Mesh browser draw error:", repr(e))
        cv.unclip()
        self.hotspots = cv.hotspots

    # -- events ------------------------------------------------------------
    def modal(self, context, event):
        if self.area is None or not self.area.regions:
            return self._finish(context, cancelled=True)
        self.area.tag_redraw()
        if event.type == "TIMER":
            if self._saved is None:
                self._hide_viewport_ui()
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
            self.hover = canvas.Canvas.hit(self.hotspots, mx, my)
            return {"RUNNING_MODAL"}
        if event.type in {"WHEELUPMOUSE", "WHEELDOWNMOUSE"}:
            self.scroll_at(mx, my, -1 if event.type == "WHEELUPMOUSE" else 1)
            return {"RUNNING_MODAL"}
        if event.type in {"TRACKPADPAN"}:
            self.scroll_at(mx, my, (event.mouse_prev_y - event.mouse_y) / 40.0)
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
            if action == ("ok",):
                return self._finish(context, cancelled=not self.accept(context))
            if action == ("cancel",):
                return self._finish(context, cancelled=True)
            if action[0] == "focus":
                self.focus = action[1]
            else:
                self.focus = None
                if self.on_action(action, double) == "ACCEPT":
                    return self._finish(context, cancelled=not self.accept(context))
            return {"RUNNING_MODAL"}
        if event.value != "PRESS":
            return {"RUNNING_MODAL"}
        if event.type in {"ESC", "RIGHTMOUSE"}:
            if self.focus and event.type == "ESC":
                self.focus = None
                return {"RUNNING_MODAL"}
            return self._finish(context, cancelled=True)
        if event.type in {"RET", "NUMPAD_ENTER"}:
            return self._finish(context, cancelled=not self.accept(context))
        # Typing goes to the filter (like Hammer), even without clicking it.
        text = self.filter_text()
        if event.type == "BACK_SPACE":
            self.focus = "filter"
            self.on_text("" if event.ctrl else text[:-1])
        elif event.unicode and event.unicode.isprintable() and not event.ctrl and not event.alt:
            self.focus = "filter"
            self.on_text(text + event.unicode)
        elif event.type in {"PAGE_UP", "PAGE_DOWN"}:
            self.scroll_at(mx, my, -8 if event.type == "PAGE_UP" else 8)
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

    def scroll_at(self, mx, my, delta):
        pass

    def filter_text(self):
        return ""

    def on_text(self, text):
        pass
