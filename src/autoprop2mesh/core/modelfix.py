"""Prop2Mesh's model fixer (lua/prop2mesh/cl_modelfixer.lua).

In game P2M turns some models 90 degrees about their own up axis when it
builds them, either the whole model or only some of its submeshes. The model
and folder lists are read from the installed Prop2Mesh. Its special cases are
Lua functions there, so they're copied here.
"""

import numpy as np

LUA_PATH = "lua/prop2mesh/cl_modelfixer.lua"

IDENTITY = np.eye(3)
RZ90 = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
RX90 = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])

ALL = "all"            # every submesh turned (P2M's generic fix)
FORWARD = "forward"    # every submesh turned, then 90 degrees about forward

# rule = ALL, FORWARD, or the 1-based submeshes that get turned
_SPECIAL_MODELS = {}
for _size in (15, 20, 25, 30):
    _SPECIAL_MODELS["models/sprops/trans/miscwheels/thin_moto%d.mdl" % _size] = (1,)
    _SPECIAL_MODELS["models/sprops/trans/miscwheels/thick_moto%d.mdl" % _size] = (1,)
    _SPECIAL_MODELS["models/sprops/trans/miscwheels/tank%d.mdl" % _size] = (1, 2)
for _model in ("models/props_mining/diesel_generator_crank.mdl",
               "models/props/de_nuke/hr_nuke/nuke_vent_bombsite/nuke_vent_bombsite_breakable_a.mdl",
               "models/props/de_nuke/hr_nuke/nuke_vent_bombsite/nuke_vent_bombsite_breakable_b.mdl",
               "models/props/de_nuke/hr_nuke/nuke_vent_bombsite/nuke_vent_bombsite_breakable_c.mdl"):
    _SPECIAL_MODELS[_model] = FORWARD
_SPECIAL_FOLDERS = {
    "models/sprops/trans/wheel_b/": (1,),
    "models/sprops/trans/wheel_d/": (1, 2),
}


def _lua_list(text, name):
    """Lines of a `name = [[...]]` long string."""
    start = text.find(name)
    if start < 0:
        return set()
    start = text.find("[[", start)
    end = text.find("]]", start)
    if start < 0 or end < 0:
        return set()
    return {line.strip().lower() for line in text[start + 2:end].splitlines() if line.strip()}


class Fixer:
    def __init__(self, lua_text=""):
        self.models = _lua_list(lua_text, "genericmodellist =")
        self.folders = _lua_list(lua_text, "genericfolderlist =")

    def rule(self, model):
        """None when P2M builds the model as is. Same lookup order as prop2mesh.getModelFix."""
        model = model.replace("\\", "/").lower()
        if model in _SPECIAL_MODELS:
            return _SPECIAL_MODELS[model]
        if model in self.models:
            return ALL
        folder = model.rsplit("/", 1)[0] + "/"
        if folder in _SPECIAL_FOLDERS:
            return _SPECIAL_FOLDERS[folder]
        if folder in self.folders:
            return ALL
        return None


def submesh_transform(rule, submesh, scale, holo=False):
    """(F, s, C) for one 1-based submesh. P2M scales the raw vertices by s,
    clips them with C @ normal, then turns them by F in the part's frame."""
    scale = np.asarray(scale, dtype=np.float64)
    sx, sy, sz = scale
    swapped = np.array((sy, sx, sz) if holo else (sx, sz, sy))
    if rule is None:
        return IDENTITY, scale, IDENTITY
    if rule == FORWARD:
        # P2M uses the plain scale and clip normal here
        return RZ90 @ RX90, scale, IDENTITY
    if rule == ALL or submesh in rule:
        return RZ90, swapped, RZ90.T
    return IDENTITY, scale, IDENTITY
