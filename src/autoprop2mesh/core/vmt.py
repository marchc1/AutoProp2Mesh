"""Valve Material (VMT) parsing: just what the preview needs."""

from . import keyvalues


class Material:
    def __init__(self, path):
        self.path = path
        self.shader = ""
        self.params = {}

    @property
    def base_texture(self):
        tex = self.params.get("$basetexture")
        return texture_path(tex) if tex else None

    @property
    def translucent(self):
        return _truthy(self.params.get("$translucent")) or _truthy(self.params.get("$alphatest"))

    @property
    def alpha_test(self):
        return _truthy(self.params.get("$alphatest"))


def _truthy(v):
    if v is None:
        return False
    try:
        return float(v) != 0
    except ValueError:
        return False


def texture_path(name):
    name = name.replace("\\", "/").strip().strip("/").lower()
    if name.startswith("materials/"):
        name = name[len("materials/"):]
    if name.endswith(".vtf"):
        name = name[:-4]
    return "materials/%s.vtf" % name


def material_path(name):
    """Normalizes "hunter/myplastic" style names to a VMT file path."""
    name = name.replace("\\", "/").strip().strip("/").lower()
    if name.startswith("materials/"):
        name = name[len("materials/"):]
    if name.endswith(".vmt"):
        name = name[:-4]
    return "materials/%s.vmt" % name


def load(read, path, _depth=0):
    """`read(path) -> bytes|None`. Returns Material or None."""
    data = read(path)
    if data is None:
        return None
    root = keyvalues.parse(data.decode("utf-8", "replace"))
    if not root.items:
        return None
    shader, body = root.items[0]
    mat = Material(path)
    mat.shader = shader.lower()
    if not isinstance(body, keyvalues.KVNode):
        return mat

    if mat.shader == "patch" and _depth < 8:
        include = body.get("include")
        base = load(read, material_path(include), _depth + 1) if include else None
        if base:
            mat.shader = base.shader
            mat.params = dict(base.params)
        for section in ("insert", "replace"):
            node = body.get_node(section)
            if node:
                mat.params.update(node.to_dict())
        return mat

    mat.params = body.to_dict()
    return mat
