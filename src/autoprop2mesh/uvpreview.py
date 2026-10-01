"""Geometry Nodes modifier reproducing Prop2Mesh's box-projected UVs, so the
texture preview matches the game.

Port of getBoxDir/getBoxUV (cl_meshlab.lua). For each face, the dominant axis
of the face normal (in entity space) picks the projection; UVs are the
entity-local vertex position in Source units times 1/uvs (1/48 when uvs is 0):

    |n.x| largest:  u =  p.z * sign(n.x),  v = p.y
    |n.y| largest:  u =  p.x,              v = p.z * sign(n.y)
    otherwise:      u = -p.x * sign(n.z),  v = p.y

with sign(0) = +1. Source samples with v pointing down, so v' = 1 - v.

Model parts with uvs = 0 keep the model's own UVs instead (P2M only
box-projects model parts when the controller has a UV scale).

The result is stored as "p2m_uv" (read by the controller material) and also
under the mesh's active UV map name, which Solid > Texture view uses.
"""

import bpy

GROUP_NAME = "P2M UV Preview"
MODIFIER_NAME = "P2M UV Preview"
_VERSION = 2


def _sock(node, name, outputs=False, index=None):
    socks = node.outputs if outputs else node.inputs
    if index is not None:
        return socks[index]
    s = socks.get(name)
    if s is None:
        raise KeyError("%s has no socket %s" % (node.bl_idname, name))
    return s


def _build(ng):
    ng.nodes.clear()
    iface = ng.interface
    iface.clear()
    iface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    iface.new_socket("Entity", in_out="INPUT", socket_type="NodeSocketObject")
    s = iface.new_socket("Unit Scale", in_out="INPUT", socket_type="NodeSocketFloat")
    s.default_value = 1.0
    s = iface.new_socket("Texel Scale", in_out="INPUT", socket_type="NodeSocketFloat")
    s.default_value = 1.0 / 48.0
    s = iface.new_socket("UV Name", in_out="INPUT", socket_type="NodeSocketString")
    s.default_value = "UVMap"
    s = iface.new_socket("Box UV", in_out="INPUT", socket_type="NodeSocketBool")
    s.default_value = True
    iface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")

    N = ng.nodes
    L = ng.links.new
    x = [0]

    def node(idname, **attrs):
        n = N.new(idname)
        for k, v in attrs.items():
            setattr(n, k, v)
        n.location = (x[0], 0)
        x[0] += 40
        return n

    def math(op, a=None, b=None, c=None, av=None, bv=None, cv=None):
        n = node("ShaderNodeMath", operation=op)
        for i, (sock, val) in enumerate(((a, av), (b, bv), (c, cv))):
            if sock is not None:
                L(sock, n.inputs[i])
            elif val is not None:
                n.inputs[i].default_value = val
        return n.outputs[0]

    gin = node("NodeGroupInput")
    gout = node("NodeGroupOutput")

    info = node("GeometryNodeObjectInfo", transform_space="RELATIVE")
    L(gin.outputs["Entity"], _sock(info, "Object"))
    inv = node("FunctionNodeInvertMatrix")
    L(_sock(info, "Transform", outputs=True), inv.inputs[0])
    inv_m = inv.outputs[0]

    pos = node("GeometryNodeInputPosition")
    tp = node("FunctionNodeTransformPoint")
    L(pos.outputs[0], tp.inputs[0])
    L(inv_m, tp.inputs[1])
    scale = node("ShaderNodeVectorMath", operation="SCALE")
    L(tp.outputs[0], scale.inputs[0])
    L(gin.outputs["Unit Scale"], _sock(scale, "Scale"))
    sep_p = node("ShaderNodeSeparateXYZ")
    L(scale.outputs["Vector"], sep_p.inputs[0])
    px, py, pz = sep_p.outputs[0], sep_p.outputs[1], sep_p.outputs[2]

    normal = node("GeometryNodeInputNormal")
    on_face = node("GeometryNodeFieldOnDomain", domain="FACE", data_type="FLOAT_VECTOR")
    L(normal.outputs[0], on_face.inputs[0])
    td = node("FunctionNodeTransformDirection")
    L(on_face.outputs[0], td.inputs[0])
    L(inv_m, td.inputs[1])
    sep_n = node("ShaderNodeSeparateXYZ")
    L(td.outputs[0], sep_n.inputs[0])
    nx, ny, nz = sep_n.outputs[0], sep_n.outputs[1], sep_n.outputs[2]

    ax, ay, az = math("ABSOLUTE", nx), math("ABSOLUTE", ny), math("ABSOLUTE", nz)

    def sign(n):  # n < 0 ? -1 : 1
        return math("MULTIPLY_ADD", math("LESS_THAN", n, bv=0.0), bv=-2.0, cv=1.0)

    def greater(a, b):
        c = node("FunctionNodeCompare", data_type="FLOAT", operation="GREATER_THAN")
        L(a, c.inputs[0])
        L(b, c.inputs[1])
        return c.outputs[0]

    both = node("FunctionNodeBooleanMath", operation="AND")
    L(greater(ax, ay), both.inputs[0])
    L(greater(ax, az), both.inputs[1])
    cond_x = both.outputs[0]
    cond_y = greater(ay, az)

    def combine(u, v):
        c = node("ShaderNodeCombineXYZ")
        L(u, c.inputs[0])
        L(v, c.inputs[1])
        return c.outputs[0]

    uv_x = combine(math("MULTIPLY", pz, sign(nx)), py)
    uv_y = combine(px, math("MULTIPLY", pz, sign(ny)))
    uv_z = combine(math("MULTIPLY", px, math("MULTIPLY", sign(nz), bv=-1.0)), py)

    def switch(cond, a_true, b_false):
        sw = node("GeometryNodeSwitch", input_type="VECTOR")
        L(cond, _sock(sw, "Switch"))
        L(b_false, _sock(sw, "False"))
        L(a_true, _sock(sw, "True"))
        return sw.outputs[0]

    uv = switch(cond_x, uv_x, switch(cond_y, uv_y, uv_z))
    sep_uv = node("ShaderNodeSeparateXYZ")
    L(uv, sep_uv.inputs[0])
    k = gin.outputs["Texel Scale"]
    u = math("MULTIPLY", sep_uv.outputs[0], k)
    v = math("MULTIPLY_ADD", sep_uv.outputs[1], math("MULTIPLY", k, bv=-1.0), cv=1.0)
    box_uv = combine(u, v)
    own_uv = node("GeometryNodeInputNamedAttribute", data_type="FLOAT_VECTOR")
    L(gin.outputs["UV Name"], _sock(own_uv, "Name"))
    final = switch(gin.outputs["Box UV"], box_uv, _sock(own_uv, "Attribute", outputs=True))

    def store(geo, name_socket=None, name=None):
        st = node("GeometryNodeStoreNamedAttribute", data_type="FLOAT2", domain="CORNER")
        L(geo, _sock(st, "Geometry"))
        if name_socket is not None:
            L(name_socket, _sock(st, "Name"))
        else:
            _sock(st, "Name").default_value = name
        L(final, _sock(st, "Value"))
        return st.outputs[0]

    geo = store(gin.outputs["Geometry"], name="p2m_uv")
    geo = store(geo, name_socket=gin.outputs["UV Name"])
    L(geo, gout.inputs[0])
    ng["p2m_version"] = _VERSION


def node_group():
    ng = bpy.data.node_groups.get(GROUP_NAME)
    if ng is None or ng.get("p2m_version") != _VERSION:
        if ng is None:
            ng = bpy.data.node_groups.new(GROUP_NAME, "GeometryNodeTree")
        _build(ng)
    return ng


def _input_ids(ng):
    return {item.name: item.identifier for item in ng.interface.items_tree
            if getattr(item, "item_type", "SOCKET") == "SOCKET" and item.in_out == "INPUT"}


def _get_input(mod, ident):
    """Geometry Nodes modifier input value (Blender 4.2-4.x and 5.x APIs)."""
    props = getattr(mod, "properties", None)
    if props is not None and hasattr(props, "inputs"):
        sock = getattr(props.inputs, ident, None)
        return sock.value if sock is not None else None
    try:
        return mod[ident]
    except KeyError:
        return None


def _set_input(mod, ident, value):
    props = getattr(mod, "properties", None)
    if props is not None and hasattr(props, "inputs"):
        sock = getattr(props.inputs, ident, None)
        if sock is not None:
            sock.value = value
        return
    mod[ident] = value


def _active_uv_name(obj):
    data = obj.data
    layers = getattr(data, "uv_layers", None)
    if layers and layers.active is not None:
        return layers.active.name
    return "UVMap"


def ensure(obj, entity, unit_scale, uvs, box_uv=True):
    """Adds/updates the preview modifier. Returns False if unsupported."""
    mod = obj.modifiers.get(MODIFIER_NAME)
    if mod is None:
        try:
            mod = obj.modifiers.new(MODIFIER_NAME, "NODES")
        except Exception:
            return False
        if mod is None:
            return False
        mod.show_expanded = False
    ng = node_group()
    if mod.node_group is not ng:
        mod.node_group = ng
    ids = _input_ids(ng)
    texel = 1.0 / uvs if uvs else 1.0 / 48.0
    values = {"Entity": entity, "Unit Scale": float(unit_scale), "Texel Scale": texel,
              "UV Name": _active_uv_name(obj), "Box UV": bool(box_uv)}
    changed = False
    for name, value in values.items():
        ident = ids.get(name)
        if ident is None:
            continue
        current = _get_input(mod, ident)
        if isinstance(value, float) and isinstance(current, float):
            same = abs(current - value) < 1e-9
        else:
            same = current == value
        if not same:
            _set_input(mod, ident, value)
            changed = True
    if changed:
        obj.update_tag()
    # Keep it last so it sees the final geometry.
    if hasattr(mod, "use_pin_to_last"):
        if not mod.use_pin_to_last:
            mod.use_pin_to_last = True
    elif obj.modifiers[-1] != mod:
        try:
            obj.modifiers.move(list(obj.modifiers).index(mod), len(obj.modifiers) - 1)
        except Exception:
            pass
    return True


def remove(obj):
    mod = obj.modifiers.get(MODIFIER_NAME)
    if mod is not None:
        obj.modifiers.remove(mod)
