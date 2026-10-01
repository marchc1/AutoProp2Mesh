"""Source studio model loading (MDL + VVD + VTX), LOD 0, skin 0, any body
group combination.

Positions are returned in model space (Source units) and triangles are
returned counter-clockwise, ready for Blender.
"""

import struct

import numpy as np

MDL_ID = b"IDST"
VVD_ID = b"IDSV"
VVD_VERTEX_SIZE = 48
VTX_SUFFIXES = (".dx90.vtx", ".dx80.vtx", ".sw.vtx", ".vtx")


class ModelError(Exception):
    pass


class ModelData:
    def __init__(self):
        self.positions = None   # (N, 3) float32
        self.normals = None     # (N, 3) float32
        self.uvs = None         # (N, 2) float32, Blender convention (v up)
        self.triangles = None   # (M, 3) int32, CCW
        self.tri_material = None  # (M,) int32 index into `materials`
        self.materials = []     # list of candidate lists of VMT paths
        self.name = ""
        # [(bodypart name, base, [submodel names])] - for body group UIs
        self.bodyparts = []
        self.bodygroups = []    # chosen submodel per body part
        self.bodygroup_mask = 0  # sum(choice * base), as util.GetModelMeshes takes


def bodygroup_mask(bodyparts, choices):
    mask = 0
    for i, (_name, base, models) in enumerate(bodyparts):
        c = choices[i] if i < len(choices) else 0
        if 0 <= c < len(models):
            mask += c * base
    return mask


def _cstr(data, offset):
    end = data.find(b"\0", offset)
    if end < 0:
        end = len(data)
    return data[offset:end].decode("utf-8", "replace")


def _parse_mdl(mdl):
    if mdl[:4] != MDL_ID:
        raise ModelError("not an MDL file")
    version = struct.unpack_from("<i", mdl, 4)[0]
    if not 44 <= version <= 49:
        raise ModelError("unsupported MDL version %d" % version)

    (num_textures, texture_index, num_cdtextures, cdtexture_index,
     num_skinref, num_skinfamilies, skin_index,
     num_bodyparts, bodypart_index) = struct.unpack_from("<9i", mdl, 204)

    textures = []
    for i in range(num_textures):
        base = texture_index + i * 64
        name_off = struct.unpack_from("<i", mdl, base)[0]
        textures.append(_cstr(mdl, base + name_off))

    cddirs = []
    for i in range(num_cdtextures):
        off = struct.unpack_from("<i", mdl, cdtexture_index + i * 4)[0]
        cddirs.append(_cstr(mdl, off))

    skin0 = list(struct.unpack_from("<%dh" % num_skinref, mdl, skin_index)) if num_skinref else []

    bodyparts = []
    for b in range(num_bodyparts):
        bbase = bodypart_index + b * 16
        name_off, num_models, base, model_index = struct.unpack_from("<4i", mdl, bbase)
        models = []
        for m in range(num_models):
            mbase = bbase + model_index + m * 148
            num_meshes, mesh_index, num_vertices, vertex_index = struct.unpack_from("<4i", mdl, mbase + 72)
            meshes = []
            for k in range(num_meshes):
                kbase = mbase + mesh_index + k * 116
                material, _mi, mesh_numverts, mesh_vertoff = struct.unpack_from("<4i", mdl, kbase)
                meshes.append((material, mesh_numverts, mesh_vertoff))
            models.append({"name": _cstr(mdl, mbase)[:64], "vstart": vertex_index // VVD_VERTEX_SIZE,
                           "meshes": meshes})
        bodyparts.append({"name": _cstr(mdl, bbase + name_off), "base": max(base, 1), "models": models})

    return {
        "version": version,
        "name": _cstr(mdl, 12),
        "checksum": struct.unpack_from("<i", mdl, 8)[0],
        "textures": textures,
        "cddirs": cddirs,
        "skin0": skin0,
        "bodyparts": bodyparts,
    }


def _parse_vvd(vvd):
    if vvd[:4] != VVD_ID:
        raise ModelError("not a VVD file")
    (_ver, checksum, _numlods) = struct.unpack_from("<3i", vvd, 4)
    lod_verts = struct.unpack_from("<8i", vvd, 16)
    num_fixups, fixup_start, vertex_start, _tangent_start = struct.unpack_from("<4i", vvd, 48)

    count = lod_verts[0]
    raw = np.frombuffer(vvd, dtype=np.uint8, count=count * VVD_VERTEX_SIZE, offset=vertex_start)
    raw = raw.reshape(count, VVD_VERTEX_SIZE)

    if num_fixups:
        # Rebuild the LOD 0 vertex order from the fixup table.
        chunks = []
        for i in range(num_fixups):
            lod, src, n = struct.unpack_from("<3i", vvd, fixup_start + i * 12)
            if lod >= 0:
                chunks.append(raw[src:src + n])
        raw = np.concatenate(chunks) if chunks else raw

    raw = np.ascontiguousarray(raw)
    floats = raw[:, 16:48].copy().view(np.float32)  # pos(3) normal(3) uv(2)
    return checksum, floats[:, 0:3], floats[:, 3:6], floats[:, 6:8]


def _parse_vtx(vtx, mdl_info, choices, strip_group_size, strip_size):
    """Returns list of (material, triangles as global vertex ids) for the
    chosen submodel of each body part."""
    (_ver, _cache, _mbs, _mbt, _mbv, _checksum, _numlods, _matrepl,
     num_bodyparts, bodypart_offset) = struct.unpack_from("<iiHHiiiiii", vtx, 0)
    unpack = struct.unpack_from
    results = []
    mdl_bodyparts = mdl_info["bodyparts"]
    for b in range(min(num_bodyparts, len(mdl_bodyparts))):
        bbase = bodypart_offset + b * 8
        num_models, model_offset = unpack("<ii", vtx, bbase)
        mdl_models = mdl_bodyparts[b]["models"]
        choice = choices[b] if b < len(choices) else 0
        if not (0 <= choice < min(num_models, len(mdl_models))):
            choice = 0
        if num_models == 0 or not mdl_models:
            continue
        mbase = bbase + model_offset + choice * 8
        num_lods, lod_offset = unpack("<ii", vtx, mbase)
        if num_lods == 0:
            continue
        lbase = mbase + lod_offset  # LOD 0
        num_meshes, mesh_offset, _switch = unpack("<iif", vtx, lbase)
        model_vstart = mdl_models[choice]["vstart"]
        mdl_meshes = mdl_models[choice]["meshes"]
        for k in range(min(num_meshes, len(mdl_meshes))):
            kbase = lbase + mesh_offset + k * 9
            num_groups, group_offset = unpack("<ii", vtx, kbase)
            material, mesh_nverts, mesh_vertoff = mdl_meshes[k]
            tris = []
            for g in range(num_groups):
                gbase = kbase + group_offset + g * strip_group_size
                nverts, vert_off, nidx, idx_off, nstrips, strip_off = unpack("<6i", vtx, gbase)
                if nverts <= 0 or nidx <= 0:
                    continue
                vrec = np.frombuffer(vtx, dtype=np.uint8, count=nverts * 9, offset=gbase + vert_off).reshape(nverts, 9)
                orig = vrec[:, 4].astype(np.int64) | (vrec[:, 5].astype(np.int64) << 8)
                if orig.max() >= mesh_nverts:
                    raise ModelError("vtx vertex out of range")
                indices = np.frombuffer(vtx, dtype=np.uint16, count=nidx, offset=gbase + idx_off)
                if indices.max() >= nverts:
                    raise ModelError("vtx index out of range")
                for s in range(nstrips):
                    sbase = gbase + strip_off + s * strip_size
                    s_nidx, s_idx_off = unpack("<ii", vtx, sbase)
                    flags = vtx[sbase + 18]
                    sidx = indices[s_idx_off:s_idx_off + s_nidx]
                    if flags & 0x02:  # triangle strip
                        t = []
                        for i in range(len(sidx) - 2):
                            a, b2, c = sidx[i], sidx[i + 1], sidx[i + 2]
                            if a == b2 or b2 == c or a == c:
                                continue
                            t.append((a, b2, c) if i % 2 == 0 else (b2, a, c))
                        sidx = np.array(t, dtype=np.int64).reshape(-1)
                    sidx = sidx[:len(sidx) - len(sidx) % 3]
                    tris.append(orig[sidx.astype(np.int64)].reshape(-1, 3))
            if tris:
                t = np.concatenate(tris) + (model_vstart + mesh_vertoff)
                results.append((material, t))
    return results


def load(read, model_path, bodygroups=()):
    """`read(path) -> bytes|None` reads from the virtual filesystem.
    `bodygroups` is the chosen submodel index per body part (default 0)."""
    model_path = model_path.replace("\\", "/").lower()
    if not model_path.endswith(".mdl"):
        model_path += ".mdl"
    base = model_path[:-4]

    mdl = read(model_path)
    if mdl is None:
        raise ModelError("model not found: %s" % model_path)
    vvd = read(base + ".vvd")
    if vvd is None:
        raise ModelError("missing %s.vvd" % base)
    vtx = None
    for suffix in VTX_SUFFIXES:
        vtx = read(base + suffix)
        if vtx is not None:
            break
    if vtx is None:
        raise ModelError("missing %s.dx90.vtx" % base)

    info = _parse_mdl(mdl)
    _checksum, positions, normals, uvs = _parse_vvd(vvd)

    # Newer studiomdl builds add topology fields to strip groups and strips.
    meshes = None
    last_err = None
    for sg_size, s_size in ((25, 27), (33, 35)):
        try:
            meshes = _parse_vtx(vtx, info, list(bodygroups), sg_size, s_size)
            break
        except (ModelError, struct.error, ValueError) as e:
            last_err = e
    if meshes is None:
        raise ModelError("could not parse VTX: %s" % last_err)

    data = ModelData()
    data.name = model_path
    data.bodyparts = [(bp["name"], bp["base"], [m["name"] for m in bp["models"]]) for bp in info["bodyparts"]]
    data.bodygroups = [(bodygroups[i] if i < len(bodygroups) else 0) for i in range(len(data.bodyparts))]
    data.bodygroup_mask = bodygroup_mask(data.bodyparts, data.bodygroups)
    all_tris = []
    all_mats = []
    mat_slots = {}
    for material, tris in meshes:
        if not len(tris):
            continue
        if tris.max() >= len(positions):
            raise ModelError("triangle references missing vertex")
        tex = info["skin0"][material] if material < len(info["skin0"]) else material
        if tex not in mat_slots:
            mat_slots[tex] = len(data.materials)
            name = info["textures"][tex] if tex < len(info["textures"]) else ""
            data.materials.append(material_candidates(name, info["cddirs"]))
        all_tris.append(tris[:, ::-1])  # Source is clockwise; Blender is CCW
        all_mats.append(np.full(len(tris), mat_slots[tex], dtype=np.int32))

    if not all_tris:
        if any(len(bp[2]) > 1 for bp in data.bodyparts):
            # A body group combination can legitimately be empty.
            data.positions = np.zeros((0, 3), np.float32)
            data.normals = np.zeros((0, 3), np.float32)
            data.uvs = np.zeros((0, 2), np.float32)
            data.triangles = np.zeros((0, 3), np.int32)
            data.tri_material = np.zeros(0, np.int32)
            return data
        raise ModelError("model has no LOD 0 geometry")

    tris = np.concatenate(all_tris)
    used, inverse = np.unique(tris.reshape(-1), return_inverse=True)
    data.triangles = inverse.reshape(-1, 3).astype(np.int32)
    data.positions = positions[used].astype(np.float32)
    data.normals = normals[used].astype(np.float32)
    uv = uvs[used].astype(np.float32)
    uv[:, 1] = 1.0 - uv[:, 1]
    data.uvs = uv
    data.tri_material = np.concatenate(all_mats)
    return data


def material_candidates(texture_name, cddirs):
    """VMT paths to try, in the engine's order, for an MDL texture."""
    name = texture_name.replace("\\", "/").strip("/").lower()
    if name.endswith(".vmt"):
        name = name[:-4]
    out = []
    for d in cddirs:
        d = d.replace("\\", "/").strip("/").lower()
        out.append("materials/%s/%s.vmt" % (d, name) if d else "materials/%s.vmt" % name)
    out.append("materials/%s.vmt" % name)
    seen = set()
    return [p.replace("//", "/") for p in out if not (p in seen or seen.add(p))]
