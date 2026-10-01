"""A small software rasterizer for model browser thumbnails (numpy only, so
it works the same on every Blender version and GPU backend)."""

import numpy as np

# Camera looks at the model's front-right from above, like Hammer's model
# browser. Source: +X forward, +Y left, +Z up.
VIEW_DIR = np.array([0.72, -0.55, 0.42])
LIGHT_DIR = np.array([0.45, -0.25, 0.85])


def _norm(v):
    return v / (np.linalg.norm(v) or 1.0)


def render(model, textures, size=320, supersample=2, background=(0.16, 0.16, 0.16, 1.0)):
    """model: studiomdl.ModelData. textures: list (per material) of RGBA
    arrays (h, w, 4, top row first) or None. Returns (size, size, 4) float32,
    top row first."""
    res = size * supersample
    pos = model.positions.astype(np.float64)
    tris = model.triangles
    d = _norm(VIEW_DIR)
    up = np.array([0.0, 0.0, 1.0])
    right = _norm(np.cross(up, d))
    cam_up = np.cross(d, right)

    sx, sy, depth = pos @ right, pos @ cam_up, pos @ d
    lo = np.array([sx.min(), sy.min()])
    hi = np.array([sx.max(), sy.max()])
    extent = max(hi - lo) or 1.0
    scale = res * 0.86 / extent
    center = (lo + hi) / 2
    px = (sx - center[0]) * scale + res / 2
    py = res / 2 - (sy - center[1]) * scale  # screen y down

    color = np.zeros((res, res, 3), dtype=np.float32)
    color[:] = background[:3]
    zbuf = np.full((res, res), -np.inf)

    p0, p1, p2 = pos[tris[:, 0]], pos[tris[:, 1]], pos[tris[:, 2]]
    normals = np.cross(p1 - p0, p2 - p0)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    light = _norm(LIGHT_DIR)
    lambert = np.abs(normals @ light)
    facing = normals @ d
    shade = (0.35 + 0.65 * lambert) * np.where(facing >= 0, 1.0, 0.6)

    uvs = model.uvs
    mats = model.tri_material

    X = px[tris]  # (m, 3)
    Y = py[tris]
    Z = depth[tris]
    xmin = np.clip(np.floor(X.min(1)), 0, res - 1).astype(int)
    xmax = np.clip(np.ceil(X.max(1)), 0, res - 1).astype(int)
    ymin = np.clip(np.floor(Y.min(1)), 0, res - 1).astype(int)
    ymax = np.clip(np.ceil(Y.max(1)), 0, res - 1).astype(int)

    # Draw far triangles first so most pixels are written once.
    order = np.argsort(Z.mean(1))
    for t in order:
        x0, x1, y0, y1 = xmin[t], xmax[t], ymin[t], ymax[t]
        gx, gy = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
        ax, bx, cx = X[t]
        ay, by, cy = Y[t]
        den = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
        if abs(den) < 1e-12:
            continue
        w0 = ((by - cy) * (gx - cx) + (cx - bx) * (gy - cy)) / den
        w1 = ((cy - ay) * (gx - cx) + (ax - cx) * (gy - cy)) / den
        w2 = 1.0 - w0 - w1
        inside = (w0 >= -1e-6) & (w1 >= -1e-6) & (w2 >= -1e-6)
        if not inside.any():
            continue
        z = w0 * Z[t, 0] + w1 * Z[t, 1] + w2 * Z[t, 2]
        sub = zbuf[y0:y1 + 1, x0:x1 + 1]
        win = inside & (z > sub)
        if not win.any():
            continue
        sub[win] = z[win]
        tex = textures[mats[t]] if mats[t] < len(textures) else None
        if tex is not None:
            tri_uv = uvs[tris[t]]
            u = w0 * tri_uv[0, 0] + w1 * tri_uv[1, 0] + w2 * tri_uv[2, 0]
            v = w0 * tri_uv[0, 1] + w1 * tri_uv[1, 1] + w2 * tri_uv[2, 1]
            th, tw = tex.shape[:2]
            # model.uvs are Blender-style (v up); textures are top row first
            tu = (np.mod(u[win], 1.0) * tw).astype(int) % tw
            tv = (np.mod(1.0 - v[win], 1.0) * th).astype(int) % th
            c = tex[tv, tu, :3]
        else:
            c = np.array([0.75, 0.75, 0.75], dtype=np.float32)
        color[y0:y1 + 1, x0:x1 + 1][win] = c * shade[t]

    alpha = np.where(np.isfinite(zbuf), 1.0, background[3]).astype(np.float32)
    rgba = np.dstack([color, alpha])
    if supersample > 1:
        rgba = rgba.reshape(size, supersample, size, supersample, 4).mean((1, 3))
    return rgba.astype(np.float32)
