"""A small software rasterizer for model browser thumbnails (numpy only, so
it works the same on every Blender version and GPU backend).

All triangles are rasterized at once: every (triangle, pixel-in-its-bounding-
box) pair is tested in bulk, the nearest hit per pixel wins (z-buffer), and
shading/texturing is done once per pixel afterwards.
"""

import numpy as np

# Camera looks at the model's front-right from above, like Hammer's model
# browser. Source: +X forward, +Y left, +Z up.
VIEW_DIR = np.array([0.72, -0.55, 0.42])
LIGHT_DIR = np.array([0.45, -0.25, 0.85])
MAX_PAIRS = 2_000_000  # (triangle, pixel) pairs tested per batch


def _norm(v):
    return v / (np.linalg.norm(v) or 1.0)


def render(model, textures, size=320, supersample=2, background=(0.16, 0.16, 0.16, 1.0)):
    """model: studiomdl.ModelData. textures: list (per material) of RGBA
    arrays (h, w, 4, top row first) or None. Returns (size, size, 4) float32,
    top row first."""
    res = size * supersample
    rgba = np.empty((res * res, 4), dtype=np.float32)
    rgba[:] = background
    pos = model.positions.astype(np.float64)
    tris = model.triangles
    if len(tris) == 0:
        return _downsample(rgba, size, supersample)

    d = _norm(VIEW_DIR)
    right = _norm(np.cross(np.array([0.0, 0.0, 1.0]), d))
    cam_up = np.cross(d, right)
    sx, sy, depth = pos @ right, pos @ cam_up, pos @ d
    lo = np.array([sx.min(), sy.min()])
    hi = np.array([sx.max(), sy.max()])
    scale = res * 0.86 / (max(hi - lo) or 1.0)
    center = (lo + hi) / 2
    px = (sx - center[0]) * scale + res / 2
    py = res / 2 - (sy - center[1]) * scale  # screen y points down

    X, Y, Z = px[tris], py[tris], depth[tris]
    p0, p1, p2 = pos[tris[:, 0]], pos[tris[:, 1]], pos[tris[:, 2]]
    normals = np.cross(p1 - p0, p2 - p0)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    shade = (0.35 + 0.65 * np.abs(normals @ _norm(LIGHT_DIR))) * np.where(normals @ d >= 0, 1.0, 0.6)

    # Pixel centres are at +0.5. Each triangle covers the rows whose centres
    # lie within its vertical extent; on each row it covers an exact span.
    ymin = np.clip(np.ceil(Y.min(1) - 0.5), 0, res).astype(np.int64)
    ymax = np.clip(np.floor(Y.max(1) - 0.5), -1, res - 1).astype(np.int64)
    h = ymax - ymin + 1
    den = (Y[:, 1] - Y[:, 2]) * (X[:, 0] - X[:, 2]) + (X[:, 2] - X[:, 1]) * (Y[:, 0] - Y[:, 2])
    live = np.nonzero((h > 0) & (np.abs(den) > 1e-12))[0]

    # One entry per (triangle, row): intersect the row's centre line with
    # the triangle's three edges to get the covered x range.
    rt = np.repeat(live, h[live])
    ry = ymin[rt] + (np.arange(len(rt)) - np.repeat(np.cumsum(h[live]) - h[live], h[live])) + 0.5
    lo = np.full(len(rt), np.inf)
    hi = np.full(len(rt), -np.inf)
    for a, b in ((0, 1), (1, 2), (2, 0)):
        xa, ya, xb, yb = X[rt, a], Y[rt, a], X[rt, b], Y[rt, b]
        dy = yb - ya
        crosses = ((ya - ry) * (yb - ry) <= 0) & (dy != 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            x = np.where(crosses, xa + (ry - ya) * (xb - xa) / np.where(dy == 0, 1, dy), np.nan)
        lo = np.where(crosses, np.minimum(lo, x), lo)
        hi = np.where(crosses, np.maximum(hi, x), hi)
    x0 = np.clip(np.ceil(lo - 0.5 - 1e-9), 0, res).astype(np.int64)
    x1 = np.clip(np.floor(hi - 0.5 + 1e-9), -1, res - 1).astype(np.int64)
    span = np.where(np.isfinite(lo), x1 - x0 + 1, 0)
    keep = span > 0
    rt, ry, x0, span = rt[keep], ry[keep], x0[keep], span[keep]

    zbuf = np.full(res * res, -np.inf)
    tri_buf = np.full(res * res, -1, dtype=np.int64)
    b0_buf = np.zeros(res * res)
    b1_buf = np.zeros(res * res)

    start = 0
    while start < len(rt):
        # Take as many rows as fit in one batch (at least one).
        csum = np.cumsum(span[start:])
        end = start + max(1, int(np.searchsorted(csum, MAX_PAIRS, side="right")))
        n = span[start:end]
        t = np.repeat(rt[start:end], n)
        gy = np.repeat(ry[start:end], n)
        gx = np.repeat(x0[start:end], n) + (np.arange(int(n.sum())) - np.repeat(np.cumsum(n) - n, n)) + 0.5
        start = end
        ax, bx, cx = X[t, 0], X[t, 1], X[t, 2]
        ay, by, cy = Y[t, 0], Y[t, 1], Y[t, 2]
        dt = den[t]
        b0 = ((by - cy) * (gx - cx) + (cx - bx) * (gy - cy)) / dt
        b1 = ((cy - ay) * (gx - cx) + (ax - cx) * (gy - cy)) / dt
        b2 = 1.0 - b0 - b1
        inside = (b0 >= -1e-6) & (b1 >= -1e-6) & (b2 >= -1e-6)
        if not inside.any():
            continue
        t, b0, b1, b2 = t[inside], b0[inside], b1[inside], b2[inside]
        pix = (gy[inside].astype(np.int64)) * res + gx[inside].astype(np.int64)
        z = b0 * Z[t, 0] + b1 * Z[t, 1] + b2 * Z[t, 2]

        # Nearest candidate per pixel within this batch...
        order = np.lexsort((-z, pix))
        pix, z, t, b0, b1 = pix[order], z[order], t[order], b0[order], b1[order]
        first = np.ones(len(pix), dtype=bool)
        first[1:] = pix[1:] != pix[:-1]
        pix, z, t, b0, b1 = pix[first], z[first], t[first], b0[first], b1[first]
        # ...then against what earlier batches drew.
        win = z > zbuf[pix]
        pix = pix[win]
        zbuf[pix] = z[win]
        tri_buf[pix] = t[win]
        b0_buf[pix] = b0[win]
        b1_buf[pix] = b1[win]

    hit = np.nonzero(tri_buf >= 0)[0]
    if len(hit):
        t = tri_buf[hit]
        b0, b1 = b0_buf[hit], b1_buf[hit]
        b2 = 1.0 - b0 - b1
        color = np.empty((len(hit), 3), dtype=np.float32)
        color[:] = 0.75
        mats = model.tri_material[t]
        for m in np.unique(mats):
            tex = textures[m] if m < len(textures) else None
            if tex is None:
                continue
            sel = mats == m
            tt = tris[t[sel]]
            uv = (model.uvs[tt[:, 0]] * b0[sel, None] + model.uvs[tt[:, 1]] * b1[sel, None]
                  + model.uvs[tt[:, 2]] * b2[sel, None])
            th, tw = tex.shape[:2]
            # model.uvs are Blender-style (v up); textures are top row first
            tu = (np.mod(uv[:, 0], 1.0) * tw).astype(np.int64) % tw
            tv = (np.mod(1.0 - uv[:, 1], 1.0) * th).astype(np.int64) % th
            color[sel] = tex[tv, tu, :3]
        rgba[hit, :3] = color * shade[t, None]
        rgba[hit, 3] = 1.0
    return _downsample(rgba, size, supersample)


def _downsample(rgba, size, supersample):
    img = rgba.reshape(size * supersample, size * supersample, 4)
    if supersample > 1:
        img = img.reshape(size, supersample, size, supersample, 4).mean((1, 3))
    return np.ascontiguousarray(img, dtype=np.float32)
