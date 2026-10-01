"""Plane clipping of triangle soups, matching Prop2Mesh's applyClippingPlane
(cl_meshlab.lua): keeps the side where dot(p, n) >= d, cuts triangles that
straddle the plane and leaves the cut open (no cap). Winding is preserved."""

import numpy as np


def clip(soup, normal, dist):
    """soup: (m, 3, F) float array; columns 0..2 are positions, the rest
    (normals, uvs, ...) are interpolated along cut edges."""
    if len(soup) == 0:
        return soup
    n = np.asarray(normal, dtype=np.float64)
    s = soup[:, :, :3] @ n - dist          # (m, 3) signed distances
    inside = s >= 0
    count = inside.sum(1)

    keep = soup[count == 3]
    out = [keep]

    def lerp(a, b, sa, sb):
        t = (sa / (sa - sb))[:, None]
        return a + (b - a) * t

    # Roll each triangle so the odd vertex is first (cyclic: keeps winding).
    for want_inside, odd_is_inside in ((1, True), (2, False)):
        sel = count == want_inside
        if not sel.any():
            continue
        tris = soup[sel]
        dist3 = s[sel]
        odd = np.argmax(inside[sel] == odd_is_inside, axis=1)
        idx = (odd[:, None] + np.arange(3)[None, :]) % 3
        tris = np.take_along_axis(tris, idx[:, :, None], 1)
        dist3 = np.take_along_axis(dist3, idx, 1)
        v0, v1, v2 = tris[:, 0], tris[:, 1], tris[:, 2]
        s0, s1, s2 = dist3[:, 0], dist3[:, 1], dist3[:, 2]
        v01 = lerp(v0, v1, s0, s1)
        v02 = lerp(v0, v2, s0, s2)
        if odd_is_inside:
            # only v0 kept: (v0, v01, v02)
            out.append(np.stack([v0, v01, v02], 1))
        else:
            # v0 cut away: quad (v01, v1, v2, v02)
            out.append(np.stack([v01, v1, v2], 1))
            out.append(np.stack([v01, v2, v02], 1))
    return np.concatenate(out) if out else soup[:0]


def clip_all(soup, planes):
    for normal, dist in planes:
        soup = clip(soup, normal, dist)
    return soup
