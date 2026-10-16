"""Period proxy geometry as extruded polygons ("prisms"), in local ENU meters.

Every shape is a list of prisms {"ring": (n, 2) counter-clockwise, "z0", "z1", "kind"}; the
Blender side extrudes them the same way as OSM buildings. Proxies are deliberately simple
massing models: the video model adds surface detail, the proxies fix layout and scale.
"""

from __future__ import annotations

import math

import numpy as np

KINDS = ("land", "house", "temple", "shrine", "stairs", "wall", "palace", "causeway", "bridge", "garden")


def rect(cx: float, cy: float, w: float, d: float, angle_deg: float = 0.0) -> np.ndarray:
    """Counter-clockwise rectangle centered at (cx, cy), w along x and d along y before rotation."""
    a = math.radians(angle_deg)
    c, s = math.cos(a), math.sin(a)
    local = np.array([[-w / 2, -d / 2], [w / 2, -d / 2], [w / 2, d / 2], [-w / 2, d / 2]])
    return local @ np.array([[c, s], [-s, c]]) + [cx, cy]


def circle(cx: float, cy: float, r: float, n: int = 24) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([cx + r * np.cos(t), cy + r * np.sin(t)])


def prism(ring: np.ndarray, z0: float, z1: float, kind: str) -> dict:
    if kind not in KINDS:
        raise ValueError(f"unknown kind {kind}")
    return {"ring": np.asarray(ring, dtype=np.float64), "z0": float(z0), "z1": float(z1), "kind": kind}


def box(cx, cy, w, d, z0, h, kind, angle_deg=0.0) -> dict:
    return prism(rect(cx, cy, w, d, angle_deg), z0, z0 + h, kind)


# Unit vectors for the side a stairway faces.
FACING = {"west": (-1.0, 0.0), "east": (1.0, 0.0), "north": (0.0, 1.0), "south": (0.0, -1.0)}


def stepped_pyramid(cx, cy, base_w, base_d, height, tiers, z0, stairs="west", stair_width=None,
                    top_ratio=0.45, kind="temple") -> list[dict]:
    """Stacked terraces shrinking from base to top_ratio of the base, with a stairway on one
    face built from thin steps (each step one prism) running from the ground to the top."""
    out = []
    th = height / tiers
    for i in range(tiers):
        f = 1 - (1 - top_ratio) * i / max(tiers - 1, 1)
        out.append(box(cx, cy, base_w * f, base_d * f, z0 + i * th, th, kind))
    if stairs:
        out += stairway(cx, cy, base_w, base_d, height, z0, stairs, top_ratio,
                        stair_width or 0.35 * (base_d if stairs in ("west", "east") else base_w))
    return out


def stairway(cx, cy, base_w, base_d, height, z0, facing, top_ratio, width, steps: int = 12) -> list[dict]:
    """A straight ramp of `steps` boxes from the base edge to the top platform on one face."""
    ux, uy = FACING[facing]
    half_base = (base_w if ux else base_d) / 2
    half_top = half_base * top_ratio
    run = half_base - half_top + 4.0  # stairs project 4 m beyond the base
    out = []
    for k in range(steps):
        t0, t1 = k / steps, (k + 1) / steps
        dist_out = half_base + 4.0 - run * t0  # outer edge of this step from the center
        dist_in = half_base + 4.0 - run * t1
        mid = (dist_out + dist_in) / 2
        depth = dist_out - dist_in
        sx, sy = cx + ux * mid, cy + uy * mid
        w, d = (depth, width) if ux else (width, depth)
        out.append(box(sx, sy, w, d, z0, height * t1, "stairs"))
    return out


def twin_temple(cx, cy, base_w, base_d, height, shrine_height, tiers, z0, stairs="west") -> list[dict]:
    """Templo Mayor type: one stepped base, two stairways and two shrines side by side.

    The shrines (and stairways) sit north and south of the axis; facing west, north is the
    Tlaloc shrine and south the Huitzilopochtli shrine (sources_1519.md, layout facts)."""
    out = stepped_pyramid(cx, cy, base_w, base_d, height, tiers, z0, stairs=None)
    ux, uy = FACING[stairs]
    px, py = -uy, ux  # perpendicular to the facing direction
    side = 0.25 * (base_d if ux else base_w)
    for sgn in (1, -1):
        sx, sy = cx + px * side * sgn, cy + py * side * sgn
        out += stairway(sx, sy, base_w, base_d, height, z0, stairs, 0.45,
                        0.3 * (base_d if ux else base_w), steps=12)
        tw, td = 0.18 * base_w, 0.18 * base_d
        # shrine pushed toward the back of the top platform
        bx, by = sx - ux * 0.08 * base_w, sy - uy * 0.08 * base_d
        out.append(box(bx, by, tw, td, z0 + height, shrine_height, "shrine"))
    return out


def round_temple(cx, cy, r, platform_h, shrine_r, shrine_h, z0, stairs="east") -> list[dict]:
    """Ehecatl-Quetzalcoatl type: circular platform with a cylindrical shrine and one stair."""
    out = [prism(circle(cx, cy, r), z0, z0 + platform_h, "temple"),
           prism(circle(cx, cy, shrine_r), z0 + platform_h, z0 + platform_h + shrine_h, "shrine")]
    out += stairway(cx, cy, 2 * r, 2 * r, platform_h, z0, stairs, 0.9, 0.5 * r, steps=6)
    return out


def ballcourt(cx, cy, length, width, wall_h, z0, angle_deg=0.0) -> list[dict]:
    """I-shaped court: two long parallel walls plus end zones, long axis along x (east-west)."""
    a = math.radians(angle_deg)
    ex, ey = math.cos(a), math.sin(a)
    nx, ny = -ey, ex
    out = []
    for sgn in (1, -1):
        out.append(box(cx + nx * sgn * width / 2, cy + ny * sgn * width / 2, length * 0.6, 6, z0, wall_h,
                       "wall", angle_deg))
        out.append(box(cx + ex * sgn * length * 0.42, cy + ey * sgn * length * 0.42, length * 0.16,
                       width * 1.8, z0, wall_h * 0.6, "wall", angle_deg))
    return out


def wall_around(x0, y0, x1, y1, thickness, height, z0, gate_width=0.0) -> list[dict]:
    """Rectangular enclosure; optional gates centered on each side."""
    out = []
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    w, d = x1 - x0, y1 - y0
    for (sx, sy, length, horiz) in ((cx, y0, w, True), (cx, y1, w, True), (x0, cy, d, False), (x1, cy, d, False)):
        segs = [(-length / 2, -gate_width / 2), (gate_width / 2, length / 2)] if gate_width else [(-length / 2, length / 2)]
        for a, b in segs:
            m, L = (a + b) / 2, b - a
            if horiz:
                out.append(box(sx + m, sy, L, thickness, z0, height, "wall"))
            else:
                out.append(box(sx, sy + m, thickness, L, z0, height, "wall"))
    return out


def courtyard_house(cx, cy, w, d, h, z0, wing: float, angle_deg=0.0, kind="house") -> list[dict]:
    """Four wings around an open courtyard (elite house or palace)."""
    a = math.radians(angle_deg)
    c, s = math.cos(a), math.sin(a)

    def at(x, y):
        return cx + x * c - y * s, cy + x * s + y * c

    parts = [((0, -(d - wing) / 2), w, wing), ((0, (d - wing) / 2), w, wing),
             ((-(w - wing) / 2, 0), wing, d - 2 * wing), (((w - wing) / 2, 0), wing, d - 2 * wing)]
    return [box(*at(*off), pw, pd, z0, h, kind, angle_deg) for off, pw, pd in parts]
