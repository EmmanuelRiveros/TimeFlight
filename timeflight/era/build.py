"""Era manifest -> prisms in the place's ENU frame (water, island, canals, houses, precinct, causeways).

Pure numpy and deterministic for a given seed. z values are meters relative to the place
origin's ground height, like the terrain and OSM buildings.
"""

from __future__ import annotations

import math

import numpy as np

from ..geo.enu import Origin, geodetic_to_enu
from ..geo.osm import point_in_ring
from . import proxies as P


def _enu_xy(lat, lon, origin: Origin) -> np.ndarray:
    return geodetic_to_enu(lat, lon, origin.h, origin)[..., :2]


def _seg_dist(pts: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ab = b - a
    t = np.clip(((pts - a) @ ab) / max(ab @ ab, 1e-9), 0, 1)
    return np.linalg.norm(pts - (a + t[:, None] * ab), axis=1)


def polyline_dist(pts: np.ndarray, line: np.ndarray) -> np.ndarray:
    return np.min([_seg_dist(pts, line[i], line[i + 1]) for i in range(len(line) - 1)], axis=0)


def ring_dist(pts: np.ndarray, ring: np.ndarray) -> np.ndarray:
    return polyline_dist(pts, np.vstack([ring, ring[:1]]))


def in_rects(pts: np.ndarray, rects: list[tuple[float, float, float, float]]) -> np.ndarray:
    hit = np.zeros(len(pts), bool)
    for x0, y0, x1, y1 in rects:
        hit |= (pts[:, 0] >= x0) & (pts[:, 0] <= x1) & (pts[:, 1] >= y0) & (pts[:, 1] <= y1)
    return hit


def build_era(m: dict, origin: Origin, seed: int = 1519) -> dict:
    rng = np.random.default_rng(seed)
    water = m["water"]["level_m"]
    land = water + m["water"]["land_height_m"]
    anchor = _enu_xy(m["anchor"]["lat"], m["anchor"]["lon"], origin)
    ax, ay = anchor

    def rel(bounds):  # anchor-relative [x0, y0, x1, y1] -> ENU
        x0, y0, x1, y1 = bounds
        return (ax + x0, ay + y0, ax + x1, ay + y1)

    prisms: list[dict] = []
    island = _enu_xy(*np.array(m["island"]["polygon_latlon"]).T, origin)
    if _signed_area(island) < 0:
        island = island[::-1]

    # Exclusion zones for houses: precinct, plazas, palaces and other structures.
    pr = m["precinct"]
    keepout = [rel(pr["bounds_m"])] + [rel(p["bounds_m"]) for p in m.get("plazas", [])]
    struct_rects = []
    for s in m["structures"]:
        ox, oy = s["offset_m"]
        w, d = s.get("size_m") or s.get("base_m") or [2 * s.get("radius_m", 10)] * 2
        struct_rects.append(rel((ox - w / 2 - 8, oy - d / 2 - 8, ox + w / 2 + 8, oy + d / 2 + 8)))
    keepout += struct_rects
    causeways = [_enu_xy(*np.array(c["points_latlon"]).T, origin) for c in m["causeways"]]

    # Land blocks separated by canals; chinampa strips near the shore.
    c = m["canals"]
    pitch = c["block_m"] + c["canal_m"]
    lo, hi = island.min(axis=0), island.max(axis=0)
    gx = np.arange(math.floor((lo[0] - ax) / pitch), math.ceil((hi[0] - ax) / pitch) + 1) * pitch + ax
    gy = np.arange(math.floor((lo[1] - ay) / pitch), math.ceil((hi[1] - ay) / pitch) + 1) * pitch + ay
    centers = np.array([(x, y) for x in gx for y in gy])
    inside = np.array([point_in_ring(p, island) for p in centers])
    centers = centers[inside]
    shore = ring_dist(centers, island)
    houses_cfg = m["houses"]
    lot = houses_cfg["lot_m"]
    n_houses = n_gardens = 0
    for (cx, cy), sd in zip(centers, shore):
        if sd < c["chinampa_band_m"]:
            # Chinampas: long garden strips with water between them, no land platform.
            for k in range(-2, 3):
                strip_y = cy + k * c["block_m"] / 5
                prisms.append(P.box(cx, strip_y, c["block_m"] * 0.95, c["block_m"] / 5 - 5, water - 2, 2.6, "garden"))
                n_gardens += 1
            continue
        prisms.append(P.box(cx, cy, c["block_m"], c["block_m"], water - 3, land - water + 3, "land"))
        # Houses on a lot grid inside the block.
        n = int(c["block_m"] // lot)
        offs = (np.arange(n) - (n - 1) / 2) * lot
        lots = np.array([(cx + u, cy + v) for u in offs for v in offs])
        drop = in_rects(lots, keepout)
        for line in causeways:
            drop |= polyline_dist(lots, line) < 12  # avenues along the causeway axes
        for (hx, hy) in lots[~drop]:
            ang = rng.uniform(-3, 3)
            if rng.random() < houses_cfg["courtyard_share"]:
                h = rng.uniform(*houses_cfg["elite_height_m"])
                prisms += P.courtyard_house(hx, hy, lot - 3, lot - 3, h, land - 0.5, wing=5.5, angle_deg=ang)
            else:
                h = rng.uniform(*houses_cfg["commoner_height_m"])
                w, d = rng.uniform(9, lot - 6), rng.uniform(9, lot - 6)
                prisms.append(P.box(hx + rng.uniform(-2, 2), hy + rng.uniform(-2, 2), w, d, land - 0.5, h + 0.5,
                                    "house", ang))
            n_houses += 1

    # Precinct: solid land, wall with gates, plazas as land.
    x0, y0, x1, y1 = rel(pr["bounds_m"])
    prisms.append(P.prism(P.rect((x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0), water - 3, land + 0.3, "land"))
    prisms += wall_with_gates(x0, y0, x1, y1, pr["wall"]["thickness_m"], pr["wall"]["height_m"], land + 0.3,
                              pr["gates"], ax, ay)
    for p in m.get("plazas", []):
        px0, py0, px1, py1 = rel(p["bounds_m"])
        prisms.append(P.prism(P.rect((px0 + px1) / 2, (py0 + py1) / 2, px1 - px0, py1 - py0), water - 3, land, "land"))

    for s, (sx0, sy0, sx1, sy1) in zip(m["structures"], struct_rects):
        # Solid ground under every named structure, so no canal runs through a palace court.
        prisms.append(P.prism(P.rect((sx0 + sx1) / 2, (sy0 + sy1) / 2, sx1 - sx0, sy1 - sy0), water - 3, land, "land"))
        prisms += structure(s, ax, ay, land + 0.3)

    for cw, line in zip(m["causeways"], causeways):
        prisms += causeway(line, cw, water, island)

    return {"prisms": prisms, "water_level": water, "land_level": land, "island_enu": island,
            "stats": {"houses": n_houses, "garden_strips": n_gardens, "prisms": len(prisms),
                      "island_area_km2": round(abs(_signed_area(island)) / 1e6, 2)}}


def _signed_area(ring: np.ndarray) -> float:
    x, y = ring[:, 0], ring[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def wall_with_gates(x0, y0, x1, y1, thick, height, z0, gates, ax, ay, gate_w=14.0) -> list[dict]:
    """Enclosure wall; gates are given as positions along a side, relative to the anchor."""
    sides = {"south": ((x0, y0), (x1, y0)), "north": ((x0, y1), (x1, y1)),
             "west": ((x0, y0), (x0, y1)), "east": ((x1, y0), (x1, y1))}
    out = []
    for side, ((sx0, sy0), (sx1, sy1)) in sides.items():
        horiz = sy0 == sy1
        a, b = (sx0, sx1) if horiz else (sy0, sy1)
        cuts = sorted((g["at_m"] + (ax if horiz else ay)) for g in gates if g["side"] == side)
        edges = [a] + [v for g in cuts for v in (g - gate_w / 2, g + gate_w / 2)] + [b]
        for s0, s1 in zip(edges[0::2], edges[1::2]):
            if s1 - s0 < 1:
                continue
            m = (s0 + s1) / 2
            if horiz:
                out.append(P.box(m, sy0, s1 - s0, thick, z0, height, "wall"))
            else:
                out.append(P.box(sx0, m, thick, s1 - s0, z0, height, "wall"))
    return out


def structure(s: dict, ax: float, ay: float, z0: float) -> list[dict]:
    x, y = ax + s["offset_m"][0], ay + s["offset_m"][1]
    t = s["type"]
    if t == "twin_temple":
        return P.twin_temple(x, y, *s["base_m"], s["height_m"], s["shrine_height_m"], s["tiers"], z0, s["stairs"])
    if t == "stepped_pyramid":
        return P.stepped_pyramid(x, y, *s["base_m"], s["height_m"], s["tiers"], z0, s["stairs"])
    if t == "round_temple":
        return P.round_temple(x, y, s["radius_m"], s["platform_height_m"], s["shrine_radius_m"],
                              s["shrine_height_m"], z0, s["stairs"])
    if t == "ballcourt":
        return P.ballcourt(x, y, s["length_m"], s["width_m"], s["wall_height_m"], z0, s.get("angle_deg", 0))
    if t == "platform":
        return [P.box(x, y, *s["size_m"], z0, s["height_m"], "temple")]
    if t == "courtyard":
        # Named courtyard complexes (palaces, schools) are not part of the house fill.
        return P.courtyard_house(x, y, *s["size_m"], s["height_m"], z0, s["wing_m"], kind="palace")
    raise ValueError(f"unknown structure type {t}")


def causeway(line: np.ndarray, cw: dict, water: float, island: np.ndarray) -> list[dict]:
    """Raised road over the water, cut by bridge gaps; segments inside the island are skipped
    (there the causeway continues as an avenue kept free of houses)."""
    width, top = cw["width_m"], water + cw["height_above_water_m"]
    every, gap = cw["bridge_every_m"], 10.0
    out, run = [], 0.0
    for a, b in zip(line[:-1], line[1:]):
        seg = b - a
        length = float(np.linalg.norm(seg))
        u = seg / length
        ang = math.degrees(math.atan2(u[1], u[0]))
        pos = 0.0
        while pos < length:
            step = min(every - (run % every), length - pos)
            s0, s1 = pos, pos + step
            mid = a + u * (s0 + s1) / 2
            if not point_in_ring(mid, island):
                piece = s1 - s0 - (gap if s1 - s0 > gap * 2 and abs((run + step) % every) < 1e-6 else 0)
                c = a + u * (s0 + piece / 2)
                out.append(P.box(c[0], c[1], piece, width, water - 3, top - water + 3, "causeway", ang))
                if piece < s1 - s0:  # removable wooden bridge over the gap
                    g = a + u * (s0 + piece + gap / 2)
                    out.append(P.box(g[0], g[1], gap + 1, width * 0.8, top - 0.5, 0.5, "bridge", ang))
            run += step
            pos = s1
    return out
