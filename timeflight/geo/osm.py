"""OSM building footprints -> extrusion-ready polygons in ENU (pure Python/numpy).

Reading the .osm.pbf happens in scripts/fetch_geodata.py. This module only turns
(tags, lat/lon ring) records into footprints with a height.

OSM data is ODbL: derived files stay in data/ (gitignored) and are never committed.
"""

from __future__ import annotations

import re

import numpy as np

from .enu import Origin, geodetic_to_enu

LEVEL_HEIGHT_M = 3.2
DEFAULT_HEIGHT_M = 9.0  # typical 2-3 storey building in the historic centre
RUINS_HEIGHT_M = 3.0  # excavations such as the Templo Mayor zone are low, open structures
KEEP_TAGS = ("name", "building", "building:part", "historic", "amenity", "tourism", "heritage", "building:levels", "height")

_NUM = re.compile(r"^\s*([0-9]+(?:[.,][0-9]+)?)\s*(m|ft|')?\s*$")


def parse_length(value: str | None) -> float | None:
    """'12', '12 m', '12.5', '40 ft' -> meters; anything else -> None."""
    if not value:
        return None
    m = _NUM.match(value)
    if not m:
        return None
    x = float(m.group(1).replace(",", "."))
    return x * 0.3048 if m.group(2) in ("ft", "'") else x


def building_height(tags: dict) -> tuple[float, str]:
    """Height in meters and where it came from: 'height', 'levels' or 'default'."""
    h = parse_length(tags.get("height"))
    if h and h > 0:
        return h, "height"
    levels = parse_length(tags.get("building:levels"))
    if levels and levels > 0:
        return levels * LEVEL_HEIGHT_M + 1.0, "levels"
    if tags.get("historic") == "ruins":
        return RUINS_HEIGHT_M, "ruins"
    return DEFAULT_HEIGHT_M, "default"


def signed_area(ring: np.ndarray) -> float:
    x, y = ring[:, 0], ring[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def footprint(osm_id: str, tags: dict, latlon: list[tuple[float, float]], origin: Origin) -> dict | None:
    """One building record, or None if the ring is degenerate. The ring is counter-clockwise,
    without the closing duplicate point; xy are ENU meters (east, north)."""
    pts = np.asarray(latlon, dtype=np.float64)
    if len(pts) >= 2 and np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]
    if len(pts) < 3:
        return None
    enu = geodetic_to_enu(pts[:, 0], pts[:, 1], np.full(len(pts), origin.h), origin)
    ring = enu[:, :2]
    area = signed_area(ring)
    if abs(area) < 4.0:
        return None
    if area < 0:
        ring = ring[::-1]
    height, height_src = building_height(tags)
    return {"id": osm_id, "ring": ring.round(3).tolist(), "height": round(height, 2),
            "height_source": height_src, "area_m2": round(abs(area), 1),
            "tags": {k: tags[k] for k in KEEP_TAGS if k in tags}}


def is_part(record: dict) -> bool:
    """building:part records. Older extracts kept no building:part tag, but every extracted record
    has either building or building:part, so a record without building is a part."""
    tags = record["tags"]
    return "building:part" in tags or "building" not in tags


def point_in_ring(pt: np.ndarray, ring: np.ndarray) -> bool:
    x, y = pt
    xi, yi = ring[:, 0], ring[:, 1]
    xj, yj = np.roll(xi, 1), np.roll(yi, 1)
    crosses = (yi > y) != (yj > y)
    with np.errstate(divide="ignore", invalid="ignore"):
        xs = (xj - xi) * (y - yi) / (yj - yi) + xi
    return bool(np.count_nonzero(crosses & (x < xs)) % 2)


def drop_outlines_with_parts(records: list[dict]) -> list[dict]:
    """Simple 3D Buildings rule: an outline that contains building:part records is not rendered;
    its parts carry the real shape (towers, domes, naves)."""
    parts = [r for r in records if is_part(r)]
    centers = [np.asarray(r["ring"]).mean(axis=0) for r in parts]
    kept = []
    for r in records:
        if not is_part(r):
            ring = np.asarray(r["ring"])
            lo, hi = ring.min(axis=0), ring.max(axis=0)
            if any((lo <= c).all() and (c <= hi).all() and point_in_ring(c, ring) for c in centers):
                continue
        kept.append(r)
    return kept
