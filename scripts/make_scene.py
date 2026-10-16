"""Place data -> Blender-ready scene: terrain grid, building prisms and per-frame camera.

  python scripts/make_scene.py places/tenochtitlan_zocalo/place.json --frames 241

Reads data/places/<id>/{dem.npz,buildings.json} (from scripts/fetch_geodata.py) and writes
data/places/<id>/scene/:
  terrain.npy    (ny, nx, 3) ENU vertices, float32
  buildings.npz  flat arrays: ring vertices, ring lengths, base z, heights
  camera.json    per-frame position and look-at target (ENU), fps, lens
All coordinates are ENU meters around the origin (x east, y north, z up).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from timeflight.geo.dem import Dem, grey_opening, terrain_grid
from timeflight.geo.enu import A, Origin, enu_to_geodetic
from timeflight.geo.osm import building_height, drop_outlines_with_parts
from timeflight.geo.path import sample_path

REPO = Path(__file__).resolve().parents[1]


def load_dem(path: Path, opening_px: int) -> Dem:
    d = np.load(path)
    heights = d["heights"].astype(np.float64)
    if opening_px > 1:
        heights = grey_opening(heights, opening_px)  # the DSM includes buildings; OSM supplies them
    return Dem(heights, float(d["lon0"]), float(d["dlon"]), float(d["lat0"]), float(d["dlat"]))


def building_arrays(buildings: list[dict], dem: Dem, origin: Origin, radius_m: float) -> dict:
    rings, lengths, base, height = [], [], [], []
    for b in buildings:
        ring = np.asarray(b["ring"])
        c = ring.mean(axis=0)
        if np.hypot(*c) > radius_m:
            continue
        lat, lon, _ = enu_to_geodetic(np.array([*c, 0.0]), origin)
        ground = float(dem.sample(lat, lon)) - origin.h - (c @ c) / (2 * A)  # local curvature drop
        rings.append(ring)
        lengths.append(len(ring))
        base.append(ground)
        height.append(b["height"])
    return {"ring_xy": np.vstack(rings).astype(np.float32), "ring_len": np.array(lengths, np.int32),
            "base_z": np.array(base, np.float32), "height": np.array(height, np.float32)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("place", type=Path)
    p.add_argument("--frames", type=int, default=241)
    p.add_argument("--lens", type=float, default=24.0, help="focal length, mm (36 mm sensor width)")
    args = p.parse_args()

    place = json.loads(args.place.read_text(encoding="utf-8"))
    data = REPO / "data" / "places" / place["id"]
    out = data / "scene"
    out.mkdir(parents=True, exist_ok=True)
    tcfg = place["terrain"]

    raw = np.load(data / "dem.npz")
    raw_dem = Dem(raw["heights"].astype(np.float64), float(raw["lon0"]), float(raw["dlon"]),
                  float(raw["lat0"]), float(raw["dlat"]))
    o = place["origin"]
    origin = Origin(o["lat"], o["lon"], float(raw_dem.sample(o["lat"], o["lon"])))
    dem = load_dem(data / "dem.npz", tcfg["dsm_opening_px"])
    # Ground height under the origin after removing buildings from the DSM.
    origin = Origin(origin.lat, origin.lon, float(dem.sample(origin.lat, origin.lon)))

    grid = terrain_grid(dem, origin, tcfg["half_size_m"], tcfg["step_m"])
    np.save(out / "terrain.npy", grid.astype(np.float32))

    blds = json.loads((data / "buildings.json").read_text(encoding="utf-8"))["buildings"]
    blds = drop_outlines_with_parts(blds)
    for b in blds:  # heights from the current rules, so old extracts stay usable
        b["height"] = building_height(b["tags"])[0]
    arrays = building_arrays(blds, dem, origin, tcfg.get("buildings_radius_m", 3000))
    np.savez_compressed(out / "buildings.npz", **arrays)

    cam = place["path"]
    path = sample_path(cam["keyframes"], origin, args.frames, cam.get("ease", 0.3))
    speed = path["mean_step_m"] * cam["fps"]
    (out / "camera.json").write_text(json.dumps({
        "fps": cam["fps"], "frames": args.frames, "lens_mm": args.lens,
        "origin": {"lat": origin.lat, "lon": origin.lon, "h": origin.h},
        "length_m": round(path["length_m"], 1), "mean_speed_mps": round(speed, 1),
        "position": path["position"].round(3).tolist(), "target": path["target"].round(3).tolist(),
    }), encoding="utf-8")
    print(f"terrain {grid.shape[:2]}, z {grid[..., 2].min():.0f}..{grid[..., 2].max():.0f} m; "
          f"{len(arrays['height'])} buildings; path {path['length_m']:.0f} m at {speed:.0f} m/s "
          f"over {args.frames / cam['fps']:.1f} s; origin ground {origin.h:.1f} m")


if __name__ == "__main__":
    main()
