"""Merge two generations of the same era clip by world zone: the era's dry zones (precinct, plazas,
named structures) come from one clip, the rest of the city (canals, houses, lake) from the other.

  .venv/Scripts/python scripts/era_merge.py \
      --base out/m3/clips/P_past_1519_s1.00_inverse_1519_r1024_361.mp4 \
      --zones out/m3/full/P_past_1519_plaza2_s1.00_inverse_1519_r1024_361.mp4 \
      --depth out/m2/r1024_361 --place places/tenochtitlan_zocalo/place.json --era 1519 \
      --out out/m4/1519_merged.mp4 --sheet out/m4/1519_merged_sheet.jpg

Both clips follow the same depth video, so a world position per pixel (from the Blender depth
pass) is valid for both. The seam sits on the precinct wall and structure margins.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from timeflight.era.build import dry_zones
from timeflight.geo.enu import Origin
from timeflight.passes import load_pass
from timeflight.timewave import zone_merge_frames
from timeflight.video import probe, read_video, write_video

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base", type=Path, required=True, help="clip for everything outside the zones")
    p.add_argument("--zones", type=Path, required=True, help="clip for the dry zones")
    p.add_argument("--depth", type=Path, required=True, help="era render folder (depth/ + render.json)")
    p.add_argument("--place", type=Path, required=True)
    p.add_argument("--era", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--feather", type=float, default=6.0, help="soft seam width, m")
    p.add_argument("--zone-kinds", nargs="+", default=["precinct", "plazas", "structures"])
    p.add_argument("--sheet", type=Path)
    args = p.parse_args()

    manifest = json.loads((args.place.parent / "eras" / f"{args.era}.json").read_text(encoding="utf-8"))
    camera_json = REPO / "data" / "places" / json.loads(args.place.read_text(encoding="utf-8"))["id"] / "scene" / "camera.json"
    rp = load_pass(args.depth, camera_json)
    zones = dry_zones(manifest, Origin(**rp.origin))
    rects = [r for k in args.zone_kinds for r in zones[k]]
    ia = probe(args.base)
    out = np.stack(list(zone_merge_frames(read_video(args.base), read_video(args.zones), rp, rects, args.feather)))
    write_video(out, args.out, ia["fps"], crf=12)
    print(f"wrote {args.out}")
    if args.sheet:
        idx = [0, 120, 200, 260, 310, 360]
        rows = [np.concatenate([out[a], out[b]], 1) for a, b in zip(idx[0::2], idx[1::2])]
        Image.fromarray(np.concatenate(rows, 0)).resize((1280, 1080)).save(args.sheet, quality=88)
        print(f"wrote {args.sheet}")


if __name__ == "__main__":
    main()
