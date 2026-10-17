"""Composite a present-day clip and a past clip with a time wave growing from a world point.

  .venv/Scripts/python scripts/timewave.py \
      --present out/m3/clips/P_present_2026_s1.00_inverse_present_r1024_361.mp4 \
      --past out/m3/full/P_past_1519_plaza2_s1.00_inverse_1519_r1024_361.mp4 \
      --present-depth out/m1/r1024_361 --past-depth out/m2/r1024_361 \
      --place places/tenochtitlan_zocalo/place.json --era 1519 --out out/m4/timewave.mp4

Both clips must come from the same camera path (same frames, size and start). The depth folders
are the Blender render folders (depth/####.png + render.json). The wave center defaults to the era
manifest's anchor (the Templo Mayor for 1519). Distance is horizontal, so the front is a vertical
cylinder: a tower changes era all at once instead of being sliced.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from timeflight.geo.enu import Origin, geodetic_to_enu
from timeflight.passes import decode_depth_png
from timeflight.timewave import composite, front_glow, past_alpha, unproject, wave_radius
from timeflight.video import probe, read_video, write_video

REPO = Path(__file__).resolve().parents[1]


def load_depth(folder: Path, frame: int) -> tuple[np.ndarray, np.ndarray]:
    meta = json.loads((folder / "render.json").read_text())
    z = decode_depth_png(np.asarray(Image.open(folder / "depth" / f"{frame + 1:04d}.png")), meta["zscale"])
    return z, z >= meta["zscale"] * 0.999


def horizontal_distance(z, sky, pos, tgt, lens, center, sky_dist):
    p = unproject(z, pos, tgt, lens)
    d = np.hypot(p[..., 0] - center[0], p[..., 1] - center[1])
    d[sky] = sky_dist
    return d


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--present", type=Path, required=True)
    p.add_argument("--past", type=Path, required=True)
    p.add_argument("--present-depth", type=Path, required=True)
    p.add_argument("--past-depth", type=Path, required=True)
    p.add_argument("--place", type=Path, required=True)
    p.add_argument("--era", required=True, help="era manifest name; its anchor is the wave center")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--start-s", type=float, default=3.5, help="wave starts (s)")
    p.add_argument("--end-s", type=float, default=12.5, help="wave covers everything (s)")
    p.add_argument("--r0", type=float, default=30.0, help="radius when the wave starts, m")
    p.add_argument("--r1", type=float, default=15000.0, help="radius when it ends, m")
    p.add_argument("--softness", type=float, default=25.0, help="width of the soft edge, m")
    p.add_argument("--glow", type=float, default=0.6, help="front glow strength (0 = none)")
    p.add_argument("--glow-fade", type=float, default=80.0, help="radius at which the glow reaches full strength, m")
    p.add_argument("--sheet", type=Path, help="also write a contact sheet of 8 frames")
    args = p.parse_args()

    place = json.loads(args.place.read_text(encoding="utf-8"))
    manifest = json.loads((args.place.parent / "eras" / f"{args.era}.json").read_text(encoding="utf-8"))
    cam = json.loads((REPO / "data" / "places" / place["id"] / "scene" / "camera.json").read_text())
    origin = Origin(**cam["origin"])
    center = geodetic_to_enu(manifest["anchor"]["lat"], manifest["anchor"]["lon"], origin.h, origin)[:2]

    info_a, info_b = probe(args.present), probe(args.past)
    if (info_a["width"], info_a["height"], info_a["frames"]) != (info_b["width"], info_b["height"], info_b["frames"]):
        raise SystemExit(f"clips differ: {info_a} vs {info_b}")
    fps, n = info_a["fps"], info_a["frames"]
    start = json.loads((args.present_depth / "render.json").read_text())["start"]
    present, past = read_video(args.present), read_video(args.past)

    out = np.empty_like(present)
    f0, f1 = args.start_s * fps, args.end_s * fps
    for i in range(n):
        f = start + i
        pos, tgt = cam["position"][f], cam["target"][f]
        r = wave_radius(i, f0, f1, args.r0, args.r1)
        za, sa = load_depth(args.present_depth, i)
        zb, sb = load_depth(args.past_depth, i)
        sky_dist = args.r1 * 0.8  # the sky changes late, once the wave reaches the horizon
        da = horizontal_distance(za, sa, pos, tgt, cam["lens_mm"], center, sky_dist)
        db = horizontal_distance(zb, sb, pos, tgt, cam["lens_mm"], center, sky_dist)
        za[sa], zb[sb] = np.inf, np.inf
        alpha = past_alpha(da, db, za, zb, r, args.softness)
        glow = front_glow(db, r, max(args.softness, 0.04 * r), args.glow_fade) * (~sb) if args.glow > 0 else None
        out[i] = composite(present[i], past[i], alpha, glow, glow_strength=args.glow)
        if i % 60 == 0:
            print(f"frame {i}: radius {r:.0f} m, past share {alpha.mean():.2f}", flush=True)

    write_video(out, args.out, fps, crf=12)
    print(f"wrote {args.out}")
    if args.sheet:
        idx = np.linspace(0, n - 1, 8).astype(int)
        rows = [np.concatenate([out[a], out[b]], 1) for a, b in zip(idx[0::2], idx[1::2])]
        Image.fromarray(np.concatenate(rows, 0)).resize((1024, 1152)).save(args.sheet, quality=88)
        print(f"wrote {args.sheet}")


if __name__ == "__main__":
    main()
