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

from timeflight.era.build import anchor_xy
from timeflight.geo.enu import Origin
from timeflight.passes import load_pass
from timeflight.timewave import WaveParams, timewave_frames
from timeflight.video import probe, read_video, write_video

REPO = Path(__file__).resolve().parents[1]


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
    p.add_argument("--curtain", type=float, default=0.0, help="dust curtain opacity on the front (0 = none)")
    p.add_argument("--curtain-height", type=float, default=80.0, help="curtain height above the ground, m")
    p.add_argument("--sheet", type=Path, help="also write a contact sheet of 8 frames")
    args = p.parse_args()

    manifest = json.loads((args.place.parent / "eras" / f"{args.era}.json").read_text(encoding="utf-8"))
    camera_json = REPO / "data" / "places" / json.loads(args.place.read_text(encoding="utf-8"))["id"] / "scene" / "camera.json"
    pa, pb = load_pass(args.present_depth, camera_json), load_pass(args.past_depth, camera_json)
    center = anchor_xy(manifest, Origin(**pa.origin))
    present, past = read_video(args.present), read_video(args.past)
    fps, n = probe(args.present)["fps"], len(present)
    params = WaveParams(args.start_s, args.end_s, args.r0, args.r1, args.softness, args.glow, args.glow_fade,
                        curtain=args.curtain, curtain_height=args.curtain_height)
    out = np.stack(list(timewave_frames(present, past, pa, pb, center, fps, params)))
    write_video(out, args.out, fps, crf=12)
    print(f"wrote {args.out}")
    if args.sheet:
        idx = np.linspace(0, n - 1, 8).astype(int)
        rows = [np.concatenate([out[a], out[b]], 1) for a, b in zip(idx[0::2], idx[1::2])]
        Image.fromarray(np.concatenate(rows, 0)).resize((1024, 1152)).save(args.sheet, quality=88)
        print(f"wrote {args.sheet}")


if __name__ == "__main__":
    main()
