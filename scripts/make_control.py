"""Blender render folder -> control videos for ComfyUI.

  .venv/Scripts/python scripts/make_control.py out/m0/r512 --modes inverse linear

Writes next to the render:
  control_<mode>.mp4   gray depth control (fixed near/far for the whole clip, near = bright)
  edges.mp4            depth-discontinuity edges (log-depth gradient)
  rgb.mp4              Workbench reference of the same motion (for the flicker metric)
  control.json         near/far and settings, so results are reproducible
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from timeflight.passes import (decode_depth_png, depth_to_control, edges_from_depth,
                               fixed_range, to_uint8)
from timeflight.video import write_video


def load_sequence(folder: Path) -> list[np.ndarray]:
    files = sorted(folder.glob("*.png"))
    if not files:
        raise SystemExit(f"No PNGs in {folder}")
    return [np.array(Image.open(f)) for f in files]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("render_dir", type=Path)
    p.add_argument("--modes", nargs="+", default=["inverse", "linear"], choices=["inverse", "linear"])
    p.add_argument("--near", type=float, help="override near (m)")
    p.add_argument("--far", type=float, help="override far (m)")
    args = p.parse_args()

    meta = json.loads((args.render_dir / "render.json").read_text())
    zscale, fps = meta["zscale"], meta["fps"]
    depths = [decode_depth_png(d, zscale) for d in load_sequence(args.render_dir / "depth")]
    near, far = fixed_range(depths, sky_depth=zscale)
    near, far = args.near or near, args.far or far

    for mode in args.modes:
        ctrl = np.stack([to_uint8(depth_to_control(z, near, far, mode)) for z in depths])
        write_video(ctrl, args.render_dir / f"control_{mode}.mp4", fps)

    write_video(np.stack([edges_from_depth(z) for z in depths]), args.render_dir / "edges.mp4", fps)

    rgb_dir = args.render_dir / "rgb"
    if rgb_dir.exists():
        write_video(np.stack(load_sequence(rgb_dir))[..., :3], args.render_dir / "rgb.mp4", fps)

    info = {"near_m": near, "far_m": far, "modes": args.modes, "frames": len(depths),
            "fps": fps, "width": meta["width"], "height": meta["height"]}
    (args.render_dir / "control.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info))


if __name__ == "__main__":
    main()
