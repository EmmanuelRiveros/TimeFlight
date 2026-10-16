# SPDX-License-Identifier: GPL-3.0-or-later
"""Render the depth and RGB passes of a real place along its geographic camera path.

Usage (Blender 4.4):
  blender -b --factory-startup -P blender/render_place.py -- \
      --scene data/places/tenochtitlan_zocalo/scene --out out/m1/r576_49 \
      --frames 49 --width 576 --height 320

The scene folder comes from scripts/make_scene.py. --frames may be smaller than the number of
camera frames in camera.json: the render then covers frames [--start, --start + --frames) of the
real path, at real speed (cheap tests keep the same motion per frame as the final clip).

Writes <out>/depth/####.png (16-bit, value = planar depth / zscale, sky = 1.0),
<out>/rgb/####.png and <out>/render.json.
"""

import argparse
import json
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tf_render import (check_ltx_size, configure_common, grid_faces,  # noqa: E402
                       mesh_from_arrays, render_depth, render_rgb, reset_scene)


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--scene", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--frames", type=int, default=49)
    p.add_argument("--start", type=int, default=0, help="first camera frame (0-based)")
    p.add_argument("--width", type=int, default=576)
    p.add_argument("--height", type=int, default=320)
    p.add_argument("--zscale", type=float, default=40000.0, help="meters mapped to PNG value 1.0")
    p.add_argument("--samples", type=int, default=16)
    p.add_argument("--skip-rgb", action="store_true")
    return p.parse_args(argv)


def build_terrain(scene_dir):
    grid = np.load(scene_dir / "terrain.npy")
    ny, nx = grid.shape[:2]
    loops, totals = grid_faces(ny, nx)
    z = grid[..., 2].reshape(-1)
    zq = np.stack([z[loops[0::4]], z[loops[1::4]], z[loops[2::4]], z[loops[3::4]]], axis=1).mean(axis=1)
    t = np.clip((zq - zq.min()) / max(np.ptp(zq), 1.0), 0, 1)[:, None]
    colors = (1 - t) * np.array([0.42, 0.40, 0.34]) + t * np.array([0.30, 0.36, 0.26])
    mesh_from_arrays("Terrain", grid.reshape(-1, 3), loops, totals, colors)


def build_buildings(scene_dir):
    b = np.load(scene_dir / "buildings.npz")
    xy, lens, base, height = b["ring_xy"], b["ring_len"], b["base_z"], b["height"]
    verts, loops, totals, colors = [], [], [], []
    rng = np.random.default_rng(0)
    offset, v0 = 0, 0
    for n, z0, h in zip(lens, base, height):
        ring = xy[offset:offset + n]
        offset += n
        bottom = np.column_stack([ring, np.full(n, z0 - 2.0)])  # sink 2 m to hide DEM error
        top = np.column_stack([ring, np.full(n, z0 + h)])
        verts += [bottom, top]
        idx = np.arange(n)
        nxt = (idx + 1) % n
        sides = np.stack([v0 + idx, v0 + nxt, v0 + n + nxt, v0 + n + idx], axis=1)
        loops += [sides.ravel(), v0 + n + idx]
        totals += [np.full(n, 4), [n]]
        shade = rng.uniform(0.6, 0.9)
        colors += [np.tile([shade, shade * 0.96, shade * 0.9], (n + 1, 1))]
        v0 += 2 * n
    mesh_from_arrays("Buildings", np.vstack(verts), np.concatenate(loops),
                     np.concatenate(totals), np.vstack(colors))
    return len(lens)


def add_camera(scene, cam_json, start, frames):
    pos = np.asarray(cam_json["position"])[start:start + frames]
    tgt = np.asarray(cam_json["target"])[start:start + frames]
    if len(pos) < frames:
        raise SystemExit(f"camera.json has {len(cam_json['position'])} frames; need {start + frames}")

    data = bpy.data.cameras.new("Cam")
    data.lens = cam_json["lens_mm"]
    data.sensor_width = 36.0
    data.clip_start = 1.0
    data.clip_end = 60000.0
    cam = bpy.data.objects.new("Cam", data)
    scene.collection.objects.link(cam)
    scene.camera = cam
    cam.rotation_mode = "QUATERNION"
    for f in range(frames):
        p, t = Vector(pos[f]), Vector(tgt[f])
        cam.location = p
        cam.rotation_quaternion = (t - p).to_track_quat("-Z", "Y")
        cam.keyframe_insert("location", frame=f + 1)
        cam.keyframe_insert("rotation_quaternion", frame=f + 1)
    return pos


def main():
    args = parse_args()
    check_ltx_size(args.width, args.height, args.frames)
    scene_dir = Path(args.scene).resolve()
    out = Path(args.out).resolve()
    cam_json = json.loads((scene_dir / "camera.json").read_text())
    fps = cam_json["fps"]

    scene = reset_scene()
    configure_common(scene, args.frames, fps, args.width, args.height)
    build_terrain(scene_dir)
    n_buildings = build_buildings(scene_dir)
    add_camera(scene, cam_json, args.start, args.frames)
    render_depth(scene, out, args.zscale, args.samples)
    if not args.skip_rgb:
        render_rgb(scene, out)

    meta = {"frames": args.frames, "width": args.width, "height": args.height, "fps": fps,
            "start": args.start, "path_frames": cam_json["frames"], "zscale": args.zscale,
            "depth": "planar camera Z, value = z / zscale", "sky_value": 1.0,
            "origin": cam_json["origin"], "lens_mm": cam_json["lens_mm"], "buildings": n_buildings}
    (out / "render.json").write_text(json.dumps(meta, indent=2))
    print(f"[timeflight] wrote {out}")


if __name__ == "__main__":
    main()
