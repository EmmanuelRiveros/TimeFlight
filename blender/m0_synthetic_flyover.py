# SPDX-License-Identifier: GPL-3.0-or-later
"""M0 smoke-test scene: a procedural lake city seen from a descending drone.

No geodata: this only tests whether the video model can restyle *synthetic* aerial
depth (hard edges, no texture), which is exactly what the real pipeline will feed it.

Usage (Blender 4.4):
  blender -b --factory-startup -P blender/m0_synthetic_flyover.py -- \
      --out out/m0/render --frames 49 --width 512 --height 288

Writes:
  <out>/depth/0001.png ...  16-bit gray, value = planar camera depth / zscale (sky = 1.0)
  <out>/rgb/0001.png ...    Workbench shading of the same motion (flicker reference)
  <out>/render.json         parameters needed to decode the depth
"""

import argparse
import json
import math
import random
import sys
from pathlib import Path

import bpy
from mathutils import Vector


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--frames", type=int, default=49)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--height", type=int, default=288)
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--speed", type=float, default=22.0, help="camera speed, m/s")
    p.add_argument("--zscale", type=float, default=4000.0, help="meters mapped to PNG value 1.0")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--samples", type=int, default=16)
    p.add_argument("--skip-rgb", action="store_true")
    return p.parse_args(argv)


def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    return bpy.context.scene


def add_mesh(name, verts, faces, colors=None):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    if colors is not None:
        attr = mesh.color_attributes.new("Col", "FLOAT_COLOR", "CORNER")
        for poly in mesh.polygons:
            c = colors[poly.index]
            for li in poly.loop_indices:
                attr.data[li].color = (*c, 1.0)
        mesh.color_attributes.active_color = attr
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def ground_z(x, y):
    """Gentle slope rising to the east, flat lake bed to the west."""
    return 0.0 if x < 0 else 0.015 * x + 2.0 * math.sin(y / 90.0)


def build_city(seed):
    rng = random.Random(seed)
    verts, faces, colors = [], [], []

    def box(cx, cy, sx, sy, h, color):
        z0 = ground_z(cx, cy) - 1.0
        base = len(verts)
        for dz in (z0, z0 + h + 1.0):
            verts.extend([(cx - sx, cy - sy, dz), (cx + sx, cy - sy, dz),
                          (cx + sx, cy + sy, dz), (cx - sx, cy + sy, dz)])
        quads = [(0, 1, 2, 3), (4, 7, 6, 5), (0, 4, 5, 1), (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0)]
        for q in quads:
            faces.append(tuple(base + i for i in q))
            colors.append(color)

    # City blocks east of the shore (x > 0), streets every 60 m.
    for bx in range(20, 1400, 60):
        for by in range(-700, 700, 60):
            if abs(by) < 25:  # main avenue along the causeway axis
                continue
            for _ in range(rng.randint(2, 5)):
                sx, sy = rng.uniform(5, 14), rng.uniform(5, 14)
                cx = bx + rng.uniform(sx, 50 - sx)
                cy = by + rng.uniform(sy, 50 - sy)
                shade = rng.uniform(0.45, 0.85)
                box(cx, cy, sx, sy, rng.uniform(4, 22), (shade, shade * 0.95, shade * 0.85))

    # Stepped pyramid at the end of the avenue (Templo Mayor stand-in).
    px, py = 520.0, 0.0
    for i, (half, h) in enumerate([(45, 8), (37, 8), (29, 8), (21, 8), (13, 6)]):
        box(px, py, half, half, 8 * i + h, (0.8, 0.78, 0.7))

    # Causeway across the lake, west of the shore.
    box(-500, 0, 520, 9, 2.0, (0.6, 0.55, 0.45))

    add_mesh("City", verts, faces, colors)

    # Terrain: one grid for land and lake bed.
    tv, tf = [], []
    n, size = 80, 1600.0
    for j in range(n + 1):
        for i in range(n + 1):
            x = -1200 + 2 * size * i / n
            y = -size + 2 * size * j / n
            tv.append((x, y, ground_z(x, y)))
    for j in range(n):
        for i in range(n):
            a = j * (n + 1) + i
            tf.append((a, a + 1, a + n + 2, a + n + 1))
    add_mesh("Ground", tv, tf, [(0.35, 0.33, 0.28)] * len(tf))

    # Water surface over the lake bed.
    add_mesh("Lake", [(-1200, -1600, 1.5), (0, -1600, 1.5), (0, 1600, 1.5), (-1200, 1600, 1.5)],
             [(0, 1, 2, 3)], [(0.25, 0.35, 0.4)])


def add_camera(scene, frames, fps, speed):
    cam_data = bpy.data.cameras.new("Cam")
    cam_data.lens = 24.0
    cam_data.clip_start = 1.0
    cam_data.clip_end = 10000.0
    cam = bpy.data.objects.new("Cam", cam_data)
    scene.collection.objects.link(cam)
    scene.camera = cam

    # Fly east along the causeway towards the pyramid, descending.
    travel = speed * (frames - 1) / fps
    start = Vector((-250.0, 0.0, 140.0))
    end = Vector((-250.0 + travel, 0.0, 140.0 - 0.25 * travel))
    for f in range(frames):
        t = f / max(frames - 1, 1)
        s = t * t * (3 - 2 * t)  # ease in/out
        cam.location = start.lerp(end, s)
        pitch = math.radians(58.0 + 4.0 * s)  # 90 = horizon, 0 = straight down
        yaw = math.radians(-90.0 + 3.0 * math.sin(math.pi * t))  # looking +X
        cam.rotation_euler = (pitch, 0.0, yaw)
        cam.keyframe_insert("location", frame=f + 1)
        cam.keyframe_insert("rotation_euler", frame=f + 1)


def depth_material():
    mat = bpy.data.materials.new("DepthOverride")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    cam = nt.nodes.new("ShaderNodeCameraData")
    div = nt.nodes.new("ShaderNodeMath")
    div.operation = "DIVIDE"
    emit = nt.nodes.new("ShaderNodeEmission")
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(cam.outputs["View Z Depth"], div.inputs[0])
    nt.links.new(div.outputs[0], emit.inputs["Color"])
    nt.links.new(emit.outputs[0], out.inputs["Surface"])
    return mat, div


def configure_common(scene, args):
    scene.frame_start, scene.frame_end = 1, args.frames
    scene.render.fps = args.fps
    scene.render.resolution_x, scene.render.resolution_y = args.width, args.height
    scene.render.resolution_percentage = 100
    scene.render.dither_intensity = 0.0
    scene.render.use_file_extension = True
    img = scene.render.image_settings
    img.file_format = "PNG"
    img.compression = 15


def render_depth(scene, args, out):
    scene.render.engine = "CYCLES"
    scene.cycles.samples = args.samples
    scene.cycles.use_denoising = False
    scene.cycles.max_bounces = 0
    scene.cycles.pixel_filter_type = "BOX"  # no blending across depth edges beyond 1 px
    scene.view_settings.view_transform = "Raw"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    scene.render.image_settings.color_mode = "BW"
    scene.render.image_settings.color_depth = "16"

    world = bpy.data.worlds.new("SkyDepth")
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)  # sky = zscale meters
    bg.inputs["Strength"].default_value = 1.0
    scene.world = world

    mat, div = depth_material()
    div.inputs[1].default_value = args.zscale
    scene.view_layers[0].material_override = mat
    scene.render.filepath = str(out / "depth" / "####")
    bpy.ops.render.render(animation=True)
    scene.view_layers[0].material_override = None


def render_rgb(scene, out):
    scene.render.engine = "BLENDER_WORKBENCH"
    shading = scene.display.shading
    shading.light = "STUDIO"
    shading.color_type = "VERTEX"
    shading.show_shadows = True
    shading.show_cavity = True
    scene.display.render_aa = "8"
    scene.view_settings.view_transform = "Standard"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "8"
    scene.render.filepath = str(out / "rgb" / "####")
    bpy.ops.render.render(animation=True)


def main():
    args = parse_args()
    for name in ("width", "height"):
        if getattr(args, name) % 32:
            raise SystemExit(f"--{name} must be divisible by 32 (LTX constraint)")
    if args.frames % 8 != 1:
        raise SystemExit("--frames must satisfy frames % 8 == 1 (LTX constraint)")

    out = Path(args.out).resolve()
    scene = reset_scene()
    configure_common(scene, args)
    build_city(args.seed)
    add_camera(scene, args.frames, args.fps, args.speed)
    render_depth(scene, args, out)
    if not args.skip_rgb:
        render_rgb(scene, out)

    meta = {"frames": args.frames, "width": args.width, "height": args.height,
            "fps": args.fps, "zscale": args.zscale, "depth": "planar camera Z, value = z / zscale",
            "sky_value": 1.0, "speed_mps": args.speed, "seed": args.seed}
    (out / "render.json").write_text(json.dumps(meta, indent=2))
    print(f"[timeflight] wrote {out}")


if __name__ == "__main__":
    main()
