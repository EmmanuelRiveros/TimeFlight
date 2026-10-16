# SPDX-License-Identifier: GPL-3.0-or-later
"""Shared Blender helpers: fast mesh creation and the depth / RGB passes.

Imported by the scripts in this folder (they add this folder to sys.path).
"""

import bpy
import numpy as np


def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    return bpy.context.scene


def mesh_from_arrays(name, verts, loop_verts, loop_totals, poly_colors=None):
    """One mesh object from flat arrays: verts (N, 3); polygons given as a flat list of vertex
    indices plus the vertex count of each polygon; optional per-polygon RGB colors (P, 3)."""
    verts = np.asarray(verts, dtype=np.float32).reshape(-1, 3)
    loop_verts = np.asarray(loop_verts, dtype=np.int32)
    loop_totals = np.asarray(loop_totals, dtype=np.int32)
    loop_starts = np.concatenate([[0], np.cumsum(loop_totals)[:-1]]).astype(np.int32)

    mesh = bpy.data.meshes.new(name)
    mesh.vertices.add(len(verts))
    mesh.vertices.foreach_set("co", verts.ravel())
    mesh.loops.add(len(loop_verts))
    mesh.loops.foreach_set("vertex_index", loop_verts)
    mesh.polygons.add(len(loop_totals))
    mesh.polygons.foreach_set("loop_start", loop_starts)
    mesh.polygons.foreach_set("loop_total", loop_totals)
    mesh.update(calc_edges=True)
    mesh.validate()
    mesh.shade_flat()  # polygons.add() defaults to smooth; from_pydata() is flat
    if poly_colors is not None:
        colors = np.asarray(poly_colors, dtype=np.float32).reshape(-1, 3)
        per_loop = np.repeat(colors, loop_totals, axis=0)
        rgba = np.concatenate([per_loop, np.ones((len(per_loop), 1), np.float32)], axis=1)
        attr = mesh.color_attributes.new("Col", "FLOAT_COLOR", "CORNER")
        attr.data.foreach_set("color", rgba.ravel())
        mesh.color_attributes.active_color = attr
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def add_mesh(name, verts, faces, colors=None):
    """Convenience wrapper for small meshes given as Python lists of faces."""
    loop_verts = [i for f in faces for i in f]
    totals = [len(f) for f in faces]
    return mesh_from_arrays(name, verts, loop_verts, totals, colors)


def grid_faces(ny, nx):
    """Quad faces for a (ny, nx) vertex grid, row-major."""
    j, i = np.meshgrid(np.arange(ny - 1), np.arange(nx - 1), indexing="ij")
    a = (j * nx + i).ravel()
    quads = np.stack([a, a + 1, a + nx + 1, a + nx], axis=1)
    return quads.ravel(), np.full(len(a), 4)


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


def configure_common(scene, frames, fps, width, height):
    scene.frame_start, scene.frame_end = 1, frames
    scene.render.fps = fps
    scene.render.resolution_x, scene.render.resolution_y = width, height
    scene.render.resolution_percentage = 100
    scene.render.dither_intensity = 0.0
    scene.render.use_file_extension = True
    img = scene.render.image_settings
    img.file_format = "PNG"
    img.compression = 15


def use_gpu(scene):
    """Use OptiX or CUDA for Cycles when available; otherwise stay on CPU. Returns the device type."""
    prefs = bpy.context.preferences.addons["cycles"].preferences
    for kind in ("OPTIX", "CUDA"):
        try:
            prefs.compute_device_type = kind
            prefs.get_devices()
        except TypeError:
            continue
        gpus = [d for d in prefs.devices if d.type == kind]
        if gpus:
            for d in prefs.devices:
                d.use = d.type == kind
            scene.cycles.device = "GPU"
            return kind
    scene.cycles.device = "CPU"
    return "CPU"


def render_depth(scene, out, zscale, samples):
    """16-bit gray PNGs, value = planar camera depth / zscale, sky = 1.0."""
    scene.render.engine = "CYCLES"
    print(f"[timeflight] Cycles device: {use_gpu(scene)}")
    scene.cycles.samples = samples
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
    div.inputs[1].default_value = zscale
    scene.view_layers[0].material_override = mat
    scene.render.filepath = str(out / "depth" / "####")
    bpy.ops.render.render(animation=True)
    scene.view_layers[0].material_override = None


def render_rgb(scene, out):
    """Workbench shading with vertex colors: a stable reference of the same motion."""
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


def check_ltx_size(width, height, frames):
    for name, v in (("width", width), ("height", height)):
        if v % 64:
            raise SystemExit(f"--{name} must be divisible by 64 (LTX: 32, times 2 for the ref0.5 IC-LoRA guide)")
    if frames % 8 != 1:
        raise SystemExit("--frames must satisfy frames % 8 == 1 (LTX constraint)")
