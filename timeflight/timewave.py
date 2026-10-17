"""Time wave: a sphere of the past era grows from a world point (e.g. the Templo Mayor).

Inside the sphere the scene is in the past era, outside it is the present. Both eras are
rendered along the same camera path, so every pixel of each clip has a known world position
(from the Blender depth pass). Per pixel and frame we decide which layer the camera would see:

- a past surface is real only inside the sphere, a present surface only outside it;
- if both are real, the nearer one wins;
- if neither is (a present building removed, revealing past ground still outside the
  sphere), the past layer is shown, since no render exists for "present without that building".

Everything is soft (a band of `softness_m` meters), so the front is a smooth, glowing edge.
Pure numpy; the camera model matches blender/render_place.py (planar depth, 36 mm sensor
width, horizontal sensor fit, camera up as close to world +Z as possible).
"""

from __future__ import annotations

import numpy as np

SENSOR_WIDTH_MM = 36.0


def camera_basis(position: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Forward, right, up unit vectors (ENU) of a camera looking from position to target."""
    fwd = np.asarray(target, float) - np.asarray(position, float)
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0.0, 0.0, 1.0])
    n = np.linalg.norm(right)
    if n < 1e-9:
        raise ValueError("camera looks straight up or down: roll is undefined")
    right /= n
    up = np.cross(right, fwd)
    return fwd, right, up


def pixel_rays(width: int, height: int, lens_mm: float) -> tuple[np.ndarray, np.ndarray]:
    """Camera-space offsets per pixel center at planar depth 1: (x right, y up), each (H, W)."""
    half_w = SENSOR_WIDTH_MM / 2 / lens_mm
    half_h = half_w * height / width
    x = ((np.arange(width) + 0.5) / width * 2 - 1) * half_w
    y = (1 - (np.arange(height) + 0.5) / height * 2) * half_h
    return np.meshgrid(x, y)


def unproject(depth: np.ndarray, position, target, lens_mm: float) -> np.ndarray:
    """Planar depth (H, W) in meters -> world ENU points (H, W, 3)."""
    h, w = depth.shape
    fwd, right, up = camera_basis(np.asarray(position, float), np.asarray(target, float))
    x, y = pixel_rays(w, h, lens_mm)
    d = depth[..., None]
    return np.asarray(position, float) + d * (fwd + x[..., None] * right + y[..., None] * up)


def project(points: np.ndarray, position, target, lens_mm: float, width: int, height: int) -> np.ndarray:
    """World ENU points (..., 3) -> pixel coordinates (..., 2) and planar depth (...,) as (..., 3)."""
    fwd, right, up = camera_basis(np.asarray(position, float), np.asarray(target, float))
    rel = np.asarray(points, float) - np.asarray(position, float)
    z = rel @ fwd
    half_w = SENSOR_WIDTH_MM / 2 / lens_mm
    half_h = half_w * height / width
    u = ((rel @ right) / z / half_w + 1) / 2 * width
    v = (1 - (rel @ up) / z / half_h) / 2 * height
    return np.stack([u, v, z], axis=-1)


def wave_radius(frame: float, start: float, end: float, r0: float, r1: float) -> float:
    """Front radius at a frame: 0 before `start`, then geometric growth r0 -> r1 (slow at the
    center, fast at the horizon, so the wave reaches distant terrain in time), r1 after `end`."""
    if frame < start:
        return 0.0
    u = min((frame - start) / max(end - start, 1e-9), 1.0)
    u = u * u * (3 - 2 * u)  # smoothstep: ease in and out
    return float(r0 * (r1 / r0) ** u)


def _inside(dist: np.ndarray, radius: float, softness: float) -> np.ndarray:
    if radius <= 0:
        return np.zeros_like(dist, dtype=np.float32)
    return (1.0 / (1.0 + np.exp(np.clip((dist - radius) / softness, -60, 60)))).astype(np.float32)


def past_alpha(dist_present: np.ndarray, dist_past: np.ndarray, z_present: np.ndarray, z_past: np.ndarray,
               radius: float, softness: float = 30.0) -> np.ndarray:
    """Weight of the past layer per pixel, in [0, 1] (see the module docstring for the rule)."""
    a = _inside(dist_past, radius, softness)              # past surface is real
    b = 1.0 - _inside(dist_present, radius, softness)     # present surface is real
    past_nearer = z_past <= z_present
    return np.where(past_nearer, 1.0 - (1.0 - a) * b, 1.0 - b).astype(np.float32)


def front_glow(dist_past: np.ndarray, radius: float, width: float) -> np.ndarray:
    """Band of light on the past surfaces at the front, in [0, 1]."""
    if radius <= 0:
        return np.zeros_like(dist_past, dtype=np.float32)
    return np.exp(-(((dist_past - radius) / width) ** 2)).astype(np.float32)


def composite(present: np.ndarray, past: np.ndarray, alpha: np.ndarray, glow: np.ndarray | None = None,
              glow_color=(255, 214, 150), glow_strength: float = 0.6) -> np.ndarray:
    """Blend uint8 RGB frames (H, W, 3) with alpha (H, W); add the glow band on top."""
    out = present.astype(np.float32) * (1 - alpha[..., None]) + past.astype(np.float32) * alpha[..., None]
    if glow is not None:
        g = (glow * glow_strength)[..., None]
        out = out * (1 - g) + np.asarray(glow_color, np.float32) * g
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)
