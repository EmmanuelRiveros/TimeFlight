"""Time wave: a sphere of the past era grows from a world point (e.g. the Templo Mayor).

Inside the sphere the scene is in the past era, outside it is the present. Both eras are
rendered along the same camera path, so every pixel of each clip has a known world position
(from the Blender depth pass). Per pixel and frame we decide which layer the camera would see:

- a past surface is real only inside the sphere, a present surface only outside it;
- if both are real, the nearer one wins;
- if neither is (a present building removed, revealing past ground still outside the
  sphere), the past layer is shown, since no render exists for "present without that building".

Everything is soft (a band of `softness_m` meters), so the front is a smooth, glowing edge.
Pure numpy; the camera model matches blender/render_place.py (planar depth, 36 mm sensor on the
larger image dimension as in Blender's AUTO fit, camera up as close to world +Z as possible).
"""

from __future__ import annotations

from dataclasses import dataclass

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


def half_extents(width: int, height: int, lens_mm: float) -> tuple[float, float]:
    """Half image size at planar depth 1. Blender's sensor fit AUTO puts the 36 mm sensor on the larger
    image dimension: the width in landscape, the height in portrait."""
    half_major = SENSOR_WIDTH_MM / 2 / lens_mm
    if width >= height:
        return half_major, half_major * height / width
    return half_major * width / height, half_major


def pixel_rays(width: int, height: int, lens_mm: float) -> tuple[np.ndarray, np.ndarray]:
    """Camera-space offsets per pixel center at planar depth 1: (x right, y up), each (H, W)."""
    half_w, half_h = half_extents(width, height, lens_mm)
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
    half_w, half_h = half_extents(width, height, lens_mm)
    u = ((rel @ right) / z / half_w + 1) / 2 * width
    v = (1 - (rel @ up) / z / half_h) / 2 * height
    return np.stack([u, v, z], axis=-1)


def wave_radius(frame: float, start: float, end: float, r0: float, r1: float) -> float:
    """Front radius at a frame: 0 until `start`, r1 from `end` on. In between it grows continuously
    from 0, about exponentially once it is well above the scale r0 (slow at the center, fast toward
    the horizon, so the wave reaches distant terrain in time) and eased in and out."""
    if frame <= start:
        return 0.0
    u = min((frame - start) / max(end - start, 1e-9), 1.0)
    u = u * u * (3 - 2 * u)  # smoothstep: ease in and out
    return float(r0 * ((1 + r1 / r0) ** u - 1))


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


def front_glow(dist_past: np.ndarray, radius: float, width: float, fade_in_m: float = 80.0) -> np.ndarray:
    """Band of light on the past surfaces at the front, in [0, 1]. It fades in while the radius
    grows to `fade_in_m`, so the first few meters of the wave are not a bright blob."""
    if radius <= 0:
        return np.zeros_like(dist_past, dtype=np.float32)
    fade = min(radius / fade_in_m, 1.0) ** 2 if fade_in_m > 0 else 1.0
    return (fade * np.exp(-(((dist_past - radius) / width) ** 2))).astype(np.float32)


def composite(present: np.ndarray, past: np.ndarray, alpha: np.ndarray, glow: np.ndarray | None = None,
              glow_color=(255, 214, 150), glow_strength: float = 0.6) -> np.ndarray:
    """Blend uint8 RGB frames (H, W, 3) with alpha (H, W); add the glow band on top."""
    out = present.astype(np.float32) * (1 - alpha[..., None]) + past.astype(np.float32) * alpha[..., None]
    if glow is not None:
        g = (glow * glow_strength)[..., None]
        out = out * (1 - g) + np.asarray(glow_color, np.float32) * g
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def resize_mask(m: np.ndarray, height: int, width: int) -> np.ndarray:
    """Bilinear resize of a float mask (h, w) to (height, width), pixel centers aligned. Masks are
    computed at the depth pass resolution and applied to frames of any size (e.g. a 2x upscale)."""
    h, w = m.shape
    if (h, w) == (height, width):
        return m

    def axis(n_out: int, n_in: int):
        x = np.clip((np.arange(n_out) + 0.5) * n_in / n_out - 0.5, 0, n_in - 1)
        i0 = np.floor(x).astype(int)
        i1 = np.minimum(i0 + 1, n_in - 1)
        return i0, i1, (x - i0).astype(np.float32)

    y0, y1, fy = axis(height, h)
    x0, x1, fx = axis(width, w)
    rows = m[y0] * (1 - fy[:, None]) + m[y1] * fy[:, None]
    return (rows[:, x0] * (1 - fx) + rows[:, x1] * fx).astype(np.float32)


def _check_aspect(clip: np.ndarray, depth: np.ndarray) -> None:
    (h, w), (ph, pw) = clip.shape[1:3], depth.shape[1:3]
    if abs(w * ph - pw * h) > 0.01 * pw * h:
        raise ValueError(f"clip {w}x{h} and depth pass {pw}x{ph} differ in aspect")


def rects_mask(xy: np.ndarray, rects, feather: float) -> np.ndarray:
    """Soft membership (..., ) in [0, 1] of points (..., 2) in a union of axis-aligned rects
    [x0, y0, x1, y1]; 0.5 on the edges, `feather` meters wide."""
    out = np.zeros(xy.shape[:-1], np.float32)
    for x0, y0, x1, y1 in rects:
        # Signed distance to the rect, negative inside.
        dx = np.maximum(x0 - xy[..., 0], xy[..., 0] - x1)
        dy = np.maximum(y0 - xy[..., 1], xy[..., 1] - y1)
        sd = np.where((dx > 0) | (dy > 0), np.hypot(np.maximum(dx, 0), np.maximum(dy, 0)), np.maximum(dx, dy))
        out = np.maximum(out, 1.0 / (1.0 + np.exp(np.clip(sd / max(feather, 1e-6), -60, 60))))
    return out


def horizontal_distance(depth: np.ndarray, sky: np.ndarray, position, target, lens_mm: float, center,
                        sky_dist: float) -> np.ndarray:
    """Horizontal distance (m) from `center` (ENU x, y) of each pixel's surface; sky = sky_dist."""
    p = unproject(depth, position, target, lens_mm)
    d = np.hypot(p[..., 0] - center[0], p[..., 1] - center[1])
    d[sky] = sky_dist
    return d


@dataclass
class WaveParams:
    start_s: float = 3.5
    end_s: float = 12.5
    r0: float = 30.0
    r1: float = 15000.0
    softness: float = 25.0
    glow: float = 0.6
    glow_fade: float = 80.0


def timewave_frames(present: np.ndarray, past: np.ndarray, pass_present, pass_past, center, fps: float,
                    params: WaveParams = WaveParams()):
    """Yield composited uint8 frames. `pass_*` are timeflight.passes.RenderPass of the same path; the
    clips may be larger than the passes (same aspect), the masks are then resized to the clips."""
    if present.shape != past.shape or len(present) != pass_present.frames or len(past) != pass_past.frames:
        raise ValueError(f"clips {present.shape} / {past.shape} and passes "
                         f"{pass_present.frames} / {pass_past.frames} frames must match")
    f0, f1 = params.start_s * fps, params.end_s * fps
    sky_dist = params.r1 * 0.8  # the sky changes late, once the wave reaches the horizon
    h, w = present.shape[1:3]
    _check_aspect(present, pass_present.depth)
    for i in range(len(present)):
        r = wave_radius(i, f0, f1, params.r0, params.r1)
        pos, tgt, lens = pass_present.position[i], pass_present.target[i], pass_present.lens_mm
        da = horizontal_distance(pass_present.depth[i], pass_present.sky[i], pos, tgt, lens, center, sky_dist)
        db = horizontal_distance(pass_past.depth[i], pass_past.sky[i], pos, tgt, lens, center, sky_dist)
        za = np.where(pass_present.sky[i], np.inf, pass_present.depth[i])
        zb = np.where(pass_past.sky[i], np.inf, pass_past.depth[i])
        alpha = resize_mask(past_alpha(da, db, za, zb, r, params.softness).astype(np.float32), h, w)
        glow = None
        if params.glow > 0:
            glow = front_glow(db, r, max(params.softness, 0.04 * r), params.glow_fade) * (~pass_past.sky[i])
            glow = resize_mask(glow.astype(np.float32), h, w)
        yield composite(present[i], past[i], alpha, glow, glow_strength=params.glow)


def zone_merge_frames(base: np.ndarray, zones_clip: np.ndarray, pass_era, rects, feather: float = 6.0):
    """Yield uint8 frames: `zones_clip` inside the world rects, `base` elsewhere (same depth path).
    The clips may be larger than the pass (same aspect); the mask is resized to them."""
    if base.shape != zones_clip.shape or len(base) != pass_era.frames:
        raise ValueError(f"clips {base.shape} / {zones_clip.shape} and pass {pass_era.frames} frames must match")
    h, w_px = base.shape[1:3]
    _check_aspect(base, pass_era.depth)
    for i in range(len(base)):
        xy = unproject(pass_era.depth[i], pass_era.position[i], pass_era.target[i], pass_era.lens_mm)[..., :2]
        w = rects_mask(xy, rects, feather)
        w[pass_era.sky[i]] = 0.0
        w = resize_mask(w, h, w_px)
        yield np.clip(base[i] * (1 - w[..., None]) + zones_clip[i] * w[..., None] + 0.5, 0, 255).astype(np.uint8)
