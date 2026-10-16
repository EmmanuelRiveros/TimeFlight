"""Turn raw render passes into control signals for the video model.

Blender writes planar camera depth (meters) as 16-bit PNG: value = z / zscale.
The control video uses a *fixed* near/far for the whole clip, so the brightness of
a surface only changes when its distance changes (no per-frame renormalization).
Convention matches Video-Depth-Anything "gray": near = bright, far / sky = dark.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

DepthMode = Literal["inverse", "linear"]


def decode_depth_png(values: np.ndarray, zscale: float) -> np.ndarray:
    """uint16 PNG values -> depth in meters."""
    return values.astype(np.float32) / 65535.0 * zscale


def fixed_range(depths: list[np.ndarray] | np.ndarray, sky_depth: float,
                lo_pct: float = 0.5, hi_pct: float = 99.5) -> tuple[float, float]:
    """Near/far for a whole sequence from robust percentiles, ignoring sky pixels."""
    stack = np.concatenate([np.ravel(d) for d in depths])
    geo = stack[stack < sky_depth * 0.999]
    if geo.size == 0:
        raise ValueError("No geometry pixels found (everything is sky)")
    near, far = np.percentile(geo, [lo_pct, hi_pct])
    return float(max(near, 1e-3)), float(far)


def depth_to_control(z: np.ndarray, near: float, far: float,
                     mode: DepthMode = "inverse") -> np.ndarray:
    """Depth in meters -> float32 control in [0, 1], near = 1."""
    if not 0 < near < far:
        raise ValueError(f"Need 0 < near < far, got {near}, {far}")
    z = np.clip(z.astype(np.float32), near, far)
    if mode == "inverse":
        # Disparity-like, like monocular depth estimators: more detail up close.
        c = (1.0 / z - 1.0 / far) / (1.0 / near - 1.0 / far)
    elif mode == "linear":
        c = (far - z) / (far - near)
    else:
        raise ValueError(f"Unknown mode {mode!r}")
    return np.clip(c, 0.0, 1.0).astype(np.float32)


def edges_from_depth(z: np.ndarray, threshold: float = 0.02) -> np.ndarray:
    """Binary depth-discontinuity edges (uint8 0/255) from depth in meters.

    Uses the gradient of log depth, i.e. relative depth change, so a 3 m wall reads the
    same at 100 m and at 1000 m.
    """
    gy, gx = np.gradient(np.log(np.maximum(z.astype(np.float32), 1e-3)))
    mag = np.hypot(gx, gy)
    return np.where(mag > threshold, 255, 0).astype(np.uint8)


def to_uint8(control: np.ndarray) -> np.ndarray:
    return np.round(np.clip(control, 0.0, 1.0) * 255.0).astype(np.uint8)
