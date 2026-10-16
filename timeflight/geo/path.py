"""Camera path: geographic keyframes -> one camera position and look-at target per frame.

A keyframe is {"lat", "lon", "alt", "look_at": {"lat", "lon", "alt"}}. `alt` is meters above
the origin's ground height. Positions and targets go through a centripetal Catmull-Rom spline,
then the path is re-timed by arc length so the drone moves at a steady speed, with optional
ease-in/out at the ends.
"""

from __future__ import annotations

import numpy as np

from .enu import Origin, geodetic_to_enu


def keyframes_to_enu(keyframes: list[dict], origin: Origin) -> tuple[np.ndarray, np.ndarray]:
    def enu(p):
        return geodetic_to_enu(p["lat"], p["lon"], origin.h + p["alt"], origin)

    pos = np.array([enu(k) for k in keyframes])
    tgt = np.array([enu(k["look_at"]) for k in keyframes])
    return pos, tgt


def catmull_rom(points: np.ndarray, samples_per_segment: int = 200, alpha: float = 0.5) -> np.ndarray:
    """Dense centripetal Catmull-Rom curve through `points` (N, 3); ends are extrapolated."""
    p = np.asarray(points, dtype=np.float64)
    if len(p) < 2:
        raise ValueError("Need at least 2 points")
    ext = np.vstack([2 * p[0] - p[1], p, 2 * p[-1] - p[-2]])
    out = []
    for i in range(len(p) - 1):
        p0, p1, p2, p3 = ext[i:i + 4]
        t0 = 0.0
        t1 = t0 + max(np.linalg.norm(p1 - p0) ** alpha, 1e-6)
        t2 = t1 + max(np.linalg.norm(p2 - p1) ** alpha, 1e-6)
        t3 = t2 + max(np.linalg.norm(p3 - p2) ** alpha, 1e-6)
        last = i == len(p) - 2
        t = np.linspace(t1, t2, samples_per_segment + (1 if last else 0), endpoint=last)[:, None]
        a1 = (t1 - t) / (t1 - t0) * p0 + (t - t0) / (t1 - t0) * p1
        a2 = (t2 - t) / (t2 - t1) * p1 + (t - t1) / (t2 - t1) * p2
        a3 = (t3 - t) / (t3 - t2) * p2 + (t - t2) / (t3 - t2) * p3
        b1 = (t2 - t) / (t2 - t0) * a1 + (t - t0) / (t2 - t0) * a2
        b2 = (t3 - t) / (t3 - t1) * a2 + (t - t1) / (t3 - t1) * a3
        out.append((t2 - t) / (t2 - t1) * b1 + (t - t1) / (t2 - t1) * b2)
    return np.vstack(out)


def ease(t: np.ndarray, amount: float) -> np.ndarray:
    """Blend between linear (0) and smoothstep (1) timing."""
    return (1 - amount) * t + amount * t * t * (3 - 2 * t)


def sample_path(keyframes: list[dict], origin: Origin, frames: int, ease_amount: float = 0.3) -> dict:
    """Per-frame camera positions and targets in ENU meters, plus speed statistics."""
    pos_k, tgt_k = keyframes_to_enu(keyframes, origin)
    pos_d = catmull_rom(pos_k)
    tgt_d = catmull_rom(tgt_k)
    seg = np.linalg.norm(np.diff(pos_d, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    s = ease(np.linspace(0.0, 1.0, frames), ease_amount) * arc[-1]
    idx = np.arange(len(arc))
    u = np.interp(s, arc, idx)  # fractional dense index for each frame

    def at(dense):
        return np.stack([np.interp(u, idx, dense[:, k]) for k in range(3)], axis=1)

    pos, tgt = at(pos_d), at(tgt_d)
    step = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    return {"position": pos, "target": tgt, "length_m": float(arc[-1]),
            "max_step_m": float(step.max()), "mean_step_m": float(step.mean())}
