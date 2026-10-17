"""Turn raw render passes into control signals for the video model.

Blender writes planar camera depth (meters) as 16-bit PNG: value = z / zscale.
The control video uses a *fixed* near/far for the whole clip, so the brightness of
a surface only changes when its distance changes (no per-frame renormalization).
Convention matches Video-Depth-Anything "gray": near = bright, far / sky = dark.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
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


@dataclass
class RenderPass:
    """A Blender render folder: planar depth per frame plus the camera that made it."""
    depth: np.ndarray        # (F, H, W) float32 meters; sky pixels = zscale
    sky: np.ndarray          # (F, H, W) bool
    position: np.ndarray     # (F, 3) ENU
    target: np.ndarray       # (F, 3) ENU
    lens_mm: float
    origin: dict             # {"lat", "lon", "h"}
    fps: float
    era: str

    @property
    def frames(self) -> int:
        return len(self.depth)


def load_pass(folder: Path, camera_json: Path | None = None) -> RenderPass:
    """Read depth/####.png and render.json. Older renders without a "camera" entry need the
    place's camera.json (the window starts at render.json "start")."""
    from PIL import Image  # lazy: only loaders need PIL

    folder = Path(folder)
    meta = json.loads((folder / "render.json").read_text())
    cam = meta.get("camera")
    if cam is None:
        if camera_json is None:
            raise ValueError(f"{folder}/render.json has no camera; pass the place's camera.json")
        full = json.loads(Path(camera_json).read_text())
        a, b = meta["start"], meta["start"] + meta["frames"]
        cam = {"position": full["position"][a:b], "target": full["target"][a:b]}
    files = sorted((folder / "depth").glob("*.png"))
    if len(files) != meta["frames"]:
        raise ValueError(f"{folder}: {len(files)} depth frames, render.json says {meta['frames']}")
    depth = np.stack([decode_depth_png(np.asarray(Image.open(f)), meta["zscale"]) for f in files])
    return RenderPass(depth=depth, sky=depth >= meta["zscale"] * 0.999,
                      position=np.asarray(cam["position"], float), target=np.asarray(cam["target"], float),
                      lens_mm=float(meta["lens_mm"]), origin=meta["origin"], fps=float(meta["fps"]),
                      era=meta.get("era", "present"))
