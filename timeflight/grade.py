"""Finishing grade for generated frames: local contrast ("clarity") and film grain.

Diffusion output tends to look smooth and plastic: midtone texture is weak and flat areas have no noise.
Two cheap, standard finishing steps counter that without touching the model:

- `clarity` adds back the luminance detail above a large blur radius (an unsharp mask at ~1% of the
  frame width), which lifts texture on roofs, paving and walls without sharpening edges into halos.
- `grain` adds monochrome noise, slightly blurred so it has a film-like size, strongest in the
  midtones and new every frame. It is the last step, after any resize.

Pure numpy, one frame at a time.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)  # Rec. 709


@dataclass
class GradeParams:
    clarity: float = 0.3      # amount of large-radius luminance detail added back (0 = off)
    radius: float = 0.01      # clarity blur radius, as a fraction of the frame width
    grain: float = 0.025      # grain standard deviation in the midtones, in 0..1 units (0 = off)
    grain_size: float = 0.7   # grain blur sigma in pixels at 1920 px width (scaled with the frame)


def box_blur(img: np.ndarray, r: int, axis: int) -> np.ndarray:
    """Mean over a window of 2r+1 along `axis`, edges clamped."""
    if r < 1:
        return img
    pad = [(0, 0)] * img.ndim
    pad[axis] = (r + 1, r)
    c = np.cumsum(np.pad(img, pad, mode="edge"), axis=axis, dtype=np.float32)
    n = img.shape[axis]
    hi = np.take(c, np.arange(2 * r + 1, 2 * r + 1 + n), axis=axis)
    lo = np.take(c, np.arange(0, n), axis=axis)
    return (hi - lo) / (2 * r + 1)


def gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    """Approximate gaussian on the first two axes: three box passes per axis."""
    if sigma < 0.5:
        return img
    r = max(1, int(round((np.sqrt(4 * sigma * sigma + 1) - 1) / 2)))  # 3 boxes of 2r+1 ~ sigma
    out = img
    for axis in (0, 1):
        for _ in range(3):
            out = box_blur(out, r, axis)
    return out


def grade_frame(frame: np.ndarray, params: GradeParams, rng: np.random.Generator) -> np.ndarray:
    """uint8 (H, W, 3) -> graded uint8 (H, W, 3)."""
    h, w = frame.shape[:2]
    x = frame.astype(np.float32) / 255.0
    luma = x @ LUMA
    if params.clarity > 0:
        detail = luma - gaussian_blur(luma, params.radius * w)
        x = x + params.clarity * detail[..., None]
        luma = np.clip(luma + params.clarity * detail, 0.0, 1.0)
    if params.grain > 0:
        n = rng.standard_normal((h, w), dtype=np.float32)
        sigma = params.grain_size * w / 1920.0
        if sigma >= 0.5:
            n = gaussian_blur(n, sigma)
            n /= max(float(n.std()), 1e-6)   # keep the requested strength after the blur
        # Midtones get the most grain; deep shadows and highlights much less (as on film).
        amp = params.grain * (0.3 + 2.8 * luma * (1.0 - luma))
        x = x + (n * amp)[..., None]
    return np.clip(x * 255.0 + 0.5, 0, 255).astype(np.uint8)


def grade_frames(frames, params: GradeParams, seed: int = 0):
    """Yield graded frames; the grain is new every frame and reproducible with `seed`."""
    rng = np.random.default_rng(seed)
    for f in frames:
        yield grade_frame(np.asarray(f), params, rng)
