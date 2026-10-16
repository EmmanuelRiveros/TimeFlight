"""Go/no-go metrics for M0 (see docs/smoke_test.md). Pure numpy."""

from __future__ import annotations

import numpy as np


def _ranks(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="stable")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(x.size, dtype=np.float64)
    return ranks


def spearman(a: np.ndarray, b: np.ndarray, max_samples: int = 50_000,
             seed: int = 0) -> float:
    """Spearman rank correlation between two same-shape arrays (ties broken arbitrarily)."""
    a = np.ravel(a).astype(np.float64)
    b = np.ravel(b).astype(np.float64)
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch {a.shape} vs {b.shape}")
    if a.size > max_samples:
        idx = np.random.default_rng(seed).choice(a.size, max_samples, replace=False)
        a, b = a[idx], b[idx]
    ra, rb = _ranks(a), _ranks(b)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / denom) if denom > 0 else 0.0


def structure_scores(pred_depths: np.ndarray, control: np.ndarray) -> np.ndarray:
    """Per-frame Spearman between estimated depth of the output and the control.

    Both must use the same convention (near = high), shape (F, H, W).
    """
    if pred_depths.shape != control.shape:
        raise ValueError(f"Shape mismatch {pred_depths.shape} vs {control.shape}")
    return np.array([spearman(p, c) for p, c in zip(pred_depths, control)])


def mean_frame_diff(frames: np.ndarray) -> float:
    """Mean absolute difference between consecutive frames, frames in [0, 255], (F, H, W[, C])."""
    f = frames.astype(np.float32)
    if f.shape[0] < 2:
        raise ValueError("Need at least 2 frames")
    return float(np.abs(f[1:] - f[:-1]).mean())


def flicker_ratio(output: np.ndarray, reference: np.ndarray) -> float:
    """Frame-to-frame change of the output relative to a stable render of the same motion.

    1.0 means the output changes as much as the reference; much higher means flicker/boiling.
    """
    ref = mean_frame_diff(reference)
    if ref == 0:
        raise ValueError("Reference has no motion")
    return mean_frame_diff(output) / ref


def unstable_fraction(frames: np.ndarray, threshold: float = 20.0, step: int = 1) -> float:
    """Share of pixels that change after motion compensation (flicker, boiling, popping).

    For each frame pair, Farneback optical flow warps frame t+1 back onto frame t. Pixels whose
    forward and backward flows disagree by more than 1 px are occlusions and are skipped.
    Returns the fraction of the remaining pixels that still differ by more than `threshold`
    gray levels. A stable video scores near 0. Gray frames (F, H, W) in [0, 255].
    Needs opencv (the `metrics` extra); imported lazily so the core stays numpy-only.
    """
    import cv2

    f = np.clip(frames, 0, 255).astype(np.uint8)
    h, w = f.shape[1:]
    grid = np.stack(np.meshgrid(np.arange(w), np.arange(h)), axis=-1).astype(np.float32)
    args = (None, 0.5, 4, 21, 3, 5, 1.1, 0)
    bad = total = 0
    for i in range(0, len(f) - 1, step):
        a, b = f[i], f[i + 1]
        fwd = cv2.calcOpticalFlowFarneback(a, b, *args)
        bwd = cv2.calcOpticalFlowFarneback(b, a, *args)
        pos = grid + fwd
        warped = cv2.remap(b, pos[..., 0], pos[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        back = cv2.remap(bwd, pos[..., 0], pos[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        valid = (np.linalg.norm(fwd + back, axis=-1) < 1.0)
        valid &= (pos[..., 0] >= 0) & (pos[..., 0] < w - 1) & (pos[..., 1] >= 0) & (pos[..., 1] < h - 1)
        resid = np.abs(warped.astype(np.float32) - a.astype(np.float32))
        bad += int((resid[valid] > threshold).sum())
        total += int(valid.sum())
    return bad / max(total, 1)


def pop_score(frames: np.ndarray) -> float:
    """Largest consecutive-frame difference over the median one. Spikes (>2) suggest pops."""
    f = frames.astype(np.float32)
    diffs = np.abs(f[1:] - f[:-1]).reshape(len(f) - 1, -1).mean(axis=1)
    return float(diffs.max() / np.median(diffs))


def verdict_u1(scores: np.ndarray, median_min: float = 0.75, frame_min: float = 0.5) -> bool:
    return bool(np.median(scores) >= median_min and scores.min() >= frame_min)


def verdict_u3(ratio: float, max_ratio: float = 1.5) -> bool:
    return ratio <= max_ratio


def verdict_u3_unstable(fraction: float, max_fraction: float = 0.05) -> bool:
    return fraction <= max_fraction
