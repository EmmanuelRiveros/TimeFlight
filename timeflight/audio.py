"""Audio for the time wave: the present-day soundtrack crossfades into the past one while the wave
grows, then fades out before the credit cards. Pure numpy; waveforms are float (channels, samples)."""

from __future__ import annotations

import numpy as np


def smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def mix_eras(present: np.ndarray, past: np.ndarray, sample_rate: int, start_s: float, end_s: float,
             total_s: float, clip_s: float, fade_out_s: float = 1.5) -> np.ndarray:
    """Equal-power crossfade present -> past between start_s and end_s (smoothstep, as the wave),
    a fade-out over the last `fade_out_s` of the clip (clip_s long), then silence up to total_s."""
    if present.ndim != 2 or past.ndim != 2:
        raise ValueError("waveforms must be (channels, samples)")
    channels = max(len(present), len(past))
    n_clip, n_total = round(clip_s * sample_rate), round(total_s * sample_rate)

    def fit(x: np.ndarray) -> np.ndarray:
        x = np.broadcast_to(x, (channels, x.shape[1])) if len(x) == 1 else x
        out = np.zeros((channels, n_clip), np.float32)
        n = min(n_clip, x.shape[1])
        out[:, :n] = x[:, :n]
        return out

    t = np.arange(n_clip) / sample_rate
    u = smoothstep((t - start_s) / max(end_s - start_s, 1e-6)) * (np.pi / 2)
    mixed = fit(present) * np.cos(u) + fit(past) * np.sin(u)
    if fade_out_s > 0:
        mixed *= np.clip((clip_s - t) / fade_out_s, 0.0, 1.0)
    out = np.zeros((channels, max(n_total, n_clip)), np.float32)
    out[:, :n_clip] = mixed
    return np.clip(out[:, :n_total], -1.0, 1.0)
