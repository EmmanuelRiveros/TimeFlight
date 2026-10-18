"""Minimal video I/O through the ffmpeg/ffprobe CLIs (no Python video deps)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np


def probe(path: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=width,height,r_frame_rate,nb_read_frames",
         "-of", "json", str(path)],
        check=True, capture_output=True, text=True,
    ).stdout
    s = json.loads(out)["streams"][0]
    num, den = s["r_frame_rate"].split("/")
    return {"width": int(s["width"]), "height": int(s["height"]),
            "fps": float(num) / float(den), "frames": int(s["nb_read_frames"])}


def read_video(path: Path, gray: bool = False, size: tuple[int, int] | None = None) -> np.ndarray:
    """Decode to uint8 array (F, H, W) if gray else (F, H, W, 3). Optional resize (w, h)."""
    info = probe(path)
    w, h = size or (info["width"], info["height"])
    pix, ch = ("gray", 1) if gray else ("rgb24", 3)
    cmd = ["ffmpeg", "-v", "error", "-i", str(path)]
    if size:
        cmd += ["-vf", f"scale={w}:{h}:flags=area"]
    cmd += ["-f", "rawvideo", "-pix_fmt", pix, "-"]
    raw = subprocess.run(cmd, check=True, capture_output=True).stdout
    frames = np.frombuffer(raw, np.uint8).reshape(-1, h, w, ch)
    return frames[..., 0] if gray else frames


def write_video(frames: np.ndarray, path: Path, fps: float, crf: int = 10) -> None:
    """Encode uint8 frames (F, H, W) or (F, H, W, 3) to H.264 MP4 (yuv420p for compatibility)."""
    if frames.dtype != np.uint8:
        raise ValueError("frames must be uint8")
    f, h, w = frames.shape[:3]
    pix = "gray" if frames.ndim == 3 else "rgb24"
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", pix,
         "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
         "-c:v", "libx264", "-crf", str(crf), "-preset", "slow",
         "-pix_fmt", "yuv420p", str(path)],
        input=np.ascontiguousarray(frames).tobytes(), check=True,
    )


def iter_frames(path: Path):
    """Decode an RGB video one uint8 (H, W, 3) frame at a time (constant memory for long 1080p clips)."""
    info = probe(path)
    w, h = info["width"], info["height"]
    proc = subprocess.Popen(["ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                            stdout=subprocess.PIPE)
    try:
        while True:
            raw = proc.stdout.read(w * h * 3)
            if len(raw) < w * h * 3:
                break
            yield np.frombuffer(raw, np.uint8).reshape(h, w, 3)
    finally:
        if proc.poll() is None:  # stopped early: end ffmpeg instead of letting it hit a broken pipe
            proc.kill()
        proc.stdout.close()
        proc.wait()
