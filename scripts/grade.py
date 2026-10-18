"""Apply the finishing grade (timeflight/grade.py: clarity + film grain) to a video, keeping its audio.

  .venv/Scripts/python scripts/grade.py out/flight_results_t/flight_results/hero_s2_1519.mp4 \
      --out out/grade/hero_s2_graded.mp4 --compare out/grade/hero_s2_compare.mp4

--compare also writes the original (top) over the graded clip (bottom), each scaled to half height, for
review. Frames are streamed, so a 1080p hero never sits in memory whole.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from timeflight.grade import GradeParams, grade_frames
from timeflight.video import iter_frames, probe


def encode(frames, path: Path, width: int, height: int, fps: float, audio_from: Path | None, crf: int) -> int:
    """Stream uint8 frames to H.264 (yuv420p), muxing the audio of `audio_from` if it has any."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
           "-r", str(fps), "-i", "-"]
    if audio_from:
        cmd += ["-i", str(audio_from), "-map", "0:v", "-map", "1:a?", "-c:a", "copy"]
    cmd += ["-c:v", "libx264", "-crf", str(crf), "-preset", "slow", "-pix_fmt", "yuv420p", str(path)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    n = 0
    for f in frames:
        proc.stdin.write(f.tobytes())
        n += 1
    proc.stdin.close()
    if proc.wait() != 0:
        raise SystemExit(f"ffmpeg failed writing {path}")
    return n


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("video", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--compare", type=Path, help="also write original (top) over graded (bottom)")
    p.add_argument("--clarity", type=float, default=GradeParams.clarity)
    p.add_argument("--radius", type=float, default=GradeParams.radius, help="clarity radius, fraction of width")
    p.add_argument("--grain", type=float, default=GradeParams.grain)
    p.add_argument("--grain-size", type=float, default=GradeParams.grain_size, help="px at 1920 wide")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--crf", type=int, default=14, help="x264 CRF; low, so the grain survives the encode")
    args = p.parse_args()

    info = probe(args.video)
    params = GradeParams(args.clarity, args.radius, args.grain, args.grain_size)
    n = encode(grade_frames(iter_frames(args.video), params, args.seed), args.out, info["width"], info["height"],
               info["fps"], args.video, args.crf)
    print(f"wrote {args.out} ({n} frames, {params})")
    if args.compare:
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(args.video), "-i", str(args.out), "-filter_complex",
                        "[0:v]scale=-2:ih/2[a];[1:v]scale=-2:ih/2[b];[a][b]vstack", "-map", "1:a?",
                        "-c:v", "libx264", "-crf", str(args.crf), "-pix_fmt", "yuv420p", str(args.compare)],
                       check=True)
        print(f"wrote {args.compare}")


if __name__ == "__main__":
    main()
