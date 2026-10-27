"""Cut the demo-video material from finished runs (CPU, ffmpeg): clean clips, labelled comps, stills.

  python scripts/demo_material.py --passes out/demo/src --hero out/hero_s2_polish2 --out out/demo

--passes holds present_r1024_361/ and 1519_r1024_361/ (from polish2_inputs.zip); --hero holds the
hero_s2 outputs. Every flight clip is 1920x1080, 24 fps, 361 frames and starts on the same camera
frame, so they cut and dissolve against each other directly in an editor.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

FONT = "data/raw/fonts/SourceSans3-Semibold.ttf"  # OFL, logged in assets.jsonl
W, H, N = 1920, 1080, 361
ENC = ["-c:v", "libx264", "-preset", "slow", "-crf", "16", "-pix_fmt", "yuv420p", "-r", "24"]
WAVE_FRAME = 187  # middle of the time wave (3.5-12.5 s) in the hero


def ff(*args: str) -> None:
    subprocess.run(["ffmpeg", "-v", "error", "-y", *args], check=True, stdin=subprocess.DEVNULL)


def label(text: str, size: int = 40) -> str:
    text = text.replace(":", r"\:").replace("'", r"\'")
    return (f"drawtext=fontfile={FONT}:text='{text}':fontsize={size}:fontcolor=white:x=40:y=h-th-40:"
            "box=1:boxcolor=black@0.55:boxborderw=14")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--passes", type=Path, required=True)
    p.add_argument("--hero", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--skip-clips", action="store_true", help="reuse clips/ and labelled comps from a previous run")
    a = p.parse_args()
    clips, comps, stills = (a.out / d for d in ("clips", "comps", "stills"))
    for d in (clips, comps, stills):
        d.mkdir(parents=True, exist_ok=True)

    src = {
        "depth_present": a.passes / "present_r1024_361" / "control_inverse.mp4",
        "depth_1519": a.passes / "1519_r1024_361" / "control_inverse.mp4",
        "blender_present": a.passes / "present_r1024_361" / "rgb.mp4",
        "blender_1519": a.passes / "1519_r1024_361" / "rgb.mp4",
        "edges_present": a.passes / "present_r1024_361" / "edges.mp4",
        "edges_1519": a.passes / "1519_r1024_361" / "edges.mp4",
        "ltx_present": a.hero / "hero_s2_1519_present.mp4",
        "ltx_city_1519": a.hero / "hero_s2_1519_past_base.mp4",
        "ltx_zones_1519": a.hero / "hero_s2_1519_past_zones.mp4",
        "final_flight": a.hero / "hero_s2_1519.mp4",
    }
    labels = {
        "depth_present": "1  Depth, today: OpenStreetMap buildings on Copernicus DEM, rendered in Blender",
        "depth_1519": "1  Depth, 1519: era geometry from the manifest, rendered in Blender",
        "blender_present": "2  Blender colour pass, today",
        "blender_1519": "2  Blender colour pass, 1519",
        "ltx_present": "3  LTX-2.5 + Union Control IC-LoRA, guided by the depth: today",
        "ltx_city_1519": "3  LTX-2.5, 1519 city (canals, houses, lake)",
        "ltx_zones_1519": "3  LTX-2.5, 1519 dry zones (precinct, plazas, palaces)",
        "final_flight": "4  TimeFlight nodes: era merge + time wave + credits",
    }

    # Clean clips: same size, rate and length, no labels (and the hero's audio on the final flight).
    for i, (name, path) in enumerate(src.items() if not a.skip_clips else [], 1):
        audio = ["-c:a", "aac", "-b:a", "192k", "-t", f"{N / 24}"] if name == "final_flight" else ["-an"]
        ff("-i", str(path), "-frames:v", str(N), "-vf", f"scale={W}:{H}:flags=lanczos,setsar=1",
           *ENC, *audio, str(clips / f"{i:02d}_{name}.mp4"))
    if not a.skip_clips:
        ff("-i", str(src["final_flight"]), "-c", "copy", str(clips / "11_hero_full_with_credits.mp4"))
    clip = {name: clips / f"{i:02d}_{name}.mp4" for i, name in enumerate(src, 1)}

    # Labelled versions of the same clips.
    for name, text in labels.items() if not a.skip_clips else []:
        ff("-i", str(clip[name]), "-vf", label(text), *ENC, "-an", str(comps / f"labelled_{name}.mp4"))

    # Full-screen dissolves along the flight: depth -> Blender colour -> LTX (-> final for 1519).
    def chain(names: list[str], out: Path, seg: float) -> None:
        # Every input runs on the flight clock: trim each to its own window, so the camera never jumps.
        inputs = sum((["-i", str(comps / f"labelled_{n}.mp4")] for n in names), [])
        parts = []
        for k, n in enumerate(names):
            start = k * seg
            parts.append(f"[{k}:v]trim=start={start}:duration={seg + 1},setpts=PTS-STARTPTS,fps=24,settb=AVTB[s{k}]")
        acc, offset = "s0", seg
        for k in range(1, len(names)):
            parts.append(f"[{acc}][s{k}]xfade=transition=fade:duration=1:offset={offset}[x{k}]")
            acc, offset = f"x{k}", offset + seg
        ff(*inputs, "-filter_complex", ";".join(parts), "-map", f"[{acc}]", *ENC, str(out))

    chain(["depth_present", "blender_present", "ltx_present"], comps / "dissolve_today.mp4", 5.0)
    chain(["depth_1519", "blender_1519", "ltx_city_1519", "final_flight"], comps / "dissolve_1519.mp4", 3.75)

    # 2x2 mosaics with labels.
    def mosaic(names: list[str], out: Path) -> None:
        inputs = sum((["-i", str(comps / f"labelled_{n}.mp4")] for n in names), [])
        scaled = ";".join(f"[{k}:v]scale={W // 2}:{H // 2}[m{k}]" for k in range(4))
        ff(*inputs, "-filter_complex", scaled + ";" + "".join(f"[m{k}]" for k in range(4))
           + "xstack=inputs=4:layout=0_0|w0_0|0_h0|w0_h0", *ENC, str(out))

    mosaic(["depth_1519", "blender_1519", "ltx_city_1519", "final_flight"], comps / "mosaic_pipeline_1519.mp4")
    mosaic(["depth_present", "blender_present", "ltx_present", "final_flight"], comps / "mosaic_pipeline_today.mp4")
    mosaic(["ltx_present", "ltx_city_1519", "ltx_zones_1519", "final_flight"], comps / "mosaic_generations.mp4")

    # Stills: the wave, before and after, and the credit cards.
    hero = str(src["final_flight"])
    for name, frame in [("wave_mid", WAVE_FRAME), ("today_start", 30), ("1519_end", 355),
                        ("credits_sources", 400), ("credits_models", 560)]:
        ff("-i", hero, "-vf", f"select=eq(n\\,{frame})", "-frames:v", "1", str(stills / f"{name}.png"))
    for name in ("depth_1519", "blender_1519", "ltx_city_1519"):
        ff("-i", str(clip[name]), "-vf", f"select=eq(n\\,{WAVE_FRAME})", "-frames:v", "1",
           str(stills / f"{name}_f{WAVE_FRAME}.png"))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
