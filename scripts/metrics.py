"""Compute M0 go/no-go metrics (U1 structure, U3 flicker) for every run in runs.jsonl.

  python scripts/metrics.py --results out/m0/results --renders /content/m0

U1 runs Depth-Anything-V2-Small (believed Apache-2.0, check docs/decisions.md) on the output.
It is used only for evaluation, never in the pipeline. Spearman is rank-based, so the
"inverse" and "linear" control encodings are compared fairly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from timeflight import metrics
from timeflight.video import read_video

DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"


def estimate_depth(frames: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """RGB uint8 (F, H, W, 3) -> relative inverse depth (F, h, w), near = high."""
    import torch
    from PIL import Image
    from transformers import pipeline

    pipe = pipeline("depth-estimation", model=DEPTH_MODEL,
                    device=0 if torch.cuda.is_available() else -1)
    out = []
    for f in frames:
        pred = pipe(Image.fromarray(f))["predicted_depth"]
        pred = torch.nn.functional.interpolate(pred[None, None] if pred.ndim == 2 else pred[None],
                                               size=(size[1], size[0]), mode="bilinear")
        out.append(pred.squeeze().cpu().numpy())
    return np.stack(out)


def evaluate(run: dict, renders: Path, stride: int) -> dict:
    output = Path(run["outputs"][0])
    folder = renders / run["folder"]
    control = read_video(folder / f"control_{run['mode']}.mp4", gray=True)
    h, w = control.shape[1:]
    out_rgb = read_video(output, size=(w, h))
    n = min(len(out_rgb), len(control))
    idx = np.arange(0, n, stride)

    scores = metrics.structure_scores(estimate_depth(out_rgb[idx], (w, h)),
                                      control[idx].astype(np.float32))
    ref = read_video(folder / "rgb.mp4", gray=True)
    out_gray = read_video(output, gray=True, size=(w, h))
    m = min(len(ref), len(out_gray))
    ratio = metrics.flicker_ratio(out_gray[:m], ref[:m])
    return {"name": run["name"], "u1_median": round(float(np.median(scores)), 3),
            "u1_min": round(float(scores.min()), 3), "u1_pass": metrics.verdict_u1(scores),
            "u3_ratio": round(ratio, 3), "u3_pass": metrics.verdict_u3(ratio),
            "wall_s": run.get("wall_s")}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--renders", type=Path, required=True)
    p.add_argument("--stride", type=int, default=4, help="evaluate every Nth frame for U1")
    args = p.parse_args()

    runs = [json.loads(l) for l in (args.results / "runs.jsonl").read_text().splitlines() if l.strip()]
    done = set()
    metrics_log = args.results / "metrics.jsonl"
    if metrics_log.exists():
        done = {json.loads(l)["name"] for l in metrics_log.read_text().splitlines() if l.strip()}

    for run in runs:
        if run.get("error") or not run.get("outputs") or run["name"] in done:
            continue
        rec = evaluate(run, args.renders, args.stride)
        print(json.dumps(rec), flush=True)
        with metrics_log.open("a") as f:
            f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
