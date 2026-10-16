"""M0 smoke-test driver: runs the test matrix through the Comfy SDK.

Needs ComfyUI on :8188, comfy-api-proxy on :8189, and COMFY_BASE_URL=http://127.0.0.1:8189.

  python app/run_m0.py --step A --renders /content/m0          # 12 drafts, 576x320x49
  python app/run_m0.py --step B --strength 1.0 --mode inverse  # 3 prompts, 1024x576x121
  python app/run_m0.py --step C --strength 1.0 --mode inverse --prompt past_1519
  python app/run_m0.py --step A --dry-run                      # print the plan only

`--renders` holds one folder per render (r576_49, r1024_121, r1024_241) as produced by
scripts/make_control.py. Results go to out/m0/results/, one line per run in runs.jsonl.
Exits with code 1 if any variant failed; failures include ComfyUI's exception message.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import shutil
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
WORKFLOW = REPO / "workflows" / "api" / "m0_union_depth_api.json"
PROMPTS = REPO / "places" / "m0_synthetic" / "prompts.json"

STEPS = {
    # step: (render folder, width, height, frames)
    "A": ("r576_49", 576, 320, 49),
    "B": ("r1024_121", 1024, 576, 121),
    "C": ("r1024_241", 1024, 576, 241),
    # Real place (M1 passes, 15 s hero length): --renders out/m1 --prompts places/<id>/prompts.json
    "P": ("r1024_361", 1024, 576, 361),
}

# The ref0.5 Union IC-LoRA has latent_downscale_factor 2: the guide is VAE-encoded at half the
# output size, which must itself be divisible by 32. 288 and 544 fail inside the VAE encoder.
SIZE_MULTIPLE = 64


def check_size(width: int, height: int, frames: int) -> None:
    if width % SIZE_MULTIPLE or height % SIZE_MULTIPLE:
        raise ValueError(f"{width}x{height}: width and height must be multiples of {SIZE_MULTIPLE}")
    if frames % 8 != 1:
        raise ValueError(f"{frames} frames: LTX needs frames % 8 == 1")


def comfy_error() -> str:
    """Exception message and last traceback lines of the newest failed prompt, read from ComfyUI."""
    url = os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188")
    try:
        with urllib.request.urlopen(f"{url}/history?max_items=5", timeout=10) as r:
            history = json.load(r)
    except OSError as e:
        return f"(could not read {url}/history: {e})"
    for entry in reversed(list(history.values())):
        for kind, data in entry.get("status", {}).get("messages", []):
            if kind == "execution_error":
                tb = "".join(data.get("traceback", [])[-6:])
                return f"{data.get('node_type')}: {data.get('exception_message', '').strip()}\n{tb}"
    return "(no execution_error in ComfyUI history)"


def plan(step: str, prompts: list[str], strengths: list[float], modes: list[str]) -> list[dict]:
    folder, w, h, n = STEPS[step]
    check_size(w, h, n)
    return [
        {"step": step, "prompt": p, "strength": s, "mode": m, "folder": folder,
         "width": w, "height": h, "frames": n,
         "name": f"{step}_{p}_s{s:.2f}_{m}"}
        for p, s, m in itertools.product(prompts, strengths, modes)
    ]


def node_ids_by_title(workflow: dict) -> dict[str, str]:
    ids = {}
    for node_id, node in workflow.items():
        title = node.get("_meta", {}).get("title")
        if title in ids:
            raise ValueError(f"Duplicate node title {title!r}")
        ids[title] = node_id
    return ids


def run_variant(client, v: dict, renders: Path, results: Path, texts: dict,
                comfy_input_dir: Path | None, seed: int) -> dict:
    ids = node_ids_by_title(json.loads(WORKFLOW.read_text()))
    wf = client.workflows.from_file(str(WORKFLOW))

    control = renders / v["folder"] / f"control_{v['mode']}.mp4"
    if not control.exists():
        raise FileNotFoundError(control)
    if comfy_input_dir:
        # Fallback when asset upload through the proxy is unavailable: same machine, copy the file.
        target = comfy_input_dir / f"tf_{v['folder']}_{v['mode']}.mp4"
        shutil.copy2(control, target)
        wf.set_input(ids["depth_video"], "file", target.name)
    else:
        wf.set_input(ids["depth_video"], "file", client.assets.from_file(str(control)))

    wf.set_input(ids["positive_prompt"], "text", texts["prompts"][v["prompt"]])
    wf.set_input(ids["negative_prompt"], "text", texts.get("negative", ""))
    wf.set_input(ids["video_latent"], "width", v["width"])
    wf.set_input(ids["video_latent"], "height", v["height"])
    wf.set_input(ids["video_latent"], "length", v["frames"])
    wf.set_input(ids["audio_latent"], "frames_number", v["frames"])
    wf.set_input(ids["guide"], "strength", v["strength"])
    wf.set_input(ids["seed"], "noise_seed", seed)
    wf.set_input(ids["save"], "filename_prefix", f"timeflight/m0/{v['name']}")

    t0 = time.time()
    job = client.run(wf)
    wall = time.time() - t0

    saved = []
    for i, output in enumerate(job.get_outputs(ids["save"])):
        suffix = Path(getattr(output, "name", "") or "out.mp4").suffix or ".mp4"
        path = results / f"{v['name']}{'' if i == 0 else f'_{i}'}{suffix}"
        output.to_file(str(path))
        saved.append(str(path))
    return {**v, "seed": seed, "wall_s": round(wall, 1), "outputs": saved}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--step", choices=sorted(STEPS), required=True)
    p.add_argument("--renders", type=Path, default=REPO / "out" / "m0")
    p.add_argument("--prompts", type=Path, default=PROMPTS, help="prompts.json (default: M0 synthetic)")
    p.add_argument("--results", type=Path, default=REPO / "out" / "m0" / "results")
    p.add_argument("--prompt", nargs="+", help="prompt keys (default: all)")
    p.add_argument("--strength", nargs="+", type=float, help="guide strengths (default A: 1.0 0.7)")
    p.add_argument("--mode", nargs="+", choices=["inverse", "linear"], help="depth encodings (default A: both)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--comfy-input-dir", type=Path, help="copy inputs here instead of uploading assets")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    texts = json.loads(args.prompts.read_text(encoding="utf-8"))
    prompts = args.prompt or list(texts["prompts"])
    strengths = args.strength or ([1.0, 0.7] if args.step == "A" else [1.0])
    modes = args.mode or (["inverse", "linear"] if args.step == "A" else ["inverse"])
    variants = plan(args.step, prompts, strengths, modes)

    for v in variants:
        print(v["name"])
    if args.dry_run:
        return

    if not os.environ.get("COMFY_BASE_URL"):
        raise SystemExit("Set COMFY_BASE_URL (e.g. http://127.0.0.1:8189) so the SDK targets the local proxy")
    from comfy_sdk import Comfy  # imported late so --dry-run works without the SDK

    client = Comfy()
    args.results.mkdir(parents=True, exist_ok=True)
    log = args.results / "runs.jsonl"
    done = set()
    if log.exists():
        done = {r["name"] for r in map(json.loads, log.read_text().splitlines()) if not r.get("error")}
    failed = 0
    for v in variants:
        if v["name"] in done:
            print(f"skip {v['name']} (already in runs.jsonl)")
            continue
        print(f"-> {v['name']}", flush=True)
        try:
            rec = run_variant(client, v, args.renders, args.results, texts, args.comfy_input_dir, args.seed)
        except Exception as e:  # keep going; one failed cell should not kill the matrix
            rec = {**v, "error": f"{type(e).__name__}: {e}", "comfy_error": comfy_error()}
            failed += 1
        print(json.dumps(rec), flush=True)
        if rec.get("comfy_error"):
            print(rec["comfy_error"], flush=True)
        with log.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
    if failed:
        print(f"{failed} of {len(variants)} variants failed", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
