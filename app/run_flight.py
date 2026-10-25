"""Run the full TimeFlight graph (workflows/api/timeflight_flight_api.json) through the Comfy SDK:
three generations, era merge, time wave, captions and credits -> one hero video.

  python app/run_flight.py --renders /content/place --results /content/flight_results
  python app/run_flight.py --dry-run          # print the inputs that would be set
  python app/run_flight.py --preset teaser     # 9:16, 10 s
  python app/run_flight.py --preset hero_s2    # 1920x1080: x2 latent upscale + 3 refine steps (stage 2)

Needs ComfyUI on :8188 with the TimeFlight nodes installed (colab/setup_comfy.sh), comfy-api-proxy on
:8189 and COMFY_BASE_URL=http://127.0.0.1:8189. Render folders are read by the TimeFlight nodes on the
ComfyUI machine, so --renders is a path on that machine.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path, PurePosixPath

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "app"))
from run_m0 import check_size, comfy_error, node_ids_by_title  # noqa: E402

WORKFLOWS = {1: REPO / "workflows" / "api" / "timeflight_flight_api.json",
             2: REPO / "workflows" / "api" / "timeflight_flight_s2_api.json"}
OUTPUTS = ("save", "save_present", "save_past_base", "save_past_zones")

# Output formats. Times are seconds into the clip. Render folders come from blender/render_place.py
# (+ scripts/make_control.py); the teaser flies the last 10 s of the hero path in portrait. `size` is the
# stage 1 size (= the depth passes); stage 2 presets generate at twice that and resize to `out_size`.
PRESETS = {
    "hero_s2": dict(size=(1024, 576), frames=361, folders=("present_r1024_361", "1519_r1024_361"),
                    wave=(3.5, 12.5), present_caption=(0.5, 3.3), past_caption=(12.0, 15.0), card_s=6.0,
                    stages=2, out_size=(1920, 1080)),
    # Decode check: 121 frames would be 2 temporal tiles at the old 64-frame setting; with one tile
    # there must be no soft frame at 56, and this shows decode memory without the temporal split.
    "hero_s2_seams_test": dict(size=(576, 320), frames=121, folders=("present_r576x320_121", "1519_r576x320_121"),
                               wave=(1.0, 4.0), present_caption=(0.0, 0.8), past_caption=(4.0, 5.0), card_s=1.0,
                               stages=2, out_size=(1152, 640)),
    # Small end-to-end check of stage 2 (last 2 s of the path) before the full hero.
    "hero_s2_test": dict(size=(576, 320), frames=49, folders=("present_r576x320_49", "1519_r576x320_49"),
                         wave=(0.3, 1.5), present_caption=(0.0, 0.3), past_caption=(1.5, 2.0), card_s=1.0,
                         stages=2, out_size=(1152, 640)),
    "hero": dict(size=(1024, 576), frames=361, folders=("present_r1024_361", "1519_r1024_361"),
                 wave=(3.5, 12.5), present_caption=(0.5, 3.3), past_caption=(12.0, 15.0), card_s=6.0),
    "teaser": dict(size=(576, 1024), frames=241, folders=("present_v576x1024_241", "1519_v576x1024_241"),
                   wave=(1.5, 7.0), present_caption=(0.2, 1.4), past_caption=(7.0, 10.0), card_s=4.0),
    # Small end-to-end check of the portrait graph before the teaser (CLAUDE.md: test small first).
    "teaser_test": dict(size=(320, 576), frames=49, folders=("present_v320x576_49", "1519_v320x576_49"),
                        wave=(0.3, 1.5), present_caption=(0.0, 0.3), past_caption=(1.5, 2.0), card_s=1.0),
}


def plan(args) -> dict[tuple[str, str], object]:
    """(node title, input name) -> value."""
    texts = json.loads(args.prompts.read_text(encoding="utf-8"))["prompts"]
    place_dir = args.place_dir.as_posix()
    pre = PRESETS[args.preset]
    (w, h), n = pre["size"], pre["frames"]
    check_size(w, h, n)
    present_folder, era_folder = pre["folders"]
    values = {
        ("audio_mix", "flight_frames"): n,
        ("audio_mix", "start_s"): pre["wave"][0], ("audio_mix", "end_s"): pre["wave"][1],
        ("present_pass", "render_dir"): (args.renders / (args.present_folder or present_folder)).as_posix(),
        ("past_pass", "render_dir"): (args.renders / (args.era_folder or era_folder)).as_posix(),
        ("time_wave", "start_s"): pre["wave"][0], ("time_wave", "end_s"): pre["wave"][1],
        ("credits", "present_start_s"): pre["present_caption"][0], ("credits", "present_end_s"): pre["present_caption"][1],
        ("credits", "past_start_s"): pre["past_caption"][0], ("credits", "past_end_s"): pre["past_caption"][1],
        ("credits", "card_seconds"): pre["card_s"],
        ("present_prompt", "text"): texts[args.present_prompt],
        ("past_base_prompt", "text"): texts[args.past_base_prompt],
        ("past_zones_prompt", "text"): texts[args.past_zones_prompt],
        ("era_merge", "era"): args.era, ("time_wave", "era"): args.era, ("credits", "era"): args.era,
        ("era_merge", "place_dir"): place_dir, ("time_wave", "place_dir"): place_dir,
        ("credits", "place_dir"): place_dir,
        ("save", "filename_prefix"): f"timeflight/flight/{args.name}",
    }
    for b in ("present", "past_base", "past_zones"):
        values[(f"{b}_latent", "width")] = w
        values[(f"{b}_latent", "height")] = h
        values[(f"{b}_latent", "length")] = n
        values[(f"{b}_audio_latent", "frames_number")] = n
        values[(f"{b}_seed", "noise_seed")] = args.seed
        values[(f"{b}_guide", "strength")] = args.guide_strength
        values[(f"save_{b}", "filename_prefix")] = f"timeflight/flight/{args.name}_{b}"
    if pre.get("stages", 1) == 2:
        values[("credits", "width")], values[("credits", "height")] = pre["out_size"]
    return values


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--renders", type=PurePosixPath, default=PurePosixPath("/content/place"))
    p.add_argument("--preset", choices=sorted(PRESETS), default="hero")
    p.add_argument("--present-folder", help="override the preset's present-day render folder")
    p.add_argument("--era-folder", help="override the preset's era render folder")
    p.add_argument("--era", default="1519")
    p.add_argument("--place-dir", type=PurePosixPath, default=PurePosixPath(REPO.as_posix()) / "places" / "tenochtitlan_zocalo",
                   help="places/<id> on the ComfyUI machine (manifest, place.json)")
    p.add_argument("--prompts", type=Path, default=REPO / "places" / "tenochtitlan_zocalo" / "prompts.json")
    p.add_argument("--present-prompt", default="present_2026")
    p.add_argument("--past-base-prompt", default="past_1519")
    p.add_argument("--past-zones-prompt", default="past_1519_plaza2")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--guide-strength", type=float, default=1.0,
                   help="depth guide strength for all three generations (below 1 leaves the model more freedom)")
    p.add_argument("--name", help="output name (default: <preset>_<era>)")
    p.add_argument("--results", type=Path, default=REPO / "out" / "flight")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    args.name = args.name or f"{args.preset}_{args.era}"
    values = plan(args)
    workflow = WORKFLOWS[PRESETS[args.preset].get("stages", 1)]
    print(f"workflow: {workflow.name}")
    ids = node_ids_by_title(json.loads(workflow.read_text()))
    missing = sorted({title for title, _ in values} - set(ids))
    if missing:
        raise SystemExit(f"{workflow.name} has no nodes titled {missing}: rebuild it with workflows/build_flight.py")
    for (title, name), value in values.items():
        shown = value if not isinstance(value, str) or len(value) < 70 else value[:67] + "..."
        print(f"{title}.{name} = {shown}")
    if args.dry_run:
        return

    if not os.environ.get("COMFY_BASE_URL"):
        raise SystemExit("Set COMFY_BASE_URL (e.g. http://127.0.0.1:8189) so the SDK targets the local proxy")
    from comfy_sdk import Comfy  # imported late so --dry-run works without the SDK

    client = Comfy()
    wf = client.workflows.from_file(str(workflow))
    for (title, name), value in values.items():
        wf.set_input(ids[title], name, value)
    args.results.mkdir(parents=True, exist_ok=True)
    rec = {"name": args.name, "preset": args.preset, "seed": args.seed, "era": args.era,
           "guide_strength": args.guide_strength,
           "prompts": [args.present_prompt, args.past_base_prompt, args.past_zones_prompt]}
    t0 = time.time()
    try:
        job = client.run(wf)
    except Exception as e:
        rec |= {"error": f"{type(e).__name__}: {e}", "comfy_error": comfy_error()}
        print(json.dumps(rec), flush=True)
        print(rec["comfy_error"], flush=True)
        sys.exit(1)
    rec["wall_s"] = round(time.time() - t0, 1)
    saved = []
    for title in OUTPUTS:
        for i, output in enumerate(job.get_outputs(ids[title])):
            suffix = Path(getattr(output, "name", "") or "out.mp4").suffix or ".mp4"
            path = args.results / f"{args.name}{'' if title == 'save' else title[4:]}{'' if i == 0 else f'_{i}'}{suffix}"
            output.to_file(str(path))
            saved.append(str(path))
    rec["outputs"] = saved
    print(json.dumps(rec), flush=True)
    with (args.results / "runs.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
