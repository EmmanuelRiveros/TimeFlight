"""Run the full TimeFlight graph (workflows/api/timeflight_flight_api.json) through the Comfy SDK:
three generations, era merge, time wave, captions and credits -> one hero video.

  python app/run_flight.py --renders /content/place --results /content/flight_results
  python app/run_flight.py --dry-run          # print the inputs that would be set

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
from run_m0 import comfy_error, node_ids_by_title  # noqa: E402

WORKFLOW = REPO / "workflows" / "api" / "timeflight_flight_api.json"
OUTPUTS = ("save", "save_present", "save_past_base", "save_past_zones")


def plan(args) -> dict[tuple[str, str], object]:
    """(node title, input name) -> value."""
    texts = json.loads(args.prompts.read_text(encoding="utf-8"))["prompts"]
    place_dir = args.place_dir.as_posix()
    values = {
        ("present_pass", "render_dir"): (args.renders / args.present_folder).as_posix(),
        ("past_pass", "render_dir"): (args.renders / args.era_folder).as_posix(),
        ("present_prompt", "text"): texts[args.present_prompt],
        ("past_base_prompt", "text"): texts[args.past_base_prompt],
        ("past_zones_prompt", "text"): texts[args.past_zones_prompt],
        ("era_merge", "era"): args.era, ("time_wave", "era"): args.era, ("credits", "era"): args.era,
        ("era_merge", "place_dir"): place_dir, ("time_wave", "place_dir"): place_dir,
        ("credits", "place_dir"): place_dir,
        ("save", "filename_prefix"): f"timeflight/flight/{args.name}",
    }
    for b in ("present", "past_base", "past_zones"):
        values[(f"{b}_seed", "noise_seed")] = args.seed
        values[(f"save_{b}", "filename_prefix")] = f"timeflight/flight/{args.name}_{b}"
    return values


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--renders", type=PurePosixPath, default=PurePosixPath("/content/place"))
    p.add_argument("--present-folder", default="present_r1024_361")
    p.add_argument("--era-folder", default="1519_r1024_361")
    p.add_argument("--era", default="1519")
    p.add_argument("--place-dir", type=PurePosixPath, default=PurePosixPath("/content/TimeFlight/places/tenochtitlan_zocalo"),
                   help="places/<id> on the ComfyUI machine (manifest, place.json)")
    p.add_argument("--prompts", type=Path, default=REPO / "places" / "tenochtitlan_zocalo" / "prompts.json")
    p.add_argument("--present-prompt", default="present_2026")
    p.add_argument("--past-base-prompt", default="past_1519")
    p.add_argument("--past-zones-prompt", default="past_1519_plaza2")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--name", default="hero_1519")
    p.add_argument("--results", type=Path, default=REPO / "out" / "flight")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    values = plan(args)
    ids = node_ids_by_title(json.loads(WORKFLOW.read_text()))
    missing = sorted({title for title, _ in values} - set(ids))
    if missing:
        raise SystemExit(f"{WORKFLOW.name} has no nodes titled {missing}: rebuild it with workflows/build_flight.py")
    for (title, name), value in values.items():
        shown = value if not isinstance(value, str) or len(value) < 70 else value[:67] + "..."
        print(f"{title}.{name} = {shown}")
    if args.dry_run:
        return

    if not os.environ.get("COMFY_BASE_URL"):
        raise SystemExit("Set COMFY_BASE_URL (e.g. http://127.0.0.1:8189) so the SDK targets the local proxy")
    from comfy_sdk import Comfy  # imported late so --dry-run works without the SDK

    client = Comfy()
    wf = client.workflows.from_file(str(WORKFLOW))
    for (title, name), value in values.items():
        wf.set_input(ids[title], name, value)
    args.results.mkdir(parents=True, exist_ok=True)
    rec = {"name": args.name, "seed": args.seed, "era": args.era}
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
