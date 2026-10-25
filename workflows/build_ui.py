"""Build the UI-format (canvas) version of the TimeFlight graph, to open in ComfyUI by drag and drop.

  .venv/Scripts/python workflows/build_ui.py                       # writes workflows/ui/timeflight_hero.json
  .venv/Scripts/python workflows/build_ui.py --object-info http://127.0.0.1:8188   # refresh node specs first

The graph is the same one app/run_flight.py runs (workflows/build_flight.py, stage 2, preset hero_s2) with
the hero's prompts and timings filled in, laid out in groups with notes. Widget order comes from the node
specs in workflows/ui/node_specs.json (a subset of a live /object_info with ComfyUI-LTXVideo and the
TimeFlight nodes installed), so this runs offline.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from argparse import Namespace
from pathlib import Path, PurePosixPath

HERE = Path(__file__).parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "app"))
from build_flight import build  # noqa: E402
from run_flight import plan  # noqa: E402

SPECS = HERE / "ui" / "node_specs.json"
OUT = HERE / "ui" / "timeflight_hero.json"
WIDGET_TYPES = {"INT", "FLOAT", "STRING", "BOOLEAN", "COMBO"}
COL_W, GAP = 340, 40

BRANCHES = [  # (prefix, group title, colour)
    ("present", "Generation 1: present day (present-day depth)", "#3f789e"),
    ("past_base", "Generation 2: the 1519 city (1519 depth)", "#a1309b"),
    ("past_zones", "Generation 3: the 1519 dry zones (1519 depth, plaza prompt)", "#8A8"),
]
SHARED = ["transformer", "union_lora", "text_encoder", "video_vae", "audio_vae", "latent_upscaler",
          "negative_prompt", "sampler", "sigmas", "sampler_stage2", "sigmas_stage2"]
PASSES = ["present_pass", "past_pass"]
COMPOSITE = ["era_merge", "time_wave", "credits", "audio_mix", "hero_video", "save"]

NOTES = {
    "intro": """# TimeFlight
A drone flight over a real place that turns back time: Mexico City today → Tenochtitlan in 1519.

1. **Render passes** load Blender depth rendered along real lat/lon/alt keyframes
   (present-day city from OpenStreetMap + Copernicus DEM, and the 1519 city from the era manifest).
2. **Three LTX-2.5 generations** follow that depth through the Union Control IC-LoRA: the layout comes
   from the geometry, the model only renders the look. Stage 2 upscales ×2 in latent space for 1080p.
3. **TimeFlight nodes** merge the two 1519 clips by world zone, grow the time wave from the Templo
   Mayor across the city, and append credits generated from the era manifest.

Run it with `colab/timeflight.ipynb` or `colab/flight.sh` (README). The output is an illustrative AI
reconstruction, not a historical record.""",
    "passes": """## Paths
`render_dir` and `place_dir` are paths on the ComfyUI machine. They are set for Colab
(`/content/...`); change them to your unpacked `polish2_inputs.zip` and your clone of the repo.""",
    "composite": """## TimeFlight compositing (CPU, no GPU time)
- **Era Zone Merge**: city clip + dry-zone clip, merged by the manifest's world zones (6 m seam).
- **Time Wave**: present → 1519 per pixel, along a cylinder growing from the era anchor
  (30 m → 15 km), using both depth passes, with a glow and a dust curtain on the front.
- **Captions and Credits**: sources, data, models and licenses from the manifest and asset log.
- **Audio Crossfade**: the LTX soundtracks crossfade on the wave's timing.""",
}


def is_widget(spec: list) -> bool:
    return isinstance(spec[0], list) or spec[0] in WIDGET_TYPES


def node_size(n_slots: int, n_widgets: int, tall: bool) -> list[int]:
    return [COL_W - 20, 46 + 22 * n_slots + 26 * n_widgets + (160 if tall else 0)]


def to_ui(api: dict, specs: dict) -> dict:
    nodes, links, ui = {}, [], {}
    for nid, node in api.items():
        spec = specs[node["class_type"]]
        order = spec["input_order"].get("required", []) + spec["input_order"].get("optional", [])
        known = {**spec["input"].get("required", {}), **spec["input"].get("optional", {})}
        slots, widgets, tall = [], [], False
        for name in order:
            s, value = known[name], node["inputs"].get(name)
            if is_widget(s) and not isinstance(value, list):
                opts = s[1] if len(s) > 1 and isinstance(s[1], dict) else {}
                widgets.append(opts.get("default", "") if value is None else value)
                tall |= bool(opts.get("multiline"))
                if opts.get("control_after_generate"):
                    widgets.append("fixed")
            else:
                slots.append({"name": name, "type": s[0] if not isinstance(s[0], list) else "COMBO",
                              "link": None, **({"widget": {"name": name}} if is_widget(s) else {})})
        outputs = [{"name": nm, "type": t, "links": [], "slot_index": i}
                   for i, (nm, t) in enumerate(zip(spec["output_name"], spec["output"]))]
        title = node["_meta"]["title"]
        ui[nid] = {"id": int(nid), "type": node["class_type"], "pos": [0, 0],
                   "size": node_size(max(len(slots), len(outputs)), len(widgets), tall),
                   "flags": {}, "order": len(ui), "mode": 0, "title": title, "inputs": slots,
                   "outputs": outputs, "properties": {"Node name for S&R": node["class_type"]},
                   "widgets_values": widgets}
        nodes[title] = ui[nid]
    for nid, node in api.items():
        for slot in ui[nid]["inputs"]:
            value = node["inputs"].get(slot["name"])
            if isinstance(value, list):
                src, idx = value
                link_id = len(links) + 1
                links.append([link_id, int(src), idx, int(nid), ui[nid]["inputs"].index(slot), slot["type"]])
                slot["link"] = link_id
                ui[src]["outputs"][idx]["links"].append(link_id)
    return {"nodes": nodes, "links": links}


def layout(graph: dict) -> tuple[list[dict], list[dict]]:
    """Place nodes in columns inside groups; return (groups, notes)."""
    nodes, groups, notes = graph["nodes"], [], []

    def column(titles: list[str], x: int, y: int) -> int:
        for t in titles:
            nodes[t]["pos"] = [x, y]
            y += nodes[t]["size"][1] + GAP
        return y

    def group(title: str, color: str, members: list[str], pad: int = 30) -> None:
        xs = [nodes[t]["pos"][0] for t in members] + [nodes[t]["pos"][0] + nodes[t]["size"][0] for t in members]
        ys = [nodes[t]["pos"][1] for t in members] + [nodes[t]["pos"][1] + nodes[t]["size"][1] for t in members]
        groups.append({"title": title, "color": color, "font_size": 24,
                       "bounding": [min(xs) - pad, min(ys) - pad - 40, max(xs) - min(xs) + 2 * pad,
                                    max(ys) - min(ys) + 2 * pad + 40]})

    def note(key: str, x: int, y: int, w: int, h: int) -> None:
        notes.append({"type": "MarkdownNote", "pos": [x, y], "size": [w, h], "text": NOTES[key]})

    note("intro", 0, -400, 2 * COL_W - 20, 320)
    shared = [t for t in SHARED if t in nodes]
    column(shared[:6], 0, 80)
    column(shared[6:], COL_W, 80)
    group("Models and sampling", "#444", shared)
    column(PASSES, 2 * COL_W + 60, 80)
    group("Render passes (Blender depth along the real path)", "#b06634", PASSES)
    note("passes", 2 * COL_W + 60, 80 + sum(nodes[t]["size"][1] + GAP for t in PASSES), COL_W - 20, 170)

    x0, y = 3 * COL_W + 160, 80
    for prefix, title, color in BRANCHES:
        members = [t for t in nodes if (t.startswith(prefix + "_") or t == "save_" + prefix) and t not in PASSES]
        cols = [members[i:i + 3] for i in range(0, len(members), 3)]
        bottom = y
        for i, col in enumerate(cols):
            bottom = max(bottom, column(col, x0 + i * COL_W, y))
        group(title, color, members)
        y = bottom + 120
    branch_right = max(n["pos"][0] + n["size"][0] for n in nodes.values())
    xc = branch_right + 160
    column(COMPOSITE[:3], xc, 80)
    column(COMPOSITE[3:], xc + COL_W, 80)
    group("TimeFlight: era merge, time wave, captions and credits", "#b58b2a", COMPOSITE)
    note("composite", xc, -400, 2 * COL_W - 20, 320)
    return groups, notes


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--object-info", help="ComfyUI URL (or a saved /object_info JSON) to refresh "
                   "workflows/ui/node_specs.json from")
    p.add_argument("--out", type=Path, default=OUT)
    args = p.parse_args()

    api = build(stage2=True)
    if args.object_info:
        if Path(args.object_info).is_file():
            info = json.loads(Path(args.object_info).read_text(encoding="utf-8"))
        else:
            with urllib.request.urlopen(args.object_info.rstrip("/") + "/object_info") as r:
                info = json.load(r)
        keep = ("input", "input_order", "output", "output_name", "display_name")
        specs = {c: {k: info[c][k] for k in keep if k in info[c]}
                 for c in sorted({n["class_type"] for n in api.values()})}
        SPECS.write_text(json.dumps(specs, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {SPECS} ({len(specs)} node classes)")
    specs = json.loads(SPECS.read_text(encoding="utf-8"))

    # The hero's values, as app/run_flight.py would set them on Colab.
    run = Namespace(preset="hero_s2", renders=PurePosixPath("/content/place"), present_folder=None,
                    era_folder=None, era="1519", place_dir=PurePosixPath("/content/TimeFlight/places/tenochtitlan_zocalo"),
                    prompts=REPO / "places" / "tenochtitlan_zocalo" / "prompts.json",
                    present_prompt="present_2026", past_base_prompt="past_1519",
                    past_zones_prompt="past_1519_plaza2", seed=42, guide_strength=1.0, name="hero_s2_1519")
    by_title = {n["_meta"]["title"]: n for n in api.values()}
    for (title, name), value in plan(run).items():
        by_title[title]["inputs"][name] = value

    graph = to_ui(api, specs)
    groups, notes = layout(graph)
    nodes = sorted(graph["nodes"].values(), key=lambda n: n["id"])
    next_id = max(n["id"] for n in nodes) + 1
    for i, nt in enumerate(notes):
        nodes.append({"id": next_id + i, "type": nt["type"], "pos": nt["pos"], "size": nt["size"], "flags": {},
                      "order": len(nodes), "mode": 0, "inputs": [], "outputs": [], "properties": {},
                      "widgets_values": [nt["text"]], "color": "#432", "bgcolor": "#653"})
    wf = {"last_node_id": max(n["id"] for n in nodes), "last_link_id": len(graph["links"]),
          "nodes": nodes, "links": graph["links"], "groups": groups, "config": {},
          "extra": {"ds": {"scale": 0.35, "offset": [200, 700]}}, "version": 0.4}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(wf, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {args.out} ({len(graph['nodes'])} nodes, {len(graph['links'])} links, {len(groups)} groups)")


if __name__ == "__main__":
    main()
