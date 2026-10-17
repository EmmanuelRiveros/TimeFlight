"""Generate colab/timeflight_m0.ipynb (keeps the notebook diff-able: edit this file, not the .ipynb).

  .venv/Scripts/python colab/make_notebook.py
"""

import json
from pathlib import Path

CELLS = [
    ("md", """# TimeFlight M0: aerial smoke test (go/no-go)
Protocol and pass criteria: `docs/smoke_test.md`. **Hard cap: 10 GPU hours.** The timer cell prints the hours used.

Before running:
1. Colab secrets: `HF_TOKEN` (with access granted to `Lightricks/LTX-2.5` and
   `Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control`). Add `GH_TOKEN` too if the repo is private.
2. Upload `m0_inputs.zip` (made locally with `scripts/pack_m0.py`) to Drive: `MyDrive/timeflight/m0_inputs.zip`.
3. Runtime: A100. Model downloads and setup take ~15 min of GPU time, so do everything in one session."""),
    ("code", """!nvidia-smi --query-gpu=name,memory.total --format=csv
import time; T0 = time.time()
def gpu_hours(): print(f"GPU session time: {(time.time() - T0) / 3600:.2f} h")"""),
    ("code", """import os
from google.colab import userdata, drive
drive.mount('/content/drive')
os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
try:
    gh = userdata.get("GH_TOKEN")
except Exception:
    gh = None
url = f"https://{gh}@github.com/EmmanuelRiveros/TimeFlight" if gh else "https://github.com/EmmanuelRiveros/TimeFlight"
!test -d /content/TimeFlight && git -C /content/TimeFlight pull -q || git clone -q {url} /content/TimeFlight"""),
    ("code", """!bash /content/TimeFlight/colab/setup_comfy.sh
gpu_hours()"""),
    ("code", """%cd /content/TimeFlight
!python scripts/download_models.py --comfy-dir /content/ComfyUI --precision auto
!python workflows/build_m0.py --models-json /content/ComfyUI/timeflight_models.json
gpu_hours()"""),
    ("code", """os.environ["COMFY_BASE_URL"] = "http://127.0.0.1:8189"  # the SDK targets the local proxy
!mkdir -p /content/m0 && unzip -oq /content/drive/MyDrive/timeflight/m0_inputs.zip -d /content/m0 && ls /content/m0
!bash /content/TimeFlight/colab/start_servers.sh
!python scripts/validate_workflow.py workflows/api/m0_union_depth_api.json
!python scripts/validate_workflow.py workflows/api/m3_union_depth_init_api.json"""),
    ("md", """## Step A: drafts at 576×320, 49 frames
Run **one** cell first. Each run cell stops with an AssertionError if a variant failed; the ComfyUI error is printed above it. If it fails on the asset upload, add `--comfy-input-dir /content/ComfyUI/input`."""),
    ("code", """RUN = "python app/run_m0.py --renders /content/m0 --results /content/m0_results"
!{RUN} --step A --prompt past_1519 --strength 1.0 --mode inverse
gpu_hours(); assert _exit_code == 0, 'variant failed: send the error above before running more'"""),
    ("code", """!{RUN} --step A   # the remaining 11 variants (completed ones are skipped)
assert _exit_code == 0, "some variants failed"
!python scripts/metrics.py --results /content/m0_results --renders /content/m0
gpu_hours()"""),
    ("md", """## Step B: 1024×576, 121 frames, best setting from A
Edit `--strength` and `--mode` to the winner of step A."""),
    ("code", """!{RUN} --step B --strength 1.0 --mode inverse
assert _exit_code == 0, "some variants failed"
!python scripts/metrics.py --results /content/m0_results --renders /content/m0
gpu_hours()"""),
    ("md", "## Step C: 241 frames (can we avoid chaining?)"),
    ("code", """!{RUN} --step C --strength 1.0 --mode inverse --prompt past_1519
gpu_hours()"""),
    ("md", """## Real place, 15 s (361 frames): present day and 1519
Upload `place_inputs.zip` (made locally with
`scripts/pack_m0.py --src out/place --folders present_r1024_361 1519_r1024_361 --out out/place_inputs.zip`)
to `MyDrive/timeflight/`. 361 frames is a new length: if it fails or drifts, fall back to 2 × 241 with an overlap."""),
    ("code", """ZIP = "/content/drive/MyDrive/timeflight/place_inputs.zip"
assert os.path.exists(ZIP), f"{ZIP} not found: upload it to Drive (MyDrive/timeflight/) and wait for it to sync"
!mkdir -p /content/place && unzip -oq /content/drive/MyDrive/timeflight/place_inputs.zip -d /content/place && ls /content/place
PLACE = "python app/run_m0.py --renders /content/place --results /content/m0_results --step P --prompts places/tenochtitlan_zocalo/prompts.json --strength 1.0 --mode inverse"
!{PLACE} --folder present_r1024_361 --prompt present_2026
gpu_hours(); assert _exit_code == 0, 'variant failed: send the error above'"""),
    ("code", """!{PLACE} --folder 1519_r1024_361 --prompt past_1519
gpu_hours(); assert _exit_code == 0, 'variant failed: send the error above'"""),
    ("md", """## M3 test: RGB init, last 5 s of the 1519 flight at 576×320
In run 3 the precinct, plaza and palace courtyards came out as water: depth alone cannot tell flat ground
from flat water. Here the Blender color render (water blue, land brown) is the starting point and the
sampler starts part-way through its schedule. Lower init sigma = closer to the Blender colors.
Tests 1–2 (0.975 / 0.909 / 0.725): only 0.725 kept the land, and it looked like the render. The current
matrix is in `colab/m3_test.sh`. Upload `m3_inputs.zip` to `MyDrive/timeflight/`."""),
    ("code", """ZIP = "/content/drive/MyDrive/timeflight/m3_inputs.zip"
assert os.path.exists(ZIP), f"{ZIP} not found: upload it to Drive (MyDrive/timeflight/) and wait for it to sync"
!mkdir -p /content/m3 && unzip -oq /content/drive/MyDrive/timeflight/m3_inputs.zip -d /content/m3 && ls /content/m3
!bash colab/m3_test.sh   # the test matrix lives in the repo, so a git pull updates it
gpu_hours(); assert _exit_code == 0, 'variant failed: send the error above'"""),
    ("md", """## Full flight (M4): three generations, era merge, time wave, credits
One graph through the Comfy SDK (`workflows/build_flight.py`, `app/run_flight.py`). Upload `flight_inputs.zip`
(made with `scripts/pack_m0.py --src out/place --folders present_r1024_361 1519_r1024_361 --out out/flight_inputs.zip --with-depth`)
to `MyDrive/timeflight/`. About 3 × 2 min of generation plus a few minutes of compositing.
Results are copied to `MyDrive/timeflight/flight_results/`."""),
    ("code", """!bash /content/TimeFlight/colab/flight.sh
gpu_hours(); assert _exit_code == 0, 'flight failed: send the error above'"""),
    ("code", """!mkdir -p /content/drive/MyDrive/timeflight/m0_results
!cp -r /content/m0_results/. /content/drive/MyDrive/timeflight/m0_results/
!cp /content/versions.txt /content/ComfyUI/timeflight_models.json /content/drive/MyDrive/timeflight/m0_results/
!grep -E -A3 "Prompt executed|Error|error" /content/comfy.log | tail -40
gpu_hours()"""),
]


def cell(kind: str, src: str) -> dict:
    lines = src.splitlines(keepends=True)
    if kind == "md":
        return {"cell_type": "markdown", "metadata": {}, "source": lines}
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": lines}


nb = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {"accelerator": "GPU", "colab": {"gpuType": "A100", "machine_shape": "hm"},
                 "kernelspec": {"name": "python3", "display_name": "Python 3"}},
    "cells": [cell(k, s) for k, s in CELLS],
}
out = Path(__file__).with_name("timeflight_m0.ipynb")
out.write_text(json.dumps(nb, indent=1) + "\n", encoding="utf-8")
print(f"wrote {out}")
