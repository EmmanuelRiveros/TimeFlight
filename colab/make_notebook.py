"""Generate the Colab notebooks (keeps them diff-able: edit this file, not the .ipynb files).
  colab/timeflight.ipynb      run TimeFlight: setup, models, a small test, then the 1080p hero
  colab/timeflight_m0.ipynb   the full development log: M0 smoke test, M3, M4, polish and texture tests

  .venv/Scripts/python colab/make_notebook.py
"""

import json
from pathlib import Path

M0_CELLS = [
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
    ("md", """## 9:16 teaser (10 s): small portrait check, then the teaser
Upload `teaser_inputs.zip` (made with `scripts/pack_m0.py --src out/teaser --folders present_v576x1024_241
1519_v576x1024_241 present_v320x576_49 1519_v320x576_49 --out out/teaser_inputs.zip --with-depth`) to `MyDrive/timeflight/`."""),
    ("code", """!git -C /content/TimeFlight pull -q
!bash /content/TimeFlight/colab/flight.sh teaser_inputs.zip teaser_test teaser
gpu_hours(); assert _exit_code == 0, 'teaser failed: send the error above'"""),
    ("md", """## Hero polish: stage 2 (x2 latent upscale + 3 refine steps), 1920x1080, LTX audio
Upload `polish_inputs.zip` (made with `scripts/pack_m0.py --src out/polish --folders present_r1024_361 1519_r1024_361
present_r576x320_49 1519_r576x320_49 --out out/polish_inputs.zip --with-depth`) to `MyDrive/timeflight/`.
Runs a 2 s stage-2 check at 1152x640 first, then the full hero."""),
    ("code", """!git -C /content/TimeFlight pull -q
!bash /content/TimeFlight/colab/flight.sh polish_inputs.zip hero_s2_test hero_s2
gpu_hours(); assert _exit_code == 0, 'polish failed: send the error above'"""),
    ("md", """## Hero polish 2: decode without temporal seams, dust curtain on the wave front
Upload `polish2_inputs.zip` (made with `scripts/pack_m0.py --src out/polish --folders present_r576x320_121 1519_r576x320_121
present_r1024_361 1519_r1024_361 --out out/polish2_inputs.zip --with-depth`) to `MyDrive/timeflight/`.
Runs a 5 s decode check at 1152x640 first (121 frames in one temporal tile), then the full hero."""),
    ("code", """!git -C /content/TimeFlight pull -q
!bash /content/TimeFlight/colab/flight.sh polish2_inputs.zip hero_s2_seams_test hero_s2
gpu_hours(); assert _exit_code == 0, 'polish 2 failed: send the error above'"""),
    ("md", """## Texture test: material prompts and a looser depth guide (4 x 2 s at 1152x640)
Uses `polish_inputs.zip` (already uploaded for Hero polish). Same seed for all four: `tex_base`, `tex_mat`
(material-rich prompts), `tex_g08` (guide strength 0.8), `tex_mat_g08` (both). About 10 min of GPU."""),
    ("code", """!git -C /content/TimeFlight pull -q
!bash /content/TimeFlight/colab/texture_test.sh
gpu_hours(); assert _exit_code == 0, 'texture test failed: send the error above'"""),
    ("md", """## Texture test 2: "lived-in" prompts (1 x 2 s at 1152x640)
Round 1 (run 11): guide 0.8 drifts from the depth; the *_mat prompts read as decay. This runs the *_live
prompts at strength 1.0 with the same seed, so it compares directly with `tex_base`. About 3 min of GPU."""),
    ("code", """!git -C /content/TimeFlight pull -q
!bash /content/TimeFlight/colab/texture_test.sh tex_live
gpu_hours(); assert _exit_code == 0, 'texture test 2 failed: send the error above'"""),
    ("code", """!mkdir -p /content/drive/MyDrive/timeflight/m0_results
!cp -r /content/m0_results/. /content/drive/MyDrive/timeflight/m0_results/
!cp /content/versions.txt /content/ComfyUI/timeflight_models.json /content/drive/MyDrive/timeflight/m0_results/
!grep -E -A3 "Prompt executed|Error|error" /content/comfy.log | tail -40
gpu_hours()"""),
]


RUN_CELLS = [
    ("md", """# TimeFlight: a drone flight that turns back time
Mexico City's historic centre today → Tenochtitlan in 1519, generated with LTX-2.5 in ComfyUI and guided by
depth rendered from open geodata. Repo, licenses and how it works: https://github.com/EmmanuelRiveros/TimeFlight

**You need:**
1. An **A100 80GB** runtime (Runtime → Change runtime type). Smaller GPUs run out of memory on the 1080p hero.
2. A Hugging Face token with access to the gated `Lightricks/LTX-2.5` and
   `Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control` repos (accept the LTX-2.x Community License on each model
   page), saved as the Colab secret `HF_TOKEN`.
3. Nothing else: the depth passes are downloaded from the repo's `inputs-v1` release. To use your own
   renders (README, "Locally"), clear `INPUTS_URL` and upload `polish2_inputs.zip` to `MyDrive/timeflight/`.

Setup and model downloads take about 10 min (66 GB of weights). The small test takes about 4 min and the
1080p hero about 35 min. The output is an illustrative AI reconstruction, not a historical record."""),
    ("code", """!nvidia-smi --query-gpu=name,memory.total --format=csv
import os, time; T0 = time.time()
def gpu_hours(): print(f"GPU session time: {(time.time() - T0) / 3600:.2f} h")
from google.colab import userdata
os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
!test -d /content/TimeFlight && git -C /content/TimeFlight pull -q || git clone -q --depth 1 https://github.com/EmmanuelRiveros/TimeFlight /content/TimeFlight"""),
    ("code", """# ComfyUI, ComfyUI-LTXVideo, Comfy SDK + proxy, the TimeFlight nodes; then the model weights
!bash /content/TimeFlight/colab/setup_comfy.sh
%cd /content/TimeFlight
!python scripts/download_models.py --comfy-dir /content/ComfyUI --precision auto --with-upscaler
os.environ["COMFY_BASE_URL"] = "http://127.0.0.1:8189"  # the SDK talks to ComfyUI through the local proxy
!bash colab/start_servers.sh
gpu_hours()"""),
    ("code", """# Depth passes: the published zip (INPUTS_URL), or polish2_inputs.zip in MyDrive/timeflight/ if empty
INPUTS_URL = "https://github.com/EmmanuelRiveros/TimeFlight/releases/download/inputs-v1/polish2_inputs.zip"
if INPUTS_URL:
    !mkdir -p /content/inputs && wget -q -O /content/inputs/polish2_inputs.zip "{INPUTS_URL}"
    os.environ["TF_INPUTS"] = "/content/inputs"
else:
    from google.colab import drive
    drive.mount("/content/drive")
zip_path = os.path.join(os.environ.get("TF_INPUTS", "/content/drive/MyDrive/timeflight"), "polish2_inputs.zip")
assert os.path.exists(zip_path), f"{zip_path} not found"
print(zip_path, os.path.getsize(zip_path) // 2**20, "MB")"""),
    ("md", """## Small test: 5 s at 1152×640
Three generations (present day, the 1519 city, the 1519 dry zones), era merge, time wave, captions and
credits, in one ComfyUI graph."""),
    ("code", """!bash colab/flight.sh polish2_inputs.zip hero_s2_seams_test
gpu_hours(); assert _exit_code == 0, "test failed: see the ComfyUI log above"
from IPython.display import Video
Video("/content/flight_results/hero_s2_seams_test_1519.mp4", embed=True, width=768)"""),
    ("md", """## The hero: 15 s at 1920×1080, plus credits (about 35 min)"""),
    ("code", """!bash colab/flight.sh polish2_inputs.zip hero_s2
gpu_hours(); assert _exit_code == 0, "hero failed: see the ComfyUI log above"
from google.colab import files
files.download("/content/flight_results/hero_s2_1519.mp4")"""),
]


def cell(kind: str, src: str) -> dict:
    lines = src.splitlines(keepends=True)
    if kind == "md":
        return {"cell_type": "markdown", "metadata": {}, "source": lines}
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": lines}


def write(name: str, cells: list[tuple[str, str]]) -> None:
    nb = {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {"accelerator": "GPU", "colab": {"gpuType": "A100", "machine_shape": "hm"},
                     "kernelspec": {"name": "python3", "display_name": "Python 3"}},
        "cells": [cell(k, s) for k, s in cells],
    }
    out = Path(__file__).with_name(name)
    out.write_text(json.dumps(nb, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {out}")


write("timeflight.ipynb", RUN_CELLS)
write("timeflight_m0.ipynb", M0_CELLS)
