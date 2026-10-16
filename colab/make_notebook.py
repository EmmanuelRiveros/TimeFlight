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
    ("code", """!mkdir -p /content/m0 && unzip -oq /content/drive/MyDrive/timeflight/m0_inputs.zip -d /content/m0 && ls /content/m0
!bash /content/TimeFlight/colab/start_servers.sh
!python scripts/validate_workflow.py workflows/api/m0_union_depth_api.json"""),
    ("md", """## Step A: drafts at 576×320, 49 frames
Run **one** cell first. Each run cell stops with an AssertionError if a variant failed; the ComfyUI error is printed above it. If it fails on the asset upload, add `--comfy-input-dir /content/ComfyUI/input`."""),
    ("code", """os.environ["COMFY_BASE_URL"] = "http://127.0.0.1:8189"
RUN = "python app/run_m0.py --renders /content/m0 --results /content/m0_results"
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
    ("md", """## M1: real place, present day, 15 s (361 frames)
Upload `m1_inputs.zip` (made locally with `scripts/pack_m0.py --src out/m1 --folders r1024_361 --out out/m1_inputs.zip`)
to `MyDrive/timeflight/`. 361 frames is a new length: if it fails or drifts, fall back to 2 × 241 with an overlap."""),
    ("code", """!mkdir -p /content/m1 && unzip -oq /content/drive/MyDrive/timeflight/m1_inputs.zip -d /content/m1 && ls /content/m1
!{RUN.replace('/content/m0 ', '/content/m1 ')} --step P --prompts places/tenochtitlan_zocalo/prompts.json --prompt present_2026 --strength 1.0 --mode inverse
gpu_hours(); assert _exit_code == 0, 'variant failed: send the error above'"""),
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
