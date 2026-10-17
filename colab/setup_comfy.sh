#!/usr/bin/env bash
# Install ComfyUI + ComfyUI-LTXVideo + Comfy SDK/proxy on a Colab runtime.
# Usage: bash colab/setup_comfy.sh   (repo cloned at /content/TimeFlight)
set -euo pipefail

ROOT=/content
REPO=$ROOT/TimeFlight

cd "$ROOT"
[ -d ComfyUI ] || git clone -q --depth 1 https://github.com/comfyanonymous/ComfyUI
pip install -q -r ComfyUI/requirements.txt

cd ComfyUI/custom_nodes
[ -d ComfyUI-LTXVideo ] || git clone -q --depth 1 https://github.com/Lightricks/ComfyUI-LTXVideo
pip install -q -r ComfyUI-LTXVideo/requirements.txt

pip install -q "comfy-sdk>=0.5" "comfy-api-proxy>=0.1.6" huggingface_hub hf_transfer
pip install -q -e "$REPO[metrics]"

# TimeFlight nodes: link the node pack into ComfyUI (it imports the timeflight package installed above),
# and fetch the credit-card font (logged in assets.jsonl).
ln -sfn "$REPO/comfy_nodes" "$ROOT/ComfyUI/custom_nodes/ComfyUI-TimeFlight"
python "$REPO/scripts/fetch_fonts.py"

# Record exact versions for reproducibility (copied into docs/smoke_test.md).
{
  echo "date: $(date -u +%FT%TZ)"
  echo "ComfyUI: $(git -C $ROOT/ComfyUI rev-parse --short HEAD)"
  echo "ComfyUI-LTXVideo: $(git -C $ROOT/ComfyUI/custom_nodes/ComfyUI-LTXVideo rev-parse --short HEAD)"
  pip show comfy-sdk comfy-api-proxy torch 2>/dev/null | grep -E "^(Name|Version)"
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
} | tee $ROOT/versions.txt
