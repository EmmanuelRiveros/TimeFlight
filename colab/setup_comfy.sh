#!/usr/bin/env bash
# Install ComfyUI + ComfyUI-LTXVideo + Comfy SDK/proxy, and link the TimeFlight node pack.
#   bash colab/setup_comfy.sh                              # Colab: everything under /content
#   TF_WORK=~/timeflight-work bash colab/setup_comfy.sh    # your own Linux GPU machine (activate a venv first)
# TF_WORK: where ComfyUI, logs and results go (default /content). TF_COMFY: an existing ComfyUI checkout
# to use instead of cloning one into $TF_WORK/ComfyUI. The repo is the one this script lives in.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
WORK=${TF_WORK:-/content}
COMFY=${TF_COMFY:-$WORK/ComfyUI}
mkdir -p "$WORK"

[ -d "$COMFY" ] || git clone -q --depth 1 https://github.com/comfyanonymous/ComfyUI "$COMFY"
pip install -q -r "$COMFY/requirements.txt"

cd "$COMFY/custom_nodes"
[ -d ComfyUI-LTXVideo ] || git clone -q --depth 1 https://github.com/Lightricks/ComfyUI-LTXVideo
pip install -q -r ComfyUI-LTXVideo/requirements.txt

pip install -q "comfy-sdk>=0.5" "comfy-api-proxy>=0.1.6" huggingface_hub hf_transfer
pip install -q -e "$REPO[metrics]"

# TimeFlight nodes: link the node pack into ComfyUI (it imports the timeflight package installed above),
# and fetch the credit-card font (logged in assets.jsonl).
ln -sfn "$REPO/comfy_nodes" "$COMFY/custom_nodes/ComfyUI-TimeFlight"
python "$REPO/scripts/fetch_fonts.py"

# Record exact versions for reproducibility (copied into docs/smoke_test.md).
{
  echo "date: $(date -u +%FT%TZ)"
  echo "ComfyUI: $(git -C "$COMFY" rev-parse --short HEAD)"
  echo "ComfyUI-LTXVideo: $(git -C "$COMFY/custom_nodes/ComfyUI-LTXVideo" rev-parse --short HEAD)"
  pip show comfy-sdk comfy-api-proxy torch 2>/dev/null | grep -E "^(Name|Version)"
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
} | tee "$WORK/versions.txt"
