#!/usr/bin/env bash
# Start ComfyUI (:8188) and comfy-api-proxy (:8189) in the background, wait until ready.
set -euo pipefail

cd /content/ComfyUI
if ! curl -sf http://127.0.0.1:8188/object_info > /dev/null; then
  nohup python main.py --listen 127.0.0.1 --port 8188 > /content/comfy.log 2>&1 &
  for i in $(seq 1 120); do
    curl -sf http://127.0.0.1:8188/object_info > /dev/null && break
    sleep 3
  done
fi
curl -sf http://127.0.0.1:8188/object_info > /dev/null || { tail -50 /content/comfy.log; exit 1; }
echo "ComfyUI ready"

if ! curl -s http://127.0.0.1:8189 > /dev/null; then
  nohup comfy-api-proxy --comfyui http://127.0.0.1:8188 --port 8189 \
    --comfyui-base-dir /content/ComfyUI > /content/proxy.log 2>&1 &
  sleep 3
fi
echo "proxy started (log: /content/proxy.log)"
