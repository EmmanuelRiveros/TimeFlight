#!/usr/bin/env bash
# Full TimeFlight run (M4): 3 generations -> era merge -> time wave -> captions and credits -> hero video.
# Needs flight_inputs.zip on Drive (scripts/pack_m0.py ... --with-depth) and the servers started after
# colab/setup_comfy.sh (which installs the TimeFlight nodes).
set -euo pipefail
cd /content/TimeFlight
ZIP=/content/drive/MyDrive/timeflight/flight_inputs.zip
if [ ! -d /content/place/1519_r1024_361/depth ]; then
  test -f "$ZIP" || { echo "$ZIP not found: upload it to MyDrive/timeflight/"; exit 1; }
  mkdir -p /content/place && unzip -oq "$ZIP" -d /content/place
fi
ls /content/place
nodes_loaded() { [[ "$(curl -sf http://127.0.0.1:8188/object_info/TimeFlightTimeWave || true)" == *TimeFlightTimeWave* ]]; }
if ! nodes_loaded; then
  # ComfyUI loads custom nodes only at startup: restart it if it was started before setup linked them.
  test -e /content/ComfyUI/custom_nodes/ComfyUI-TimeFlight || { echo "node pack not linked: run colab/setup_comfy.sh"; exit 1; }
  echo "restarting ComfyUI to load the TimeFlight nodes"
  pkill -f "main.py --listen 127.0.0.1 --port 8188" || true
  for i in $(seq 1 20); do curl -sf http://127.0.0.1:8188/ > /dev/null || break; sleep 1; done
  bash colab/start_servers.sh
  nodes_loaded || { grep -n -i -A8 "timeflight" /content/comfy.log | tail -40; echo "TimeFlight nodes failed to load (log above)"; exit 1; }
fi
echo "TimeFlight nodes loaded"
python workflows/build_flight.py --models-json /content/ComfyUI/timeflight_models.json
python scripts/validate_workflow.py workflows/api/timeflight_flight_api.json
python app/run_flight.py --renders /content/place --results /content/flight_results
mkdir -p /content/drive/MyDrive/timeflight/flight_results
cp -r /content/flight_results/. /content/drive/MyDrive/timeflight/flight_results/
