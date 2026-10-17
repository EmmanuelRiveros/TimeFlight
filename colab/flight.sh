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
curl -sf http://127.0.0.1:8188/object_info/TimeFlightTimeWave | grep -q TimeFlightTimeWave \
  || { echo "TimeFlight nodes not loaded: rerun setup, then restart ComfyUI (see /content/comfy.log)"; exit 1; }
python workflows/build_flight.py --models-json /content/ComfyUI/timeflight_models.json
python scripts/validate_workflow.py workflows/api/timeflight_flight_api.json
python app/run_flight.py --renders /content/place --results /content/flight_results
mkdir -p /content/drive/MyDrive/timeflight/flight_results
cp -r /content/flight_results/. /content/drive/MyDrive/timeflight/flight_results/
