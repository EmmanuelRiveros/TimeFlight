#!/usr/bin/env bash
# Full TimeFlight run (M4): 3 generations -> era merge -> time wave -> captions and credits -> video.
#   bash colab/flight.sh                                         # hero (flight_inputs.zip)
#   bash colab/flight.sh teaser_inputs.zip teaser_test teaser    # small portrait check, then the teaser
#   bash colab/flight.sh polish_inputs.zip hero_s2_test hero_s2  # stage 2 (x2 latent upscale), 1920x1080
#   bash colab/flight.sh polish2_inputs.zip hero_s2_seams_test hero_s2  # decode check (121 frames), then the hero
# The zip (scripts/pack_m0.py ... --with-depth) must be in MyDrive/timeflight/. Presets run in order and
# the script stops at the first failure. Servers must be started after colab/setup_comfy.sh, which
# installs the TimeFlight nodes; ComfyUI is restarted here if it predates them.
set -euo pipefail
cd /content/TimeFlight
ZIP=/content/drive/MyDrive/timeflight/${1:-flight_inputs.zip}
shift || true
PRESETS=("$@")
[ ${#PRESETS[@]} -gt 0 ] || PRESETS=(hero)

test -f "$ZIP" || { echo "$ZIP not found: upload it to MyDrive/timeflight/"; exit 1; }
# -o: the zip wins. Folders unpacked earlier in the session (e.g. from m3_inputs.zip) share names
# with these but may hold older render.json files without the camera entry.
mkdir -p /content/place && unzip -oq "$ZIP" -d /content/place
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

STAGE2=0
[[ " ${PRESETS[*]} " == *_s2* ]] && STAGE2=1
if [ $STAGE2 = 1 ]; then
  # Stage 2 needs the latent upscaler; files already on disk are not downloaded again.
  python scripts/download_models.py --comfy-dir /content/ComfyUI --precision auto --with-upscaler | tail -3
fi
python workflows/build_flight.py --models-json /content/ComfyUI/timeflight_models.json
python scripts/validate_workflow.py workflows/api/timeflight_flight_api.json
[ $STAGE2 = 0 ] || python scripts/validate_workflow.py workflows/api/timeflight_flight_s2_api.json
mkdir -p /content/drive/MyDrive/timeflight/flight_results
for preset in "${PRESETS[@]}"; do
  echo "== $preset"
  if ! python app/run_flight.py --preset "$preset" --renders /content/place --results /content/flight_results; then
    # If ComfyUI itself died, say why: the kernel's OOM killer, or the last lines of its log.
    curl -sf http://127.0.0.1:8188/ > /dev/null || echo "ComfyUI is not running any more"
    dmesg 2>/dev/null | grep -i -E "out of memory|killed process" | tail -5 || true
    tail -30 /content/comfy.log
    free -g | head -2
    exit 1
  fi
  free -g | head -2; nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
  cp -r /content/flight_results/. /content/drive/MyDrive/timeflight/flight_results/
done
