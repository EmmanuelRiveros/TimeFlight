#!/usr/bin/env bash
# Full TimeFlight run (M4): 3 generations -> era merge -> time wave -> captions and credits -> video.
#   bash colab/flight.sh                                         # hero (flight_inputs.zip)
#   bash colab/flight.sh teaser_inputs.zip teaser_test teaser    # small portrait check, then the teaser
#   bash colab/flight.sh polish_inputs.zip hero_s2_test hero_s2  # stage 2 (x2 latent upscale), 1920x1080
#   bash colab/flight.sh polish2_inputs.zip hero_s2_seams_test hero_s2  # decode check (121 frames), then the hero
# FLIGHT_ARGS is passed to every app/run_flight.py call (e.g. "--name x --guide-strength 0.8"); see colab/texture_test.sh.
# The first argument is a zip (scripts/pack_m0.py ... --with-depth) in $TF_INPUTS (default MyDrive/timeflight/
# on Colab), a path to a zip, or a folder that already holds the render folders (e.g. out/place on the
# machine that rendered them). Presets run in order and the script stops at the first failure.
# On your own machine: TF_WORK=~/timeflight-work bash colab/flight.sh out/place hero_s2 (see README).
# Servers must be started after colab/setup_comfy.sh, which installs the TimeFlight nodes; ComfyUI is restarted here if it predates them.
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
WORK=${TF_WORK:-/content}
COMFY=${TF_COMFY:-$WORK/ComfyUI}
INPUTS=${TF_INPUTS:-/content/drive/MyDrive/timeflight}
RESULTS=$WORK/flight_results
cd "$REPO"
SRC=${1:-flight_inputs.zip}
shift || true
PRESETS=("$@")
[ ${#PRESETS[@]} -gt 0 ] || PRESETS=(hero)

if [ -d "$SRC" ]; then
  RENDERS=$(cd "$SRC" && pwd)
else
  ZIP=$SRC; [ -f "$ZIP" ] || ZIP=$INPUTS/$SRC
  test -f "$ZIP" || { echo "$SRC not found (nor $ZIP): upload it to MyDrive/timeflight/ or pass a path"; exit 1; }
  # -o: the zip wins. Folders unpacked earlier in the session (e.g. from m3_inputs.zip) share names
  # with these but may hold older render.json files without the camera entry.
  RENDERS=$WORK/place
  mkdir -p "$RENDERS" && unzip -oq "$ZIP" -d "$RENDERS"
fi
ls "$RENDERS"

nodes_loaded() { [[ "$(curl -sf http://127.0.0.1:8188/object_info/TimeFlightTimeWave || true)" == *TimeFlightTimeWave* ]]; }
if ! nodes_loaded; then
  # ComfyUI loads custom nodes only at startup: restart it if it was started before setup linked them.
  test -e "$COMFY/custom_nodes/ComfyUI-TimeFlight" || { echo "node pack not linked: run colab/setup_comfy.sh"; exit 1; }
  echo "restarting ComfyUI to load the TimeFlight nodes"
  pkill -f "main.py --listen 127.0.0.1 --port 8188" || true
  for i in $(seq 1 20); do curl -sf http://127.0.0.1:8188/ > /dev/null || break; sleep 1; done
  bash colab/start_servers.sh
  nodes_loaded || { grep -n -i -A8 "timeflight" "$WORK/comfy.log" | tail -40; echo "TimeFlight nodes failed to load (log above)"; exit 1; }
fi
echo "TimeFlight nodes loaded"

STAGE2=0
[[ " ${PRESETS[*]} " == *_s2* ]] && STAGE2=1
if [ $STAGE2 = 1 ]; then
  # Stage 2 needs the latent upscaler; files already on disk are not downloaded again.
  python scripts/download_models.py --comfy-dir "$COMFY" --precision auto --with-upscaler | tail -3
fi
python workflows/build_flight.py --models-json "$COMFY/timeflight_models.json"
python scripts/validate_workflow.py workflows/api/timeflight_flight_api.json
[ $STAGE2 = 0 ] || python scripts/validate_workflow.py workflows/api/timeflight_flight_s2_api.json
for preset in "${PRESETS[@]}"; do
  echo "== $preset"
  # shellcheck disable=SC2086  # FLIGHT_ARGS is a list of flags
  if ! python app/run_flight.py --preset "$preset" --renders "$RENDERS" --results "$RESULTS" \
       --place-dir "$REPO/places/tenochtitlan_zocalo" ${FLIGHT_ARGS:-}; then
    # If ComfyUI itself died, say why: the kernel's OOM killer, or the last lines of its log.
    curl -sf http://127.0.0.1:8188/ > /dev/null || echo "ComfyUI is not running any more"
    dmesg 2>/dev/null | grep -i -E "out of memory|killed process" | tail -5 || true
    tail -30 "$WORK/comfy.log"
    free -g | head -2
    exit 1
  fi
  free -g | head -2; nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
  # Copy to Drive on Colab (or to TF_INPUTS when it is set and exists); locally the results stay in $RESULTS.
  if [ -d "$INPUTS" ]; then mkdir -p "$INPUTS/flight_results" && cp -r "$RESULTS/." "$INPUTS/flight_results/"; fi
done
