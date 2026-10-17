#!/usr/bin/env bash
# M3: full 15 s 1519 clip (1024x576x361) with the dry-plaza prompts. Needs place_inputs.zip on Drive.
set -e
cd /content/TimeFlight
ZIP=/content/drive/MyDrive/timeflight/place_inputs.zip
if [ ! -d /content/place/1519_r1024_361 ]; then
  test -f "$ZIP" || { echo "$ZIP not found: upload it to MyDrive/timeflight/"; exit 1; }
  mkdir -p /content/place && unzip -oq "$ZIP" -d /content/place
fi
python app/run_m0.py --renders /content/place --results /content/m0_results --step P --folder 1519_r1024_361 \
  --prompts places/tenochtitlan_zocalo/prompts.json --prompt past_1519_plaza past_1519_plaza2 --strength 1.0 --mode inverse
