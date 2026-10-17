#!/usr/bin/env bash
# M3 RGB-init test on the last 5 s of the 1519 flight (576x320x121). Edit here, not in the notebook:
# the notebook cell only calls this script, so a git pull updates the test.
set -e
cd /content/TimeFlight
M3="python app/run_m0.py --renders /content/m3 --results /content/m0_results --step T --folder 1519_r576_121_end --prompts places/tenochtitlan_zocalo/prompts.json --strength 1.0 --mode inverse"
# Test 3: prompt that says the precinct is dry paving, with and without RGB init between schedule steps.
$M3 --prompt past_1519_plaza
$M3 --prompt past_1519 past_1519_plaza --init-sigma 0.85 0.8
