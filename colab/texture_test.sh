#!/usr/bin/env bash
# Texture test (2026-10-18): does a material-rich prompt or a looser depth guide reduce the synthetic look?
# Four runs of hero_s2_test (last 2 s of the hero path, 576x320x49 -> stage 2 -> 1152x640), same seed:
#   tex_base     current prompts, guide strength 1.0 (reference)
#   tex_mat      *_mat prompts (same content, surface and camera wording), strength 1.0
#   tex_g08      current prompts, strength 0.8
#   tex_mat_g08  both
# Each run also saves the three raw generations. Needs polish_inputs.zip in MyDrive/timeflight/.
set -euo pipefail
cd /content/TimeFlight
MAT="--present-prompt present_2026_mat --past-base-prompt past_1519_mat --past-zones-prompt past_1519_plaza2_mat"
FLIGHT_ARGS="--name tex_base" bash colab/flight.sh polish_inputs.zip hero_s2_test
FLIGHT_ARGS="--name tex_mat $MAT" bash colab/flight.sh polish_inputs.zip hero_s2_test
FLIGHT_ARGS="--name tex_g08 --guide-strength 0.8" bash colab/flight.sh polish_inputs.zip hero_s2_test
FLIGHT_ARGS="--name tex_mat_g08 --guide-strength 0.8 $MAT" bash colab/flight.sh polish_inputs.zip hero_s2_test
grep tex_ /content/flight_results/runs.jsonl
