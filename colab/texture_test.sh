#!/usr/bin/env bash
# Texture test: does a different prompt or a looser depth guide reduce the synthetic look?
# Each variant is one run of hero_s2_test (last 2 s of the hero path, 576x320x49 -> stage 2 -> 1152x640),
# same seed, and also saves the three raw generations. Needs polish_inputs.zip in MyDrive/timeflight/.
#   bash colab/texture_test.sh                      # round 1 (run 11): tex_base tex_mat tex_g08 tex_mat_g08
#   bash colab/texture_test.sh tex_live             # round 2: "lived-in" prompts, strength 1.0
# Variants:
#   tex_base     current prompts, guide strength 1.0 (reference)
#   tex_mat      *_mat prompts (weathered/worn wording), strength 1.0   -> read as decay (run 11)
#   tex_g08      current prompts, strength 0.8                          -> geometry drifts (run 11)
#   tex_mat_g08  both
#   tex_live     *_live prompts (lived-in wording, no decay words), strength 1.0
set -euo pipefail
cd /content/TimeFlight
VARIANTS=("$@")
[ ${#VARIANTS[@]} -gt 0 ] || VARIANTS=(tex_base tex_mat tex_g08 tex_mat_g08)
prompts() { echo "--present-prompt present_2026_$1 --past-base-prompt past_1519_$1 --past-zones-prompt past_1519_plaza2_$1"; }
for v in "${VARIANTS[@]}"; do
  case $v in
    tex_base)    extra="" ;;
    tex_mat)     extra="$(prompts mat)" ;;
    tex_g08)     extra="--guide-strength 0.8" ;;
    tex_mat_g08) extra="--guide-strength 0.8 $(prompts mat)" ;;
    tex_live)    extra="$(prompts live)" ;;
    *) echo "unknown variant $v"; exit 1 ;;
  esac
  FLIGHT_ARGS="--name $v $extra" bash colab/flight.sh polish_inputs.zip hero_s2_test
done
grep tex_ /content/flight_results/runs.jsonl
