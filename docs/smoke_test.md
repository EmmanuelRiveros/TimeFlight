# M0: aerial smoke test (go/no-go)

**Question:** can depth-guided LTX-2.5 (Union Control IC-LoRA) turn a synthetic aerial flyover
into a usable, temporally consistent historical restyle?

**Gate:** Sat Oct 17, 22:00, or 10 GPU hours, whichever comes first.

## Input
- `blender/m0_synthetic_flyover.py`: procedural block city on a gentle slope with a lake plane.
  The camera moves forward and descends, like the hero path. Renders a depth pass (fixed near/far)
  and a Workbench RGB pass of the same motion. Rendered locally.
- `scripts/make_control.py`: depth EXR/PNG → control video (inverse or linear encoding) plus the RGB reference video.

## Test matrix (stage 1 only, seed 42, 24 fps)
| Step | Size | Frames | Variants | Purpose |
|---|---|---|---|---|
| A | 576×320 | 49 | 3 prompts × strength {1.0, 0.7} × depth encoding {inverse, linear} = 12 | Pick the best setting cheaply |
| B | 1024×576 | 121 | best of A × 3 prompts | Go/no-go evidence |
| C | 1024×576 | 241 | best of A, 1 prompt | Do long clips hold up, so we can avoid chaining? |
| D (only if LTX fails) | 832×480 | 81 | Wan 2.1 VACE, same depth | Switch candidate |

Prompts live in `places/m0_synthetic/prompts.json`: `past_1519`, `present_2026`, `colonial_1790`.

## "Usable": every criterion must pass on step B
| # | Criterion | How measured | Pass |
|---|---|---|---|
| U1 | Structure follows depth | `scripts/metrics.py structure`: Depth-Anything-V2-Small on the output, Spearman correlation with the control per frame | median ≥ 0.75 and no frame < 0.5 |
| U2 | Camera fidelity | Visual: motion matches the path; no added pans, zooms or cuts | yes |
| U3 | Temporal stability | `scripts/metrics.py flicker`: mean abs frame difference of the output ÷ that of the RGB reference. Plus a 3× slow-motion review | ratio ≤ 1.5 and ≤ 2 noticeable pops in 5 s |
| U4 | Reads as the era | 3 people with no context: "what place and time is this?" Count frames with modern artifacts | ≥ 2/3 say Aztec / pre-Hispanic / lake city; modern artifacts in < 10% of frames |
| U5 | Affordable | Wall time and units per 121-frame clip | the final hero (15 s, present + past, stage 2) fits in ≤ 30% of the remaining units |

## Decision rules
- **GO (LTX):** U1–U5 pass.
- **SWITCH (Wan VACE):** LTX fails U1 or U3 and VACE passes step D. We lose the LTX bonus.
- **DESCOPE:** U1 passes but U3 fails on both models. Use ≤ 5 s clips joined by time-wave transitions,
  or restyled keyframes with depth parallax. Geo bridge, era geometry and provenance stay in scope.

## Run log
| Date | Step | Variant | GPU | Wall time | Units | U1 | U3 | Notes |
|---|---|---|---|---|---|---|---|---|
| 2026-10-16 | A, B, C | all 16 | A100-SXM4 80GB | <2 s each (failed) | ~0.10 GPU h total, incl. setup | – | – | Every job failed in the guide's VAE encode: 288 and 544 are not multiples of 64 (see decisions.md, "Guide size rule"). Re-rendered at 576×320 and 1024×576. Setup (models + ComfyUI) took 0.08 h. |

## Decision
_pending_
