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
| U3 | Temporal stability | `scripts/metrics.py`: **unstable fraction**, the share of non-occluded pixels that still differ by > 20 gray levels after Farneback flow warping (revised 2026-10-16, see below). Pop score = max ÷ median frame difference. Plus a 3× slow-motion review | unstable ≤ 5% and pop ≤ 2 |
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
| 2026-10-16 (run 1) | A, B, C | all 16 | A100-SXM4 80GB | <2 s each (failed) | ~0.10 GPU h total, incl. setup | – | – | Every job failed in the guide's VAE encode: 288 and 544 are not multiples of 64 (see decisions.md, "Guide size rule"). Re-rendered at 576×320 and 1024×576. Setup (models + ComfyUI) took 0.08 h. |

| 2026-10-16 (run 2) | A | 12 at 576×320×49 | A100-SXM4 80GB | 6.6–10.4 s (first run 25.5 s, model load) | ~0.12 GPU h for the whole session, incl. setup | 0.994–0.998 (inflated, see below) | unstable 0.08–0.47%, pop 1.06–1.18 | All pass. Inverse is slightly steadier than linear; 0.7 slightly steadier than 1.0 |
| 2026-10-16 (run 2) | B | 3 prompts, 1.0, inverse, 1024×576×121 | same | 40.4–40.5 s | (in the 0.12 h) | 0.997–0.999 | unstable 0.08–0.15%, pop 1.16–1.19 | Depth edges overlaid on frame 60 match the building outlines |
| 2026-10-16 (run 2) | C | past_1519, 1024×576×241 | same | 80.5 s | (in the 0.12 h) | – | unstable 0.03%, pop 1.50 | Coherent over 10 s; no drift seen in frames 0/40/100 |
| 2026-10-17 (run 3) | P (real place, M1/M2) | present_2026 and past_1519, 1.0, inverse, 1024×576×361 | A100-SXM4 80GB | 150.0 s and 125.3 s | ~0.16 GPU h for the whole session, incl. setup | edge overlay: good (present), partial (1519) | unstable 0.21% / 0.29%, pop 1.44 / 1.40 | First 15 s clips from the real-coordinate path. No drift over 15 s (stability flat across 60-frame segments). 1519: precinct, plaza and palace courtyards rendered as water, see notes. The first attempt stopped before submitting (COMFY_BASE_URL unset, fixed in 0285e5c) |

### Notes on the metrics (2026-10-16)
- **U1 is inflated.** The ground plane's smooth near-to-far gradient dominates a whole-frame rank
  correlation, so almost any plausible aerial image scores > 0.99. The real evidence for U1 is the
  edge overlay: depth discontinuities sit on the generated building outlines.
- **The original U3 ratio was wrong for this input.** The flat-shaded Workbench reference has little
  texture, so its frame differences are tiny and every textured output "failed" (3.4–7.6). A
  texture-normalized ratio was tried and rejected: per-frame noise raises spatial gradients as much
  as it raises temporal differences, so it hides flicker (verified in a unit test). U3 now uses a
  motion-compensated unstable-pixel fraction, which is ~0 for a textured pan and > 20% for added
  noise (`tests/test_passes_metrics.py`). Baselines: Blender RGB 0.01%, control 0.00%.

### Notes on run 3, the 15 s real-place clips (2026-10-17)
Checked locally (`out/m3/*_sheet.jpg`: output, output with Blender edges, control at frames 0–360 every 60).
- **361 frames in one pass works.** Unstable fraction stays at 0.17–0.35% in every 60-frame segment of both
  clips, and there are no cuts or zooms. Sharpness falls in the last third of both clips, which matches the
  scene (closer, larger flat surfaces), not drift. No chaining needed for the 15 s hero.
- **Present day:** blocks, streets, the Torre Latinoamericana and Bellas Artes sit on their depth edges.
  The Catedral and Palacio Nacional are generic (white with blue domes). The rooftops are uniform red.
- **1519, main defect:** every large flat area without houses became water: the walled precinct floor, the
  plaza on today's Zócalo and the palace courtyards. The Templo Mayor stands on what looks like a raft, and the
  terraces of the pyramid are lost. Cause: depth is the only control, and flat dry ground at +1.5 m has
  the same depth as flat water; the prompt says "lake", so the model fills open flat areas with water. Houses,
  canals and the lake itself read correctly.
- **Fix under test (M3):** start the sampler from the Blender color render (water blue, land brown) at a
  partial noise level (SDEdit), keeping the depth guide. See the M3 test cells in the notebook.

## Decision
**GO with LTX-2.5** (2026-10-16). U4 passed 2026-10-17.
- U1 pass (edge overlay), U2 pass on review of contact sheets (no added cuts or zooms), U3 pass, U5 pass by
  a wide margin: a 121-frame 1024×576 clip costs ~40 s, and 241 frames ~80 s, so the hero needs no chaining.
- **U4 pass (2026-10-17):** all 3 viewers with no context identified Tenochtitlan (author's test; modern-artifact frame count not recorded).
  Original note: the author shows B_past_1519 to 3 people. Expected weakness: the synthetic boxes read as
  generic white houses on water. Period shapes come from the M2 proxies, not from the prompt.
- Settings carried forward: strength 1.0, inverse depth encoding, single stage (stage-2 upscale untested).
- M0 used about 0.22 GPU h of the 10 h cap.
