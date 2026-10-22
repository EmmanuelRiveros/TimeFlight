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
| 2026-10-17 (run 7) | Flight (M4, one ComfyUI graph via comfy-sdk) | 3 generations (present_2026, past_1519, past_1519_plaza2) + era merge + time wave + credits, 1024×576×361 | A100-SXM4 80GB | 621 s for the whole graph | (not reported) | – | unstable 0.56% on the 15 s part; pop 2.11 at frame 188 (the fast phase of the wave; local v4: 2.04 at the same frame) | First end-to-end run inside ComfyUI (ComfyUI c9ea36b, ComfyUI-LTXVideo 44c8a36). Output matches the local composite in layout; the generations differ slightly from the earlier session (mean 6–9 levels of 255), the credit cards are identical (0.4, encoding only). First attempt stopped: ComfyUI had been started before the nodes were linked (fixed in 21a6007). |
| 2026-10-17 (run 8) | Teaser (M4, same graph, preset `teaser`) | portrait 576×1024×241 (last 10 s of the hero path), wave 1.5–7.0 s, two 4 s portrait cards; preceded by `teaser_test` 320×576×49 | A100-SXM4 80GB | 370.8 s for the graph (test: 35.8 s) | (not reported) | – | unstable 0.61% on the 10 s part; pop 2.28 at frame 99 (the fast phase of the wave, as in the hero); raw generations unstable 0.18–0.20%, pop 1.11–1.29 | 433 frames = 241 + 2 × 96 card frames. Portrait camera model holds: the wave ring centers on the Templo Mayor and the dry precinct merges cleanly into the canal city. Upscaled locally to 1080×1920 (lanczos, CRF 16): `out/teaser/final/teaser_1519_1080x1920.mp4`. First attempt ran the hero cell and failed on stale render folders from `place_inputs.zip` (fixed in a9da025). |
| 2026-10-17 (run 9) | Hero polish, stage 2 (`hero_s2_test`, then `hero_s2`) | test: 576×320×49 → ×2 latent upscale + 3 steps → 1152×640, last 2 s of the path, with LTX audio | A100-SXM4 80GB | test 315.7 s (incl. loading the upscaler); full hero: the runtime crashed (session reset, ComfyUI log lost) | 1.25 GPU h for the session | depth edges still line up after stage 2 (cathedral, precinct structures) | – | Stage 2 is far more detailed than stage 1 upscaled (gradient energy 13.8–14.2 vs 5.3–5.7 at 1152×640); the present-day city reads as real streets, cars and trees instead of flat red slabs. Audio: 48 kHz stereo, present steady at about −7 dBFS RMS, past about −12 dBFS; the author listens before it stays. Full-hero crash: system RAM (167 GB) exhausted, most likely by whole-clip float copies in the TimeFlight nodes and ImageScale; fixed in a4c9fd8 (frame-by-frame conversion, resize inside the credits node). |
| 2026-10-18 (run 10) | Hero polish, stage 2 (`hero_s2_test`, then `hero_s2`) | 1024×576×361 → ×2 latent upscale + 3 steps → 2048×1152, composited, resized to 1920×1080 in the credits node; LTX audio crossfaded | A100-SXM4 80GB | test 102.1 s, full hero 1662.1 s for the graph | (not reported) | – | unstable 0.42% on the 15 s part; pop 2.09 (run 7: 0.56%, 2.11) | First 1080p hero, 649 frames with 48 kHz audio (about −24 dBFS RMS, peak 0.27). Same composition and timing as run 7 (stage 2 keeps the stage 1 layout); edges, windows, cars and shrine roofs are crisp where the 1024×576 hero upscaled is soft (gradient energy 6.8 vs 5.3 at frame 60, 5.2 vs 4.7 at frame 300). The memory fix held: no crash. Run 9's first retry stopped in 0.6 s on a NameError in TimeFlight Load Render Pass (fixed in 18214e3). |
| 2026-10-18 (run 11) | Texture test (`colab/texture_test.sh`): `hero_s2_test` × {current prompts, `*_mat` prompts} × guide strength {1.0, 0.8}, seed 42 | 576×320×49 → stage 2 → 1152×640 | A100-SXM4 80GB | 112.4 s (incl. load), then 75.5 s each | (not reported) | – | unstable 0.12–0.45%, about the same in all four | **Guide 0.8 rejected:** geometry drifts from the depth (in `past_base` the pyramid in the distance disappears and the layout behind it changes; in `past_zones` a palace becomes a pitched-roof mound), which would break the era merge and the time wave. **Material prompts:** the present day looks more like a photo (varied roof tones, clutter, dust), but the wording reads as decay: roofs look damaged and the Zócalo turns into a dirt lot. In 1519 the precinct reads as ruins, which is the wrong story for a living city. Texture numbers (surface detail on flat areas) moved by under 10%. Review sheets in `out/texture/review/`. |
| 2026-10-21 (run 12) | Texture test 2 (`colab/texture_test.sh tex_live`): `hero_s2_test` with the `*_live` prompts (surface and light wording, no decay words), guide strength 1.0, seed 42 | 576×320×49 → stage 2 → 1152×640 | A100-SXM4 80GB | 107.2 s (incl. load) | (not reported) | – | unstable 0.19–0.52% (base 0.16–0.45%) | **Rejected.** Surface detail moved by under 10% again, and the wording changes what the model builds, not only its surfaces: in `past_zones` a pitched-roof mound appears inside the palace enclosure (the same failure as guide 0.8), and in the present day the cathedral's domes turn into a flat tower. The 1519 city (`past_base`) looks slightly more photographic, but not enough to justify a third prompt set. **The current prompts stay**; the synthetic look is left to the optional finishing grade (`scripts/grade.py`). Sheets: `out/texture/review/live_*`. |

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
- **M3 test 1 (2026-10-17, 576×320×121, last 5 s):** all four runs worked (no init 51 s incl. load; init ~7–10 s).
  Init at 0.975 and 0.909: still water, the colors are ignored. Init at 0.725: land is land and the pyramid
  has its terraces, but the frames look like the flat-shaded render. Unstable fraction 0.75% (no init) →
  0.03% (0.725). `out/m3/init_test_sheet.jpg`. Test 2: brighter palette with clear land/water contrast and a
  light sky, starting at 0.909 / 0.85 / 0.8 (the last two are between official schedule steps).
- **M3 test 2 (2026-10-17):** the Colab notebook copy still had the test 1 cell, so it reran 0.975 / 0.909 / 0.725
  with the new palette. Same outcome: 0.909 water, 0.725 land but render-like. The palette alone does not move
  the threshold. The matrix now lives in `colab/m3_test.sh`. Test 3: a prompt that calls the precinct a dry paved
  plaza (`past_1519_plaza`), alone and with init 0.85 / 0.8.
- **M3 test 3 result (2026-10-17):** the prompt alone fixes it. `past_1519_plaza` with no init gives a dry,
  paved precinct and plaza with the pyramid on solid ground, and still looks like footage (unstable 1.1%, pop
  1.17). Every init run (0.85, 0.8, either prompt) still looks like the flat render. **RGB init is dropped**;
  the workflow stays in the repo as an option. Remaining issues: one palace courtyard became a pool and the
  ground is sandy rather than white stucco (`past_1519_plaza2` addresses both). Next: the full 15 s with both
  prompts (`colab/m3_full.sh`) to check that the lake and canals in the first 10 s still read correctly.
  Sheet: `out/m3/init_test3_sheet.jpg`.
- **M3 full 15 s (2026-10-17, 1024×576×361, 125.6 s and 125.5 s):** both dry-plaza prompts keep the precinct and
  plaza dry for the whole flight, the lake still surrounds the city, and stability matches the original
  (unstable 0.22% / 0.29%, flat across 60-frame segments). `past_1519_plaza2` is chosen for the hero: the
  Moctezuma palace courtyard is dry and gardens appear. Trade-off: the canal grid is much weaker than with
  `past_1519` (most canals read as dirt streets). Possible fix without GPU: both clips follow the same depth,
  so the precinct can be masked from `plaza2` and the rest of the city from `past_1519`, with the precinct wall
  as the seam. Sheet: `out/m3/full_sheet.jpg`.

## Decision
**GO with LTX-2.5** (2026-10-16). U4 passed 2026-10-17.
- U1 pass (edge overlay), U2 pass on review of contact sheets (no added cuts or zooms), U3 pass, U5 pass by
  a wide margin: a 121-frame 1024×576 clip costs ~40 s, and 241 frames ~80 s, so the hero needs no chaining.
- **U4 pass (2026-10-17):** all 3 viewers with no context identified Tenochtitlan (author's test; modern-artifact frame count not recorded).
  Original note: the author shows B_past_1519 to 3 people. Expected weakness: the synthetic boxes read as
  generic white houses on water. Period shapes come from the M2 proxies, not from the prompt.
- Settings carried forward: strength 1.0, inverse depth encoding, single stage (stage-2 upscale untested).
- M0 used about 0.22 GPU h of the 10 h cap.
