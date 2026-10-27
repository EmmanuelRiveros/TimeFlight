# Demo video script (≤ 2:00)

For the challenge submission: "Two minutes max", and every tool, model or technique other than ComfyUI
"must be explained in your demo video". Narration is about 270 words (about 2:00 at a calm pace); cut the
lines marked *(optional)* if it runs long.

Material is cut by `scripts/demo_material.py` into `out/demo/` (not committed):
`clips/` (clean 1920×1080 clips, 361 frames, all on the same camera clock), `comps/` (labelled clips,
dissolves, 2×2 mosaics), `stills/` (wave, credits, plan view). Rows marked **REC** are screen
recordings the author makes.

| Time | Picture | Narration / on-screen text |
|---|---|---|
| 0:00–0:10 | `clips/10_final_flight.mp4` from 0:03 (the wave leaves the Templo Mayor), with its LTX audio | *(no voice; title)* **TimeFlight** — a drone flight that turns back time. |
| 0:10–0:22 | `clips/07_ltx_present.mp4` → `stills/plan_1519_path.png` | "This is a drone flight over Mexico City's historic centre. But the camera is not invented: it follows real latitude, longitude and altitude keyframes, and the city under it is real data." |
| 0:22–0:38 | `clips/03_blender_present.mp4` → `clips/01_depth_present.mp4` | "Buildings come from OpenStreetMap, terrain from the Copernicus 30-metre elevation model. Blender builds the scene headless and renders a depth pass along the exact path." |
| 0:38–0:55 | **REC 4**: `places/tenochtitlan_zocalo/eras/1519.json` in an editor, one element with its sources and `confidence` → `clips/04_blender_1519.mp4` | "For 1519, an era manifest removes the modern city and adds the lake, causeways, canals, the sacred precinct and the Templo Mayor. Every element cites its sources and a confidence level: five of twenty-one are high or medium, the rest are labelled illustrative." |
| 0:55–1:12 | `comps/mosaic_pipeline_1519.mp4` (or `comps/dissolve_1519.mp4`) | "In ComfyUI, LTX-2.5 with the Union Control IC-LoRA follows that depth. The geometry fixes the layout; the model only renders the look. Three generations: today, the 1519 city, and its dry plazas, because depth alone can't tell a flat plaza from flat water." |
| 1:12–1:30 | **REC 1**: Comfy Desktop with `workflows/ui/timeflight_hero.json`: whole graph, then zoom into the TimeFlight group | "Four custom TimeFlight nodes do the rest on the CPU: an era merge by world zone, a time wave that grows from the Templo Mayor using both depth passes, so a tower changes era as a whole, and an audio crossfade." |
| 1:30–1:42 | `stills/credits_sources.png` → `stills/credits_models.png` | "The credits are generated from the same manifest: sources, data, models and their licenses. The node refuses a final render while any license is unverified. LTX is open-weight under a revenue-gated license, so this is not fully open source, and the video says so." |
| 1:42–1:52 | **REC 2 / REC 3**: the README "Open in Colab" badge, the notebook running on an A100; the GitHub release | "Everything runs as one ComfyUI graph driven through the Comfy SDK. The repo has a Colab notebook and the depth passes, so anyone with an A100 can reproduce it." *(optional)* |
| 1:52–2:00 | `clips/10_final_flight.mp4`, last seconds (`stills/1519_end.png` to hold) + repo URL | "TimeFlight. A real place, real coordinates, and a past you can check." **github.com/EmmanuelRiveros/TimeFlight · #OpenSourceHiggsfield** |

## Tools to name (challenge rule) — all covered above
ComfyUI · Comfy SDK + comfy-api-proxy · LTX-2.5 (Lightricks) · LTX-2.3 Union Control IC-LoRA · Gemma 4
text encoder (inside LTX) · Blender 4.4 · OpenStreetMap (Geofabrik) · Copernicus DEM GLO-30 · Colab A100.

## Notes for the edit
- **Audio:** use the hero's own LTX soundtrack (`clips/10_final_flight.mp4`, `clips/11_hero_full_with_credits.mp4`).
  Any music must be licensed for this use; none is in the repo.
- **Disclaimer on screen** whenever 1519 footage plays for long: the hero already burns in
  "Tenochtitlan, 1519 · illustrative AI reconstruction".
- **Do not show** Google Maps/Earth or any other excluded service for context; use `stills/plan_1519_path.png`.
- **Thumbnail:** `stills/wave_mid.png`.
- **9:16 cut for the social post:** the teaser from run 8 (`teaser_1519.mp4` in `MyDrive/timeflight/flight_results/`).
