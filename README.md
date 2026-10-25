# TimeFlight

**A drone flight over a real place that turns back time as it flies.** The camera follows real
coordinates over today's city. A wave spreads out from a historic landmark, and behind it the city
becomes what stood there in an earlier year. The hero flight goes from Mexico City's historic centre
to Tenochtitlan in 1519.

TimeFlight is a ComfyUI pipeline built from open geodata, period geometry with cited sources,
and LTX-2.5 video generation guided by depth. Every source behind the reconstruction, and every
asset and model it uses, appears in the credits with its license. The credits are generated from
the same manifest that builds the geometry.

> Entry for the Comfy "Open Source Higgsfield" challenge (October 2026), category Most Innovative,
> with the LTX VFX bonus.
>
> **The output is an illustrative AI reconstruction, not a historical record.** Of the 21 elements in
> the 1519 scene, 5 are placed with high or medium confidence. The other 16 are low-confidence or
> illustrative, and the credits say which is which.

## How it works

```
place.json (lat/lon/alt keyframes)    eras/1519.json (proxies + sources + licenses)
        │                                      │
        ▼                                      ▼
 Copernicus DEM + OSM ──► Blender (headless) ──► depth passes along the real path
                                │                   present-day city   │   1519 city
                                ▼                                      ▼
                     ComfyUI: LTX-2.5 + Union Control IC-LoRA, depth-guided
                       present        past_base (city)      past_zones (precinct)
                                │                   └── Era Zone Merge ──┘
                                └──────────► Time Wave ◄─────────┘
                                               │  (+ dust curtain on the front)
                                               ▼
                          Captions and Credits (from the manifest) ──► hero.mp4
```

1. **Real camera path.** Keyframes in WGS84 lat/lon/alt are converted to a local ENU frame around the
   place's origin and eased into a per-frame camera (`timeflight/geo`, `places/<id>/place.json`).
2. **Present-day scene from open data.** The terrain comes from Copernicus GLO-30 and the building
   footprints and heights from OpenStreetMap (Geofabrik extract read with pyosmium). Blender builds the
   scene headless and renders 16-bit depth along the path (`scripts/fetch_geodata.py`,
   `scripts/make_scene.py`, `blender/render_place.py`).
3. **Era geometry.** An era manifest (`places/<id>/eras/1519.json`) removes the modern buildings and
   adds period proxies: the lake, the causeways, the canal grid, the houses, the walled sacred precinct,
   the Templo Mayor and the palaces. Each element cites its sources, their licenses and a confidence
   level. The 1519 depth pass therefore describes the layout of that time, not today's.
4. **Depth-guided generation.** The Blender depth, encoded as inverse depth, is fed straight into
   `LTXAddVideoICLoRAGuide` with the Union Control IC-LoRA, with no depth estimator in between. The
   model only re-renders the look; the layout comes from the geometry. There are three generations:
   the present day, the 1519 city, and the 1519 dry zones (precinct and plazas, which a "lake city"
   prompt otherwise floods). Optional stage 2 upscales ×2 in latent space and refines for 1080p.
5. **Era Zone Merge.** The two 1519 generations follow the same depth, so they are merged by world
   zone from the manifest, with a soft 6 m seam.
6. **Time Wave.** The present and past clips are composited per pixel, using both depth passes: a
   soft vertical cylinder grows from the era's anchor (the Templo Mayor) from 0 m to 15 km. The front
   follows the real geometry, so a tower changes era as a whole instead of being sliced. A warm glow
   and a wall of glowing dust on the front hide tall buildings while they change. All of this is
   compositing, so no GPU time is spent on it.
7. **Captions and credits from the manifest.** The cards list the historical sources, the data, the
   models, the software and the fonts, each with its license, plus the confidence levels and a
   disclaimer. The node refuses a final render while any license is unverified.
8. **Audio.** LTX-2.5 generates audio with each clip. The present soundtrack crossfades into the
   1519 one on the wave's timing.

Everything runs as one ComfyUI API graph driven through the Comfy SDK (`comfy-sdk` → `comfy-api-proxy`
→ ComfyUI).

To see the graph, drag [`workflows/ui/timeflight_hero.json`](workflows/ui/timeflight_hero.json) onto the
ComfyUI canvas (it needs ComfyUI-LTXVideo and the TimeFlight node pack). It is the 1080p hero graph,
82 nodes in groups with notes, generated from the same builder by `workflows/build_ui.py`, with the
hero's prompts and timings filled in.

## ComfyUI nodes (`comfy_nodes/`)

| Node | What it does |
|---|---|
| TimeFlight: Load Render Pass | Loads a Blender render folder: the depth control video for the LTX guide, plus the depth and camera data for compositing |
| TimeFlight: Era Zone Merge | Merges two era generations by the manifest's world zones |
| TimeFlight: Time Wave | Present → past composite along a growing world-space front, with glow and dust curtain |
| TimeFlight: Captions and Credits | Burns in captions and appends credit cards generated from the manifest and the asset log |
| TimeFlight: Audio Crossfade | Crossfades the present and past soundtracks on the wave's timing and fades out before the cards |

The nodes are thin wrappers over `timeflight/`, a pure-Python package (numpy and Pillow) with pytest
coverage. The same functions drive the local scripts.

## Running it

TimeFlight runs in two stages. The Blender stage runs on any machine with Blender 4.4. The generation
needs a large GPU. We used a Colab A100 80GB; the same scripts also run on your own Linux GPU machine
(section 3).

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EmmanuelRiveros/TimeFlight/blob/main/colab/timeflight.ipynb)
`colab/timeflight.ipynb` sets everything up, runs a 5 s test and then the 1080p hero.

### 1. Locally: geodata, scenes, depth passes

```bash
uv venv --python 3.11 && uv pip install -e ".[geo,dev]"
python scripts/fetch_geodata.py places/tenochtitlan_zocalo/place.json
python scripts/make_scene.py places/tenochtitlan_zocalo/place.json
python scripts/make_scene.py places/tenochtitlan_zocalo/place.json --era places/tenochtitlan_zocalo/eras/1519.json

# Depth passes: present day and 1519 (start small: --frames 49 --width 576 --height 320)
blender -b --factory-startup -P blender/render_place.py -- --scene data/places/tenochtitlan_zocalo/scene \
    --out out/place/present_r1024_361 --frames 361 --width 1024 --height 576
blender -b --factory-startup -P blender/render_place.py -- --scene data/places/tenochtitlan_zocalo/scene \
    --out out/place/1519_r1024_361 --frames 361 --width 1024 --height 576 --era 1519
python scripts/make_control.py out/place/present_r1024_361 --modes inverse
python scripts/make_control.py out/place/1519_r1024_361 --modes inverse

python scripts/pack_m0.py --src out/place --folders present_r1024_361 1519_r1024_361 \
    --out out/flight_inputs.zip --with-depth
```

Upload the zip to `MyDrive/timeflight/`.

### 2. On Colab: generation and compositing

Open `colab/timeflight.ipynb` (badge above) on an A100 80GB runtime. It clones the repo, then runs
`colab/setup_comfy.sh` (ComfyUI, ComfyUI-LTXVideo, the Comfy SDK and proxy, and the TimeFlight node
pack), `scripts/download_models.py` (66 GB of weights) and `colab/start_servers.sh`. It needs the Colab
secret `HF_TOKEN`, from a Hugging Face account that has accepted the license on the gated
`Lightricks/LTX-2.5` and `Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control` repos. The depth passes for the
hero are downloaded from the [`inputs-v1` release](https://github.com/EmmanuelRiveros/TimeFlight/releases/tag/inputs-v1)
(renders produced from OpenStreetMap and Copernicus data, attribution in the release notes), so step 1
is optional; to use your own renders, upload the zip to `MyDrive/timeflight/` instead. `colab/timeflight_m0.ipynb` is the full development log (every test in
`docs/smoke_test.md`). Both notebooks are generated by `colab/make_notebook.py`.

```bash
bash colab/flight.sh flight_inputs.zip hero        # 1024×576, 15 s + credits
bash colab/flight.sh polish2_inputs.zip hero_s2    # stage 2, 1920×1080
```

### 3. On your own GPU machine

Tested only on an A100 80GB with about 170 GB of system RAM (a run used about 80 GB of it).
Smaller GPUs are untested. Linux or WSL2, bash, git, ffmpeg and about 70 GB of free disk for the weights.

```bash
git clone https://github.com/EmmanuelRiveros/TimeFlight && cd TimeFlight
uv venv --python 3.11 && source .venv/bin/activate
uv pip install pip      # then install CUDA PyTorch for your driver: https://pytorch.org/get-started/locally/
export TF_WORK=~/timeflight-work        # ComfyUI, logs and results go here (default on Colab: /content)
export HF_TOKEN=hf_...                  # access to the gated Lightricks repos (see above)
bash colab/setup_comfy.sh               # or TF_COMFY=/path/to/ComfyUI to use an existing checkout
python scripts/download_models.py --comfy-dir $TF_WORK/ComfyUI --precision auto --with-upscaler
bash colab/start_servers.sh             # ComfyUI on :8188, comfy-api-proxy on :8189
export COMFY_BASE_URL=http://127.0.0.1:8189

# Pass the folder that holds the render folders from step 1, a zip made by pack_m0.py, or the
# release zip (github.com/EmmanuelRiveros/TimeFlight/releases/tag/inputs-v1):
bash colab/flight.sh out/place hero_s2  # results in $TF_WORK/flight_results/
```

`app/run_flight.py --preset <name> --dry-run` prints every input it would set. The presets cover the
16:9 hero, the 9:16 teaser and a small test for each (576×320 or 320×576, 49 frames). Run the small
test first. To change the graph, edit `workflows/build_flight.py` and check the result with
`scripts/validate_workflow.py`.

An optional finishing grade adds local contrast and fine film grain to counter the smooth, synthetic
look of diffusion output. It runs on CPU, after the GPU stage, and grades the whole file, credit cards
included:

```bash
python scripts/grade.py hero.mp4 --out hero_graded.mp4 --compare compare.mp4   # compare: original above, graded below
```

### Tests

```bash
python -m pytest
```

The tests run on CPU and need neither ComfyUI nor Blender.

## Models, data and licenses

The stack uses open-weight models, but **it is not fully open source**: LTX-2.5 and the Union Control
IC-LoRA are under a revenue-gated license.

| Component | License |
|---|---|
| LTX-2.5 (distilled transformer, video and audio VAEs, ×2 latent upscaler), Lightricks | LTX-2.x Community License: free for entities under $10M annual revenue |
| LTX-2.3 IC-LoRA Union Control (ref0.5), Lightricks | LTX-2.x Community License |
| Gemma 4 12B text encoder (LTX projection) | Apache-2.0 (ships in the LTX-2.5 repo) |
| ComfyUI | GPL-3.0 (run as a separate program) |
| ComfyUI-LTXVideo | LTX-2 Community License Agreement |
| Blender 4.4 | GPL-3.0-or-later (run as a separate program) |
| comfy-sdk, comfy-api-proxy | MIT |
| Copernicus DEM GLO-30 | Copernicus WorldDEM-30 licence (free use with attribution) |
| OpenStreetMap, Geofabrik extract | ODbL 1.0; derived data is not committed (`data/` is gitignored) |
| Source Sans 3 (captions and cards) | SIL Open Font License 1.1 |
| Depth-Anything-V2-Small (evaluation only, not in the pipeline) | Apache-2.0 |

- **Excluded services.** No Google Maps/Earth/3D Tiles, Esri, Mapbox, Bing, Cesium ion, Mapillary or
  David Rumsey data is used, and no FLUX model.
- **Asset log.** Every downloaded asset is logged with its source, license, attribution and retrieval
  date in `assets.jsonl`; `ASSETS.md` is generated from it.
- **Historical sources.** They are listed with their licenses in
  `places/tenochtitlan_zocalo/credits_1519.md`: the 1524 Nuremberg map (Newberry Library), the Uppsala
  map (ca. 1550), the Florentine Codex, Cortés' Second Letter and Díaz del Castillo (public-domain
  translations), and dimensions after Matos Moctezuma (facts only). The review of each element is in
  `docs/era_1519_review.md` and `docs/sources_1519.md`.

## Limitations

- **The reconstruction is illustrative.** No exact record of the 1519 shoreline or canal grid exists,
  and most proxies are labeled low-confidence or illustrative.
- **The 1519 buildings are simple proxies.** Their shapes come from the manifest geometry; the model
  adds surface detail but cannot add historical accuracy.
- **Depth guidance cannot tell flat dry ground from flat water.** Open plazas are kept dry by a
  separate prompt and the zone merge (see `docs/smoke_test.md`, M3).
- **Running the generation needs a large GPU and gated model access.** The local Blender stage is
  free.

## Repository layout

| Path | Contents |
|---|---|
| `timeflight/` | Pure-Python core: geo math, era manifests, provenance and credits, time wave, era merge, audio mix, metrics (Apache-2.0) |
| `comfy_nodes/` | The ComfyUI-TimeFlight node pack |
| `blender/` | Headless Blender scripts, **GPL-3.0**, always run as `blender -b -P …` subprocesses |
| `workflows/` | Graph builders: API format in `workflows/api/` (run by the SDK), canvas format in `workflows/ui/` (open in ComfyUI) |
| `app/` | Comfy SDK drivers (`run_flight.py`) |
| `colab/` | Setup and run scripts for the GPU machine, and the notebook generator |
| `places/` | Per-place camera paths, prompts, era manifests and generated credits |
| `scripts/` | Geodata fetch, scene build, control videos, packaging, credits, local compositing tools |
| `docs/` | `decisions.md` (dated decisions and verified facts), `smoke_test.md` (every GPU run, with timings), source reviews |

## License

Apache-2.0 (`LICENSE`), except `blender/`, which is GPL-3.0 (`blender/LICENSE`).
Models, data and fonts keep their own licenses, listed above and in `ASSETS.md`.

#OpenSourceHiggsfield
