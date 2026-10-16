# TimeFlight

**Documentary drone flyovers over real places, built from open geodata, then re-rendered as the
place looked in an earlier era.**

> Status: early development for the Comfy "Open Source Higgsfield" challenge (Oct 2026).
> Nothing here is usable yet.

## Idea
1. Input: a place, camera-path coordinates (lat/lon/alt keyframes) and a target year.
2. Blender (headless) builds the scene from open data only (Copernicus DEM terrain and
   OpenStreetMap buildings) and renders depth, edge and RGB passes along the real path.
3. Era geometry: modern buildings are removed and simple period proxies are added from an
   era manifest, so the depth describes the layout of that time, not today's.
4. ComfyUI: depth-guided video restyle with LTX-2.5 and the Union Control IC-LoRA.
5. "Time wave": a depth-based mask sweeps the past over the present during the flight.
6. The era manifest (place, year, references, sources, licenses) generates the credits and an SRT.
7. Everything is driven through the Comfy SDK.

Output is an **illustrative AI reconstruction**, not a historical record. Every source is cited.

## Models and licenses (honest version)
The stack uses open-weights models, but not all of them are OSI-open:
- LTX-2.5 and the LTX-2.3 Union Control IC-LoRA: LTX-2.x Community License (free for
  entities under $10M annual revenue).
- Gemma 4 text encoder: license terms to be documented.

We use no Google, Esri, Mapbox, Cesium ion or Bing data. See `ASSETS.md` (generated) for every
asset with its source and license.

## Repository layout
- `timeflight/`: pure-Python core (geo math, era manifest, provenance, time-wave masks, metrics)
- `blender/`: headless Blender scripts (**GPL-3.0**, run as subprocesses)
- `workflows/`: ComfyUI graphs (API format in `workflows/api/`)
- `app/`: comfy-sdk drivers
- `colab/`: notebook for running on a Colab A100
- `places/`: per-place camera paths, prompts and era manifests

## License
Apache-2.0, except `blender/`, which is GPL-3.0.
