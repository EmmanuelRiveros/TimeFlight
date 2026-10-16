# TimeFlight: rules for Claude

Open-source ComfyUI project for the Comfy "Open Source Higgsfield" challenge,
category **Most Innovative** plus the **LTX VFX bonus**.
Deadline: Fri Oct 30 2026 09:00 PT. **We submit Thu Oct 29 before 21:00.**
Budget: about 40 h total (Oct 16–29). Hero demo: Tenochtitlan (1519) → Zócalo, Mexico City.

- `docs/research.md`: original research (Spanish). It contains unverified model names.
- `docs/decisions.md`: verified facts and dated decisions. **This file wins over research.md.**
- `docs/smoke_test.md`: M0 go/no-go test protocol and results.

## Data and model licensing (hard rules)
- Use only open data, and models whose licenses allow commercial use. Revenue-gated
  licenses (LTX-2.x Community, Gemma) are allowed. Record them in docs/decisions.md and
  describe them honestly in the README. Never call the stack "fully open".
- NEVER use: Google 3D Tiles / Google Earth / Google Maps, Cesium ion, Esri, Mapbox,
  Bing, David Rumsey, Mapillary, FLUX (any FLUX Kommunity-licensed model). Do not use
  library defaults that point to them (CesiumJS ion/Bing, Blosm Google import, etc.).
- Log every downloaded asset (DEM, OSM extract, image, model weight, voice) with
  `timeflight.assets_log.log_asset(...)`: source URL, license, attribution, retrieval date.
  If an asset has no log entry, do not use it.
- Do not commit derived OSM databases (ODbL). Commit recipes and filters only; `data/` is gitignored.
- Historical references must be public domain, CC0 or CC BY, with provenance in the
  era manifest. The user checks every source before it is modeled.

## Code
- Code, comments, docs, README and commit messages are in English.
- `timeflight/` is pure Python (numpy allowed; no `bpy`, no ComfyUI imports) and is
  tested with pytest on Windows CPU.
- `blender/` is GPL-3.0. Run it only as a subprocess:
  `blender -b -P blender/<script>.py -- <args>`. Target Blender 4.4.
  Local binary: `E:\Archivos de programa\Blender\blender.exe` (env var `BLENDER`).
- GPL tools (e.g. Piper) are called as subprocesses, never imported.
- Everything outside `blender/` is Apache-2.0.
- ComfyUI graphs: `workflows/api/` holds API-format JSON used by comfy-sdk
  (`COMFY_BASE_URL` → comfy-api-proxy :8189 → ComfyUI :8188). Node input names
  must match the node source or live `/object_info`; validate with
  `scripts/validate_workflow.py` before spending GPU time.
- Local Python: `py -3.11` (no `python` on PATH). Use uv for environments.

## Compute
- Code is written locally on Windows (RTX 3070 Ti 8 GB, enough for Blender, not for LTX).
  All diffusion runs on a Colab A100.
- **Test small first**: 49 frames at 512×288 before 121 frames at 960×544.
- LTX constraints: frames % 8 == 1; width and height divisible by 32; no NVFP4 on A100.
- **M0 hard cap: 10 GPU hours.** Log time and units per run in docs/smoke_test.md.
- Render Blender passes (depth, RGB, edges) locally, never on Colab.

## Verification
- Check model names, versions, filenames and APIs against official docs, repos or
  source before depending on them. Say explicitly what could not be verified.

## Scope
- Never cut: the real-coordinate camera path, era-edited geometry, manifest-driven credits.
- Already cut (as of Oct 16): narration/TTS (comes back only if M4 finishes early), the Gradio app,
  the live Comfy API link, the Sonora shot, and FLUX keyframes.
- If late, cut in this order: clip chaining, then the 9:16 teaser.
- Deliverables: a 15 s 16:9 hero, a 9:16 teaser, and a ≤2 min demo video.
