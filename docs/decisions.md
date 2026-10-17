# Decisions and verified facts

This file overrides `docs/research.md`. Newest entries go at the bottom of each section.

## Decisions

| ID | Date | Decision | Why |
|---|---|---|---|
| D1 | 2026-10-16 | Enter **Most Innovative** plus the **LTX VFX bonus**. | Only one main category is allowed. The innovation case (geo → camera, era geometry, provenance) is our strongest. |
| D2 | 2026-10-16 | Revenue-gated licenses (LTX-2.x Community, Gemma) are acceptable. The README says so plainly and never claims "fully open". | The author's revenue is far below the thresholds. Honesty matters more than the label. |
| D3 | 2026-10-16 | Drop FLUX 3 Klein. | FLUX Kommunity License is non-commercial by default. |
| D4 | 2026-10-16 | Main code is Apache-2.0; `blender/` is GPL-3.0 and runs only as a subprocess. | Scripts that use `bpy` are treated as GPL. The process boundary keeps the node pack Apache. |
| D5 | 2026-10-16 | Cut now: narration/TTS (back only if M4 finishes early), Gradio app, live Comfy API link, Sonora shot. | Real budget is ~40 h. |
| D6 | 2026-10-16 | Deliverables: 15 s hero in 16:9 plus a 9:16 teaser. Path: from the Tacuba causeway (west), flying east to above Templo Mayor / Zócalo. | Agreed with the author. |
| D7 | 2026-10-16 | M0 hard cap is 10 GPU hours. If LTX is not usable by then, choose switch (Wan VACE) or descope using the data already collected. | Limited Colab units. |
| D8 | 2026-10-16 | M0 uses a hand-built API-format graph (`workflows/build_m0.py`), not the official template. It feeds Blender depth straight into `LTXAddVideoICLoRAGuide` with no annotator. | The official template always derives depth with VideoDepthAnything. We want our own exact, stable depth, and a graph the SDK can drive. |
| D9 | 2026-10-16 | Narration script (if restored): original text written from the listed sources. No copied Wikipedia text. | Author's request; avoids CC BY-SA share-alike. |
| D10 | 2026-10-16 | **M0 GO: LTX-2.5 + Union Control IC-LoRA** with Blender depth injected directly. Strength 1.0, inverse encoding, 1024×576, up to 241 frames in one pass (no chaining). U4 (era read) passed 2026-10-17: 3 of 3 viewers identified Tenochtitlan. | docs/smoke_test.md: structure and stability pass; ~40 s per 121-frame clip on A100 80GB. |
| D11 | 2026-10-16 | OSM comes from the Geofabrik Mexico extract read with pyosmium, not Overpass. Building outlines that contain `building:part`s are dropped (Simple 3D Buildings rule). The DSM gets a 7 px grey opening (~210 m) so OSM buildings are not doubled by the DSM's own building bumps. | Public Overpass instances returned 500/502/504 for real queries on 2026-10-16. The extract is dated and checksummed, so it is reproducible. |
| D12 | 2026-10-17 | 1519 proxy layout approved. Shoreline and canal grid stay labeled illustrative. Templo Mayor and precinct dimensions cite Matos Moctezuma, *The Great Temple of the Aztecs* (1988) and *Vida y muerte en el Templo Mayor* (1986), facts only. | Author review of docs/era_1519_review.md. No exact micro-topographic record of the shoreline or canals exists. |
| D13 | 2026-10-17 | 1519 hero prompt is `past_1519_plaza2` (no RGB init). The time wave is a composite of the present and 1519 clips, not a third generation: a soft vertical cylinder grows from the Templo Mayor (30 m → 15 km, 3.5–12.5 s), with per-pixel layer choice from both Blender depth passes and a warm glow at the front. | RGB init made frames look like the flat render at every setting that kept land dry (smoke_test.md, M3 tests 1–3). Compositing costs no GPU time and the depth passes are exact, so the front follows real geometry. |
| D14 | 2026-10-17 | The 1519 layer is a merge of two generations along the same depth: the era's dry zones (precinct, plazas, named structures, from `timeflight.era.build.dry_zones`) come from `past_1519_plaza2`, the rest of the city from `past_1519` (`scripts/era_merge.py`, 6 m soft seam). The time wave now grows continuously from radius 0 and its glow fades in over the first 250 m. | Author review of timewave_v2: the canals were lost with the dry-plaza prompt, and the Templo Mayor flashed yellow before the ring formed (the radius jumped from 0 to 30 m with a 25 m glow band). |

## Verified facts (checked 2026-10-16 against primary sources)

| Item | Fact | Source |
|---|---|---|
| LTX-2.5 | 22B, gated on HF. Distilled and dev bf16, Comfy int8, NVFP4 (Blackwell only). frames % 8 == 1; W/H divisible by 32. | huggingface.co/Lightricks/LTX-2.5 |
| LTX license | LTX-2.x Community License: free commercial use for entities under $10M annual revenue. | same |
| Union Control | No LTX-2.5-specific adapter exists. The official LTX-2.5 workflow reuses `ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors` from `Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control` (depth/canny/pose). | ComfyUI-LTXVideo `example_workflows/2.5/LTX-2.5_ICLoRA_Union_Control_Distilled.json` |
| Official files | `diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors`, `text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors`, `vae/ltx-2.5-video-vae-bf16.safetensors` (or `-conv-`), `vae/ltx-2.5-audio-vae-bf16.safetensors`, `latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors` | same workflow, "Model Links" note |
| Template pipeline | Reference video → VideoDepthAnything (vits, "gray") → resized so the short side is 544 → stage 1: 8 distilled steps (ManualSigmas `1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0`, euler_ancestral, cfg 1) → `LTXVCropGuides` → stage 2: ×2 latent upscale, 3 steps. Default 960×544, 121 frames, 24 fps. | same |
| IC-LoRA nodes | `LTXICLoRALoaderModelOnly(model, lora_name, strength_model)` → (model, latent_downscale_factor read from LoRA metadata). `LTXAddVideoICLoRAGuide(positive, negative, vae, latent, image, frame_idx, strength, latent_downscale_factor, crop, use_tiled_encode, tile_size, tile_overlap, [use_streaming, tile_frames])`. **`attention_strength` and `attention_mask` exist only on `LTXAddVideoICLoRAGuideAdvanced`.** | ComfyUI-LTXVideo `iclora.py` |
| Core LTX nodes | `EmptyLTXVLatentVideo(width,height,length,batch_size)`, `LTXVConditioning(positive,negative,frame_rate)`, `LTXVCropGuides(positive,negative,latent)`, `LTXVConcatAVLatent(video_latent,audio_latent)`, `LTXVSeparateAVLatent(av_latent)`, `LTXVEmptyLatentAudio(frames_number,frame_rate,batch_size,+audio_vae)` | ComfyUI `comfy_extras/nodes_lt.py`, `nodes_lt_audio.py` |
| Comfy SDK | `pip install comfy-sdk` (0.5.0). `COMFY_BASE_URL` env var. `client.workflows.from_file`, `wf.set_input(node, input, value)`, `client.assets.from_file`, `client.run`, `job.get_outputs(node)`, `output.to_file`. | docs.comfy.org/development/api-development/sdks |
| comfy-api-proxy | MIT, 0.1.6. Defaults: ComfyUI at 127.0.0.1:8188, serves on :8189. `--comfyui-base-dir` enables uploads. Ships `demo/fake_comfyui.py` for no-GPU tests. | PyPI / github.com/Comfy-Org/comfy-api-proxy |
| Challenge | Deadline Oct 30 9am PT. One main category plus one model bonus. LTX bonus requires LTX as the core video model. Video ≤ 2 min. #OpenSourceHiggsfield. Must be built with Comfy API/Router/SDK. | comfy.org/open-source-higgsfield |
| FLUX 3 Klein | FLUX Kommunity License: non-commercial by default. | HF black-forest-labs/flux-3-image-klein-comfy |
| Wan 2.1 VACE 14B | Apache-2.0; 480p/720p, ~81 frames. | HF Wan-AI/Wan2.1-VACE-14B |
| Piper | The old repo is archived (MIT). Development moved to OHF-Voice/piper1-gpl (GPL-3.0, `pip install piper-tts`). | GitHub |
| Local machine | Blender 4.4.3, RTX 3070 Ti 8 GB, Python 3.11 (`py`), uv, ffmpeg. | checked locally |
| Guide size rule | With the ref0.5 Union IC-LoRA, `LTXAddVideoICLoRAGuide.encode` resizes the guide to `latent_size × 32 / latent_downscale_factor` and VAE-encodes it, so that size must itself be divisible by 32. The first Colab run (2026-10-16) failed at 512×288 and 960×544 with an einops error in the VAE encoder (`causal_video_autoencoder.py` space-to-depth), consistent with a factor of 2 (288/2 = 144, 144/32 = 4.5). **Width and height must be multiples of 64.** M0 now uses 576×320 and 1024×576. The official template avoids this because stage 1 runs at half resolution. | ComfyUI-LTXVideo `iclora.py` @44c8a36; ComfyUI @fc04da9; Colab log |
| Copernicus DEM GLO-30 Public | Licence "for the use of the Copernicus WorldDEM-30": free, worldwide, rights of reproduction, distribution, communication to the public, adaptation and combination; **no purpose restriction**, so commercial use is allowed. Modified data needs the notice "produced using Copernicus WorldDEM-30 © DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018 provided under COPERNICUS by the European Union and ESA; all rights reserved", plus the sentence "The organisations in charge of the Copernicus programme by law or by delegation do not incur any liability for any use of the Copernicus WorldDEM-30". No endorsement implied. Note: the separate GLO-30-**R** licence (restricted users) is non-commercial; it does not apply to the public tiles. Tiles: `s3://copernicus-dem-30m/Copernicus_DSM_COG_10_N19_00_W100_00_DEM/...tif` (public HTTPS, no account). It is a **surface** model (buildings and trees included); heights are orthometric (EGM2008). | License-COPDEM-30.pdf (documentation.dataspace.copernicus.eu), registry.opendata.aws/copernicus-dem |
| OSM extract | Geofabrik `north-america/mexico-latest.osm.pbf`, 617 MB, data to 2026-10-15T20:21Z, ODbL 1.0, md5 checked. Historic centre: 7220 footprints, 95% without height or levels (default 9 m). Landmarks with heights: Catedral 67 m (towers; parts: dome 55 m, naves 40 m), Bellas Artes 53 m, Torre Latinoamericana parts up to 182 m. Templo Mayor: 19.43489, −99.13172. | download.geofabrik.de; extracted locally |
| Libraries | rasterio 1.4.4 (BSD-3, bundles GDAL), pyosmium 4.3.1 (BSD-2). Both are only used by `scripts/`. | PyPI |
| Asset upload via proxy | `client.assets.from_file` with core `LoadVideo` works through comfy-api-proxy (M0 run 2 used it). | M0 run 2 |

## Not yet verified (open)
- Gemma 4 license terms (text encoder) — **must verify before M5**.
- Whether self-hosted ComfyUI driven through comfy-sdk qualifies (author is asking on Discord).
- Comfy int8 filenames in the LTX-2.5 repo (`scripts/download_models.py` lists the repo and picks by pattern).
- Depth-Anything-V2-Small license (believed Apache-2.0; Base/Large are CC BY-NC). Used only to compute metrics, never in the pipeline.
- Licenses of Piper voices (only relevant if narration returns).
