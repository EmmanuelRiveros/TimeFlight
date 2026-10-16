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

## Not yet verified (open)
- Gemma 4 license terms (text encoder) — **must verify before M5**.
- Exact `reference_downscale_factor` stored in the Union Control LoRA metadata (we assume "ref0.5" means 2). The loader node reads it automatically, so nothing depends on our guess.
- Whether self-hosted ComfyUI driven through comfy-sdk qualifies (author is asking on Discord).
- Whether `client.assets.from_file` works with core `LoadVideo` through the proxy (fallback: copy the file into `ComfyUI/input`).
- Comfy int8 filenames in the LTX-2.5 repo (`scripts/download_models.py` lists the repo and picks by pattern).
- Depth-Anything-V2-Small license (believed Apache-2.0; Base/Large are CC BY-NC). Used only to compute metrics, never in the pipeline.
- Licenses of Piper voices (only relevant if narration returns).
