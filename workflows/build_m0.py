"""Build the M0 API-format graph: Blender depth video -> LTX-2.5 + Union Control (stage 1).

  .venv/Scripts/python workflows/build_m0.py   # writes workflows/api/m0_union_depth_api.json

Mirrors stage 1 of the official `LTX-2.5_ICLoRA_Union_Control_Distilled.json`
(ComfyUI-LTXVideo, example_workflows/2.5), minus the depth annotator and prompt enhancer:
our depth video goes straight into LTXAddVideoICLoRAGuide. Input names come from the node
source (see docs/decisions.md). Check them against a live server with
scripts/validate_workflow.py before running.

Nodes that app/run_m0.py changes are found by their `_meta.title`.
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).parent / "api" / "m0_union_depth_api.json"

# Official distilled schedule (8 steps) from the template.
DISTILLED_SIGMAS = "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"

MODEL_FILES = {
    "transformer": "ltx-2.5-22b-distilled-transformer-bf16.safetensors",
    "text_encoder": "gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
    "video_vae": "ltx-2.5-video-vae-bf16.safetensors",
    "audio_vae": "ltx-2.5-audio-vae-bf16.safetensors",
    "union_lora": "ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors",
}


class Graph:
    def __init__(self) -> None:
        self.nodes: dict[str, dict] = {}

    def add(self, class_type: str, title: str, **inputs) -> str:
        node_id = str(len(self.nodes) + 1)
        self.nodes[node_id] = {"class_type": class_type, "inputs": inputs, "_meta": {"title": title}}
        return node_id


def out(node_id: str, index: int = 0) -> list:
    return [node_id, index]


def build(width: int = 512, height: int = 288, length: int = 49, fps: float = 24.0,
          seed: int = 42, guide_strength: float = 1.0, lora_strength: float = 1.0) -> dict:
    g = Graph()
    unet = g.add("UNETLoader", "transformer", unet_name=MODEL_FILES["transformer"], weight_dtype="default")
    lora = g.add("LTXICLoRALoaderModelOnly", "union_lora", model=out(unet),
                 lora_name=MODEL_FILES["union_lora"], strength_model=lora_strength)
    clip = g.add("CLIPLoader", "text_encoder", clip_name=MODEL_FILES["text_encoder"], type="ltxv", device="default")
    pos = g.add("CLIPTextEncode", "positive_prompt", text="", clip=out(clip))
    neg = g.add("CLIPTextEncode", "negative_prompt", text="", clip=out(clip))
    cond = g.add("LTXVConditioning", "conditioning", positive=out(pos), negative=out(neg), frame_rate=fps)
    vae = g.add("VAELoader", "video_vae", vae_name=MODEL_FILES["video_vae"])
    avae = g.add("VAELoader", "audio_vae", vae_name=MODEL_FILES["audio_vae"])

    video = g.add("LoadVideo", "depth_video", file="control.mp4")
    frames = g.add("GetVideoComponents", "depth_frames", video=out(video))

    latent = g.add("EmptyLTXVLatentVideo", "video_latent", width=width, height=height, length=length, batch_size=1)
    guide = g.add("LTXAddVideoICLoRAGuide", "guide",
                  positive=out(cond, 0), negative=out(cond, 1), vae=out(vae), latent=out(latent),
                  image=out(frames, 0), frame_idx=0, strength=guide_strength,
                  latent_downscale_factor=out(lora, 1), crop="disabled",
                  use_tiled_encode=False, tile_size=256, tile_overlap=64)
    alat = g.add("LTXVEmptyLatentAudio", "audio_latent", frames_number=length, frame_rate=fps,
                 batch_size=1, audio_vae=out(avae))
    av = g.add("LTXVConcatAVLatent", "av_latent", video_latent=out(guide, 2), audio_latent=out(alat))

    noise = g.add("RandomNoise", "seed", noise_seed=seed)
    guider = g.add("CFGGuider", "guider", model=out(lora), positive=out(guide, 0), negative=out(guide, 1), cfg=1.0)
    sampler = g.add("KSamplerSelect", "sampler", sampler_name="euler_ancestral")
    sigmas = g.add("ManualSigmas", "sigmas", sigmas=DISTILLED_SIGMAS)
    sampled = g.add("SamplerCustomAdvanced", "sample", noise=out(noise), guider=out(guider),
                    sampler=out(sampler), sigmas=out(sigmas), latent_image=out(av))
    sep = g.add("LTXVSeparateAVLatent", "split_av", av_latent=out(sampled))
    crop = g.add("LTXVCropGuides", "crop_guides", positive=out(guide, 0), negative=out(guide, 1), latent=out(sep, 0))

    decoded = g.add("VAEDecodeTiled", "decode", samples=out(crop, 2), vae=out(vae),
                    tile_size=512, overlap=64, temporal_size=64, temporal_overlap=8)
    audio = g.add("LTXVAudioVAEDecode", "decode_audio", samples=out(sep, 1), audio_vae=out(avae))
    vid = g.add("CreateVideo", "create_video", images=out(decoded), fps=fps, audio=out(audio))
    g.add("SaveVideo", "save", video=out(vid), filename_prefix="timeflight/m0", format="auto", codec="auto")
    return g.nodes


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--models-json", type=Path,
                   help="timeflight_models.json from scripts/download_models.py (e.g. int8 filenames)")
    args = p.parse_args()
    if args.models_json:
        chosen = json.loads(args.models_json.read_text())
        MODEL_FILES.update({k: v for k, v in chosen.items() if k in MODEL_FILES})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(build(), indent=2) + "\n")
    print(f"wrote {OUT} with {MODEL_FILES}")
