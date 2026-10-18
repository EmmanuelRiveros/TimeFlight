"""Build the full TimeFlight API-format graph: three LTX-2.5 generations -> era merge -> time wave ->
captions and credits -> one video.

  .venv/Scripts/python workflows/build_flight.py   # writes timeflight_flight_api.json and timeflight_flight_s2_api.json

Generations (all on the same model, guide strength 1.0, inverse depth, distilled schedule):
  present     present-day control, prompt `present`
  past_base   era control, prompt `past_base`   (the city: canals, houses, lake)
  past_zones  era control, prompt `past_zones`  (dry precinct, plazas and palaces)
TimeFlight nodes (comfy_nodes/) then merge the two era clips by world zone, composite the time wave
from the era anchor, burn captions and append manifest-driven credits. app/run_flight.py sets the
inputs through the Comfy SDK; nodes it changes are found by `_meta.title`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from build_m0 import DISTILLED_SIGMAS, MODEL_FILES, Graph, out  # noqa: E402

OUT = Path(__file__).parent / "api" / "timeflight_flight_api.json"
OUT_S2 = Path(__file__).parent / "api" / "timeflight_flight_s2_api.json"
UPSCALER_FILE = "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"
# Stage 2 as in the official LTX-2.5 Union Control template: x2 latent upscale, then the last three
# distilled steps (euler, cfg 1) on the cropped-guide conditioning, with no new depth guide.
STAGE2_SIGMAS = "0.909375, 0.725, 0.421875, 0.0"
PLACE_DIR = "/content/TimeFlight/places/tenochtitlan_zocalo"
BRANCHES = ("present", "past_base", "past_zones")


def build(width: int = 1024, height: int = 576, length: int = 361, fps: float = 24.0, seed: int = 42,
          guide_strength: float = 1.0, stage2: bool = False, out_size: tuple[int, int] = (1920, 1080)) -> dict:
    """`width`/`height` are the stage 1 size (the depth passes' size). With `stage2`, each generation is
    upscaled x2 in latent space and refined, and the composite is resized to `out_size`."""
    g = Graph()
    unet = g.add("UNETLoader", "transformer", unet_name=MODEL_FILES["transformer"], weight_dtype="default")
    lora = g.add("LTXICLoRALoaderModelOnly", "union_lora", model=out(unet),
                 lora_name=MODEL_FILES["union_lora"], strength_model=1.0)
    clip = g.add("CLIPLoader", "text_encoder", clip_name=MODEL_FILES["text_encoder"], type="ltxv", device="default")
    vae = g.add("VAELoader", "video_vae", vae_name=MODEL_FILES["video_vae"])
    avae = g.add("VAELoader", "audio_vae", vae_name=MODEL_FILES["audio_vae"])
    neg = g.add("CLIPTextEncode", "negative_prompt", text="", clip=out(clip))
    sampler = g.add("KSamplerSelect", "sampler", sampler_name="euler_ancestral")
    sigmas = g.add("ManualSigmas", "sigmas", sigmas=DISTILLED_SIGMAS)
    if stage2:
        upscaler = g.add("LatentUpscaleModelLoader", "latent_upscaler", model_name=UPSCALER_FILE)
        sampler2 = g.add("KSamplerSelect", "sampler_stage2", sampler_name="euler")
        sigmas2 = g.add("ManualSigmas", "sigmas_stage2", sigmas=STAGE2_SIGMAS)

    present_pass = g.add("TimeFlightLoadPass", "present_pass", render_dir="/content/place/present_r1024_361")
    past_pass = g.add("TimeFlightLoadPass", "past_pass", render_dir="/content/place/1519_r1024_361")

    def branch(name: str, render_pass: str) -> tuple[str, str]:
        pos = g.add("CLIPTextEncode", f"{name}_prompt", text="", clip=out(clip))
        cond = g.add("LTXVConditioning", f"{name}_conditioning", positive=out(pos), negative=out(neg), frame_rate=fps)
        latent = g.add("EmptyLTXVLatentVideo", f"{name}_latent", width=width, height=height, length=length,
                       batch_size=1)
        guide = g.add("LTXAddVideoICLoRAGuide", f"{name}_guide",
                      positive=out(cond, 0), negative=out(cond, 1), vae=out(vae), latent=out(latent),
                      image=out(render_pass, 1), frame_idx=0, strength=guide_strength,
                      latent_downscale_factor=out(lora, 1), crop="disabled",
                      use_tiled_encode=False, tile_size=256, tile_overlap=64)
        alat = g.add("LTXVEmptyLatentAudio", f"{name}_audio_latent", frames_number=length, frame_rate=fps,
                     batch_size=1, audio_vae=out(avae))
        av = g.add("LTXVConcatAVLatent", f"{name}_av_latent", video_latent=out(guide, 2), audio_latent=out(alat))
        noise = g.add("RandomNoise", f"{name}_seed", noise_seed=seed)
        guider = g.add("CFGGuider", f"{name}_guider", model=out(lora), positive=out(guide, 0),
                       negative=out(guide, 1), cfg=1.0)
        sampled = g.add("SamplerCustomAdvanced", f"{name}_sample", noise=out(noise), guider=out(guider),
                        sampler=out(sampler), sigmas=out(sigmas), latent_image=out(av))
        sep = g.add("LTXVSeparateAVLatent", f"{name}_split_av", av_latent=out(sampled))
        crop = g.add("LTXVCropGuides", f"{name}_crop_guides", positive=out(guide, 0), negative=out(guide, 1),
                     latent=out(sep, 0))
        video_latent, audio_latent = out(crop, 2), out(sep, 1)
        if stage2:
            up = g.add("LTXVLatentUpsampler", f"{name}_upsample", samples=video_latent,
                       upscale_model=out(upscaler), vae=out(vae))
            av2 = g.add("LTXVConcatAVLatent", f"{name}_av_latent_stage2", video_latent=out(up),
                        audio_latent=audio_latent)
            noise2 = g.add("RandomNoise", f"{name}_seed_stage2", noise_seed=seed)
            guider2 = g.add("CFGGuider", f"{name}_guider_stage2", model=out(lora), positive=out(crop, 0),
                            negative=out(crop, 1), cfg=1.0)
            sampled2 = g.add("SamplerCustomAdvanced", f"{name}_sample_stage2", noise=out(noise2),
                             guider=out(guider2), sampler=out(sampler2), sigmas=out(sigmas2),
                             latent_image=out(av2))
            sep2 = g.add("LTXVSeparateAVLatent", f"{name}_split_av_stage2", av_latent=out(sampled2))
            video_latent, audio_latent = out(sep2, 0), out(sep2, 1)
        decoded = g.add("VAEDecodeTiled", f"{name}_decode", samples=video_latent, vae=out(vae),
                        tile_size=512, overlap=64, temporal_size=64, temporal_overlap=8)
        audio = g.add("LTXVAudioVAEDecode", f"{name}_audio_decode", samples=audio_latent, audio_vae=out(avae))
        # Keep each raw generation too: cheap, and lets the composite be redone locally without GPU.
        vid = g.add("CreateVideo", f"{name}_video", images=out(decoded), fps=fps, audio=out(audio))
        g.add("SaveVideo", f"save_{name}", video=out(vid), filename_prefix=f"timeflight/flight/{name}",
              format="auto", codec="auto")
        return decoded, audio

    present, present_audio = branch("present", present_pass)
    past_base, past_audio = branch("past_base", past_pass)
    past_zones, _ = branch("past_zones", past_pass)

    merged = g.add("TimeFlightEraMerge", "era_merge", base=out(past_base), zones=out(past_zones),
                   era_pass=out(past_pass, 0), place_dir=PLACE_DIR, era="1519", feather_m=6.0)
    wave = g.add("TimeFlightTimeWave", "time_wave", present=out(present), past=out(merged),
                 present_pass=out(present_pass, 0), past_pass=out(past_pass, 0), place_dir=PLACE_DIR, era="1519",
                 fps=fps, start_s=3.5, end_s=12.5, r0_m=30.0, r1_m=15000.0, softness_m=25.0, glow=0.6,
                 glow_fade_m=80.0)
    if stage2:
        wave = g.add("ImageScale", "output_size", image=out(wave), upscale_method="lanczos",
                     width=out_size[0], height=out_size[1], crop="disabled")
    credits = g.add("TimeFlightCredits", "credits", images=out(wave), place_dir=PLACE_DIR, era="1519", fps=fps,
                    present_caption="Mexico City, today", present_start_s=0.5, present_end_s=3.3,
                    past_caption="", past_start_s=12.0, past_end_s=15.0, card_seconds=6.0, allow_draft=False)
    # The present soundtrack crossfades into the city's (past_base) on the wave's timing.
    mix = g.add("TimeFlightAudioMix", "audio_mix", images=out(credits, 0), present=out(present_audio),
                past=out(past_audio), fps=fps, flight_frames=length, start_s=3.5, end_s=12.5, fade_out_s=1.5)
    vid = g.add("CreateVideo", "hero_video", images=out(credits, 0), fps=fps, audio=out(mix))
    g.add("SaveVideo", "save", video=out(vid), filename_prefix="timeflight/flight/hero", format="auto", codec="auto")
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
    for path, stage2 in ((OUT, False), (OUT_S2, True)):
        wf = build(stage2=stage2)
        path.write_text(json.dumps(wf, indent=2) + "\n")
        print(f"wrote {path} ({len(wf)} nodes) with {MODEL_FILES}")
