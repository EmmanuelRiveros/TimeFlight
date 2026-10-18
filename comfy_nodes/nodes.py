"""TimeFlight ComfyUI nodes. Thin wrappers: all logic lives in the `timeflight` package.

IMAGE tensors are float [frames, H, W, 3] in 0..1; they are converted to uint8 numpy for
timeflight and back one frame at a time, so a 2K clip (10 GB as float) is never copied whole.
torch is imported lazily so the module can be inspected without it (tests check every node's
inputs against its function signature).
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np

CATEGORY = "TimeFlight"
PASS = "TF_PASS"  # a timeflight.passes.RenderPass: depth + camera of one Blender render folder


def _frame_u8(frame) -> np.ndarray:
    import torch

    return frame.clamp(0, 1).mul(255).round().to(torch.uint8).cpu().numpy()


class _Frames:
    """Read-only uint8 view of an IMAGE tensor, converted frame by frame on access."""

    def __init__(self, images):
        self.images = images
        self.shape = tuple(images.shape)

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, i: int) -> np.ndarray:
        return _frame_u8(self.images[i])


def _frames(images):
    return _Frames(images)


def _to_uint8(images, size: tuple[int, int] | None = None) -> np.ndarray:
    """IMAGE -> uint8 (F, H, W, 3); with `size` (w, h) each frame is resized (lanczos) on the way."""
    n, h, w = images.shape[:3]
    if size is None or tuple(size) == (w, h):
        size = None
    out = np.empty((n, *(size[::-1] if size else (h, w)), 3), np.uint8)
    for i in range(n):
        f = _frame_u8(images[i])
        if size:
            from PIL import Image

            f = np.asarray(Image.fromarray(f).resize(size, Image.Resampling.LANCZOS))
        out[i] = f
    return out


def _collect(frames, count: int):
    """uint8 frames (an iterable of (H, W, 3)) -> IMAGE tensor, filled in place."""
    import torch

    out = None
    for i, f in enumerate(frames):
        if out is None:
            out = torch.empty((count, *f.shape), dtype=torch.float32)
        out[i] = torch.from_numpy(np.ascontiguousarray(f)).float().div_(255.0)
    return out


def _audio_np(audio) -> tuple[np.ndarray, int]:
    """ComfyUI AUDIO {"waveform": [batch, channels, samples], "sample_rate"} -> (channels, samples), rate."""
    return audio["waveform"][0].float().cpu().numpy(), int(audio["sample_rate"])


def _to_audio(waveform: np.ndarray, sample_rate: int) -> dict:
    import torch

    return {"waveform": torch.from_numpy(np.ascontiguousarray(waveform))[None], "sample_rate": sample_rate}


def _manifest(place_dir: str, era: str) -> dict:
    return json.loads((Path(place_dir) / "eras" / f"{era}.json").read_text(encoding="utf-8"))


class TimeFlightLoadPass:
    """A Blender render folder (depth/, render.json, control_inverse.mp4) made by blender/render_place.py."""

    CATEGORY = CATEGORY
    FUNCTION = "load"
    RETURN_TYPES = (PASS, "IMAGE", "INT", "FLOAT")
    RETURN_NAMES = ("render_pass", "depth_control", "frames", "fps")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"render_dir": ("STRING", {"default": "/content/place/present_r1024_361"})},
                "optional": {"camera_json": ("STRING", {"default": ""})}}

    def load(self, render_dir, camera_json=""):
        from timeflight.passes import load_pass
        from timeflight.video import read_video

        rp = load_pass(Path(render_dir), Path(camera_json) if camera_json else None)
        control = read_video(Path(render_dir) / "control_inverse.mp4")
        if len(control) != rp.frames:
            raise ValueError(f"control has {len(control)} frames, depth has {rp.frames}")
        return (rp, _to_image(control), rp.frames, rp.fps)


class TimeFlightEraMerge:
    """Take the era's dry zones (precinct, plazas, named structures) from one generation and the rest
    of the city from another. Both must follow the same depth control."""

    CATEGORY = CATEGORY
    FUNCTION = "merge"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "base": ("IMAGE",), "zones": ("IMAGE",), "era_pass": (PASS,),
            "place_dir": ("STRING", {"default": "/content/TimeFlight/places/tenochtitlan_zocalo"}),
            "era": ("STRING", {"default": "1519"}),
            "feather_m": ("FLOAT", {"default": 6.0, "min": 0.1, "max": 100.0, "step": 0.5}),
        }}

    def merge(self, base, zones, era_pass, place_dir, era, feather_m):
        from timeflight.era.build import dry_zones
        from timeflight.geo.enu import Origin
        from timeflight.timewave import zone_merge_frames

        z = dry_zones(_manifest(place_dir, era), Origin(**era_pass.origin))
        rects = z["precinct"] + z["plazas"] + z["structures"]
        frames = zone_merge_frames(_frames(base), _frames(zones), era_pass, rects, feather_m)
        return (_collect(frames, len(base)),)


class TimeFlightTimeWave:
    """Present -> past: a soft vertical cylinder grows from the era anchor (e.g. the Templo Mayor)."""

    CATEGORY = CATEGORY
    FUNCTION = "wave"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)

    @classmethod
    def INPUT_TYPES(cls):
        f = lambda d, lo, hi, st: ("FLOAT", {"default": d, "min": lo, "max": hi, "step": st})  # noqa: E731
        return {"required": {
            "present": ("IMAGE",), "past": ("IMAGE",), "present_pass": (PASS,), "past_pass": (PASS,),
            "place_dir": ("STRING", {"default": "/content/TimeFlight/places/tenochtitlan_zocalo"}),
            "era": ("STRING", {"default": "1519"}),
            "fps": f(24.0, 1.0, 120.0, 1.0),
            "start_s": f(3.5, 0.0, 600.0, 0.1), "end_s": f(12.5, 0.0, 600.0, 0.1),
            "r0_m": f(30.0, 1.0, 1000.0, 1.0), "r1_m": f(15000.0, 100.0, 100000.0, 100.0),
            "softness_m": f(25.0, 0.5, 500.0, 0.5),
            "glow": f(0.6, 0.0, 1.0, 0.05), "glow_fade_m": f(80.0, 0.0, 2000.0, 10.0),
        }}

    def wave(self, present, past, present_pass, past_pass, place_dir, era, fps, start_s, end_s,
             r0_m, r1_m, softness_m, glow, glow_fade_m):
        from timeflight.era.build import anchor_xy
        from timeflight.geo.enu import Origin
        from timeflight.timewave import WaveParams, timewave_frames

        center = anchor_xy(_manifest(place_dir, era), Origin(**present_pass.origin))
        params = WaveParams(start_s, end_s, r0_m, r1_m, softness_m, glow, glow_fade_m)
        frames = timewave_frames(_frames(present), _frames(past), present_pass, past_pass, center, fps, params)
        return (_collect(frames, len(present)),)


class TimeFlightCredits:
    """Burn the two captions in and append the credit cards, all generated from the era manifest
    and assets.jsonl. Stops on draft credits (an unverified license) unless allow_draft."""

    CATEGORY = CATEGORY
    FUNCTION = "credits"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "credits_markdown")

    @classmethod
    def INPUT_TYPES(cls):
        f = lambda d, hi: ("FLOAT", {"default": d, "min": 0.0, "max": hi, "step": 0.1})  # noqa: E731
        return {"required": {
            "images": ("IMAGE",),
            "place_dir": ("STRING", {"default": "/content/TimeFlight/places/tenochtitlan_zocalo"}),
            "era": ("STRING", {"default": "1519"}),
            "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 1.0}),
            "present_caption": ("STRING", {"default": "Mexico City, today"}),
            "present_start_s": f(0.5, 600.0), "present_end_s": f(3.3, 600.0),
            "past_caption": ("STRING", {"default": ""}),
            "past_start_s": f(12.0, 600.0), "past_end_s": f(15.0, 600.0),
            "card_seconds": f(6.0, 60.0),
            "allow_draft": ("BOOLEAN", {"default": False}),
        }, "optional": {
            # Output size (0 = keep). Resizing here, frame by frame, avoids a whole-clip copy.
            "width": ("INT", {"default": 0, "min": 0, "max": 8192}),
            "height": ("INT", {"default": 0, "min": 0, "max": 8192}),
        }}

    def credits(self, images, place_dir, era, fps, present_caption, present_start_s, present_end_s,
                past_caption, past_start_s, past_end_s, card_seconds, allow_draft, width=0, height=0):
        from timeflight.assets_log import read_assets
        from timeflight.provenance.cards import caption_frames, card_frames, render_cards
        from timeflight.provenance.credits import build_credits, to_markdown

        place = json.loads((Path(place_dir) / "place.json").read_text(encoding="utf-8"))
        manifest = _manifest(place_dir, era)
        c = build_credits(manifest, read_assets(), place.get("label"))
        if c.draft and not allow_draft:
            raise ValueError("credits are a draft: " + "; ".join(c.problems))
        frames = _to_uint8(images, (width, height) if width and height else None)
        captions = [(present_start_s, present_end_s, present_caption),
                    (past_start_s, past_end_s, past_caption or f"{manifest['label']} · illustrative AI reconstruction")]
        captioned = caption_frames(frames, [cap for cap in captions if cap[2]], fps)
        cards = render_cards(c.title, c.notice, c.cards, (frames.shape[2], frames.shape[1]), c.draft)
        tail = card_frames(cards, fps, card_seconds)
        return (_collect(itertools.chain(captioned, tail), len(captioned) + len(tail)), to_markdown(c))


class TimeFlightAudioMix:
    """Soundtrack of the flight: the present clip's audio crossfades into the past clip's audio on the
    wave's timing, fades out at the end of the flight and stays silent under the credit cards."""

    CATEGORY = CATEGORY
    FUNCTION = "mix"
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)

    @classmethod
    def INPUT_TYPES(cls):
        f = lambda d, hi: ("FLOAT", {"default": d, "min": 0.0, "max": hi, "step": 0.1})  # noqa: E731
        return {"required": {
            "images": ("IMAGE",), "present": ("AUDIO",), "past": ("AUDIO",),
            "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 1.0}),
            "flight_frames": ("INT", {"default": 361, "min": 1, "max": 100000}),
            "start_s": f(3.5, 600.0), "end_s": f(12.5, 600.0), "fade_out_s": f(1.5, 60.0),
        }}

    def mix(self, images, present, past, fps, flight_frames, start_s, end_s, fade_out_s):
        from timeflight.audio import mix_eras

        (a, sr), (b, sr_b) = _audio_np(present), _audio_np(past)
        if sr != sr_b:
            raise ValueError(f"sample rates differ: {sr} and {sr_b}")
        out = mix_eras(a, b, sr, start_s, end_s, total_s=len(images) / fps, clip_s=flight_frames / fps,
                       fade_out_s=fade_out_s)
        return (_to_audio(out, sr),)


NODE_CLASS_MAPPINGS = {
    "TimeFlightLoadPass": TimeFlightLoadPass,
    "TimeFlightEraMerge": TimeFlightEraMerge,
    "TimeFlightTimeWave": TimeFlightTimeWave,
    "TimeFlightCredits": TimeFlightCredits,
    "TimeFlightAudioMix": TimeFlightAudioMix,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "TimeFlightLoadPass": "TimeFlight: Load Render Pass",
    "TimeFlightEraMerge": "TimeFlight: Era Zone Merge",
    "TimeFlightTimeWave": "TimeFlight: Time Wave",
    "TimeFlightCredits": "TimeFlight: Captions and Credits",
    "TimeFlightAudioMix": "TimeFlight: Audio Crossfade",
}
