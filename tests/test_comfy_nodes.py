import importlib.util
import inspect
import sys
from pathlib import Path

import numpy as np
import pytest

import comfy_nodes.nodes as N
from timeflight.passes import RenderPass

REPO = Path(__file__).resolve().parents[1]
PLACE_DIR = str(REPO / "places" / "tenochtitlan_zocalo")
ORIGIN = {"lat": 19.4326, "lon": -99.1332, "h": 2228.3}


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def object_info() -> dict:
    """/object_info-style specs for our own nodes (what a live ComfyUI would report)."""
    return {name: {"input": cls.INPUT_TYPES(), "output": list(cls.RETURN_TYPES)}
            for name, cls in N.NODE_CLASS_MAPPINGS.items()}


@pytest.mark.parametrize("name,cls", list(N.NODE_CLASS_MAPPINGS.items()))
def test_node_inputs_match_function_signature(name, cls):
    spec = cls.INPUT_TYPES()
    declared = list(spec["required"]) + list(spec.get("optional", {}))
    params = list(inspect.signature(getattr(cls, cls.FUNCTION)).parameters)[1:]
    assert params == declared, name
    assert len(cls.RETURN_TYPES) == len(cls.RETURN_NAMES)
    assert name in N.NODE_DISPLAY_NAME_MAPPINGS


@pytest.mark.parametrize("stage2", [False, True])
def test_flight_workflow_wires_timeflight_nodes_correctly(stage2):
    wf = _load(REPO / "workflows" / "build_flight.py", "build_flight").build(stage2=stage2)
    validate = _load(REPO / "scripts" / "validate_workflow.py", "validate_workflow").validate
    ours = {k: v for k, v in wf.items() if v["class_type"] in N.NODE_CLASS_MAPPINGS}
    assert len(ours) == 6
    assert any(v["class_type"] == "LTXVLatentUpsampler" for v in wf.values()) == stage2
    # Input names and required inputs of our nodes. Core and LTX nodes are accepted as they are here;
    # scripts/validate_workflow.py checks them against the live server on Colab.
    info = object_info()
    permissive = {v["class_type"]: {"input": {"required": {}, "optional": {n: ["*"] for n in v["inputs"]}}}
                  for v in wf.values() if v["class_type"] not in info}
    assert validate(wf, info | permissive) == []
    # Types on every link that leaves one of our nodes.
    core_expect = {"image": "IMAGE", "images": "IMAGE", "audio": "AUDIO"}
    for nid, node in wf.items():
        for inp, val in node["inputs"].items():
            if not (isinstance(val, list) and len(val) == 2 and val[0] in ours):
                continue
            produced = N.NODE_CLASS_MAPPINGS[wf[val[0]]["class_type"]].RETURN_TYPES[val[1]]
            if node["class_type"] in N.NODE_CLASS_MAPPINGS:
                req = N.NODE_CLASS_MAPPINGS[node["class_type"]].INPUT_TYPES()["required"]
                assert req[inp][0] == produced, (node["_meta"]["title"], inp)
            else:
                assert core_expect.get(inp) == produced, (node["_meta"]["title"], inp)


def _fake_pass(frames=3, h=36, w=64) -> RenderPass:
    depth = np.full((frames, h, w), 400.0, np.float32)
    depth[:, :8] = 40000.0  # sky band
    pos = np.tile([-300.0, 400.0, 200.0], (frames, 1))
    tgt = np.tile([300.0, 400.0, 0.0], (frames, 1))
    return RenderPass(depth, depth >= 39960, pos, tgt, 24.0, ORIGIN, 24.0, "x")


@pytest.fixture
def no_torch(monkeypatch):
    monkeypatch.setattr(N, "_frames", lambda x: x)
    monkeypatch.setattr(N, "_to_uint8", lambda x, size=None: x)
    monkeypatch.setattr(N, "_collect", lambda frames, n: np.stack(list(frames)))


def test_era_merge_and_time_wave_nodes_run(no_torch):
    rp = _fake_pass()
    a = np.zeros((3, 36, 64, 3), np.uint8)
    b = np.full((3, 36, 64, 3), 200, np.uint8)
    (merged,) = N.TimeFlightEraMerge().merge(a, b, rp, PLACE_DIR, "1519", 6.0)
    assert merged.shape == a.shape and merged.dtype == np.uint8
    (wave,) = N.TimeFlightTimeWave().wave(a, b, rp, rp, PLACE_DIR, "1519", 24.0, 0.0, 0.05, 30.0, 15000.0,
                                         25.0, 0.0, 80.0)
    assert wave[0].mean() < 1 and wave[-1].mean() > 199  # all present at frame 0, all past once r = r1


@pytest.mark.skipif(not (REPO / "data" / "raw" / "fonts" / "SourceSans3-Regular.ttf").exists(),
                    reason="font not downloaded (scripts/fetch_fonts.py)")
def test_credits_node_appends_cards(no_torch):
    frames = np.zeros((48, 144, 256, 3), np.uint8)
    images, md = N.TimeFlightCredits().credits(frames, PLACE_DIR, "1519", 24.0, "Mexico City, today", 0.2, 1.0,
                                              "", 1.0, 2.0, 1.0, False)
    assert len(images) > 48 and images.shape[1:] == frames.shape[1:]
    assert images[10].max() > 0  # caption burned in
    assert "Matos Moctezuma" in md and "DRAFT" not in md


def test_audio_mix_node(monkeypatch):
    monkeypatch.setattr(N, "_audio_np", lambda a: (a["waveform"], a["sample_rate"]))
    monkeypatch.setattr(N, "_to_audio", lambda w, sr: {"waveform": w, "sample_rate": sr})
    sr = 100
    present = {"waveform": np.ones((2, 2 * sr), np.float32), "sample_rate": sr}
    past = {"waveform": -np.ones((2, 2 * sr), np.float32), "sample_rate": sr}
    images = np.zeros((72, 4, 4, 3))  # 3 s at 24 fps: 2 s of flight + 1 s of cards
    (audio,) = N.TimeFlightAudioMix().mix(images, present, past, 24.0, 48, 0.5, 1.0, 0.0)
    w = audio["waveform"]
    assert w.shape == (2, 3 * sr) and w[0, 10] == 1.0 and w[0, 150] == -1.0 and not w[:, 2 * sr:].any()
