"""The UI-format workflow carries exactly the links and widget values of the API graph it is built from."""

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "workflows"))

from build_flight import build  # noqa: E402
from build_ui import SPECS, to_ui  # noqa: E402


def test_ui_links_match_api():
    api = build(stage2=True)
    graph = to_ui(api, json.loads(SPECS.read_text(encoding="utf-8")))
    api_links = sorted((int(v[0]), v[1], int(nid), name) for nid, n in api.items()
                       for name, v in n["inputs"].items() if isinstance(v, list))
    ui_nodes = {n["id"]: n for n in graph["nodes"].values()}
    ui_links = sorted((src, slot, dst, ui_nodes[dst]["inputs"][dslot]["name"])
                      for _, src, slot, dst, dslot, _ in graph["links"])
    assert ui_links == api_links


def test_ui_widgets_keep_api_values():
    api = build(stage2=True)
    graph = to_ui(api, json.loads(SPECS.read_text(encoding="utf-8")))
    wave = graph["nodes"]["time_wave"]
    assert wave["widgets_values"][:1] == [api[str(wave["id"])]["inputs"]["place_dir"]]
    seed = graph["nodes"]["present_seed"]
    assert seed["widgets_values"] == [42, "fixed"]
