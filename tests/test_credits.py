import copy
import json
from pathlib import Path

import pytest

from timeflight.assets_log import Asset
from timeflight.provenance.credits import build_credits, to_markdown, to_srt

MANIFEST = Path(__file__).resolve().parents[1] / "places" / "tenochtitlan_zocalo" / "eras" / "1519.json"
ASSETS = [
    Asset("dem_tile", "dem", "https://example.org/dem", "Copernicus licence", "produced using Copernicus WorldDEM-30."),
    Asset("ltx", "model", "https://example.org/ltx", "LTX-2.x Community License", "LTX-2.5 by Lightricks"),
    Asset("da", "model", "https://example.org/da", "Apache-2.0", "DA authors", notes="Evaluation only (U1)."),
]


@pytest.fixture(scope="module")
def manifest():
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def test_every_used_source_is_credited_with_license(manifest):
    c = build_credits(manifest, ASSETS)
    assert not c.draft, c.problems
    sources = dict(c.sections)["Historical sources"]
    for s in manifest["sources"]:
        line = next(l for l in sources if l.startswith(s["credit"]))
        assert s["license"] in line and "Used for:" in line
    assert any("Templo Mayor (medium)" in l for l in sources)


def test_unused_source_is_not_credited(manifest):
    m = copy.deepcopy(manifest)
    m["sources"].append({"id": "Z9", "title": "Unused", "url": "u", "license": "Public domain", "credit": "Unused book"})
    c = build_credits(m, ASSETS)
    assert not any("Unused book" in l for l in dict(c.sections)["Historical sources"])


def test_unverified_license_makes_a_draft(manifest):
    bad = ASSETS + [Asset("x", "model", "https://example.org/x", "Some terms - UNVERIFIED", "X")]
    c = build_credits(manifest, bad)
    assert c.draft and "license not verified" in c.problems[0]
    assert to_markdown(c).count("DRAFT") == 1


def test_evaluation_models_are_separate(manifest):
    c = build_credits(manifest, ASSETS)
    sections = dict(c.sections)
    assert not any("DA authors" in l for l in sections["Models"])
    assert any(l.startswith("da (Apache-2.0)") for l in sections["Evaluation only"])


def test_notice_is_honest_about_confidence_and_license(manifest):
    text = " ".join(build_credits(manifest, ASSETS).notice)
    assert "illustrative AI reconstruction" in text and "not fully open source" in text
    assert "of 21 elements" in text


def test_srt():
    srt = to_srt([(12.5, 15.0, "Tenochtitlan, 1519"), (0.5, 3.5, "Mexico City, today")])
    assert srt.startswith("1\n00:00:00,500 --> 00:00:03,500\nMexico City, today\n")
    assert "2\n00:00:12,500 --> 00:00:15,000\nTenochtitlan, 1519" in srt
    with pytest.raises(ValueError):
        to_srt([(3.0, 2.0, "x")])
