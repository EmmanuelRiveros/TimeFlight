import copy
import json
from pathlib import Path

import numpy as np
import pytest

from timeflight.era import proxies as P
from timeflight.era.build import build_era, polyline_dist
from timeflight.era.manifest import license_ok, validate
from timeflight.geo.enu import Origin, geodetic_to_enu
from timeflight.geo.osm import point_in_ring

MANIFEST = Path(__file__).resolve().parents[1] / "places" / "tenochtitlan_zocalo" / "eras" / "1519.json"
ORIGIN = Origin(19.4326, -99.1332, 2228.3)


@pytest.fixture(scope="module")
def manifest():
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def era(manifest):
    return build_era(manifest, ORIGIN)


def test_licenses():
    for ok in ("Public domain", "No Copyright - United States", "Public Domain Mark", "CC0 1.0",
               "CC BY 4.0", "Facts only (no text or images used)", "ODbL 1.0 (positions only)"):
        assert license_ok(ok), ok
    for bad in ("CC BY-NC 4.0", "CC BY-SA 4.0", "CC BY-NC-SA", "All rights reserved", "Non-commercial use"):
        assert not license_ok(bad), bad


def test_manifest_is_valid(manifest):
    assert validate(manifest) == []


def test_manifest_rejects_missing_provenance(manifest):
    m = copy.deepcopy(manifest)
    m["structures"][0]["sources"] = []
    m["causeways"][0]["sources"] = ["NOPE"]
    m["water"]["confidence"] = "sure"
    m["sources"].append({"id": "X", "title": "t", "url": "u", "license": "CC BY-NC 4.0", "credit": "c"})
    problems = " | ".join(validate(m))
    for needle in ("Templo Mayor: no sources", "unknown source NOPE", "water: confidence", "license not allowed"):
        assert needle in problems


def test_stepped_pyramid_shape():
    parts = P.stepped_pyramid(0, 0, 80, 100, 30, 4, z0=1, stairs="west")
    tiers = [p for p in parts if p["kind"] == "temple"]
    stairs = [p for p in parts if p["kind"] == "stairs"]
    assert len(tiers) == 4 and max(p["z1"] for p in tiers) == pytest.approx(31)
    assert np.ptp(tiers[0]["ring"][:, 0]) == pytest.approx(80) and np.ptp(tiers[0]["ring"][:, 1]) == pytest.approx(100)
    assert all(p["ring"][:, 0].mean() < 0 for p in stairs)  # stairs on the west face
    assert max(p["z1"] for p in stairs) == pytest.approx(31)


def test_twin_temple_shrines_north_and_south():
    parts = P.twin_temple(0, 0, 80, 100, 30, 15, 4, 0, stairs="west")
    shrines = [p for p in parts if p["kind"] == "shrine"]
    ys = sorted(p["ring"][:, 1].mean() for p in shrines)
    assert len(shrines) == 2 and ys[0] < 0 < ys[1]
    assert max(p["z1"] for p in shrines) == pytest.approx(45)  # 30 m base + 15 m shrines (D2: 45-60 m)


def test_rect_rotation_and_winding():
    r = P.rect(10, 5, 20, 4, angle_deg=90)
    assert np.ptp(r[:, 0]) == pytest.approx(4) and np.ptp(r[:, 1]) == pytest.approx(20)
    x, y = r[:, 0], r[:, 1]
    assert 0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) > 0


def test_era_layout(era, manifest):
    s = era["stats"]
    assert 8 <= s["island_area_km2"] <= 13.5  # D1
    assert s["houses"] > 2000 and s["garden_strips"] > 100
    tm = geodetic_to_enu(manifest["anchor"]["lat"], manifest["anchor"]["lon"], ORIGIN.h, ORIGIN)[:2]
    temples = [p for p in era["prisms"] if p["kind"] == "temple"]
    tallest = max(temples, key=lambda p: p["z1"])
    assert np.linalg.norm(tallest["ring"].mean(axis=0) - tm) < 1.0  # Templo Mayor at its OSM spot
    # no house inside the precinct
    x0, y0, x1, y1 = (tm[0] - 300, tm[1] - 190, tm[0] + 140, tm[1] + 210)
    for p in era["prisms"]:
        if p["kind"] == "house":
            cx, cy = p["ring"].mean(axis=0)
            assert not (x0 + 5 < cx < x1 - 5 and y0 + 5 < cy < y1 - 5)


def test_causeways_leave_the_island(era, manifest):
    island = geodetic_to_enu(*np.array(manifest["island"]["polygon_latlon"]).T, ORIGIN.h, ORIGIN)[:, :2]
    cws = [p for p in era["prisms"] if p["kind"] == "causeway"]
    assert len(cws) > 10
    assert all(not point_in_ring(p["ring"].mean(axis=0), island) for p in cws)
    assert any(p["kind"] == "bridge" for p in era["prisms"])


def test_polyline_dist():
    line = np.array([[0.0, 0.0], [10.0, 0.0]])
    d = polyline_dist(np.array([[5.0, 3.0], [-4.0, 0.0], [13.0, 4.0]]), line)
    assert d == pytest.approx([3.0, 4.0, 5.0])
