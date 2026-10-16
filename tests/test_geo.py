import numpy as np
import pytest

from timeflight.geo.dem import Dem, grey_opening, terrain_grid
from timeflight.geo.enu import Origin, enu_to_geodetic, geodetic_to_ecef, geodetic_to_enu
from timeflight.geo.osm import building_height, drop_outlines_with_parts, footprint, parse_length
from timeflight.geo.path import catmull_rom, sample_path

ZOCALO = Origin(19.4326, -99.1332, 2240.0)


def test_ecef_equator_and_pole():
    x, y, z = geodetic_to_ecef(0.0, 0.0, 0.0)
    assert (float(x), float(y), float(z)) == pytest.approx((6378137.0, 0.0, 0.0))
    _, _, z = geodetic_to_ecef(90.0, 0.0, 0.0)
    assert float(z) == pytest.approx(6356752.314, abs=1e-3)


def test_enu_axes_and_scale():
    east = geodetic_to_enu(ZOCALO.lat, ZOCALO.lon + 0.01, ZOCALO.h, ZOCALO)
    north = geodetic_to_enu(ZOCALO.lat + 0.01, ZOCALO.lon, ZOCALO.h, ZOCALO)
    up = geodetic_to_enu(ZOCALO.lat, ZOCALO.lon, ZOCALO.h + 100, ZOCALO)
    # 0.01 deg of longitude at 19.43 N is about 1050 m; 0.01 deg of latitude about 1107 m
    assert east[0] == pytest.approx(1050.0, rel=0.01) and abs(east[1]) < 1 and east[2] < 0
    assert north[1] == pytest.approx(1107.0, rel=0.01) and abs(north[0]) < 1e-6
    assert up == pytest.approx([0, 0, 100], abs=1e-6)


def test_enu_round_trip():
    rng = np.random.default_rng(1)
    enu = rng.uniform(-20000, 20000, (100, 3))
    lat, lon, h = enu_to_geodetic(enu, ZOCALO)
    back = geodetic_to_enu(lat, lon, h, ZOCALO)
    assert np.abs(back - enu).max() < 1e-4


def test_curvature_drop():
    p = geodetic_to_enu(*enu_to_geodetic(np.array([20000.0, 0, 0]), ZOCALO)[:2], ZOCALO.h, ZOCALO)
    assert p[2] == pytest.approx(-20000 ** 2 / (2 * 6378137), rel=0.02)  # about -31 m


def test_catmull_rom_passes_through_points():
    pts = np.array([[0, 0, 0], [100, 50, 10], [300, 40, 0], [400, 0, 5]], dtype=float)
    curve = catmull_rom(pts, samples_per_segment=50)
    for p in pts:
        assert np.linalg.norm(curve - p, axis=1).min() < 1e-6


def test_sample_path_is_even_and_hits_ends():
    kf = [{"lat": 19.4370, "lon": -99.1450, "alt": 250, "look_at": {"lat": 19.4350, "lon": -99.1320, "alt": 0}},
          {"lat": 19.4362, "lon": -99.1390, "alt": 200, "look_at": {"lat": 19.4350, "lon": -99.1310, "alt": 0}},
          {"lat": 19.4350, "lon": -99.1330, "alt": 150, "look_at": {"lat": 19.4345, "lon": -99.1290, "alt": 0}}]
    path = sample_path(kf, ZOCALO, frames=121, ease_amount=0.0)
    pos = path["position"]
    assert pos.shape == (121, 3) and path["target"].shape == (121, 3)
    start = geodetic_to_enu(19.4370, -99.1450, ZOCALO.h + 250, ZOCALO)
    end = geodetic_to_enu(19.4350, -99.1330, ZOCALO.h + 150, ZOCALO)
    assert np.allclose(pos[0], start, atol=1e-3) and np.allclose(pos[-1], end, atol=1e-3)
    steps = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    assert steps.max() / steps.min() < 1.02  # constant speed without easing


def test_dem_sample_and_opening():
    heights = np.add.outer(np.arange(10.0), np.arange(10.0))  # h = row + col
    dem = Dem(heights, lon0=-100.0, dlon=0.1, lat0=20.0, dlat=-0.1)
    # center of pixel (row 2, col 3) is lat 19.75, lon -99.65
    assert float(dem.sample(19.75, -99.65)) == pytest.approx(5.0)
    assert float(dem.sample(19.70, -99.65)) == pytest.approx(5.5)
    bump = np.zeros((20, 20))
    bump[9:11, 9:11] = 30.0  # a "building"
    bump += np.linspace(0, 10, 20)[None]  # broad slope
    opened = grey_opening(bump, 5)
    assert opened[10, 10] < 12 and abs(opened[5, 15] - bump[5, 15]) < 1e-9


def test_terrain_grid_flat_dem():
    dem = Dem(np.full((50, 50), 2240.0), lon0=-99.2, dlon=0.004, lat0=19.5, dlat=-0.004)
    grid = terrain_grid(dem, ZOCALO, half_size_m=1000, step_m=100)
    assert grid.shape == (21, 21, 3)
    assert abs(grid[10, 10, 2]) < 1e-6 and grid[0, 0, 2] < 0  # curvature makes corners lower


def test_osm_heights_and_footprint():
    assert parse_length("12 m") == 12 and parse_length("12,5") == 12.5
    assert parse_length("40 ft") == pytest.approx(12.192) and parse_length("tall") is None
    assert building_height({"height": "20"}) == (20.0, "height")
    assert building_height({"building:levels": "3"}) == (pytest.approx(10.6), "levels")
    assert building_height({}) == (9.0, "default")
    # clockwise square of about 20 x 20 m, closed ring
    d = 0.0001
    ring = [(19.4326, -99.1332), (19.4326 + 2 * d, -99.1332), (19.4326 + 2 * d, -99.1332 + 2 * d),
            (19.4326, -99.1332 + 2 * d), (19.4326, -99.1332)]
    fp = footprint("w1", {"building": "yes", "name": "X", "source": "survey"}, ring, ZOCALO)
    assert len(fp["ring"]) == 4 and 400 < fp["area_m2"] < 500
    assert fp["tags"] == {"building": "yes", "name": "X"}
    r = np.array(fp["ring"])
    assert 0.5 * (np.dot(r[:, 0], np.roll(r[:, 1], -1)) - np.dot(r[:, 1], np.roll(r[:, 0], -1))) > 0
    assert footprint("w2", {}, ring[:2], ZOCALO) is None


def test_ruins_height_and_parts_rule():
    assert building_height({"historic": "ruins", "building": "yes"}) == (3.0, "ruins")
    sq = lambda x0, s: [[x0, 0], [x0 + s, 0], [x0 + s, s], [x0, s]]
    recs = [{"id": "outline", "ring": sq(0, 100), "tags": {"building": "church", "height": "67"}},
            {"id": "dome", "ring": sq(40, 20), "tags": {"name": "Cúpula", "height": "55"}},
            {"id": "other", "ring": sq(500, 10), "tags": {"building": "yes"}}]
    assert [r["id"] for r in drop_outlines_with_parts(recs)] == ["dome", "other"]
