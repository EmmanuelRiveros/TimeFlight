import numpy as np
import pytest

from timeflight.timewave import front_glow, past_alpha, rects_mask, pixel_rays, project, unproject, wave_radius

POS = np.array([10.0, -50.0, 200.0])
TGT = np.array([300.0, 400.0, 0.0])


def test_unproject_project_round_trip():
    rng = np.random.default_rng(0)
    depth = rng.uniform(100, 5000, (36, 64))
    pts = unproject(depth, POS, TGT, 24.0)
    uvz = project(pts, POS, TGT, 24.0, 64, 36)
    u, v = np.meshgrid(np.arange(64) + 0.5, np.arange(36) + 0.5)
    assert uvz[..., 0] == pytest.approx(u, abs=1e-6)
    assert uvz[..., 1] == pytest.approx(v, abs=1e-6)
    assert uvz[..., 2] == pytest.approx(depth, rel=1e-9)


def test_center_ray_hits_target_and_frame_edges():
    # The target projects to the image center.
    u, v, _ = project(TGT, POS, TGT, 24.0, 1024, 576)
    assert (u, v) == pytest.approx((512, 288))
    # Horizontal half field of view: 18 mm / 24 mm at planar depth 1 (36 mm sensor, width fit).
    x, y = pixel_rays(1024, 576, 24.0)
    assert x[0, -1] == pytest.approx(0.75 * (1 - 1 / 1024))
    assert y[0, 0] == pytest.approx(0.75 * 576 / 1024 * (1 - 1 / 576))
    # Camera up stays near world +Z: the top row is higher than the bottom row.
    depth = np.full((576, 1024), 1000.0)
    pts = unproject(depth, POS, TGT, 24.0)
    assert pts[0, 512, 2] > pts[-1, 512, 2]


def test_wave_radius_grows_from_zero():
    assert wave_radius(10, 20, 100, 20, 20000) == 0.0
    assert wave_radius(20, 20, 100, 20, 20000) == 0.0  # no jump when the wave starts
    assert wave_radius(21, 20, 100, 20, 20000) < 5.0
    assert wave_radius(100, 20, 100, 20, 20000) == pytest.approx(20000)
    assert wave_radius(500, 20, 100, 20, 20000) == pytest.approx(20000)
    rs = [wave_radius(f, 20, 100, 20, 20000) for f in range(20, 101)]
    assert all(b > a for a, b in zip(rs, rs[1:]))
    assert wave_radius(60, 20, 100, 20, 20000) == pytest.approx(20 * (np.sqrt(1 + 20000 / 20) - 1))  # midpoint


def test_glow_fades_in():
    d = np.array([10.0])
    assert front_glow(d, 10.0, 25.0)[0] < 0.01
    assert front_glow(d * 50, 500.0, 25.0)[0] == pytest.approx(1.0)


def test_past_alpha_rules():
    r = 1000.0
    hard = dict(radius=r, softness=1e-3)
    one = np.ones(1)
    # Both far outside: present.
    assert past_alpha(5000 * one, 5000 * one, 10 * one, 10 * one, **hard)[0] == pytest.approx(0)
    # Both inside: past.
    assert past_alpha(100 * one, 100 * one, 10 * one, 10 * one, **hard)[0] == pytest.approx(1)
    # A present tower inside the wave, in front of past ground outside it: tower removed -> past.
    assert past_alpha(900 * one, 1500 * one, 50 * one, 80 * one, **hard)[0] == pytest.approx(1)
    # Past pyramid inside the wave in front of present ground outside: pyramid -> past.
    assert past_alpha(1500 * one, 900 * one, 80 * one, 50 * one, **hard)[0] == pytest.approx(1)
    # Present building outside, in front of past ground inside: the building stays.
    assert past_alpha(1500 * one, 900 * one, 50 * one, 80 * one, **hard)[0] == pytest.approx(0)
    # Soft edge: exactly at the front -> about halfway.
    assert 0.2 < past_alpha(r * one, r * one, 10 * one, 10 * one, radius=r, softness=30)[0] < 0.8


def test_rects_mask():
    pts = np.array([[5.0, 5.0], [-50.0, 5.0], [0.0, 5.0], [15.0, 30.0], [10.0 + 1e3, 5.0]])
    m = rects_mask(pts, [(0, 0, 10, 10), (12, 25, 20, 35)], feather=2.0)
    assert m[0] > 0.9 and m[1] < 0.01 and m[2] == pytest.approx(0.5) and m[3] > 0.8 and m[4] < 0.01
