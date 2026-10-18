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
    assert front_glow(d / 2, 5.0, 25.0)[0] < 0.01  # 5 m into an 80 m fade-in
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


def test_portrait_fits_the_sensor_to_the_height():
    x, y = pixel_rays(576, 1024, 24.0)
    assert y[0, 0] == pytest.approx(0.75 * (1 - 1 / 1024))       # 18 mm / 24 mm on the long side
    assert x[0, -1] == pytest.approx(0.75 * 576 / 1024 * (1 - 1 / 576))
    depth = np.random.default_rng(1).uniform(100, 5000, (64, 36))
    uvz = project(unproject(depth, POS, TGT, 24.0), POS, TGT, 24.0, 36, 64)
    u, v = np.meshgrid(np.arange(36) + 0.5, np.arange(64) + 0.5)
    assert uvz[..., 0] == pytest.approx(u, abs=1e-6) and uvz[..., 1] == pytest.approx(v, abs=1e-6)


def test_resize_mask_upscales_and_keeps_values():
    from timeflight.timewave import resize_mask

    m = np.zeros((4, 6), np.float32)
    m[:, 3:] = 1.0
    up = resize_mask(m, 8, 12)
    assert up.shape == (8, 12)
    assert up[:, :5].max() == 0.0 and up[:, 7:].min() == 1.0
    assert 0 < up[0, 5] < 1 and 0 < up[0, 6] < 1  # soft at the edge
    assert resize_mask(m, 4, 6) is m


def test_zone_merge_on_frames_larger_than_the_pass():
    from timeflight.passes import RenderPass
    from timeflight.timewave import zone_merge_frames

    depth = np.full((2, 36, 64), 400.0, np.float32)
    pos = np.tile(POS, (2, 1))
    tgt = np.tile(TGT, (2, 1))
    rp = RenderPass(depth, depth > 1e9, pos, tgt, 24.0, {"lat": 0, "lon": 0, "h": 0}, 24.0, "x")
    base = np.zeros((2, 72, 128, 3), np.uint8)
    zones = np.full((2, 72, 128, 3), 200, np.uint8)
    big = [-1e6, -1e6, 1e6, 1e6]
    out = list(zone_merge_frames(base, zones, rp, [big]))
    assert out[0].shape == (72, 128, 3) and out[0].min() >= 199
    with pytest.raises(ValueError):
        list(zone_merge_frames(base[:, :, :100], zones[:, :, :100], rp, [big]))


def test_front_curtain_stands_on_the_front_and_respects_occlusion():
    from timeflight.timewave import front_curtain

    # Camera 500 m west of the center at 100 m, looking east and level: the center row of pixels
    # runs along the ground plane height, so the ray crosses the wall at z = 100.
    pos, tgt, c = np.array([-500.0, 0.0, 100.0]), np.array([0.0, 0.0, 100.0]), (0.0, 0.0)
    far = np.full((9, 16), 1e5)
    low = front_curtain(far, pos, tgt, 24.0, c, 200.0, ground_z=50.0, height=80.0, width=30.0)
    high = front_curtain(far, pos, tgt, 24.0, c, 200.0, ground_z=-200.0, height=80.0, width=30.0)
    assert low[4, 8] > 0.2                      # wall crossed 50 m above its foot: dust
    assert high[4, 8] == 0.0                    # wall top at -120 m: the ray passes above it
    near = np.full((9, 16), 100.0)              # a surface 100 m away, in front of the wall (300 m)
    assert front_curtain(near, pos, tgt, 24.0, c, 200.0, 50.0, 80.0, 30.0)[4, 8] == 0.0
    assert front_curtain(far, pos, tgt, 24.0, c, 0.0, 50.0, 80.0, 30.0).max() == 0.0
