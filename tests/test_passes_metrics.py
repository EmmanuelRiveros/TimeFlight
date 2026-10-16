import numpy as np
import pytest

from timeflight import metrics, passes


def test_depth_to_control_endpoints_and_monotonic():
    z = np.array([10.0, 20.0, 50.0, 100.0, 5000.0])
    for mode in ("inverse", "linear"):
        c = passes.depth_to_control(z, near=10, far=100, mode=mode)
        assert c[0] == pytest.approx(1.0)
        assert c[3] == pytest.approx(0.0)
        assert c[4] == pytest.approx(0.0)  # beyond far / sky clamps to 0
        assert np.all(np.diff(c) <= 0)


def test_inverse_gives_more_range_up_close():
    z = np.array([10.0, 20.0])
    inv = passes.depth_to_control(z, 10, 100, "inverse")
    lin = passes.depth_to_control(z, 10, 100, "linear")
    assert (inv[0] - inv[1]) > (lin[0] - lin[1])


def test_fixed_range_ignores_sky():
    d = np.full((10, 10), 4000.0)
    d[:5] = np.linspace(50, 500, 50).reshape(5, 10)
    near, far = passes.fixed_range([d], sky_depth=4000.0, lo_pct=0, hi_pct=100)
    assert near == pytest.approx(50) and far == pytest.approx(500)


def test_decode_depth_png():
    assert passes.decode_depth_png(np.array([65535], np.uint16), 4000)[0] == pytest.approx(4000)


def test_edges_are_relative_to_distance():
    z = np.full((8, 8), 100.0, np.float32)
    z[:, 4:] = 105.0  # 5% jump: edge
    far = z * 10  # same 5% jump at 10x the distance: still an edge
    smooth = np.tile(np.linspace(100, 101, 8, dtype=np.float32), (8, 1))
    for d in (z, far):
        e = passes.edges_from_depth(d)
        assert e[:, 3:5].all() and not e[:, 0].any()
    assert not passes.edges_from_depth(smooth).any()


def test_spearman():
    a = np.arange(100.0)
    assert metrics.spearman(a, a ** 3) == pytest.approx(1.0)
    assert metrics.spearman(a, -a) == pytest.approx(-1.0)


def test_structure_verdict():
    ctrl = np.random.default_rng(0).random((3, 16, 16))
    scores = metrics.structure_scores(ctrl * 2 + 1, ctrl)
    assert metrics.verdict_u1(scores)
    assert not metrics.verdict_u1(metrics.structure_scores(-ctrl, ctrl))


def test_flicker_ratio():
    ref = np.stack([np.full((4, 4), i * 2.0) for i in range(5)])
    out = ref.copy()
    assert metrics.flicker_ratio(out, ref) == pytest.approx(1.0)
    noisy = out + np.random.default_rng(1).normal(0, 20, out.shape)
    assert not metrics.verdict_u3(metrics.flicker_ratio(noisy, ref))


def test_unstable_fraction_flags_boiling_not_motion():
    pytest.importorskip("cv2")
    rng = np.random.default_rng(0)
    from numpy.fft import fft2, ifft2
    noise = rng.normal(0, 1, (160, 400))
    k = np.exp(-((np.fft.fftfreq(160)[:, None] ** 2 + np.fft.fftfreq(400)[None] ** 2) / 0.002))
    tex = np.real(ifft2(fft2(noise) * k))
    tex = 128 + 90 * tex / tex.std()
    pan = np.stack([tex[:, i:i + 200] for i in range(0, 40, 2)])  # textured camera pan
    assert metrics.unstable_fraction(pan) < 0.02
    boiling = pan + rng.normal(0, 30, pan.shape)
    assert metrics.unstable_fraction(boiling) > 0.2


def test_pop_score_detects_spike():
    frames = np.stack([np.full((8, 8), i, np.float32) for i in range(20)])
    assert metrics.pop_score(frames) < 1.1
    frames[10] += 50
    assert metrics.pop_score(frames) > 2
