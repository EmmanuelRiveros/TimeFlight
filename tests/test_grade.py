import numpy as np

from timeflight.grade import GradeParams, box_blur, gaussian_blur, grade_frame, grade_frames


def test_box_blur_keeps_flat_and_mean():
    flat = np.full((20, 30), 0.4, np.float32)
    assert np.allclose(box_blur(flat, 3, 0), 0.4) and np.allclose(box_blur(flat, 3, 1), 0.4)
    ramp = np.tile(np.arange(30, dtype=np.float32), (5, 1))
    assert np.allclose(box_blur(ramp, 2, 1)[:, 5:25], ramp[:, 5:25])  # linear inside the edges


def test_gaussian_blur_spreads_an_impulse():
    img = np.zeros((41, 41), np.float32)
    img[20, 20] = 1.0
    out = gaussian_blur(img, 4.0)
    assert abs(out.sum() - 1.0) < 1e-4
    ys, xs = np.mgrid[:41, :41]
    sigma = np.sqrt((out * (xs - 20) ** 2).sum())
    assert 3.5 < sigma < 4.6


def test_off_is_identity():
    rng = np.random.default_rng(1)
    frame = rng.integers(0, 256, (36, 64, 3), dtype=np.uint8)
    out = grade_frame(frame, GradeParams(clarity=0.0, grain=0.0), np.random.default_rng(0))
    assert out.dtype == np.uint8 and np.array_equal(out, frame)


def test_clarity_leaves_flat_areas_and_boosts_detail():
    frame = np.full((64, 128, 3), 120, np.uint8)
    assert np.array_equal(grade_frame(frame, GradeParams(clarity=0.5, grain=0.0), np.random.default_rng(0)), frame)
    frame[:, 60:68] = 160  # a bright stripe gains contrast against its surroundings
    out = grade_frame(frame, GradeParams(clarity=0.5, radius=0.1, grain=0.0), np.random.default_rng(0))
    assert out[32, 64, 0] > 160 and out[32, 20, 0] <= 120


def test_grain_is_zero_mean_midtone_weighted_and_new_each_frame():
    p = GradeParams(clarity=0.0, grain=0.03, grain_size=0.0)
    mid = np.full((200, 200, 3), 128, np.uint8)
    dark = np.full((200, 200, 3), 8, np.uint8)
    a, b = list(grade_frames([mid, mid], p, seed=3))
    d = a.astype(float) - 128
    assert abs(d.mean()) < 0.3 and 6 < d.std() < 10
    assert not np.array_equal(a, b)
    assert np.array_equal(a[..., 0], a[..., 1])  # monochrome
    dd = next(grade_frames([dark], p, seed=3)).astype(float) - 8
    assert dd.std() < 0.5 * d.std()
    again = next(grade_frames([mid], p, seed=3))
    assert np.array_equal(again, a)
