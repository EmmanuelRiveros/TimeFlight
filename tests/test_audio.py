import numpy as np
import pytest

from timeflight.audio import mix_eras

SR = 1000


def test_crossfade_follows_the_wave_and_fades_out():
    present = np.full((2, 10 * SR), 0.5, np.float32)
    past = np.full((1, 10 * SR), -0.5, np.float32)  # mono is spread to both channels
    out = mix_eras(present, past, SR, start_s=2.0, end_s=6.0, total_s=12.0, clip_s=10.0, fade_out_s=1.0)
    assert out.shape == (2, 12 * SR)
    assert out[:, 1 * SR] == pytest.approx(0.5)      # before the wave: present only
    assert out[:, 7 * SR] == pytest.approx(-0.5)     # after: past only
    mid = out[0, 4 * SR]                             # halfway: equal power, the two cancel here
    assert mid == pytest.approx(0.0, abs=1e-6)
    assert out[0, int(9.5 * SR)] == pytest.approx(-0.25, abs=1e-3)  # halfway through the fade-out
    assert np.all(out[:, 10 * SR:] == 0)             # silence under the cards


def test_short_or_long_inputs_are_fit_to_the_clip():
    out = mix_eras(np.ones((1, 3 * SR)), np.ones((1, 30 * SR)), SR, 1.0, 2.0, 5.0, 5.0, 0.0)
    assert out.shape == (1, 5 * SR)
    assert out[0, 4 * SR] == pytest.approx(1.0)
