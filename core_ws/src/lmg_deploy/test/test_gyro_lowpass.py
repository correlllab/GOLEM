"""Test the gyro low-pass: a Butterworth at any sample rate, its step response, its restart after a dropout.

    python -m pytest test/test_gyro_lowpass.py
"""

import numpy as np
import pytest

from lmg_deploy.gyro_lowpass import GyroLowpass


def _lowpass_sine(cutoff, freq, dt, seconds=4.0):
    """Return the GyroLowpass's steady-state amplitude on a unit sine of freq Hz sampled every dt s."""
    lp = GyroLowpass(cutoff)
    t = np.arange(0.0, seconds, dt)
    y = np.array([lp(np.array([np.sin(2 * np.pi * freq * tk)] * 3), dt if k else 0.0)[0]
                  for k, tk in enumerate(t)])
    return np.abs(y[t > seconds / 2]).max()


def test_gyro_lowpass_is_a_butterworth_at_any_rate():
    """The low-pass has the Butterworth magnitude at 200 Hz and at a 500 Hz physics step (/lowstate's rate)."""
    for dt in (0.005, 0.002):   # 200 Hz and a 500 Hz physics step
        # Butterworth magnitude 1 / sqrt(1 + (f / fc)^4).
        for freq, gain in ((1.0, 1.0 / np.sqrt(1 + (1 / 8) ** 4)), (20.0, 1.0 / np.sqrt(1 + (20 / 8) ** 4))):
            assert _lowpass_sine(8.0, freq, dt) == pytest.approx(gain, rel=0.03)


def test_gyro_lowpass_step_matches_across_rates_and_settles():
    """The step response is the same at two sample rates, overshoots as a Butterworth's and settles."""
    out = {}
    for dt in (0.005, 0.0025):
        lp = GyroLowpass(8.0)
        lp(np.zeros(3), 0.0)
        ys = [lp(np.ones(3), dt)[0] for _ in range(int(round(0.5 / dt)))]
        out[dt] = ys[int(round(0.05 / dt)) - 1]           # value at 50 ms
        assert max(ys) == pytest.approx(1.0432, abs=2e-3)  # a Butterworth's 4.3 % step overshoot
        assert ys[-1] == pytest.approx(1.0, abs=1e-3)      # settled after 0.5 s
    assert out[0.005] == pytest.approx(out[0.0025], abs=1e-12)


def test_gyro_lowpass_restarts_after_a_dropout():
    """After a gap longer than MAX_DT the low-pass restarts at the input."""
    lp = GyroLowpass(8.0)
    lp(np.zeros(3), 0.0)
    np.testing.assert_allclose(lp(np.full(3, 2.0), 0.5), 2.0)   # gap > MAX_DT: restart at the input
