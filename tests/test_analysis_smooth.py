"""`mainspring.analysis.gaussian_smooth` and `gaussian_gain`, against closed forms.

The smoother's whole case is what a non-negative kernel cannot do: ring. So the tests
are the properties that follow from that and from the kernel being a Gaussian in time
-- a step comes out as the normal CDF, nothing goes negative, the area is kept, and the
width in ms does not depend on the scan tick -- plus the edge rule (zero padding) and
the refusals (lab record, task 41).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from mainspring.analysis import gaussian_gain, gaussian_smooth

TICKS = (0.062, 0.129)
"""The two scan ticks the lab's files carry, in ms: 62 µs and 129 µs."""
SIGMAS = (0.6, 1.0)
"""The two widths the lab's figures are choosing between, in ms."""
RISE_SIGMAS = 2 * 1.2815515655446004
"""The normal CDF's 10–90 % rise, in σ: twice the 90th percentile."""


def impulse_response(tick_ms, sigma_ms, n=8001):
    x = np.zeros(n)
    x[n // 2] = 1.0
    return gaussian_smooth(x, tick_ms=tick_ms, sigma_ms=sigma_ms)


def rise_ms(y, tick_ms):
    """The 10–90 % rise of a step response, from first upward crossings."""
    def up(level):
        j = int(np.argmax(y >= level))
        return (j - 1 + (level - y[j - 1]) / (y[j] - y[j - 1])) * tick_ms
    return up(0.9) - up(0.1)


@pytest.mark.parametrize("tick_ms", TICKS)
@pytest.mark.parametrize("sigma_ms", SIGMAS)
def test_a_step_comes_out_as_the_normal_cdf(tick_ms, sigma_ms):
    n = 1 << 14
    step = np.zeros(n)
    step[n // 4:3 * n // 4] = 1.0
    y = gaussian_smooth(step, tick_ms=tick_ms, sigma_ms=sigma_ms)
    edge = y[:n // 2]
    assert rise_ms(edge, tick_ms) == pytest.approx(RISE_SIGMAS * sigma_ms, abs=0.01)
    assert edge.min() >= 0.0 and edge.max() <= 1.0 + 1e-12
    assert np.all(np.diff(edge) >= -1e-15)


def test_a_non_negative_profile_keeps_its_area_and_its_sign():
    rng = np.random.default_rng(41)
    profile = np.zeros(6000)
    profile[500:5500] = rng.poisson(3.0, 5000) * np.exp(-np.arange(5000) / 1500)
    for tick_ms in TICKS:
        for sigma_ms in SIGMAS:
            y = gaussian_smooth(profile, tick_ms=tick_ms, sigma_ms=sigma_ms)
            assert y.min() >= 0.0
            assert y.sum() == pytest.approx(profile.sum(), rel=1e-12)


@pytest.mark.parametrize("sigma_ms", SIGMAS)
def test_the_kernel_is_the_same_in_time_whatever_the_tick(sigma_ms):
    for tick_ms in TICKS:
        h = impulse_response(tick_ms, sigma_ms)
        t = (np.arange(h.size) - h.size // 2) * tick_ms
        assert h.sum() == pytest.approx(1.0, rel=1e-12)
        assert math.sqrt((h * t ** 2).sum()) == pytest.approx(sigma_ms, rel=1e-4)
        half = h >= h.max() / 2
        width = (np.count_nonzero(half)) * tick_ms
        assert width == pytest.approx(2 * math.sqrt(2 * math.log(2)) * sigma_ms, abs=tick_ms)


def measured_gain(f_hz, tick_ms, sigma_ms, n=40000):
    """The attenuation the smoother puts on a long, steady sinusoid, by projection on the
    middle half, where the ends cannot reach."""
    t = np.arange(n) * tick_ms * 1e-3
    x = np.sin(2 * math.pi * f_hz * t)
    y = gaussian_smooth(x, tick_ms=tick_ms, sigma_ms=sigma_ms)
    mid = slice(n // 4, 3 * n // 4)
    ref = np.exp(2j * math.pi * f_hz * t[mid])
    return float(np.real(np.mean(y[mid] * np.conj(ref)) / np.mean(x[mid] * np.conj(ref))))


@pytest.mark.parametrize("tick_ms", TICKS)
@pytest.mark.parametrize("sigma_ms", (0.35, *SIGMAS))
def test_the_gain_is_what_a_sinusoid_loses(tick_ms, sigma_ms):
    for f_hz in (100.0, 321.0, 954.0, 1420.0):
        want = float(gaussian_gain(f_hz, sigma_ms=sigma_ms))
        assert measured_gain(f_hz, tick_ms, sigma_ms) == pytest.approx(want, abs=2e-6)


def sampled_gain(f_hz, tick_ms, sigma_ms):
    """The sampled, truncated kernel's own frequency response (it is symmetric, so real)."""
    h = impulse_response(tick_ms, sigma_ms)
    k = np.arange(h.size) - h.size // 2
    return (h * np.cos(2 * math.pi * np.asarray(f_hz)[:, None] * k * tick_ms * 1e-3)).sum(-1)


@pytest.mark.parametrize("tick_ms", TICKS)
def test_the_closed_form_holds_to_nyquist_from_two_scans_up(tick_ms):
    f = np.linspace(0.0, 0.5e3 / tick_ms, 2001)
    for scans in (2.0, 3.0, 10.0, 40.0):
        sigma_ms = scans * tick_ms
        gap = np.abs(sampled_gain(f, tick_ms, sigma_ms) - gaussian_gain(f, sigma_ms=sigma_ms))
        assert gap.max() < 1e-6, scans
    # and the docstring's figures below that, where the kernel's aliases come in
    one = np.abs(sampled_gain(f, tick_ms, tick_ms) - gaussian_gain(f, sigma_ms=tick_ms)).max()
    assert 5e-3 < one < 1e-2


def test_the_gain_is_one_at_zero_and_takes_any_shape():
    assert gaussian_gain(0.0, sigma_ms=1.0) == 1.0
    f = np.array([[0.0, 100.0], [1e3, 1e4]])
    g = gaussian_gain(f, sigma_ms=0.6)
    assert g.shape == f.shape and np.all(np.diff(g.ravel()) < 0)
    assert g[0, 1] == pytest.approx(math.exp(-2 * math.pi ** 2 * 0.6e-3 ** 2 * 100.0 ** 2))


def test_the_ends_are_zero_padded():
    tick_ms, sigma_ms = 0.062, 0.6
    rng = np.random.default_rng(7)
    profile = rng.uniform(0.0, 10.0, 500)
    y = gaussian_smooth(profile, tick_ms=tick_ms, sigma_ms=sigma_ms)
    pad = np.zeros(200)
    padded = gaussian_smooth(np.concatenate([pad, profile, pad]), tick_ms=tick_ms, sigma_ms=sigma_ms)
    np.testing.assert_allclose(y, padded[200:700], rtol=0, atol=1e-12)
    # a constant comes out as itself in the middle and about half at either end, not as
    # a constant (which reflection or extension would give)
    flat = gaussian_smooth(np.ones(500), tick_ms=tick_ms, sigma_ms=sigma_ms)
    assert flat[250] == pytest.approx(1.0, abs=1e-12)
    for end in (flat[0], flat[-1]):
        assert 0.5 < end < 0.55
    # and an impulse at the first point loses the half of itself that lies beyond it
    first = np.zeros(500)
    first[0] = 1.0
    assert gaussian_smooth(first, tick_ms=tick_ms, sigma_ms=sigma_ms).sum() == pytest.approx(flat[0])


def test_the_output_is_float64_and_the_profile_length_and_the_input_is_untouched():
    profile = np.arange(30, dtype=np.int32)
    before = profile.copy()
    y = gaussian_smooth(profile, tick_ms=0.129, sigma_ms=1.0)
    assert y.dtype == np.float64 and y.shape == (30,)
    np.testing.assert_array_equal(profile, before)
    # a kernel longer than the profile (±5σ is 78 scans here) still gives its length back
    short = gaussian_smooth([0.0, 1.0, 4.0, 1.0, 0.0], tick_ms=0.129, sigma_ms=2.0)
    assert short.shape == (5,) and np.all(short > 0)
    assert gaussian_smooth([], tick_ms=0.062, sigma_ms=0.6).shape == (0,)
    assert gaussian_smooth([3.0, 5.0], tick_ms=0.062, sigma_ms=0.6).shape == (2,)


def test_a_sigma_under_one_scan_tends_to_a_delta():
    profile = np.random.default_rng(3).uniform(0.0, 1.0, 100)
    y = gaussian_smooth(profile, tick_ms=0.129, sigma_ms=0.01)
    np.testing.assert_allclose(y, profile, rtol=0, atol=1e-12)


@pytest.mark.parametrize("bad", (0.0, -0.6, float("nan"), float("inf"), float("-inf")))
def test_a_width_or_tick_that_is_not_a_finite_positive_number_is_refused(bad):
    with pytest.raises(ValueError, match="sigma_ms"):
        gaussian_smooth([1.0, 2.0], tick_ms=0.062, sigma_ms=bad)
    with pytest.raises(ValueError, match="tick_ms"):
        gaussian_smooth([1.0, 2.0], tick_ms=bad, sigma_ms=0.6)
    with pytest.raises(ValueError, match="sigma_ms"):
        gaussian_gain(100.0, sigma_ms=bad)


@pytest.mark.parametrize("profile", (np.ones((3, 4)), np.float64(2.0), [[1.0], [2.0]]))
def test_a_profile_that_is_not_one_dimensional_is_refused(profile):
    with pytest.raises(ValueError, match="one-dimensional"):
        gaussian_smooth(profile, tick_ms=0.062, sigma_ms=0.6)


@pytest.mark.parametrize("bad", (float("nan"), float("inf")))
def test_a_profile_holding_a_non_finite_value_is_refused(bad):
    with pytest.raises(ValueError, match="NaN or an infinity"):
        gaussian_smooth([1.0, bad, 1.0], tick_ms=0.062, sigma_ms=0.6)


def test_both_widths_are_required_keywords():
    with pytest.raises(TypeError):
        gaussian_smooth([1.0, 2.0], 0.062, 0.6)  # positional
    with pytest.raises(TypeError):
        gaussian_smooth([1.0, 2.0], tick_ms=0.062)  # no default width
    with pytest.raises(TypeError):
        gaussian_gain(100.0)
    with pytest.raises(TypeError):
        gaussian_gain(100.0, 0.6)
