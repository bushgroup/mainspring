"""A Gaussian smoother for an arrival-time profile, and its gain.

A profile here is intensity per scan, one value per mobility scan of a frame; one scan
is the frame's `AverageTOFLength`, so the caller supplies the scan tick in ms and every
width is in ms too. A 62 µs file and a 129 µs file then get the same kernel in time.

The kernel is a normalised Gaussian, sampled at the scan tick and truncated at ±5σ. It
is non-negative, so it cannot ring: a step comes out as the normal CDF (10–90 % rise
2.563σ, no undershoot ahead of an edge and no overshoot after it), and a non-negative
profile stays non-negative. It is the arrival-time smoother the lab chose to replace a
zero-phase Butterworth, whose ringing put negative intensity ahead of a leading edge
(lab record, task 41).

**There is no default width.** The right σ depends on the figure and the data, and a
default would be a width nobody chose; both functions take `sigma_ms` as a required
keyword.

numpy only.
"""

from __future__ import annotations

import math

import numpy as np

TRUNCATE_SIGMAS = 5.0
"""The kernel's half-width, in σ. What lies beyond is under 6e-7 of the kernel's area."""


def _positive(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a finite number above zero, not {value!r}")
    return value


def _kernel(tick_ms: float, sigma_ms: float) -> np.ndarray:
    """The normalised kernel `gaussian_smooth` convolves with: odd length, centred,
    summing to 1, truncated at ±5σ."""
    tick_ms = _positive("tick_ms", tick_ms)
    sigma_ms = _positive("sigma_ms", sigma_ms)
    s = sigma_ms / tick_ms
    half = int(math.ceil(TRUNCATE_SIGMAS * s))
    x = np.arange(-half, half + 1)
    kernel = np.exp(-0.5 * (x / s) ** 2)
    return kernel / kernel.sum()


def gaussian_smooth(profile, *, tick_ms: float, sigma_ms: float) -> np.ndarray:
    """`profile` smoothed by a Gaussian of `sigma_ms`, sampled at `tick_ms` per point.

    Returns float64, the same length as `profile`, which is not modified. Both widths
    are keyword-only so the two floats cannot be passed the wrong way round, and
    neither has a default.

    **Zero padding, deliberately.** Beyond either end the profile is taken to be zero:
    a frame starts before the ions arrive and an arrival-time profile is zero outside
    it, whereas reflecting or extending the end would invent counts. So the area is
    preserved wherever the signal lies more than 5σ from both ends, and intensity
    within 5σ of an end is partly smoothed out through it.

    A σ under one scan is allowed: the kernel then tends to a delta and the profile
    comes back nearly unchanged. Raises `ValueError` for a σ or tick that is not a
    finite number above zero, for a profile that is not one-dimensional, and for a
    profile holding a NaN or an infinity, which the kernel would spread over 10σ.
    """
    kernel = _kernel(tick_ms, sigma_ms)
    values = np.asarray(profile, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"profile must be one-dimensional, not of shape {values.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError("profile holds a NaN or an infinity")
    if values.size == 0:
        return values.copy()
    half = kernel.size // 2
    # The full convolution, cut to the profile: `mode="same"` whenever the profile is
    # at least as long as the kernel, and still the profile's length when it is not.
    return np.convolve(values, kernel)[half:half + values.size]


def gaussian_gain(f_hz, *, sigma_ms: float) -> np.ndarray:
    """What `gaussian_smooth` multiplies a sinusoid of `f_hz` by: exp(−2π²σ²f²), σ in s.

    The continuous kernel's closed form, the shape of `f_hz`. The sampled, truncated
    kernel's own response differs from it by under 1e-6 at every frequency up to
    Nyquist when σ is at least two scans (0.124 ms at 62 µs, 0.258 ms at 129 µs). Below
    that the sampled kernel's aliases widen the difference: 1.5e-5 at 1.5 scans, 7e-3
    at one scan (`tests/test_analysis_smooth.py` measures both).
    """
    sigma_s = _positive("sigma_ms", sigma_ms) * 1e-3
    f = np.asarray(f_hz, dtype=np.float64)
    return np.exp(-2.0 * math.pi ** 2 * sigma_s ** 2 * f ** 2)
