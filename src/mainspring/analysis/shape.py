"""What the lab reads off an arrival-time profile: position, spread, width, edge, peaks.

Every function takes a 1-D profile, intensity per scan, and returns scans as floats
(`ScanProfile.ms` converts), or `None` for a profile with nothing in it -- no value above
zero, an empty array included. **None of them smooths.** A raw single-repetition
profile is a few hundred stored points, and which smoother, at what width, is the
caller's choice: smooth first (`gaussian_smooth`, a boxcar), then read.

The definitions are the ones the lab's scripts already used, gathered here so that two
figures quoting a "centroid" mean the same thing (lab record, task 42):

    argmax       the scan of the largest value, the first of a tie
    centroid     the intensity-weighted mean scan over the contiguous run around the
                 argmax that stays at or above half its height
    median       the scan at which the cumulative intensity first reaches half the total
    iqr          the 25th to 75th percentile span of the cumulative, interpolated within
                 a scan (1.349 sigma for a Gaussian)
    fwhm         the width at half the argmax's height, interpolated at the crossing
                 nearest the peak on each side (2.355 sigma for a Gaussian)
    rise_10_90   the leading edge: from the 10 % to the 90 % crossing walking back from
                 the argmax, interpolated
    local_maxima every local maximum at or above a fraction of the largest value, a
                 plateau counted once at its first scan, no merging of neighbours

The crossings take the profile as zero beyond either end, as `gaussian_smooth` does, so
a peak cut by the end of the frame reads a crossing just past it rather than none.

numpy only.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = ["argmax", "centroid", "fwhm", "iqr", "local_maxima", "median", "rise_10_90"]


def _values(profile) -> np.ndarray:
    values = np.asarray(profile, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"profile must be one-dimensional, not of shape {values.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError("profile holds a NaN or an infinity")
    return values


def _peak(profile) -> tuple[np.ndarray, int | None]:
    """The profile as float64 and its argmax, or None when nothing in it is above zero."""
    values = _values(profile)
    if values.size == 0:
        return values, None
    top = int(np.argmax(values))
    return values, (top if values[top] > 0.0 else None)


def _cumulative(profile) -> tuple[np.ndarray, np.ndarray | None]:
    values = _values(profile)
    if np.any(values < 0.0):
        raise ValueError("profile holds a negative value, which makes its cumulative "
                         "non-monotonic; a percentile of it is not defined")
    cumulative = np.cumsum(values)
    return values, (cumulative if cumulative.size and cumulative[-1] > 0.0 else None)


def _outside(values: np.ndarray, top: int, level: float) -> tuple[int, int]:
    """The nearest index on each side of `top` whose value is below `level`: -1 or
    `values.size` when the run at or above it reaches the end."""
    below = np.flatnonzero(values[:top] < level)
    above = np.flatnonzero(values[top + 1:] < level)
    return (int(below[-1]) if below.size else -1,
            top + 1 + int(above[0]) if above.size else values.size)


def _at(values: np.ndarray, index: int) -> float:
    return float(values[index]) if 0 <= index < values.size else 0.0


def _left_crossing(values: np.ndarray, top: int, level: float) -> float:
    i = _outside(values, top, level)[0]
    return float(i + (level - _at(values, i)) / (_at(values, i + 1) - _at(values, i)))


def argmax(profile) -> float | None:
    """The scan of the largest value; the first of a tie."""
    return None if (top := _peak(profile)[1]) is None else float(top)


def centroid(profile) -> float | None:
    """The intensity-weighted mean scan over the contiguous run of scans around the
    argmax whose values stay at or above half of it. A plain centroid, no fitting; on a
    sparse profile the run can be one scan wide."""
    values, top = _peak(profile)
    if top is None:
        return None
    lo, hi = _outside(values, top, values[top] / 2.0)
    run = values[lo + 1:hi]
    return float((np.arange(lo + 1, hi) * run).sum() / run.sum())


def median(profile) -> float | None:
    """The first scan at which the cumulative intensity reaches half the total. A whole
    scan, unlike `iqr`'s interpolated percentiles. Refuses a negative value."""
    _, cumulative = _cumulative(profile)
    if cumulative is None:
        return None
    return float(np.searchsorted(cumulative, cumulative[-1] / 2.0))


def iqr(profile) -> float | None:
    """The span of scans holding the central half of the intensity: the 75th minus the
    25th percentile scan of the cumulative, each interpolated within its scan, a scan's
    intensity taken as spread evenly over the interval ending at it. Spike-proof and
    floor-free; 1.349σ for a Gaussian. Refuses a negative value."""
    values, cumulative = _cumulative(profile)
    if cumulative is None:
        return None

    def percentile(q: float) -> float:
        target = q * cumulative[-1]
        scan = int(np.searchsorted(cumulative, target))
        before = float(cumulative[scan - 1]) if scan else 0.0
        return scan - 1 + (target - before) / values[scan]

    return float(percentile(0.75) - percentile(0.25))


def fwhm(profile) -> float | None:
    """The full width at half the argmax's height, linearly interpolated at the crossing
    nearest the peak on each side; 2.355σ for a Gaussian. Read on the profile as given,
    so on a sparse one it is the width of a spike unless the caller smooths first."""
    values, top = _peak(profile)
    if top is None:
        return None
    half = values[top] / 2.0
    j = _outside(values, top, half)[1]
    right = j - 1 + (_at(values, j - 1) - half) / (_at(values, j - 1) - _at(values, j))
    return float(right - _left_crossing(values, top, half))


def rise_10_90(profile) -> float | None:
    """The leading edge's 10–90 % rise: walking back from the argmax, the 90 % crossing
    less the 10 % one, each of the argmax's height and interpolated between scans;
    2.563σ for a step through a Gaussian, 1.6869σ for a Gaussian's own flank."""
    values, top = _peak(profile)
    if top is None:
        return None
    height = values[top]
    return (_left_crossing(values, top, 0.9 * height)
            - _left_crossing(values, top, 0.1 * height))


def local_maxima(profile, *, fraction: float) -> np.ndarray | None:
    """The scans of every local maximum at or above `fraction` of the largest value.

    A local maximum rises from the scan before it and falls to the first different
    value after it; a plateau counts once, at its first scan, and neither end of the
    profile is one. Neighbouring maxima are not merged, so on an unsmoothed profile this
    counts noise. `fraction` is required, from 0 to 1: the floor is the caller's.
    """
    fraction = float(fraction)
    if not (math.isfinite(fraction) and 0.0 <= fraction <= 1.0):
        raise ValueError(f"fraction must be a number from 0 to 1, not {fraction!r}")
    values, top = _peak(profile)
    if top is None:
        return None
    changes = np.flatnonzero(np.diff(values))
    rising = np.diff(values)[changes] > 0.0
    starts = changes[:-1][rising[:-1] & ~rising[1:]] + 1
    return starts[values[starts] >= fraction * values[top]].astype(np.float64)
