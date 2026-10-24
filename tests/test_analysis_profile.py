"""`mainspring.analysis`'s profile read, against files whose every point is known.

The window has to select exactly the bins of the closed m/z interval, through each
frame's own calibration; the sum over frames has to be the sum of the frames' profiles
and nothing else; an empty window has to come back as zeros; and a frame with no m/z
axis has to be refused rather than read (lab record, task 42).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from mainspring.analysis import (
    ScanProfile, argmax, bin_range, centroid, scan_profile, summed_profile,
)
from mainspring.uimf import Calibration
from mainspring.uimf.reader import UimfFile
from mainspring.uimf.writer import FrameSpec, GlobalSpec, UimfWriter
from synthetic import AVERAGE_TOF_LENGTH_NS, INTERCEPT, SLOPE, write_synthetic_uimf


def mz_of(bin_index, slope=SLOPE, intercept=INTERCEPT, bin_width_ns=1.0):
    """The calibration written out independently of the one under test, time clamped
    at the intercept as the documented calibration is."""
    return (slope * np.maximum(np.asarray(bin_index) * bin_width_ns / 1000.0 - intercept, 0.0)) ** 2


def expected(spec, frame, lo_mz, hi_mz):
    """The window's profile from what the fixture wrote: the closed interval on m/z."""
    out = np.zeros(spec.scans)
    for (f, scan), row in spec.scan_rows.items():
        if f != frame:
            continue
        inside = (mz_of(row.bin_index) >= lo_mz) & (mz_of(row.bin_index) <= hi_mz)
        out[scan] += float(row.intensity[inside].sum())
    return out


@pytest.fixture
def synthetic(tmp_path):
    spec = write_synthetic_uimf(tmp_path / "synth.uimf", frames=3, scans=16, bins=4096)
    return spec, UimfFile(spec.path)


def calibration():
    return Calibration(SLOPE, INTERCEPT, 1.0)


def window_of_bins(first, last):
    """An m/z window whose edges sit half a bin outside `first` and `last`."""
    return (float(mz_of(first - 0.5)), float(mz_of(last + 0.5)))


def test_the_window_selects_exactly_its_bins():
    assert bin_range(calibration(), window_of_bins(1000, 1040)) == (1000, 1041)


def test_the_bin_range_is_the_closed_interval_on_random_windows():
    rng = np.random.default_rng(42)
    bins = np.arange(4096, dtype=np.float64)
    mz = mz_of(bins)
    for _ in range(500):
        lo, hi = np.sort(rng.uniform(mz[200], mz[-1], size=2))
        start, stop = bin_range(calibration(), (lo, hi))
        closed = np.flatnonzero((mz >= lo) & (mz <= hi))
        edge = calibration().bin_of(np.array([lo, hi]))
        if np.any(np.abs(edge - np.round(edge)) < 1e-9):
            continue  # an edge on a whole bin: rounding decides, and the docstring says so
        assert (start, stop) == ((int(closed[0]), int(closed[-1]) + 1) if closed.size
                                 else (stop, stop)), (lo, hi)


def test_one_frame_gives_the_planted_points_per_scan(synthetic):
    spec, uimf = synthetic
    window = window_of_bins(100, 2100)
    for frame in spec.frames:
        whole = scan_profile(uimf.read_frame(frame), calibration(), window)
        cut = scan_profile(uimf.read_frame(frame, bin_range=bin_range(calibration(), window)),
                           calibration(), window)
        want = expected(spec, frame, *window)
        assert whole.dtype == np.float64 and whole.shape == (spec.scans,)
        assert want.sum() > 0
        np.testing.assert_array_equal(whole, want)
        np.testing.assert_array_equal(cut, want)


def test_the_sum_over_frames_is_the_sum_of_the_frames(synthetic):
    spec, uimf = synthetic
    window = window_of_bins(100, 2100)
    summed = summed_profile(uimf, [3, 1], window)
    np.testing.assert_array_equal(
        summed.values, expected(spec, 1, *window) + expected(spec, 3, *window))
    assert summed.frames == (3, 1)
    assert summed.tick_ms == pytest.approx(AVERAGE_TOF_LENGTH_NS * 1e-6, rel=1e-15)
    assert not summed.values.flags.writeable


def test_an_empty_window_is_zeros_and_no_statistic(synthetic):
    spec, uimf = synthetic
    window = (float(mz_of(5000)), float(mz_of(6000)))   # past the file's last bin
    summed = summed_profile(uimf, spec.frames, window)
    np.testing.assert_array_equal(summed.values, np.zeros(spec.scans))
    assert argmax(summed.values) is None and centroid(summed.values) is None
    assert summed.ms(argmax(summed.values)) is None


def test_the_profile_converts_scans_to_ms(synthetic):
    _, uimf = synthetic
    summed = summed_profile(uimf, [1], window_of_bins(100, 2100))
    tick = summed.tick_ms
    assert summed.ms(3) == 3 * tick and isinstance(summed.ms(3.5), float)
    np.testing.assert_array_equal(summed.ms(np.array([0, 2])), [0.0, 2 * tick])
    np.testing.assert_array_equal(summed.axis_ms, np.arange(summed.values.size) * tick)


def write_two_calibrations(path, slopes, scans, points):
    """Frames whose calibrations differ, one point each: `points[i]` is `(scan, bin)`."""
    writer = UimfWriter(path, GlobalSpec(bins=4096, bin_width_ns=1.0, instrument_name="SLIM3",
                                         date_started="8/25/2026 3:23:14 PM"),
                        journal_mode="delete", overwrite=True)
    with writer:
        for number, (slope, n, (scan, bin_index)) in enumerate(zip(slopes, scans, points), 1):
            writer.add_frame(FrameSpec(scans=n, accumulations=1, calibration_slope=slope,
                                       calibration_intercept=INTERCEPT,
                                       average_tof_length_ns=AVERAGE_TOF_LENGTH_NS),
                             frame=number)
            writer.write_scans(number, [(scan, np.array([bin_index]), np.array([100 * number],
                                                                             dtype=np.int32))])
            writer.finalise_frame(number, duration_s=0.1)
    return UimfFile(path)


def test_each_frame_is_placed_through_its_own_calibration(tmp_path):
    # The same m/z lands on bin 2000 under the first slope and on another bin under the
    # second; a sum through one calibration would miss one of the two points.
    second = 0.6
    other = round((math.sqrt(float(mz_of(2000))) / second + INTERCEPT) * 1000.0)
    assert abs(other - 2000) > 100
    near = sorted((float(mz_of(2000)), float(mz_of(other, slope=second))))
    window = (near[0] - 1e-4, near[1] + 1e-4)
    assert not window[0] <= mz_of(other) <= window[1]
    assert not window[0] <= mz_of(2000, slope=second) <= window[1]
    uimf = write_two_calibrations(tmp_path / "two.uimf", (SLOPE, second), (8, 12),
                                  ((2, 2000), (9, other)))
    summed = summed_profile(uimf, [1, 2], window)
    want = np.zeros(12)
    want[2], want[9] = 100.0, 200.0
    np.testing.assert_array_equal(summed.values, want)   # 8 scans added onto 12


def test_a_frame_with_no_mz_axis_is_refused(tmp_path):
    uimf = write_two_calibrations(tmp_path / "flat.uimf", (SLOPE, 0.0), (8, 8),
                                  ((2, 2000), (2, 2000)))
    window = window_of_bins(1990, 2010)
    summed_profile(uimf, [1], window)
    with pytest.raises(ValueError, match="frame 2"):
        summed_profile(uimf, [1, 2], window)
    with pytest.raises(ValueError, match="unusable"):
        bin_range(Calibration(0.0, INTERCEPT, 1.0), window)


@pytest.mark.parametrize("window", [(600.0, 500.0), (math.nan, 600.0), (500.0, math.inf)])
def test_a_window_that_is_not_an_ascending_finite_pair_is_refused(synthetic, window):
    _, uimf = synthetic
    with pytest.raises(ValueError, match="mz_window"):
        bin_range(calibration(), window)
    with pytest.raises(ValueError, match="mz_window"):
        summed_profile(uimf, [1], window)


def test_no_frames_is_refused(synthetic):
    _, uimf = synthetic
    with pytest.raises(ValueError, match="at least one frame"):
        summed_profile(uimf, [], window_of_bins(100, 2100))


def test_the_result_is_a_scan_profile(synthetic):
    _, uimf = synthetic
    assert isinstance(summed_profile(uimf, [1], window_of_bins(100, 120)), ScanProfile)
