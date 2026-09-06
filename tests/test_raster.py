"""Rasterisation and the axis tables, held to invariants rather than to pictures.

A heatmap is hard to test by looking at it and easy to test by conservation. Four
things must hold, and between them they pin down the whole reduction:

* a full-range `sum` image totals the frame's `TIC`, because every point lands in
  exactly one pixel and nothing is dropped or counted twice;
* a full-range `max` image's maximum is the frame's largest stored intensity;
* a zoomed window's counts equal a brute-force mask over the same window;
* a side-plot profile totals the same thing the image over that window does;
* `render_view`, which the viewer's render worker calls because computing the three
  together is cheaper, returns exactly what the three separate calls do -- an
  optimisation that changed an answer would be a bug the invariants above cannot see,
  since both halves would still conserve.

The axis tables get their own tests, because the calibration is where a wrong constant
would look plausible: `mz_axis` must never decrease, and the whole point of the
`DisplayAxes` seam is that swapping the axes and switching to raw units are a
re-assignment of two tables rather than a second code path.
"""

from __future__ import annotations

import numpy as np
import pytest

from mainspring.uimf.calib import Calibration, arrival_time_ms, scan_axis_ms
from mainspring.uimf.raster import DisplayAxes, profile, rasterise, render_view
from mainspring.uimf.reader import UimfFile

SLOPE, INTERCEPT = 0.738123, 0.07690495


@pytest.fixture
def frame_and_axes(synthetic_uimf):
    uimf = UimfFile(synthetic_uimf.path)
    params = uimf.frame_params(1)
    frame = uimf.read_frame(1)
    axes = DisplayAxes.build(
        frame, params.calibration(synthetic_uimf.bin_width_ns), params.average_tof_length_ns
    )
    return frame, axes


# --- the calibration --------------------------------------------------------------


def test_mz_is_the_formula_of_the_format_note():
    calibration = Calibration(SLOPE, INTERCEPT, 1.0)
    time_us = 31290 * 1.0 / 1000.0
    assert calibration.mz(31290) == pytest.approx((SLOPE * (time_us - INTERCEPT)) ** 2, rel=1e-15)


def test_mz_and_bin_of_are_inverses_above_the_intercept():
    calibration = Calibration(SLOPE, INTERCEPT, 1.0)
    bins = np.array([100.0, 5000.0, 31290.0, 114688.0])
    assert calibration.bin_of(calibration.mz(bins)) == pytest.approx(bins)


def test_the_mz_axis_never_decreases():
    """Below `T0` the parabola turns back up, and an axis that decreases would put the
    lowest bins on the wrong side of the image (lab record, task 01)."""
    edges = Calibration(SLOPE, INTERCEPT, 1.0).mz_axis(114688)
    assert edges.size == 114689
    assert (np.diff(edges) >= 0).all()
    assert edges[0] == 0.0 and edges[80] > 0.0


def test_the_mz_axis_is_cached_and_read_only():
    calibration = Calibration(SLOPE, INTERCEPT, 1.0)
    first = calibration.mz_axis(4096)
    assert calibration.mz_axis(4096) is first, "the same table, not a rebuild"
    with pytest.raises(ValueError):
        first[0] = 1.0


def test_an_unusable_calibration_says_so_rather_than_inventing_an_axis():
    assert not Calibration(0.0, 0.0, 1.0).usable
    with pytest.raises(ValueError, match="unusable"):
        Calibration(0.0, 0.0, 1.0).bin_of(500.0)


def test_arrival_time_is_scans_times_the_tof_length():
    assert arrival_time_ms(4992, 129003.607843137) == pytest.approx(644.0, abs=0.1)
    edges = scan_axis_ms(5000, 129003.607843137)
    assert edges.size == 5001 and edges[0] == 0.0
    assert (np.diff(edges) > 0).all()


def test_scan_axis_ms_subtracts_the_t0_offset_and_may_go_negative():
    """The viewer's own offset, not a file-native calibration term (module docstring)
    -- so unlike `mz_axis` there is no floor clamping the result to zero."""
    plain = scan_axis_ms(5000, 129003.607843137)
    shifted = scan_axis_ms(5000, 129003.607843137, 100.0)
    assert np.array_equal(shifted, plain - 100.0)
    assert shifted[0] == pytest.approx(-100.0)


# --- the axis tables as a seam ------------------------------------------------------


def test_the_default_axes_are_mz_against_arrival_time(frame_and_axes):
    _, axes = frame_and_axes
    assert axes.x_label == "m/z" and axes.y_label == "Arrival time (ms)"
    assert not axes.swapped


def test_raw_units_and_swapping_only_re_assign_the_tables(synthetic_uimf):
    uimf = UimfFile(synthetic_uimf.path)
    params = uimf.frame_params(1)
    frame = uimf.read_frame(1)
    calibration = params.calibration(synthetic_uimf.bin_width_ns)
    plain = DisplayAxes.build(frame, calibration, params.average_tof_length_ns)
    swapped = DisplayAxes.build(frame, calibration, params.average_tof_length_ns, swapped=True)
    raw = DisplayAxes.build(frame, calibration, params.average_tof_length_ns, raw_units=True)

    assert np.array_equal(swapped.x_edges, plain.y_edges)
    assert np.array_equal(swapped.y_edges, plain.x_edges)
    assert np.array_equal(swapped.bin_edges, plain.bin_edges)
    assert raw.x_label == "TOF bin" and raw.y_label == "Scan"
    assert raw.x_edges.tolist() == list(range(frame.bins + 1))


def test_t0_offset_ms_shifts_arrival_time_but_not_raw_scan(synthetic_uimf):
    uimf = UimfFile(synthetic_uimf.path)
    params = uimf.frame_params(1)
    frame = uimf.read_frame(1)
    calibration = params.calibration(synthetic_uimf.bin_width_ns)
    plain = DisplayAxes.build(frame, calibration, params.average_tof_length_ns)
    offset = DisplayAxes.build(
        frame, calibration, params.average_tof_length_ns, t0_offset_ms=100.0
    )
    raw = DisplayAxes.build(
        frame, calibration, params.average_tof_length_ns, raw_units=True, t0_offset_ms=100.0
    )

    assert np.array_equal(offset.y_edges, plain.y_edges - 100.0)
    assert np.array_equal(offset.x_edges, plain.x_edges)  # the m/z axis is untouched
    # Raw units show a scan index, not a time -- the offset does not apply to it.
    assert raw.y_edges.tolist() == list(range(frame.scans + 1))


def test_an_uncalibrated_frame_falls_back_to_bins(synthetic_uimf):
    frame = UimfFile(synthetic_uimf.path).read_frame(1)
    axes = DisplayAxes.build(frame, Calibration(0.0, 0.0, 1.0), 0.0)
    assert axes.x_label == "TOF bin" and axes.y_label == "Scan"


# --- the four invariants ------------------------------------------------------------


def test_a_full_range_sum_image_conserves_the_total(frame_and_axes, synthetic_uimf):
    frame, axes = frame_and_axes
    x_range, y_range = axes.full_range
    result = rasterise(frame, axes, x_range, y_range, 400, 300)
    assert result.points_in_view == len(frame)
    assert result.tic_in_view == synthetic_uimf.tic(1)
    assert result.image.sum(dtype=np.float64) == pytest.approx(synthetic_uimf.tic(1), rel=1e-6)


def test_a_full_range_max_image_finds_the_largest_stored_intensity(frame_and_axes):
    frame, axes = frame_and_axes
    x_range, y_range = axes.full_range
    result = rasterise(frame, axes, x_range, y_range, 400, 300, aggregate="max")
    assert result.image.max() == pytest.approx(float(frame.intensity.max()))
    assert result.max_intensity == float(frame.intensity.max())


def test_a_zoom_window_matches_a_brute_force_mask(frame_and_axes):
    frame, axes = frame_and_axes
    (x0, x1), (y0, y1) = axes.full_range
    x_range = (x0 + 0.2 * (x1 - x0), x0 + 0.6 * (x1 - x0))
    y_range = (y0 + 0.3 * (y1 - y0), y0 + 0.8 * (y1 - y0))
    result = rasterise(frame, axes, x_range, y_range, 128, 96)

    mz = 0.5 * (axes.x_edges[:-1] + axes.x_edges[1:])[frame.bin_index]
    ms = 0.5 * (axes.y_edges[:-1] + axes.y_edges[1:])[frame.scan_of()]
    inside = ((mz >= x_range[0]) & (mz < x_range[1])
              & (ms >= y_range[0]) & (ms < y_range[1]))
    assert result.points_in_view == int(inside.sum())
    assert result.tic_in_view == pytest.approx(float(frame.intensity[inside].sum()))
    assert result.image.sum(dtype=np.float64) == pytest.approx(result.tic_in_view, rel=1e-6)


def test_the_image_is_never_wider_than_the_elements_it_covers(frame_and_axes):
    """Zoomed past one bin per pixel there is nothing to gain from more columns, and the
    viewer stretches what it gets."""
    frame, axes = frame_and_axes
    # Well above the intercept: the first 77 bins share an m/z of 0, so a window down
    # there is degenerate rather than narrow.
    result = rasterise(frame, axes, (axes.x_edges[2000], axes.x_edges[2005]),
                       axes.full_range[1], 800, 600)
    assert result.image.shape[1] <= 5


def test_swapping_the_axes_transposes_the_image(frame_and_axes, synthetic_uimf):
    frame, axes = frame_and_axes
    swapped = DisplayAxes(axes.y_edges, axes.x_edges, axes.y_label, axes.x_label, True)
    plain = rasterise(frame, axes, *axes.full_range, 200, 200)
    other = rasterise(frame, swapped, *swapped.full_range, 200, 200)
    assert other.tic_in_view == plain.tic_in_view == synthetic_uimf.tic(1)
    assert np.allclose(other.image, plain.image.T)


def test_a_window_with_no_points_is_a_blank_image(frame_and_axes):
    frame, axes = frame_and_axes
    x0 = float(axes.x_edges[-1])
    result = rasterise(frame, axes, (x0 * 1.5, x0 * 2.0), axes.full_range[1], 64, 64)
    assert result.points_in_view == 0
    assert result.tic_in_view == 0.0
    assert not result.image.any()


def test_an_empty_display_range_is_refused(frame_and_axes):
    frame, axes = frame_and_axes
    with pytest.raises(ValueError, match="empty display range"):
        rasterise(frame, axes, (5.0, 5.0), axes.full_range[1], 64, 64)


def test_an_unknown_aggregate_is_refused(frame_and_axes):
    frame, axes = frame_and_axes
    with pytest.raises(ValueError, match="aggregate"):
        rasterise(frame, axes, *axes.full_range, 64, 64, aggregate="median")


# --- the side plots -------------------------------------------------------------------


@pytest.mark.parametrize("along", ["x", "y"])
def test_a_profile_totals_what_the_image_over_the_same_window_does(frame_and_axes, along):
    frame, axes = frame_and_axes
    (x0, x1), (y0, y1) = axes.full_range
    x_range = (x0 + 0.1 * (x1 - x0), x0 + 0.7 * (x1 - x0))
    y_range = (y0 + 0.2 * (y1 - y0), y0 + 0.9 * (y1 - y0))
    image = rasterise(frame, axes, x_range, y_range, 200, 150)
    edges, values = profile(frame, axes, x_range, y_range, along=along)
    assert edges.size == values.size + 1
    assert values.sum() == pytest.approx(image.tic_in_view)


def test_a_native_profile_has_one_value_per_source_element(frame_and_axes):
    frame, axes = frame_and_axes
    edges, values = profile(frame, axes, *axes.full_range, along="x")
    assert values.size == frame.bins
    edges, values = profile(frame, axes, *axes.full_range, along="y")
    assert values.size == frame.scans


def test_a_binned_profile_keeps_the_bins_it_was_asked_for(frame_and_axes):
    frame, axes = frame_and_axes
    edges, values = profile(frame, axes, *axes.full_range, along="x", bins=64)
    assert values.size == 64 and edges.size == 65
    assert values.sum() == pytest.approx(
        rasterise(frame, axes, *axes.full_range, 64, 64).tic_in_view
    )


def test_render_view_returns_what_the_three_separate_calls_do(frame_and_axes):
    frame, axes = frame_and_axes
    (x0, x1), (y0, y1) = axes.full_range
    windows = (
        ((x0, x1), (y0, y1)),
        ((x0 + 0.3 * (x1 - x0), x0 + 0.6 * (x1 - x0)), (y0, y0 + 0.5 * (y1 - y0))),
    )
    for x_range, y_range in windows:
        for aggregate in ("sum", "max"):
            result, x_profile, y_profile = render_view(
                frame, axes, x_range, y_range, 300, 200, aggregate=aggregate
            )
            separate = rasterise(
                frame, axes, x_range, y_range, 300, 200, aggregate=aggregate
            )
            assert np.array_equal(result.image, separate.image)
            assert result.tic_in_view == separate.tic_in_view
            assert result.max_intensity == separate.max_intensity
            assert result.points_in_view == separate.points_in_view
            for got, want in (
                (x_profile, profile(frame, axes, x_range, y_range, along="x")),
                (y_profile, profile(frame, axes, x_range, y_range, along="y")),
            ):
                assert np.array_equal(got[0], want[0])
                assert np.array_equal(got[1], want[1])


def test_render_view_refuses_an_unknown_aggregate(frame_and_axes):
    frame, axes = frame_and_axes
    with pytest.raises(ValueError, match="aggregate"):
        render_view(frame, axes, *axes.full_range, 64, 64, aggregate="mean")


def test_an_unknown_direction_is_refused(frame_and_axes):
    frame, axes = frame_and_axes
    with pytest.raises(ValueError, match="along"):
        profile(frame, axes, *axes.full_range, along="z")


# --- and the same invariants on a real file -------------------------------------------


def test_a_real_frame_conserves_its_own_tic(real_uimf):
    uimf = UimfFile(real_uimf)
    number = uimf.frame_numbers()[0]
    params = uimf.frame_params(number)
    frame = uimf.read_frame(number)
    _, _, bpi, tic = uimf.scan_summary(number)
    axes = DisplayAxes.build(frame, params.calibration(uimf.global_params().bin_width_ns),
                             params.average_tof_length_ns)
    result = rasterise(frame, axes, *axes.full_range, 1200, 800)
    assert result.tic_in_view == pytest.approx(float(tic.sum()))
    assert result.image.sum(dtype=np.float64) == pytest.approx(float(tic.sum()), rel=1e-5)
    peak = rasterise(frame, axes, *axes.full_range, 1200, 800, aggregate="max")
    assert peak.image.max() == pytest.approx(float(bpi.max()))


def test_render_view_agrees_with_the_separate_calls_on_a_real_file(real_uimf):
    """The same equivalence on files whose axis tables are not the fixture's.

    Worth its own run on real data: the shared selection is indexed off the axis tables,
    and a file with a different bin count, a clamped low-m/z region or scans the writer
    never stored is where an off-by-one in it would show.
    """
    uimf = UimfFile(real_uimf)
    number = uimf.frame_numbers()[0]
    params = uimf.frame_params(number)
    frame = uimf.read_frame(number)
    axes = DisplayAxes.build(frame, params.calibration(uimf.global_params().bin_width_ns),
                             params.average_tof_length_ns)
    (x0, x1), (y0, y1) = axes.full_range
    x_range = (x0 + 0.2 * (x1 - x0), x0 + 0.7 * (x1 - x0))

    result, x_profile, y_profile = render_view(frame, axes, x_range, (y0, y1), 800, 500)

    assert np.array_equal(
        result.image, rasterise(frame, axes, x_range, (y0, y1), 800, 500).image
    )
    assert np.array_equal(
        x_profile[1], profile(frame, axes, x_range, (y0, y1), along="x")[1]
    )
    assert np.array_equal(
        y_profile[1], profile(frame, axes, x_range, (y0, y1), along="y")[1]
    )
    assert float(np.sum(x_profile[1])) == pytest.approx(result.tic_in_view, rel=1e-12)
