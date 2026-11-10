"""The smoothed arrival-time plot: smoothed with real neighbours, drawn over the raw one.

The arrival-time side plot is drawn Gaussian-smoothed over a dimmed copy of the raw
projection while `Data settings > Smooth arrival time` is ticked (lab record, task 44).
What these tests hold:

* a curve near a zoom edge is the full-frame curve at the same scans, because the
  projection is widened by the kernel's reach before it is smoothed;
* the raw profiles a render returns are the same with smoothing on or off, so the peak
  readout and everything else that reads them is unchanged;
* the window draws the smoothed curve on whichever plot is arrival time, dims the raw
  one under it, passes the light-palette walk, and exports the raw curve alone.
"""

from __future__ import annotations

import numpy as np
import pytest

from mainspring.analysis import gaussian_smooth
from mainspring.uimf import DisplayAxes, UimfFile
from mainspring.uimf.raster import render_view
from mainspring.viewer import theme
from mainspring.viewer.export import export_display
from mainspring.viewer.main_window import MainWindow
from mainspring.viewer.settings import ViewerSettings
from mainspring.viewer.side_plots import peak_of
from mainspring.viewer.smoothing import SmoothingWidthDialog, parse_width
from mainspring.viewer.workers import RenderRequest, RenderWorker, smoothed_arrival

from synthetic import write_synthetic_uimf

SCANS = 200


@pytest.fixture
def long_file(tmp_path):
    """Enough scans that a zoom leaves several kernel widths on either side."""
    return write_synthetic_uimf(tmp_path / "long.uimf", frames=2, scans=SCANS, bins=4096)


def _frame(spec, swapped: bool):
    uimf = UimfFile(spec.path)
    params = uimf.frame_params(1)
    frame = uimf.read_frame(1)
    axes = DisplayAxes.build(
        frame, params.calibration(spec.bin_width_ns), params.average_tof_length_ns,
        swapped=swapped,
    )
    return frame, axes, params.average_tof_length_ns * 1e-6


def _ranges(axes: DisplayAxes, scans: "tuple[float, float]", bins: "tuple[float, float]"):
    """`(x_range, y_range)` with the scan range on whichever axis carries scans."""
    return (scans, bins) if axes.swapped else (bins, scans)


@pytest.mark.parametrize("swapped", [False, True], ids=["default", "swapped"])
def test_a_zoomed_curve_is_the_full_frame_curve_at_the_same_scans(long_file, swapped):
    frame, axes, tick_ms = _frame(long_file, swapped)
    sigma_ms = 3.0 * tick_ms  # 15 scans of reach either side
    bin_edges, scan_edges = axes.bin_edges, axes.scan_edges
    # The same m/z window both times, narrower than the frame, so the curve is the one
    # on screen and not the whole frame's.
    bins = (float(bin_edges[200]), float(bin_edges[3000]))
    arrival = 0 if swapped else 1  # which profile of a render is arrival time

    full_x, full_y = _ranges(axes, (float(scan_edges[0]), float(scan_edges[-1])), bins)
    full = render_view(frame, axes, full_x, full_y, 400, 300)[1 + arrival]
    assert full[1].size == SCANS
    expected = gaussian_smooth(full[1], tick_ms=tick_ms, sigma_ms=sigma_ms)

    for lo, hi in ((60, 120), (5, 40), (170, 200), (0, SCANS)):
        x_range, y_range = _ranges(axes, (float(scan_edges[lo]), float(scan_edges[hi])), bins)
        raw = render_view(frame, axes, x_range, y_range, 400, 300)[1 + arrival]
        assert raw[1].size == hi - lo
        got = smoothed_arrival(frame, axes, x_range, y_range, raw,
                               tick_ms=tick_ms, sigma_ms=sigma_ms)
        np.testing.assert_allclose(got, expected[lo:hi], rtol=1e-12, atol=1e-9)

    # And the zero-padded alternative is visibly not that, or the test above proves
    # nothing: smoothing the in-view values alone pulls the ends of the curve down.
    x_range, y_range = _ranges(axes, (float(scan_edges[60]), float(scan_edges[120])), bins)
    raw = render_view(frame, axes, x_range, y_range, 400, 300)[1 + arrival]
    naive = gaussian_smooth(raw[1], tick_ms=tick_ms, sigma_ms=sigma_ms)
    assert not np.allclose(naive, expected[60:120], rtol=1e-6)


def _request(frame, axes, tick_ms, sigma_ms, scans=(60, 120)):
    x_range, y_range = _ranges(
        axes, (float(axes.scan_edges[scans[0]]), float(axes.scan_edges[scans[1]])),
        (float(axes.bin_edges[0]), float(axes.bin_edges[-1])),
    )
    return RenderRequest(frame=frame, axes=axes, x_range=x_range, y_range=y_range,
                         width=400, height=300, sigma_ms=sigma_ms, scan_tick_ms=tick_ms)


@pytest.mark.parametrize("swapped", [False, True], ids=["default", "swapped"])
def test_smoothing_leaves_the_raw_profiles_of_a_render_as_they_were(long_file, swapped):
    frame, axes, tick_ms = _frame(long_file, swapped)
    off = RenderWorker._render(_request(frame, axes, tick_ms, None))
    on = RenderWorker._render(_request(frame, axes, tick_ms, 0.6))

    assert off.arrival_smoothed is None
    for a, b in ((off.x_profile, on.x_profile), (off.y_profile, on.y_profile)):
        assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    arrival = on.x_profile if swapped else on.y_profile
    assert on.arrival_smoothed.shape == arrival[1].shape
    # The area in view is the raw curve's to within what the kernel moved across the
    # zoom edges, so the smoothed curve is the same signal and not a different one.
    assert on.arrival_smoothed.sum() == pytest.approx(arrival[1].sum(), rel=0.2)


def test_a_frame_with_no_scan_time_is_not_smoothed(long_file):
    frame, axes, _ = _frame(long_file, False)
    render = RenderWorker._render(_request(frame, axes, 0.0, 0.6))
    assert render.arrival_smoothed is None


# --- the window --------------------------------------------------------------------


def _window(qtbot, spec, **settings):
    window = MainWindow(ViewerSettings(**settings))
    qtbot.addWidget(window)
    window.resize(1000, 700)
    window.show()
    with qtbot.waitSignal(window.frame_shown, timeout=20000):
        window.open_file(spec.path)
    return window


def _rendered(qtbot, window, test) -> object:
    qtbot.waitUntil(lambda: window._last_render is not None and test(window._last_render),
                    timeout=5000)
    return window._last_render


def test_on_by_default_the_arrival_plot_is_smoothed_over_a_faint_raw_curve(qtbot, long_file):
    window = _window(qtbot, long_file)
    assert window.settings.smooth_arrival_time
    assert window._smooth_arrival_action.isChecked()
    render = _rendered(qtbot, window, lambda r: r.arrival_smoothed is not None)

    plots = window.side_plots
    assert plots.faint == "y"
    x_smooth, y_smooth = plots.smoothed_curves
    assert x_smooth.xData is None or x_smooth.xData.size == 0
    assert np.array_equal(y_smooth.xData, render.arrival_smoothed)
    assert np.array_equal(plots.curves[1].xData, render.y_profile[1])  # raw still drawn
    # The status-bar peak reads the raw projection, whatever the plot shows.
    position, height = peak_of(*render.y_profile)
    assert f"({height:,.0f})" in window._peaks.text()

    # Turned off: the raw curve alone, at full strength, without rereading the frame.
    frame = window._current_frame
    window._smooth_arrival_action.setChecked(False)
    _rendered(qtbot, window, lambda r: r.arrival_smoothed is None)
    assert plots.faint is None
    assert all(c.xData is None or c.xData.size == 0 for c in plots.smoothed_curves)
    assert window._current_frame is frame
    assert not window.settings.smooth_arrival_time


def test_the_smoothed_curve_follows_swap_axes(qtbot, long_file):
    window = _window(qtbot, long_file, swap_axes=True)
    render = _rendered(qtbot, window, lambda r: r.arrival_smoothed is not None)
    assert window.side_plots.faint == "x"
    x_smooth, _ = window.side_plots.smoothed_curves
    assert np.array_equal(x_smooth.yData, render.arrival_smoothed)
    assert render.arrival_smoothed.shape == render.x_profile[1].shape


def test_a_new_width_redraws_with_it(qtbot, long_file):
    window = _window(qtbot, long_file)
    _rendered(qtbot, window, lambda r: r.arrival_smoothed is not None)
    frame, axes = window._current_frame, window._current_axes
    tick_ms = window._frame_params.average_tof_length_ns * 1e-6

    def at(sigma_ms):
        def test(render):
            expected = smoothed_arrival(
                frame, axes, render.result.x_range, render.result.y_range,
                render.y_profile, tick_ms=tick_ms, sigma_ms=sigma_ms,
            )
            return render.arrival_smoothed is not None and np.allclose(
                render.arrival_smoothed, expected)
        return test

    assert at(0.6)(window._last_render)
    assert not at(2.0)(window._last_render)
    window.set_arrival_sigma(2.0)
    assert window.settings.arrival_sigma_ms == 2.0
    _rendered(qtbot, window, at(2.0))
    # A width the smoother would refuse is not stored.
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        window.set_arrival_sigma(bad)
        assert window.settings.arrival_sigma_ms == 2.0


def test_the_width_dialog_accepts_any_positive_finite_number(qtbot):
    assert parse_width("0.6") == 0.6
    assert parse_width(" 1e-4 ") == 1e-4
    assert parse_width("250") == 250.0
    for bad in ("", "0", "-0.6", "nan", "inf", "wide"):
        assert parse_width(bad) is None

    dialog = SmoothingWidthDialog(None, 0.6)
    qtbot.addWidget(dialog)
    assert dialog.value() == 0.6
    dialog._edit.setText("-1")
    assert dialog.value() is None
    assert not dialog._buttons.button(dialog._buttons.StandardButton.Ok).isEnabled()
    dialog._edit.setText("1.5")
    assert dialog._buttons.button(dialog._buttons.StandardButton.Ok).isEnabled()
    assert dialog._edit.toolTip()


def test_both_controls_are_in_data_settings_and_explain_themselves(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    data_menu = next(a.menu() for a in window.menuBar().actions()
                     if a.text() == "&Data settings")
    for action in (window._smooth_arrival_action, window._arrival_sigma_action):
        assert action in data_menu.actions()
        assert action.toolTip()


def test_the_faint_curve_is_themed_in_both_palettes(qtbot, long_file):
    window = _window(qtbot, long_file)
    _rendered(qtbot, window, lambda r: r.arrival_smoothed is not None)
    try:
        for name in ("light", "dark"):
            palette = theme.apply(window, name)
            assert theme.themed(window) == []
            raw = window.side_plots.curves[1]
            assert raw.opts["pen"].color().name() == palette.curve_faint
    finally:
        theme.apply(window, "dark")


def test_an_export_draws_the_raw_curve_alone_and_puts_the_screen_back(qtbot, long_file,
                                                                     tmp_path):
    window = _window(qtbot, long_file)
    _rendered(qtbot, window, lambda r: r.arrival_smoothed is not None)
    plots = window.side_plots
    with plots.raw_only():
        assert plots.faint is None
        assert not any(c.isVisible() for c in plots.smoothed_curves)
        assert plots.curves[1].opts["pen"].color().name() == theme.active().curve
    assert plots.faint == "y"
    assert all(c.isVisible() for c in plots.smoothed_curves)

    export_display(window.heatmap, plots, str(tmp_path / "figure.png"), "png", 96)
    assert plots.faint == "y"
    assert all(c.isVisible() for c in plots.smoothed_curves)
