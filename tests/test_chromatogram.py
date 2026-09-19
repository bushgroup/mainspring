"""Task 31: the chromatogram dock, its two sources, its band, and its time axis.

Four groups. The pure rule first -- whether a file's start times mean anything, which is
the one piece of this that needs no Qt at all and is the trap the whole time axis rests
on. Then the panel on its own, driven directly, where the two sources, the band and the
two orientations are. Then the window, where the sources are asked for and the walk is
started and cancelled. Then what the panel must *not* touch: the export rectangle, the
tooltip walk and the two palettes.

Everything runs on the synthetic fixture, offscreen. The fixture can now space its
frames out in time (`write_synthetic_uimf(start_time_step_minutes=...)`), which is what
makes both halves of the time-axis rule testable: a file that means its start times and
a file that carries the writer's default on every frame.
"""

from __future__ import annotations

import numpy as np
import pytest

from mainspring.viewer import theme
from mainspring.viewer.chromatogram import (
    FRAME_AXIS,
    TIME_AXIS,
    ChromatogramPanel,
    elapsed_minutes,
)
from mainspring.viewer.main_window import MainWindow
from mainspring.viewer.settings import ViewerSettings

from synthetic import write_synthetic_uimf

STEP = 0.01
"""Minutes between the fixture's frames where a real time axis is wanted. 0.6 s, which
is about what a clockwork frame takes at the SLIMPHONY pusher rate."""


# --- the rule: does this file mean its start times? ------------------------------------


def test_evenly_spaced_start_times_are_an_axis():
    frames = (1, 2, 3, 4)
    times = {frame: (frame - 1) * 0.5 for frame in frames}
    assert elapsed_minutes(frames, times) == (0.0, 0.5, 1.0, 1.5)


def test_the_writers_default_on_every_frame_is_not_an_axis():
    """The trap. `FrameSpec.start_time_minutes` is 0.0 unless a client sets it and the
    writer stores it regardless, so "the parameter is there" is worth nothing."""
    frames = tuple(range(1, 5001))
    assert elapsed_minutes(frames, {frame: 0.0 for frame in frames}) is None


def test_start_times_that_go_backwards_are_not_an_axis():
    """Not a clock, whatever it is. Drawing against it would put the run out of order."""
    assert elapsed_minutes((1, 2, 3), {1: 0.0, 2: 1.0, 3: 0.5}) is None


def test_a_missing_start_time_is_not_an_axis():
    """A hole is worse than no axis: the frames either side would be drawn at the wrong
    place and nothing would say so."""
    assert elapsed_minutes((1, 2, 3), {1: 0.0, 3: 1.0}) is None


def test_one_frame_has_no_time_axis():
    """The summed companion, which holds one frame per method frame. A single point has
    no spread to draw against, and its start time is the run's rather than its own."""
    assert elapsed_minutes((1,), {1: 0.0007}) is None


def test_the_values_are_the_files_own_and_are_not_shifted_to_zero():
    """A PNNL LC excerpt starts at 465 minutes into its run, and that is what the file
    recorded. Subtracting the first frame would answer a question nobody asked."""
    assert elapsed_minutes((1, 2), {1: 465.72, 2: 465.77}) == (465.72, 465.77)


# --- the panel on its own --------------------------------------------------------------


@pytest.fixture
def panel(qtbot):
    """The panel docked in a bare main window, which is the only way it is ever used.

    Not a parentless `ChromatogramPanel()`: a `QDockWidget` with no main window around
    it answers `isFloating()` with True, and a floated panel deliberately draws
    horizontally whatever edge it was told about -- so the orientation tests would be
    testing the floating branch under another name.
    """
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QMainWindow

    host = QMainWindow()
    qtbot.addWidget(host)
    made = ChromatogramPanel(host)
    host.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, made)
    # Yielded rather than returned: the only Python reference to `host` is this frame,
    # and a collected main window takes its dock's C++ object with it.
    yield made


def _fill(panel, frames=8, step=STEP, totals=None):
    """Give the panel a series, the way `_on_chromatogram` does."""
    numbers = tuple(range(1, frames + 1))
    values = totals or {n: 100.0 * n for n in numbers}
    times = {n: (n - 1) * step for n in numbers}
    panel.set_series(values, times)
    return numbers, values


def test_the_panel_draws_one_point_per_frame(panel):
    numbers, values = _fill(panel)
    frames, totals = panel.series
    assert frames == numbers
    assert totals == tuple(values[n] for n in numbers)
    assert panel._axis_name() == TIME_AXIS


def test_a_file_without_usable_start_times_falls_back_to_the_frame_axis(panel):
    _fill(panel, step=0.0)
    assert panel._axis_name() == FRAME_AXIS
    assert "records no frame start times" in panel._readout.text()


def test_raw_units_switches_the_axis_and_has_no_visible_effect_without_times(panel):
    _fill(panel, step=0.0)
    before = panel._positions()
    panel.set_raw_units(True)
    assert panel._axis_name() == FRAME_AXIS
    assert panel._positions() == before


def test_raw_units_switches_a_timed_file_to_frame_numbers(panel):
    numbers, _ = _fill(panel)
    assert panel._positions() == tuple((n - 1) * STEP for n in numbers)
    panel.set_raw_units(True)
    assert panel._axis_name() == FRAME_AXIS
    assert panel._positions() == tuple(float(n) for n in numbers)


def test_a_band_reports_the_frames_it_covers(panel):
    _fill(panel)
    positions = panel._positions()
    panel._set_band(positions[2], positions[5])
    assert panel.selected_frames() == (3, 4, 5, 6)
    assert "frames 3 to 6" in panel._readout.text()
    assert panel._sum_button.isEnabled()


def test_a_band_changes_nothing_until_sum_selection_is_pressed(panel, qtbot):
    """The whole point of the band being a marker: drawing one must not start a sum."""
    _fill(panel)
    positions = panel._positions()
    asked: list = []
    panel.sum_requested.connect(asked.append)
    panel._set_band(positions[1], positions[3])
    assert asked == []
    panel._on_sum_clicked()
    assert asked == [(2, 3, 4)]


def test_the_band_survives_a_units_toggle_with_the_same_frames_under_it(panel):
    """Re-expressed, not cleared and not left in the old numbers.

    The same rule `_rebuild_axes` follows for the heatmap's view: a toggle says how an
    axis is labelled, and a selection that silently emptied itself would cost the user
    the span they were about to sum.
    """
    _fill(panel)
    positions = panel._positions()
    panel._set_band(positions[2], positions[5])
    covered = panel.selected_frames()
    panel.set_raw_units(True)
    assert panel.selected_frames() == covered
    panel.set_raw_units(False)
    assert panel.selected_frames() == covered


def test_a_double_click_clears_the_band(panel):
    _fill(panel)
    positions = panel._positions()
    panel._set_band(positions[1], positions[4])
    panel.clear_band()
    assert panel.selected_frames() == ()
    assert not panel._sum_button.isEnabled()


def test_a_click_picks_the_nearest_frame(panel):
    """A trace of one point per frame is never clicked exactly on a point."""
    _fill(panel)
    picked: list = []
    panel.frame_picked.connect(picked.append)
    positions = panel._positions()
    panel._pick(positions[4] + 0.3 * STEP)
    panel._pick(positions[0] - 100.0)
    assert picked == [5, 1]


def test_the_two_sources_are_held_apart(panel):
    """A half-finished walk must not be drawn over the file's own totals.

    The one way this panel could lie: a trace that was mostly stored totals with a few
    view-restricted points in it would look complete and say nothing about it.
    """
    numbers, stored = _fill(panel)
    panel.set_restricted({1: 1.0, 2: 2.0})
    assert panel.restricted
    assert panel.series == ((1, 2), (1.0, 2.0))
    assert "summed over the heatmap's view" in panel._readout.text()
    panel.clear_restricted()
    assert not panel.restricted
    assert panel.series == (numbers, tuple(stored[n] for n in numbers))


def test_a_live_poll_appends_rather_than_replacing(panel):
    """`replace=False` is the poll's form: one or two frames, merged into what is there."""
    _fill(panel, frames=3)
    panel.set_series({4: 400.0}, {4: 3 * STEP}, replace=False)
    frames, totals = panel.series
    assert frames == (1, 2, 3, 4)
    assert totals[-1] == 400.0
    assert panel._axis_name() == TIME_AXIS


def test_the_axis_reaches_the_end_of_the_method_frame(panel):
    """A run of ten repetitions looks like ten from its first frame, not like a full
    plot that rescales every second."""
    _fill(panel, frames=3)
    panel.set_span(10)
    projected = panel._span_position()
    assert projected == pytest.approx(9 * STEP)
    panel.set_raw_units(True)
    assert panel._span_position() == pytest.approx(10.0)
    panel.set_span(None)
    assert panel._span_position() is None


def test_the_panel_draws_the_other_way_round_when_it_is_docked_on_its_side(panel):
    """Time down the plot instead of across it, which is one number and not a code path."""
    from PySide6.QtCore import Qt

    _fill(panel)
    horizontal_x, _ = panel._curve.getData()
    panel._on_location_changed(Qt.DockWidgetArea.LeftDockWidgetArea)
    assert panel.vertical
    vertical_x, vertical_y = panel._curve.getData()
    assert np.array_equal(vertical_y, horizontal_x)
    assert panel._view_box._axis == 1
    assert panel._band.orientation == "horizontal"


def test_moving_the_dock_keeps_the_highlight(panel):
    """The band is rebuilt when the orientation changes, because pyqtgraph cannot turn
    one -- so the span it covered has to be carried over by hand."""
    from PySide6.QtCore import Qt

    _fill(panel)
    positions = panel._positions()
    panel._set_band(positions[1], positions[4])
    covered = panel.selected_frames()
    panel._on_location_changed(Qt.DockWidgetArea.LeftDockWidgetArea)
    assert panel.selected_frames() == covered
    assert panel._band.isVisible()


def test_nothing_inside_the_panel_dictates_its_size(panel, qtbot):
    """The info panel's invariant, widened: this dock can sit along either edge, so a
    long readout must not fix a width there or a height here."""
    from PySide6.QtWidgets import QSizePolicy

    policy = panel._readout.sizePolicy()
    assert policy.horizontalPolicy() == QSizePolicy.Policy.Ignored
    assert policy.verticalPolicy() == QSizePolicy.Policy.Ignored
    before = panel._container.minimumSizeHint().width()
    _fill(panel, frames=400)
    panel._set_band(0.0, 10.0)
    assert panel._container.minimumSizeHint().width() == before


# --- through the window ----------------------------------------------------------------


@pytest.fixture
def timed_uimf(tmp_path):
    """A grouped file whose frames really are spaced out in time: six frames, two method
    frames of three, the shape of a clockwork raw acquisition in miniature."""
    return write_synthetic_uimf(
        tmp_path / "timed.uimf", frames=6, scans=16, bins=4096,
        grouped=True, repetitions=3, start_time_step_minutes=STEP,
    )


def _open(qtbot, path, *, chromatogram=True):
    window = MainWindow(ViewerSettings(show_chromatogram=chromatogram))
    qtbot.addWidget(window)
    window.show()
    with qtbot.waitSignal(window.frame_shown, timeout=5000):
        window.open_file(path)
    return window


def test_opening_a_file_fills_the_panel_from_the_files_own_totals(qtbot, timed_uimf):
    window = _open(qtbot, timed_uimf.path)
    qtbot.waitUntil(lambda: bool(window.chromatogram.series[0]), timeout=5000)
    frames, totals = window.chromatogram.series
    assert frames == timed_uimf.frames
    for frame, total in zip(frames, totals):
        assert total == pytest.approx(timed_uimf.tic(frame))
    assert window.chromatogram._axis_name() == TIME_AXIS


def test_a_hidden_panel_costs_nothing(qtbot, timed_uimf):
    """Nothing is read for a panel nobody has opened, which is why an upgrade costs an
    existing user neither width nor two whole-file queries."""
    window = _open(qtbot, timed_uimf.path, chromatogram=False)
    qtbot.wait(200)
    assert window.chromatogram.series == ((), ())
    with qtbot.waitSignal(window._worker.chromatogram, timeout=5000):
        window._chromatogram_action.trigger()
    assert window.chromatogram.series[0] == timed_uimf.frames


def test_the_panel_is_hidden_by_default_and_remembered(qtbot, timed_uimf):
    window = MainWindow(ViewerSettings())
    qtbot.addWidget(window)
    assert window.chromatogram.isHidden()
    window._chromatogram_action.trigger()
    assert not window.chromatogram.isHidden()
    assert window.settings.show_chromatogram is True
    window.close()
    assert window.settings.show_chromatogram is True


def test_clicking_a_frame_in_the_panel_shows_it(qtbot, timed_uimf):
    window = _open(qtbot, timed_uimf.path)
    qtbot.waitUntil(lambda: bool(window.chromatogram.series[0]), timeout=5000)
    with qtbot.waitSignal(window.frame_shown, timeout=5000):
        window.chromatogram.frame_picked.emit(4)
    assert window._current_frame_number == 4


def test_summing_the_highlight_goes_through_the_ordinary_sum_path(qtbot, timed_uimf):
    """The same generator, cache and status line as `Sum all`, and the total is the
    highlighted frames' own stored `TIC` columns."""
    window = _open(qtbot, timed_uimf.path)
    qtbot.waitUntil(lambda: bool(window.chromatogram.series[0]), timeout=5000)
    positions = window.chromatogram._positions()
    window.chromatogram._set_band(positions[1], positions[3])
    covered = window.chromatogram.selected_frames()
    assert covered == (2, 3, 4)

    with qtbot.waitSignal(window.frame_shown, timeout=10000) as blocker:
        window.chromatogram._on_sum_clicked()
    result = blocker.args[0]
    assert result.tic_in_view == pytest.approx(sum(timed_uimf.tic(f) for f in covered))
    assert "highlighted span" in window.status_text()
    # And the highlight stays: moving it by a frame and asking again is one gesture.
    assert window.chromatogram.selected_frames() == covered


def test_restrict_to_view_totals_the_window_and_agrees_at_full_range(qtbot, timed_uimf):
    """At full range the walk's answer is the stored total, frame for frame -- the two
    sources meeting at the one view where they must."""
    window = _open(qtbot, timed_uimf.path)
    qtbot.waitUntil(lambda: bool(window.chromatogram.series[0]), timeout=5000)
    stored = dict(zip(*window.chromatogram.series))

    window.chromatogram._restrict_box.setChecked(True)
    qtbot.waitUntil(
        lambda: window.chromatogram.restricted
        and len(window.chromatogram.series[0]) == len(timed_uimf.frames),
        timeout=20000,
    )
    walked = dict(zip(*window.chromatogram.series))
    for frame in timed_uimf.frames:
        assert walked[frame] == pytest.approx(stored[frame])


def test_a_zoomed_view_totals_less_than_the_whole_frame(qtbot, timed_uimf):
    window = _open(qtbot, timed_uimf.path)
    qtbot.waitUntil(lambda: bool(window.chromatogram.series[0]), timeout=5000)
    stored = dict(zip(*window.chromatogram.series))

    (x0, x1), (y0, y1) = window.heatmap.view_range()
    with qtbot.waitSignal(window.frame_shown, timeout=5000):
        window.heatmap.view_box.setRange(
            xRange=(x0 + 0.4 * (x1 - x0), x0 + 0.6 * (x1 - x0)),
            yRange=(y0, y1), padding=0.0,
        )
    window.chromatogram._restrict_box.setChecked(True)
    qtbot.waitUntil(
        lambda: window.chromatogram.restricted
        and len(window.chromatogram.series[0]) == len(timed_uimf.frames),
        timeout=20000,
    )
    walked = dict(zip(*window.chromatogram.series))
    assert all(walked[f] < stored[f] for f in timed_uimf.frames)
    assert sum(walked.values()) > 0.0


def test_unticking_restrict_goes_back_without_reading_anything(qtbot, timed_uimf):
    """The stored series is still held, so this is a redraw and not two more queries."""
    window = _open(qtbot, timed_uimf.path)
    qtbot.waitUntil(lambda: bool(window.chromatogram.series[0]), timeout=5000)
    stored = window.chromatogram.series

    window.chromatogram._restrict_box.setChecked(True)
    qtbot.waitUntil(lambda: window.chromatogram.restricted, timeout=20000)
    asked: list = []
    window._worker.chromatogram.connect(lambda *a: asked.append(a))
    window.chromatogram._restrict_box.setChecked(False)
    qtbot.wait(200)
    assert not window.chromatogram.restricted
    assert window.chromatogram.series == stored
    assert asked == []


def test_the_axis_spans_the_method_frames_intended_repetitions(qtbot, timed_uimf):
    """Two method frames of three repetitions: the first frame's span ends on frame 3."""
    window = _open(qtbot, timed_uimf.path)
    qtbot.waitUntil(lambda: bool(window.chromatogram.series[0]), timeout=5000)
    assert window.chromatogram._span_frame == 3
    with qtbot.waitSignal(window.frame_shown, timeout=5000):
        window.show_frame(5)
    assert window.chromatogram._span_frame == 6


def test_an_ungrouped_file_spans_only_the_frames_it_has(qtbot, synthetic_uimf):
    """Nothing in the file says how long the method was, and inventing a length would
    be guessing (lab record, task 31)."""
    window = _open(qtbot, synthetic_uimf.path)
    qtbot.waitUntil(lambda: bool(window.chromatogram.series[0]), timeout=5000)
    assert window.chromatogram._span_frame is None


def test_opening_another_file_empties_the_panel_first(qtbot, timed_uimf, synthetic_uimf):
    window = _open(qtbot, timed_uimf.path)
    qtbot.waitUntil(lambda: bool(window.chromatogram.series[0]), timeout=5000)
    with qtbot.waitSignal(window.frame_shown, timeout=5000):
        window.open_file(synthetic_uimf.path)
    qtbot.waitUntil(
        lambda: window.chromatogram.series[0] == synthetic_uimf.frames, timeout=5000
    )


# --- what the panel must not touch -------------------------------------------------------


def test_the_export_never_contains_the_panel(qtbot, synthetic_uimf, tmp_path):
    """The panel is a dock with a scene of its own, which is half of why it is one: an
    exported figure is the heatmap's scene rendered again, so it cannot gain a
    chromatogram however the window is laid out.

    Two halves. The chromatogram's items are in a different `QGraphicsScene` and so are
    not reachable from the exported rectangle at all -- the structural guarantee, and
    the one that holds whatever the layout does. And the figure written with the panel
    on screen is still exactly that rectangle, so the panel has taken window space and
    contributed nothing to the picture.
    """
    from mainspring.viewer.export import BASE_DPI, content_rect, export_display, export_pixels

    window = _open(qtbot, synthetic_uimf.path, chromatogram=False)
    before = content_rect(window.heatmap, window.side_plots)

    window._chromatogram_action.trigger()
    qtbot.waitUntil(lambda: bool(window.chromatogram.series[0]), timeout=5000)
    assert window.chromatogram.plot.scene() is not window.heatmap.scene()
    panel_items = set(window.chromatogram.plot.scene().items())
    assert not panel_items & set(window.heatmap.scene().items())

    # And the figure a user takes away is still exactly that rectangle: the export
    # renders the heatmap's scene, so a panel on screen adds nothing to it.
    rect = content_rect(window.heatmap, window.side_plots)
    assert rect.size() != before.size(), "the dock should have taken some of the window"
    figure = tmp_path / "with-panel.png"
    size = export_display(window.heatmap, window.side_plots, str(figure), "png", BASE_DPI)
    assert size == export_pixels(rect, BASE_DPI)


def test_the_panel_is_inside_both_canvas_walks(qtbot, synthetic_uimf):
    """The check that would otherwise have been skipped silently: a plot in a dock of
    its own is not in the heatmap's scene, so a walk written around one name misses a
    whole widget rather than one pen on it (lab record, task 31)."""
    window = _open(qtbot, synthetic_uimf.path)
    assert window.chromatogram.plot in window.plot_canvases()


def test_the_bands_colors_come_from_the_palette(qtbot, synthetic_uimf):
    """`theme.themed` cannot see a `LinearRegionItem` -- the type is outside the walk on
    purpose, because the color bar's own region and handles are color-map furniture. So
    the band is checked here instead, under both palettes.
    """
    from mainspring.viewer.chromatogram import BAND_FILL_ALPHA, BAND_LINE_ALPHA

    window = _open(qtbot, synthetic_uimf.path)
    band = window.chromatogram._band
    for name in ("dark", "light"):
        palette = theme.apply(window, name)
        band = window.chromatogram._band
        wanted = theme.PALETTES[name].role("curve")
        assert band.brush.color().rgb() == wanted.rgb()
        assert band.brush.color().alpha() == BAND_FILL_ALPHA
        for line in band.lines:
            assert line.pen.color().rgb() == wanted.rgb()
            assert line.pen.color().alpha() == BAND_LINE_ALPHA
    assert palette is theme.LIGHT


def test_the_trace_follows_the_window_size_like_every_other_line(qtbot, synthetic_uimf):
    """`CURVE_WIDTH` at `REFERENCE_VIEWPORT`, scaled and clamped by the same three
    numbers the tick pens and the projections use."""
    from mainspring.viewer.heatmap import LINE_WIDTH_LIMITS

    window = _open(qtbot, synthetic_uimf.path)
    panel = window.chromatogram
    panel.fit_lines(3000, 3000)
    assert panel.curve_width == LINE_WIDTH_LIMITS[1]
    panel.fit_lines(100, 100)
    assert panel.curve_width == LINE_WIDTH_LIMITS[0]
