"""Task 06: toolbar toggles, the info panel, frame navigation and sum-all.

Four groups, roughly in the order a session would meet them: the per-push arithmetic
and the info panel widget on their own; the held colour levels and the colour-scale
transform on a bare `HeatmapView`; then the toolbar and frame navigation through a full
`MainWindow`, offscreen, on the synthetic fixture -- same style as
`test_viewer_interaction.py`, which this complements rather than repeats (that file owns
the gestures and the render path; this one owns what task 06 added on top of them).
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from mainspring.uimf import DisplayAxes, GlobalParams, RasterResult
from mainspring.viewer.heatmap import HeatmapView
from mainspring.viewer.info_panel import InfoPanel, per_push
from mainspring.viewer.main_window import MainWindow
from mainspring.viewer.workers import LoadWorker

from synthetic import write_synthetic_uimf


def _assert_same_view(actual, expected) -> None:
    """`(x_range, y_range) == (x_range, y_range)`, approximately -- `pytest.approx`
    does not look inside a tuple of tuples, so the two axes are compared separately."""
    (actual_x, actual_y), (expected_x, expected_y) = actual, expected
    assert actual_x == pytest.approx(expected_x)
    assert actual_y == pytest.approx(expected_y)


# --- per_push and the info panel widget ------------------------------------------------

def test_per_push_divides_by_accumulations_and_reports_full_scale():
    counts, percent = per_push(max_intensity=25.5, accumulations=100, detector_bits=8)
    assert counts == pytest.approx(0.255)
    assert percent == pytest.approx(100.0 * 0.255 / 255.0)


def test_per_push_never_divides_by_zero_accumulations_or_bits():
    counts, _ = per_push(max_intensity=10.0, accumulations=0, detector_bits=8)
    assert counts == pytest.approx(10.0)  # accumulations clamped to at least 1
    _, percent = per_push(max_intensity=10.0, accumulations=1, detector_bits=0)
    assert percent == pytest.approx(100.0 * 10.0 / 1.0)  # bits clamped to at least 1


def test_info_panel_shows_every_key_a_file_happens_to_carry(qtbot):
    panel = InfoPanel()
    qtbot.addWidget(panel)
    global_params = GlobalParams(extra={"InstrumentName": "SLIM3", "DatasetType": "HMS"})
    from mainspring.uimf import FrameParams

    frame_params = FrameParams(accumulations=100, extra={"FrameType": "0", "Scans": "16"})

    panel.set_file(global_params, frame_params)

    shown = {
        panel._global_root.child(i).text(0): panel._global_root.child(i).text(1)
        for i in range(panel._global_root.childCount())
    }
    assert shown == {"InstrumentName": "SLIM3", "DatasetType": "HMS"}


def test_info_panel_live_readout_states_accumulations_and_bits_with_the_number(qtbot):
    panel = InfoPanel()
    qtbot.addWidget(panel)
    axes = DisplayAxes(x_edges=np.array([0.0, 1.0]), y_edges=np.array([0.0, 1.0]),
                        x_label="X", y_label="Y")
    result = RasterResult(
        image=np.array([[10.0]]), x_range=(0.0, 1.0), y_range=(0.0, 1.0), axes=axes,
        aggregate="sum", points_in_view=5, tic_in_view=50.0, max_intensity=25.5,
    )

    panel.set_view(result, accumulations=100, detector_bits=8)

    text = panel._per_push_label.text()
    assert "100" in text and "8-bit" in text  # never a per-push number on its own


# --- held colour levels and the colour-scale transform ---------------------------------

def _fake_result(image: "np.ndarray") -> RasterResult:
    axes = DisplayAxes(x_edges=np.array([0.0, 1.0]), y_edges=np.array([0.0, 1.0]),
                        x_label="X", y_label="Y")
    return RasterResult(
        image=image, x_range=(0.0, 1.0), y_range=(0.0, 1.0), axes=axes, aggregate="sum",
        points_in_view=int(image.size), tic_in_view=float(image.sum()),
        max_intensity=float(image.max()),
    )


def test_colour_scale_compresses_the_levels_but_never_the_raw_result(qtbot):
    view = HeatmapView()
    qtbot.addWidget(view)
    result = _fake_result(np.array([[0.0, 3.0], [8.0, 15.0]]))

    view.set_image(result, colour_scale="linear")
    assert view.levels()[1] == pytest.approx(15.0)

    view.set_image(result, colour_scale="log")
    assert view.levels()[1] == pytest.approx(np.log1p(15.0))

    view.set_image(result, colour_scale="sqrt")
    assert view.levels()[1] == pytest.approx(np.sqrt(15.0))

    # The transform never touches the RasterResult the readouts quote.
    assert result.image.max() == pytest.approx(15.0)
    assert result.max_intensity == pytest.approx(15.0)


def test_set_levels_holds_the_colour_bar_across_a_new_image(qtbot):
    view = HeatmapView()
    qtbot.addWidget(view)
    view.set_image(_fake_result(np.array([[0.0, 10.0]])))
    assert view.levels() == pytest.approx((0.0, 10.0))

    view.set_levels(*view.levels())
    view.set_image(_fake_result(np.array([[0.0, 100.0]])))
    assert view.levels() == pytest.approx((0.0, 10.0))  # held, not rescaled to 100

    view.release_levels()
    view.set_image(_fake_result(np.array([[0.0, 100.0]])))
    assert view.levels() == pytest.approx((0.0, 100.0))  # auto again


# --- LoadWorker.sum_all: progress, and a genuine cancel ---------------------------------

def test_sum_all_reports_progress_and_can_be_cancelled(qtbot, tmp_path, monkeypatch):
    """A cancel mid-sum must emit nothing, not a partial total.

    `mainspring.uimf.frame.sum_frames`'s own cancellation semantics are `test_frame.py`'s
    to check; this is the plumbing above it -- the worker's progress ticks and its
    `summed` signal on a real cancel. `read_frame` is slowed down (test-only, via
    monkeypatch) so the test thread has a real window to call `cancel_sum` in rather than
    racing a decode of a few dozen tiny frames that finishes before the next event-loop
    tick.
    """
    import mainspring.uimf.reader as reader_module

    original_read_frame = reader_module.UimfFile.read_frame

    def _slow_read_frame(self, *args, **kwargs):
        time.sleep(0.02)
        return original_read_frame(self, *args, **kwargs)

    monkeypatch.setattr(reader_module.UimfFile, "read_frame", _slow_read_frame)

    spec = write_synthetic_uimf(tmp_path / "many.uimf", frames=20, scans=16, bins=4096)
    worker = LoadWorker()
    with qtbot.waitSignal(worker.opened, timeout=5000):
        worker.open(spec.path)

    progress: list[tuple[int, int]] = []
    results: list[tuple[object, object]] = []
    worker.summing_progress.connect(lambda done, total: progress.append((done, total)))
    worker.summed.connect(lambda frame, params: results.append((frame, params)))

    worker.sum_all(list(spec.frames))
    qtbot.waitUntil(lambda: len(progress) >= 1, timeout=5000)
    worker.cancel_sum()
    qtbot.waitUntil(lambda: len(results) == 1, timeout=5000)

    assert progress[0] == (1, len(spec.frames))
    assert len(progress) < len(spec.frames)  # stopped short of summing every frame
    assert results[0] == (None, None)  # cancelled: nothing summed, not a partial total

    worker.stop()
    worker.wait(2000)


# --- MainWindow: frame navigation, keep-ranges, sum-all, and the toolbar ---------------

@pytest.fixture
def opened_window(qtbot, synthetic_uimf):
    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    with qtbot.waitSignal(window.frame_shown, timeout=5000):
        window.open_file(synthetic_uimf.path)
    return window


def test_show_frame_switches_frames_and_keeps_the_view(opened_window, synthetic_uimf, qtbot):
    window = opened_window
    axes = window._current_axes
    (x0, x1), (y0, y1) = axes.full_range
    quarter = (x0 + 0.25 * (x1 - x0), x0 + 0.5 * (x1 - x0))

    with qtbot.waitSignal(window.frame_shown, timeout=5000):
        window.heatmap.view_box.setRange(xRange=quarter, yRange=(y0, y1), padding=0.0)
    zoomed_range = window.heatmap.view_range()

    with qtbot.waitSignal(window.frame_shown, timeout=5000) as blocker:
        window.show_frame(2)
    result = blocker.args[0]

    assert window._current_frame_number == 2
    assert window._current_frame.frame == 2
    _assert_same_view(window.heatmap.view_range(), zoomed_range)
    assert result.tic_in_view != pytest.approx(synthetic_uimf.tic(1))


def test_keep_ranges_preserves_the_view_across_a_reopen(qtbot, synthetic_uimf):
    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    window._keep_ranges_action.setChecked(True)
    assert window.settings.keep_ranges is True

    with qtbot.waitSignal(window.frame_shown, timeout=5000):
        window.open_file(synthetic_uimf.path)
    axes = window._current_axes
    (x0, x1), (y0, y1) = axes.full_range
    quarter = (x0 + 0.25 * (x1 - x0), x0 + 0.5 * (x1 - x0))
    with qtbot.waitSignal(window.frame_shown, timeout=5000):
        window.heatmap.view_box.setRange(xRange=quarter, yRange=(y0, y1), padding=0.0)
    zoomed = window.heatmap.view_range()

    with qtbot.waitSignal(window.frame_shown, timeout=5000):
        window.open_file(synthetic_uimf.path)

    _assert_same_view(window.heatmap.view_range(), zoomed)


def test_without_keep_ranges_reopening_resets_to_the_full_range(qtbot, synthetic_uimf):
    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    assert window.settings.keep_ranges is False

    with qtbot.waitSignal(window.frame_shown, timeout=5000):
        window.open_file(synthetic_uimf.path)
    axes = window._current_axes
    full_range = axes.full_range
    (x0, x1), (y0, y1) = full_range
    quarter = (x0 + 0.25 * (x1 - x0), x0 + 0.5 * (x1 - x0))
    with qtbot.waitSignal(window.frame_shown, timeout=5000):
        window.heatmap.view_box.setRange(xRange=quarter, yRange=(y0, y1), padding=0.0)

    with qtbot.waitSignal(window.frame_shown, timeout=5000):
        window.open_file(synthetic_uimf.path)

    _assert_same_view(window.heatmap.view_range(), full_range)


def test_sum_all_totals_every_frames_tic(opened_window, synthetic_uimf, qtbot):
    window = opened_window
    expected = synthetic_uimf.tic(1) + synthetic_uimf.tic(2)

    with qtbot.waitSignal(window.frame_shown, timeout=5000) as blocker:
        window.sum_frames()

    result = blocker.args[0]
    assert result.tic_in_view == pytest.approx(expected, rel=1e-9)
    assert window._sum_dialog is None  # closed once the result arrived


def test_sum_all_agrees_with_a_real_files_per_frame_tic(qtbot, real_uimf):
    """The same milestone arithmetic as the single-frame checks
    (`test_viewer_interaction.py`), made through `sum_frames` instead: on whatever real
    files this clone has (`conftest.real_uimf`), summing a few frames must equal the sum
    of their own stored `TIC` columns -- ground truth, not a re-derivation."""
    from mainspring.uimf import UimfFile

    file = UimfFile(real_uimf)
    numbers = file.frame_numbers()[:3]  # a few frames is enough, and keeps this fast
    expected = sum(float(file.scan_summary(n)[3].sum()) for n in numbers)

    window = MainWindow()
    qtbot.addWidget(window)
    with qtbot.waitSignal(window.frame_shown, timeout=30000):
        window.open_file(real_uimf)
    with qtbot.waitSignal(window.frame_shown, timeout=30000) as blocker:
        window.sum_frames(numbers)

    assert blocker.args[0].tic_in_view == pytest.approx(expected, rel=1e-9)


def test_aggregate_toggle_rerenders_with_the_new_aggregate(opened_window, qtbot):
    window = opened_window
    assert window.last_render.result.aggregate == "sum"

    with qtbot.waitSignal(window.frame_shown, timeout=5000):
        window._aggregate_box.setCurrentText("Max")

    assert window.settings.aggregate == "max"
    assert window.last_render.result.aggregate == "max"


def test_swap_axes_preserves_the_visible_region(opened_window, qtbot):
    window = opened_window
    axes = window._current_axes
    (x0, x1), (y0, y1) = axes.full_range
    quarter_x = (x0 + 0.25 * (x1 - x0), x0 + 0.5 * (x1 - x0))
    with qtbot.waitSignal(window.frame_shown, timeout=5000):
        window.heatmap.view_box.setRange(xRange=quarter_x, yRange=(y0, y1), padding=0.0)

    with qtbot.waitSignal(window.frame_shown, timeout=5000):
        window._swap_action.setChecked(True)

    assert window._current_axes.swapped is True
    new_x_range, new_y_range = window.heatmap.view_range()
    # x is now the old y (the full arrival-time range) and y the old x (the m/z quarter).
    assert new_x_range == pytest.approx((y0, y1), abs=1e-6)
    assert new_y_range == pytest.approx(quarter_x, abs=1e-6)


def test_frame_type_filter_narrows_the_active_frames(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    window._on_opened(GlobalParams(), [1, 2, 3], {1: 0, 2: 1, 3: 2})

    items = [window._type_filter.itemText(i) for i in range(window._type_filter.count())]
    assert items == ["All frames", "MS1", "MS2"]
    assert window._active_frame_numbers == [1, 2, 3]

    window._type_filter.setCurrentText("MS2")
    assert window._active_frame_numbers == [3]
    assert (window._frame_spin.minimum(), window._frame_spin.maximum()) == (3, 3)

    window._type_filter.setCurrentText("All frames")
    assert window._active_frame_numbers == [1, 2, 3]


# --- the info panel's size, and its toggle --------------------------------------------

def test_the_info_panel_keeps_its_width_whatever_it_is_told(qtbot):
    """A live readout must never resize the dock: the width is fixed, and the one long
    line wraps inside it. (The cursor readout, which changed on every mouse move, is
    the status bar's alone for the same reason.)"""
    from mainspring.viewer.info_panel import INFO_PANEL_WIDTH

    panel = InfoPanel()
    qtbot.addWidget(panel)
    panel.show()
    before = panel.widget().width()
    assert before == INFO_PANEL_WIDTH

    long_result = _fake_result(np.full((4, 4), 1234567890.0))
    panel.set_view(long_result, accumulations=1000000, detector_bits=32)
    panel.set_view(long_result, accumulations=1, detector_bits=1)

    assert panel.widget().width() == before
    assert panel.widget().minimumSizeHint().width() <= INFO_PANEL_WIDTH
    assert not hasattr(panel, "set_cursor")


def test_the_info_toggle_hides_the_dock_and_is_remembered(opened_window):
    window = opened_window
    assert window.settings.show_info_panel is True
    assert not window.info_panel.isHidden()

    # `trigger()` is what a click on the toolbar button or the menu entry does; Qt moves
    # the dock on the action's `triggered`, so a bare `setChecked` would not.
    window._info_action.trigger()
    assert window.info_panel.isHidden()
    assert window._info_action.isChecked() is False
    assert window.settings.show_info_panel is False

    window._info_action.trigger()
    assert not window.info_panel.isHidden()
    assert window.settings.show_info_panel is True


def test_closing_the_dock_itself_unticks_the_toggle(opened_window):
    window = opened_window
    window.info_panel.close()
    assert window.info_panel.isHidden()
    assert window._info_action.isChecked() is False
    assert window.settings.show_info_panel is False


def test_a_stored_hidden_info_panel_starts_hidden(qtbot):
    from mainspring.viewer.settings import ViewerSettings

    window = MainWindow(ViewerSettings(show_info_panel=False))
    qtbot.addWidget(window)
    window.show()
    assert window.info_panel.isHidden()
    assert window._info_action.isChecked() is False
    # The same action is reachable from the View menu and from the toolbar.
    view_menu = next(a.menu() for a in window.menuBar().actions() if a.text() == "&View")
    assert window._info_action in view_menu.actions()
    from PySide6.QtWidgets import QToolBar

    toolbar = window.findChild(QToolBar, "view_toolbar")
    assert window._info_action in toolbar.actions()
