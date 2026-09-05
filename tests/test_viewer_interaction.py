"""The gestures, the render path, and what the window says the image contains.

Three kinds of check live here, and they are separable on purpose.

`RenderMailbox` is plain Python with no Qt in it, so its newest-wins contract is tested
directly with threads rather than inferred from a window's behaviour.

The gestures are tested by handing `UimfViewBox` the events pyqtgraph would hand it.
Synthesising a real drag through the `QGraphicsScene` needs a press, a move past the
drag threshold and a release delivered to a widget with a live geometry, and offscreen
that is a test of Qt's event synthesis more than of ours; the stand-ins below carry
exactly the methods pyqtgraph's `ViewBox` reads, so the code under test -- including the
`super()` calls it deliberately delegates to -- runs unchanged. Positions come from
`mapFromView`, so a test says "zoom about this m/z" rather than "about this pixel".

The last group is the milestone: what the window shows must agree with the file's own
`TIC` and `BPI` columns at full range, and the side plots must agree with the image.
That is checked on the synthetic fixture and, where a clone has them, on real files.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pyqtgraph as pg
import pytest
from PySide6.QtCore import Qt

from mainspring.uimf import UimfFile
from mainspring.viewer.heatmap import HeatmapView, UimfViewBox, pixel_of
from mainspring.viewer.main_window import MainWindow
from mainspring.viewer.workers import RenderMailbox, RenderRequest


# --- the mailbox ----------------------------------------------------------------------

def _request(serial: int) -> RenderRequest:
    """A request that is only its serial: the mailbox never looks inside one."""
    return RenderRequest(
        frame=None, axes=None, x_range=(0.0, 1.0), y_range=(0.0, 1.0),
        width=10, height=10, serial=serial,
    )


def test_the_mailbox_keeps_only_the_newest_request():
    mailbox = RenderMailbox()
    for serial in range(5):
        mailbox.put(_request(serial))

    assert mailbox.take(0.1).serial == 4
    assert mailbox.take(0.05) is None  # and the four it overwrote are gone, not queued


def test_take_blocks_until_a_request_arrives():
    mailbox = RenderMailbox()
    taken: list[object] = []

    reader = threading.Thread(target=lambda: taken.append(mailbox.take()))
    reader.start()
    time.sleep(0.05)
    assert taken == []  # still waiting: an empty mailbox does not spin

    mailbox.put(_request(7))
    reader.join(timeout=2.0)
    assert [r.serial for r in taken] == [7]


def test_closing_wakes_a_waiting_take_and_drops_later_puts():
    mailbox = RenderMailbox()
    taken: list[object] = ["not yet"]

    reader = threading.Thread(target=lambda: taken.__setitem__(0, mailbox.take()))
    reader.start()
    time.sleep(0.05)
    mailbox.close()
    reader.join(timeout=2.0)

    assert taken == [None]
    assert mailbox.closed
    mailbox.put(_request(1))
    assert mailbox.take(0.05) is None


# --- the gestures ---------------------------------------------------------------------

class _Wheel:
    """The three methods `ViewBox.wheelEvent` reads, and nothing else."""

    def __init__(self, pos, delta: int, modifiers=Qt.KeyboardModifier.NoModifier):
        self._pos, self._delta, self._modifiers = pos, delta, modifiers
        self.accepted = False

    def pos(self):
        return self._pos

    def delta(self):
        return self._delta

    def modifiers(self):
        return self._modifiers

    def accept(self):
        self.accepted = True


class _Drag:
    """A drag in `ViewBox`-local pixels: one step of it, marked start, middle or finish."""

    def __init__(self, button, down, last, pos,
                 finish: bool = False, modifiers=Qt.KeyboardModifier.NoModifier):
        # `Point` and not `QPointF`: pyqtgraph's pan arithmetic multiplies the drag
        # delta by a numpy mask, which its own `Point` supports and a `QPointF` does not.
        self._button = button
        self._down, self._last, self._pos = pg.Point(down), pg.Point(last), pg.Point(pos)
        self._finish, self._modifiers = finish, modifiers
        self.accepted = False

    def button(self):
        return self._button

    def buttonDownPos(self, btn=None):
        return self._down

    def lastPos(self):
        return self._last

    def pos(self):
        return self._pos

    def isStart(self):
        return self._last == self._down

    def isFinish(self):
        return self._finish

    def modifiers(self):
        return self._modifiers

    def accept(self):
        self.accepted = True

    def ignore(self):
        self.accepted = False


@pytest.fixture
def view(qtbot):
    """A shown `HeatmapView` over a unit square, with a live geometry to map through."""
    widget = HeatmapView()
    qtbot.addWidget(widget)
    widget.resize(600, 400)
    widget.show()
    qtbot.waitExposed(widget)
    box = widget.view_box
    box._full_range = ((0.0, 100.0), (0.0, 50.0))
    box.setLimits(xMin=0.0, xMax=100.0, yMin=0.0, yMax=50.0,
                  maxXRange=100.0, maxYRange=50.0, minXRange=0.1, minYRange=0.05)
    box.reset_range()
    return widget


def _spans(box: UimfViewBox) -> tuple[float, float]:
    (x0, x1), (y0, y1) = box.viewRange()
    return x1 - x0, y1 - y0


def test_the_wheel_zooms_about_the_cursor_and_not_the_centre(view):
    box = view.view_box
    cursor = pg.Point(20.0, 10.0)  # deliberately off-centre in a 0-100 by 0-50 frame
    before_x, before_y = _spans(box)

    box.wheelEvent(_Wheel(box.mapFromView(cursor), 480), None)

    after_x, after_y = _spans(box)
    assert after_x < before_x and after_y < before_y
    # The data point under the pointer is the fixed point of the transform, so it is
    # still under the pointer afterwards. Zooming about the centre would move it.
    still = box.mapToView(box.mapFromView(cursor))
    assert still.x() == pytest.approx(cursor.x(), abs=0.05)
    assert still.y() == pytest.approx(cursor.y(), abs=0.05)


@pytest.mark.parametrize(
    "modifier, x_moves, y_moves",
    [
        (Qt.KeyboardModifier.NoModifier, True, True),
        (Qt.KeyboardModifier.ControlModifier, True, False),
        (Qt.KeyboardModifier.ShiftModifier, False, True),
    ],
)
def test_a_modifier_confines_the_wheel_to_one_axis(view, modifier, x_moves, y_moves):
    box = view.view_box
    before_x, before_y = _spans(box)

    box.wheelEvent(_Wheel(box.mapFromView(pg.Point(50.0, 25.0)), 480, modifier), None)

    after_x, after_y = _spans(box)
    assert (after_x < before_x) is x_moves
    assert (after_y < before_y) is y_moves


def test_a_left_drag_pans_and_keeps_the_span(view):
    box = view.view_box
    box.setRange(xRange=(20.0, 60.0), yRange=(10.0, 30.0), padding=0.0)
    before_x, before_y = _spans(box)
    start = box.mapFromView(pg.Point(40.0, 20.0))
    end = box.mapFromView(pg.Point(30.0, 20.0))

    box.mouseDragEvent(_Drag(Qt.MouseButton.LeftButton, start, start, end), None)

    (x0, x1), _ = box.viewRange()
    after_x, after_y = _spans(box)
    assert after_x == pytest.approx(before_x) and after_y == pytest.approx(before_y)
    assert x0 > 20.0  # dragging the data left moves the window right


@pytest.mark.parametrize(
    "button, modifiers",
    [
        (Qt.MouseButton.RightButton, Qt.KeyboardModifier.NoModifier),
        (Qt.MouseButton.LeftButton, Qt.KeyboardModifier.ShiftModifier),
    ],
)
def test_a_rectangle_drag_zooms_to_the_rectangle_on_release(view, button, modifiers):
    box = view.view_box
    down = box.mapFromView(pg.Point(20.0, 10.0))
    up = box.mapFromView(pg.Point(60.0, 30.0))

    box.mouseDragEvent(_Drag(button, down, down, up, modifiers=modifiers), None)
    assert box.rbScaleBox.isVisible()  # mid-drag: the band is drawn, the view has not moved
    assert _spans(box) == pytest.approx((100.0, 50.0))

    box.mouseDragEvent(_Drag(button, down, up, up, finish=True, modifiers=modifiers), None)

    (x0, x1), (y0, y1) = box.viewRange()
    assert (x0, x1) == pytest.approx((20.0, 60.0), abs=0.2)
    assert (y0, y1) == pytest.approx((10.0, 30.0), abs=0.2)
    assert not box.rbScaleBox.isVisible()


def test_a_rectangle_that_is_really_a_click_leaves_the_view_alone(view):
    box = view.view_box
    down = box.mapFromView(pg.Point(20.0, 10.0))
    nudge = pg.Point(down.x() + 1.0, down.y() + 1.0)

    box.mouseDragEvent(
        _Drag(Qt.MouseButton.RightButton, down, down, nudge, finish=True), None
    )

    assert _spans(box) == pytest.approx((100.0, 50.0))


def test_gestures_cannot_leave_the_frame(view):
    box = view.view_box

    for _ in range(20):  # far more zoom-out than the frame has
        box.wheelEvent(_Wheel(box.mapFromView(pg.Point(5.0, 5.0)), -480), None)
    (x0, x1), (y0, y1) = box.viewRange()

    assert (x0, x1) == pytest.approx((0.0, 100.0))
    assert (y0, y1) == pytest.approx((0.0, 50.0))


def test_a_double_click_resets_to_the_full_range(view, qtbot):
    box = view.view_box
    box.setRange(xRange=(30.0, 40.0), yRange=(15.0, 20.0), padding=0.0)

    class _Click:
        def double(self):
            return True

        def accept(self):
            pass

    with qtbot.waitSignal(box.reset_requested, timeout=1000):
        box.mouseClickEvent(_Click())

    assert _spans(box) == pytest.approx((100.0, 50.0))


def test_a_view_change_is_debounced_into_one_render_request(view, qtbot):
    emitted: list[object] = []
    view.view_changed.connect(lambda x, y: emitted.append((x, y)))
    box = view.view_box

    for step in range(10):  # a wheel burst: ten range changes inside the debounce window
        box.setRange(xRange=(float(step), 90.0), yRange=(0.0, 50.0), padding=0.0)
    qtbot.wait(150)

    assert len(emitted) == 1
    assert emitted[0][0][0] == pytest.approx(9.0)  # and it carries the newest range


# --- the window, end to end -----------------------------------------------------------

@pytest.fixture
def loaded_window(qtbot, synthetic_uimf):
    """A window showing the synthetic file's first frame, and its first `RasterResult`."""
    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    qtbot.waitExposed(window)
    with qtbot.waitSignal(window.frame_shown, timeout=5000) as blocker:
        window.open_file(synthetic_uimf.path)
    return window, blocker.args[0]


def test_the_full_range_view_totals_the_files_own_tic_and_bpi(loaded_window, synthetic_uimf):
    """Milestone M3's arithmetic: the image on screen is the whole frame, exactly.

    Against the file's own stored columns, not against a re-derivation -- the same
    comparison `uimf-info --verify` makes, made this time through the window, so a
    wiring mistake between the frame, the axes and the render worker cannot hide behind
    a rasteriser that is correct on its own.
    """
    _, result = loaded_window
    scans = [synthetic_uimf.scan(1, s) for s in synthetic_uimf.stored_scans(1)]

    assert result.tic_in_view == pytest.approx(sum(row.tic for row in scans), rel=1e-12)
    assert result.max_intensity == pytest.approx(max(row.bpi for row in scans))
    assert result.points_in_view == synthetic_uimf.points(1)


def test_the_side_plots_project_the_same_points_as_the_image(loaded_window, synthetic_uimf):
    window, result = loaded_window
    render = window.last_render

    for edges, values in (render.x_profile, render.y_profile):
        assert len(edges) == len(values) + 1
        assert float(np.sum(values)) == pytest.approx(result.tic_in_view, rel=1e-12)

    x_curve, y_curve = window.side_plots.curves
    # Drawn on element centres, one point per element, and the arrival-time plot with
    # its axes swapped so that its vertical axis is the heatmap's.
    assert x_curve.xData.size == render.x_profile[1].size
    assert np.allclose(x_curve.yData, render.x_profile[1])
    assert np.allclose(y_curve.xData, render.y_profile[1])


def test_zooming_narrows_the_view_and_the_side_plots_with_it(loaded_window, qtbot):
    window, full = loaded_window
    axes = window._current_axes
    (x0, x1), (y0, y1) = axes.full_range
    quarter = (x0 + 0.25 * (x1 - x0), x0 + 0.5 * (x1 - x0))

    with qtbot.waitSignal(window.frame_shown, timeout=5000) as blocker:
        window.heatmap.view_box.setRange(xRange=quarter, yRange=(y0, y1), padding=0.0)
    zoomed = blocker.args[0]

    assert zoomed.x_range == pytest.approx(quarter)
    assert zoomed.points_in_view < full.points_in_view
    assert zoomed.tic_in_view < full.tic_in_view
    # The mass spectrum is now the window's, not the frame's, and it still totals it.
    _, x_values = window.last_render.x_profile
    assert float(np.sum(x_values)) == pytest.approx(zoomed.tic_in_view, rel=1e-12)


def test_home_returns_to_the_full_range_after_a_zoom(loaded_window, qtbot):
    window, full = loaded_window
    (x0, x1), (y0, y1) = window._current_axes.full_range

    with qtbot.waitSignal(window.frame_shown, timeout=5000):
        window.heatmap.view_box.setRange(
            xRange=(x0 + 0.4 * (x1 - x0), x0 + 0.5 * (x1 - x0)), padding=0.0
        )
    with qtbot.waitSignal(window.frame_shown, timeout=5000) as blocker:
        window.heatmap.reset_range()

    back = blocker.args[0]
    assert back.x_range == pytest.approx((x0, x1))
    assert back.y_range == pytest.approx((y0, y1))
    assert back.tic_in_view == pytest.approx(full.tic_in_view)


def test_the_cursor_readout_names_every_unit_and_the_aggregation(loaded_window):
    window, result = loaded_window
    axes = window._current_axes
    # The middle of the view: guaranteed to be inside the image, whatever the widget size.
    (x0, x1), (y0, y1) = axes.full_range
    x, y = 0.5 * (x0 + x1), 0.5 * (y0 + y1)

    window._on_cursor_moved(x, y)
    text = window._readout.text()

    assert axes.x_label in text and axes.y_label in text
    assert "bin " in text and "scan " in text
    # The number is the drawn pixel's, and it is labelled with what made it.
    row, column = pixel_of(result, x, y)
    assert f"{result.aggregate} {float(result.image[row, column]):,.0f}" in text

    window._clear_readout()
    assert window._readout.text() == ""


def test_a_stale_render_is_dropped_rather_than_drawn(loaded_window):
    """A result for a superseded frame must not repaint the current one.

    The mailbox drops superseded *requests*; this is the other half -- a result already
    in flight when the frame changed. Forced here by bumping the serial by hand, because
    the race it guards against is one a test cannot reliably produce and a user with a
    slow file and a fast finger can.
    """
    window, result = loaded_window
    stale = window.last_render
    window._serial += 1
    window._last_render = None

    window._on_rendered(stale)

    assert window.last_render is None


def test_the_window_agrees_with_a_real_files_stored_columns(qtbot, real_uimf):
    """The same milestone check against a writer's own numbers, on whatever files exist.

    Skips in a bare clone (`conftest.real_uimf`); this is the check that would catch a
    frame convention the synthetic fixture does not reproduce.
    """
    file = UimfFile(real_uimf)
    frame = file.frame_numbers()[0]
    _, _, bpi, tic = file.scan_summary(frame)

    window = MainWindow()
    qtbot.addWidget(window)
    with qtbot.waitSignal(window.frame_shown, timeout=30000) as blocker:
        window.open_file(real_uimf)
    result = blocker.args[0]

    assert result.tic_in_view == pytest.approx(float(tic.sum()), rel=1e-9)
    assert result.max_intensity == pytest.approx(float(bpi.max()))
