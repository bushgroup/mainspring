"""The heatmap: a pyqtgraph image in display coordinates, and the gestures over it.

The image arrives from `mainspring.uimf.raster` already reduced to the widget's pixel
size and already linear in display space, so this module places it with a single
`ImageItem.setRect` and never transforms it per pixel. Switching axes to raw bin and
scan units, or swapping x and y, changes the `DisplayAxes` handed to the rasteriser and
nothing here.

`UimfViewBox` exists because pyqtgraph's default box does not do what an instrument
user expects. **One gesture, one meaning, no modes**: the wheel zooms about the cursor
rather than the centre, a left drag pans, a right drag draws a zoom rectangle that
applies on release, and a double-click or Home returns to the full range. PNNL's viewer
makes a zoom a multi-step rectangle operation on a stack you then have to unwind, which
is the second of the three faults this project exists to fix; the reset must therefore
be a single key, always available, and never lose the frame you were on. There is
deliberately **no zoom history** here for the same reason: a stack is what makes a
wrong zoom expensive, and with a one-gesture zoom and a one-key reset it earns nothing.

Two of those four are pyqtgraph's own behaviour and are not reimplemented: its
`wheelEvent` already scales about the cursor, and its pan-mode left drag already pans.
What this box adds is the right-drag rectangle (pyqtgraph puts that on a mode switch and
gives right-drag a one-axis-at-a-time scale instead), the modifier that confines the
wheel to one axis, the double-click reset, and a context menu that is gone so that the
right button is free to mean something.

View changes are debounced and sent to the render worker's mailbox, so a continuous
gesture repaints at the rate the rasteriser can sustain and always with the newest view
(`workers.py`). **Auto-range is off and stays off**: the view range is state the window
owns, set by a gesture or by opening a file, and an image arriving must never move it --
which is what makes "keep ranges" a matter of `MainWindow` not calling `set_frame_extent`'s
reset (`main_window._on_frame_loaded`), rather than of fighting the widget.

While a render is in flight the previous image stays where it is and the ViewBox
stretches it, so a pan or a zoom is continuous rather than blanking between frames; the
new image replaces it in place when it lands.
"""

from __future__ import annotations

import os

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor

from ..uimf import DisplayAxes
from .controls import describe
from .workers import DEBOUNCE_MS

__all__ = [
    "AXIS_HEIGHT",
    "AXIS_WIDTH",
    "COLOUR_BAR_WIDTH",
    "RIGHT_AXIS_WIDTH",
    "SIDE_PLOT_SIZE",
    "TICK_LENGTH",
    "TOP_AXIS_HEIGHT",
    "HeatmapView",
    "UimfViewBox",
    "pixel_of",
]


def _scaled(image: np.ndarray, colour_scale: str) -> np.ndarray:
    """The display transform behind the log/sqrt colour toggle. `"linear"` is a no-op."""
    if colour_scale == "log":
        return np.log1p(np.clip(image, 0.0, None))
    if colour_scale == "sqrt":
        return np.sqrt(np.clip(image, 0.0, None))
    return image

AXIS_WIDTH = 68
"""Pixels reserved for the heatmap's left axis. Fixed rather than fitted so that the
heatmap and the mass-spectrum plot above it, which are linked in x, also line up in x on
screen -- their plot areas start where their left margins end, and pyqtgraph does not
equalise those. The side plots mirror this number in their own layout grids
(`side_plots.py`)."""

AXIS_HEIGHT = 46
"""The same for the heatmap's bottom axis, so the heatmap and the arrival-time plot to
its right line up in y."""

TOP_AXIS_HEIGHT = 0
"""The heatmap's top axis shows ticks and nothing else, and inward ticks reserve no
space (`AxisItem._updateHeight` adds `max(0, tickLength)`), so it is zero pixels tall.
Named because the side plots mirror it: the spectrum above sits flush against the image."""

RIGHT_AXIS_WIDTH = 0
"""The same for the heatmap's right axis, mirrored by the arrival-time plot beside it."""

SIDE_PLOT_SIZE = 160
"""Pixels across each projection: the spectrum's height and the arrival-time plot's width."""

COLOUR_BAR_WIDTH = 80
"""Pixels for the colour bar's column: pyqtgraph's 25 px strip, its 45 px value axis and
the item's own margins."""

TICK_LENGTH = -8
"""Major tick length on the heatmap's axes. Negative points the ticks **into** the plot
(pyqtgraph's convention); minor levels are drawn at 1/1.5 and 1/2 of this. Inward and
long enough to read against the image, because a tick is how a zoomed view is read
against the axis values -- and on the top and right edges, which carry no values, the
ticks are the whole axis."""


class UimfViewBox(pg.ViewBox):
    """A `pyqtgraph.ViewBox` with the one-gesture zoom, pan and reset above.

    `set_extent` gives it the frame's full range, which is both what a reset returns to
    and what `setLimits` clamps every gesture inside: a pan cannot leave the frame and a
    zoom cannot go out past it or in past a couple of source elements. Limits are worth
    more here than in a general plot -- outside the frame there is nothing to see, and a
    user who has lost the data is exactly the user who reaches for the reset PNNL's
    viewer buries in a context menu.
    """

    reset_requested = Signal()
    """A double-click asked for the full range. The box has already reset itself; this is
    for anything that wants to hear about it (the window, which logs it to the status bar)."""

    def __init__(self, parent: "object | None" = None) -> None:
        super().__init__(parent=parent, enableMenu=False, defaultPadding=0.0)
        self.setMouseMode(pg.ViewBox.PanMode)
        self.disableAutoRange()
        self._full_range: tuple[tuple[float, float], tuple[float, float]] | None = None

    # --- extent and reset -------------------------------------------------------------

    def set_extent(self, axes: DisplayAxes, reset: bool = True) -> None:
        """Declare the frame's full range: the reset target and the limits of every gesture.

        `reset=False` keeps whatever is on screen and only moves the limits, which is
        what the keep-ranges setting uses. Even then the ranges are clamped
        into the new frame's extent by `setLimits`, since a window kept from a file with
        a different calibration may not overlap this one at all.

        The zoom-in floor is **two average source elements**, taken from the axis tables
        rather than guessed: a window narrower than the data behind it shows the user
        their own pixel grid. Two average m/z elements on a SLIMPHONY frame is about
        0.03 Da, which is well inside an isotope pattern, and the floor is what keeps a
        wheel that runs away from producing a degenerate range for the rasteriser.
        """
        self._full_range = axes.full_range
        (x0, x1), (y0, y1) = self._full_range
        x_elements = max(1, len(axes.x_edges) - 1)
        y_elements = max(1, len(axes.y_edges) - 1)
        self.setLimits(
            xMin=x0, xMax=x1, yMin=y0, yMax=y1,
            maxXRange=x1 - x0, maxYRange=y1 - y0,
            minXRange=2.0 * (x1 - x0) / x_elements,
            minYRange=2.0 * (y1 - y0) / y_elements,
        )
        if reset:
            self.reset_range()

    def reset_range(self) -> None:
        """Back to the frame's full m/z and arrival-time range."""
        if self._full_range is None:
            return
        (x0, x1), (y0, y1) = self._full_range
        self.setRange(xRange=(x0, x1), yRange=(y0, y1), padding=0.0)

    # --- gestures ---------------------------------------------------------------------

    def wheelEvent(self, ev, axis=None) -> None:
        """Zoom about the cursor; Ctrl confines it to x, Shift to y.

        pyqtgraph's own wheel handler already scales about the cursor and already takes
        an `axis` argument meaning "only this one", so the modifier is all this adds.
        """
        if axis is None:
            modifiers = ev.modifiers()
            if modifiers & Qt.KeyboardModifier.ControlModifier:
                axis = 0
            elif modifiers & Qt.KeyboardModifier.ShiftModifier:
                axis = 1
        super().wheelEvent(ev, axis)

    def mouseDragEvent(self, ev, axis=None) -> None:
        """Left drag pans; right drag, or Shift and a left drag, draws a zoom rectangle.

        Shift-left is there for the trackpads and the remote-desktop sessions where a
        right-drag is awkward to produce; it is the same gesture, not a second one.
        """
        button = ev.button()
        shift = bool(ev.modifiers() & Qt.KeyboardModifier.ShiftModifier)
        if button == Qt.MouseButton.RightButton or (
            button == Qt.MouseButton.LeftButton and shift and axis is None
        ):
            self._rect_zoom(ev)
            return
        if button == Qt.MouseButton.LeftButton:
            super().mouseDragEvent(ev, axis)
            return
        ev.ignore()

    def mouseClickEvent(self, ev) -> None:
        """A double-click resets; a single click does nothing, and there is no menu."""
        if ev.double():
            ev.accept()
            self.reset_range()
            self.reset_requested.emit()
            return
        ev.ignore()

    def _rect_zoom(self, ev) -> None:
        """The rubber band, and the range it means once the button comes up.

        Applied on release and never pushed onto a history, so the whole gesture is
        press, drag, release -- one step, undone by one key.
        """
        ev.accept()
        if not ev.isFinish():
            self.updateScaleBox(ev.buttonDownPos(ev.button()), ev.pos())
            return
        self.rbScaleBox.hide()
        rect = QRectF(
            pg.Point(ev.buttonDownPos(ev.button())), pg.Point(ev.pos())
        ).normalized()
        if rect.width() <= 2.0 or rect.height() <= 2.0:
            return  # a click that wobbled, not a rectangle; leave the view alone
        self.showAxRect(self.childGroup.mapRectFromParent(rect))


class HeatmapView(pg.GraphicsLayoutWidget):
    """The image, its axes, its colour bar, and the debounced view-changed signal.

    The layout leaves room for the side plots without knowing what they are:

        row 0   spectrum cell    .            .
        row 1   heatmap          arrival-time cell    colour bar
                col 0            col 1                col 2

    The heatmap sits at (1, 0), the colour bar at (1, 2), and `side_plot_slots()` hands
    out the two cells above and beside the image. That is why `side_plots.py` can own
    its own widgets and this module can stay about the image. Ticks are drawn on all
    four of the heatmap's axes, inward; the top and right ones carry no values, so the
    projections sit directly against the plot area they share an axis with.
    """

    view_resized = Signal(int, int)
    """Emitted with the viewport's pixel size on every resize, so the owner can
    re-rasterise at the new size (`MainWindow`, which holds the frame this needs)."""

    view_changed = Signal(object, object)
    """`(x_range, y_range)`, debounced by `DEBOUNCE_MS`, whenever the visible window
    moves. What the render mailbox is fed from; the only path a gesture takes to a new
    image."""

    cursor_moved = Signal(float, float)
    """`(x, y)` in display units whenever the pointer is over the image. Not debounced:
    a readout that lags the pointer is worse than no readout."""

    cursor_left = Signal()
    """The pointer left the image; the readout should say nothing rather than something
    stale."""

    def __init__(self, colour_map: str = "viridis", parent: "object | None" = None) -> None:
        super().__init__(parent=parent)
        self._view_box = UimfViewBox()
        self._plot = self.addPlot(row=1, col=0, viewBox=self._view_box)
        self._plot.showGrid(x=False, y=False)
        self._plot.setMenuEnabled(False)
        self._plot.getAxis("left").setWidth(AXIS_WIDTH)
        self._plot.getAxis("bottom").setHeight(AXIS_HEIGHT)
        # Every axis is created and linked to the box when the PlotItem is; the top and
        # right ones only need showing. They carry ticks and nothing else, and at zero
        # size, so that the projections meet the image edge they share.
        for name, size in (("top", TOP_AXIS_HEIGHT), ("right", RIGHT_AXIS_WIDTH)):
            self._plot.showAxis(name)
            axis = self._plot.getAxis(name)
            axis.setStyle(showValues=False)
            if name == "top":
                axis.setHeight(size)
            else:
                axis.setWidth(size)
        for name in ("left", "bottom", "top", "right"):
            axis = self._plot.getAxis(name)
            # `tickAlpha` pinned: pyqtgraph otherwise fades each minor level by half.
            axis.setStyle(tickLength=TICK_LENGTH, tickAlpha=255)
            axis.setTickPen(pg.mkPen(axis.textPen().color(), width=1.5))
        self._image_item = pg.ImageItem()
        self._plot.addItem(self._image_item)

        # The gesture contract at the top of this module is the one piece of the viewer a
        # user cannot discover by looking, because the context menu that would have
        # advertised it is deliberately off. The image is where they point.
        _GESTURES = (
            "Scroll to zoom about the pointer, drag to pan, right-drag to zoom to a box,"
            " double-click to reset."
        )
        describe(self._image_item, "Intensity across the frame. " + _GESTURES)
        describe(self._plot, "Intensity across the frame. " + _GESTURES)
        for name, what in (
            ("left", "the vertical axis"),
            ("bottom", "the horizontal axis"),
            ("top", "the horizontal axis"),
            ("right", "the vertical axis"),
        ):
            describe(
                self._plot.getAxis(name),
                f"Ticks on {what}. Ctrl-scroll zooms the horizontal axis alone and"
                " Shift-scroll the vertical.",
            )

        # The colour bar is a `PlotItem` of its own in the last column rather than
        # inserted into the heatmap's layout (`insert_in`), so that the arrival-time
        # plot can sit between the two. Its blank bottom axis is fixed to the heatmap's
        # bottom-axis height so the strip spans exactly the image's height.
        self._colour_bar = pg.ColorBarItem(colorMap=colour_map)
        self._colour_bar.setImageItem(self._image_item)
        self._colour_bar.getAxis("bottom").setHeight(AXIS_HEIGHT)
        self._colour_bar.getAxis("top").setHeight(TOP_AXIS_HEIGHT)
        _BAR_TIP = (
            "The intensity each colour stands for. Drag an end to set the limits by hand,"
            " and tick Keep levels to hold them across frames."
        )
        describe(self._colour_bar, _BAR_TIP)
        for name in ("left", "bottom", "top", "right"):
            describe(self._colour_bar.getAxis(name), _BAR_TIP)
        self.ci.addItem(self._colour_bar, row=1, col=2)

        # Row 0 and column 1 are the side plots' (`side_plot_slots`); the stretch factors
        # are what keep the image the large thing on screen once they are filled. No
        # spacing between cells: the projections should touch the image they project.
        self.ci.layout.setSpacing(0)
        self.ci.layout.setRowFixedHeight(0, SIDE_PLOT_SIZE)
        self.ci.layout.setColumnFixedWidth(1, SIDE_PLOT_SIZE)
        self.ci.layout.setColumnFixedWidth(2, COLOUR_BAR_WIDTH)
        self.ci.layout.setColumnStretchFactor(0, 1)
        self.ci.layout.setRowStretchFactor(1, 1)

        self._debug = pg.TextItem(color=QColor(220, 220, 220), anchor=(0, 0))
        self._debug.setParentItem(self._view_box)  # parented to the box: position is in pixels
        self._debug.setPos(8, 6)
        self._debug.setVisible(bool(os.environ.get("MAINSPRING_DEBUG_RENDER")))

        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(DEBOUNCE_MS)
        self._debounce.timeout.connect(self._emit_view_changed)
        self._view_box.sigRangeChanged.connect(lambda *_: self._debounce.start())
        self.scene().sigMouseMoved.connect(self._on_mouse_moved)
        self._cursor_inside = False
        self._levels_held = False

    # --- accessors --------------------------------------------------------------------

    @property
    def image_item(self) -> pg.ImageItem:
        """The underlying `ImageItem`, mostly so a test can read back its state."""
        return self._image_item

    @property
    def view_box(self) -> UimfViewBox:
        """The custom box, for a caller that wants to drive a gesture directly."""
        return self._view_box

    @property
    def plot_item(self) -> pg.PlotItem:
        """The heatmap's `PlotItem`, which the side plots link their axes to."""
        return self._plot

    def side_plot_slots(self) -> "tuple[pg.GraphicsLayout, tuple[int, int], tuple[int, int]]":
        """`(layout, y_plot_cell, x_plot_cell)`: where the two side plots go.

        `y_plot` (the projection onto the vertical axis) goes beside the image, `x_plot`
        (onto the horizontal axis) above it. The heatmap owns the grid because it owns
        the alignment -- the axis extents that make a linked plot line up are set here,
        on `AXIS_WIDTH`, `AXIS_HEIGHT` and their zero-sized top and right counterparts,
        and a caller that placed its own plots elsewhere in the layout would silently
        lose it.
        """
        return self.ci, (1, 1), (0, 0)

    def view_range(self) -> "tuple[tuple[float, float], tuple[float, float]]":
        """The visible `(x_range, y_range)` in display units."""
        (x0, x1), (y0, y1) = self._view_box.viewRange()
        return (float(x0), float(x1)), (float(y0), float(y1))

    def pixel_size(self) -> "tuple[int, int]":
        """`(width, height)` of the drawing area, in device pixels, at least 1x1.

        What `rasterise` should be asked for: the viewport rather than the whole widget,
        since the axes and the colour bar take some of it (lab record, task 04).
        """
        size = self.viewport().size()
        return max(1, size.width()), max(1, size.height())

    # --- state ------------------------------------------------------------------------

    def set_frame_extent(self, axes: DisplayAxes, reset: bool = True) -> None:
        """Hand the box a new frame's full range, then ask for a render without waiting.

        The debounce exists to coalesce a gesture, and this is not one: the range change
        it causes is already the final one, so waiting 30 ms to say so only delays the
        first paint.
        """
        self._view_box.set_extent(axes, reset=reset)
        self._emit_view_changed()

    def reset_range(self) -> None:
        """Back to the frame's full range: what Home and the double-click both call."""
        self._view_box.reset_range()

    def set_image(self, result: object, colour_scale: str = "linear") -> None:
        """Show a `RasterResult`, placing it by its display ranges.

        `colour_scale` is a display transform applied to the image before it reaches the
        colour bar -- `log1p` or `sqrt` of the (non-negative) intensity -- so that one
        bright peak does not wash out everything else in the same view. It never touches
        `result.image` itself: the readouts (the cursor, the info panel) quote the raw
        `RasterResult`, and quoting a transformed number as if it were a stored intensity
        is exactly the mistake `notes/viewer-ux.md` warns the cursor label against.

        Levels go through the colour bar's own `setLevels` rather than through
        `ImageItem`'s `autoLevels` -- once a `ColorBarItem` is attached it owns the
        applied levels and does not follow an image's own auto-scaling (no
        `sigLevelsChanged` on plain `ImageItem` in this pyqtgraph version), so
        `autoLevels=True` here would silently keep showing whatever the colour bar's
        levels happened to be, which is 0-1 until told otherwise. **Unless the levels are
        held** (`set_levels`, the keep-levels setting): then this image is shown under
        whatever levels are already pinned, computed from a possibly different image, on
        purpose -- that is the point of holding them.
        """
        axes = result.axes
        x0, x1 = result.x_range
        y0, y1 = result.y_range
        displayed = _scaled(result.image, colour_scale)
        self._image_item.setImage(displayed, autoLevels=False)
        self._image_item.setRect(x0, y0, x1 - x0, y1 - y0)
        self._plot.setLabel("bottom", axes.x_label)
        self._plot.setLabel("left", axes.y_label)
        if not self._levels_held:
            low, high = float(displayed.min()), float(displayed.max())
            self._colour_bar.setLevels((low, high if high > low else low + 1.0))

    def levels(self) -> "tuple[float, float]":
        """The colour bar's current `(low, high)`, whatever last set them."""
        low, high = self._colour_bar.levels()
        return float(low), float(high)

    def set_levels(self, low: float, high: float) -> None:
        """Pin the colour levels: every later `set_image` leaves them alone.

        What the keep-levels setting calls when it is turned on, with whatever is on
        screen right now (`levels()`) -- freezing the current scale rather than starting
        from an arbitrary one, the same choice keep-ranges makes for the view range.
        """
        self._levels_held = True
        self._colour_bar.setLevels((float(low), float(high)))

    def release_levels(self) -> None:
        """Un-pin the colour levels: the next `set_image` goes back to auto-scaling."""
        self._levels_held = False

    def set_debug_text(self, text: str) -> None:
        """The render-time overlay, drawn only when `MAINSPRING_DEBUG_RENDER` is set.

        Off by default because it is a developer's number, and always available because
        "is the render keeping up" is the first question a slow file raises in the field.
        """
        self._debug.setText(text)

    # --- events -----------------------------------------------------------------------

    def _emit_view_changed(self) -> None:
        self._debounce.stop()
        x_range, y_range = self.view_range()
        self.view_changed.emit(x_range, y_range)

    def _on_mouse_moved(self, scene_pos) -> None:
        if not self._plot.sceneBoundingRect().contains(scene_pos):
            if self._cursor_inside:
                self._cursor_inside = False
                self.cursor_left.emit()
            return
        point = self._view_box.mapSceneToView(scene_pos)
        (x0, x1), (y0, y1) = self.view_range()
        inside = x0 <= point.x() <= x1 and y0 <= point.y() <= y1
        if not inside:
            if self._cursor_inside:
                self._cursor_inside = False
                self.cursor_left.emit()
            return
        self._cursor_inside = True
        self.cursor_moved.emit(float(point.x()), float(point.y()))

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        width, height = self.pixel_size()
        self.view_resized.emit(width, height)


def pixel_of(result: object, x: float, y: float) -> "tuple[int, int] | None":
    """`(row, column)` of the display point `(x, y)` in a `RasterResult`'s image.

    Here rather than in the window because it is the inverse of what `rasterise` did to
    build the image, and the two belong within sight of each other: the image spans
    exactly `x_range` by `y_range` in `cols` by `rows` uniform cells, so the inverse is a
    division. Returns None outside the image, which is not an error -- the pointer is
    over the axes, or over a view the render has not caught up with yet.
    """
    rows, cols = result.image.shape
    x0, x1 = result.x_range
    y0, y1 = result.y_range
    if not (x1 > x0 and y1 > y0):
        return None
    column = int(np.floor((x - x0) * (cols / (x1 - x0))))
    row = int(np.floor((y - y0) * (rows / (y1 - y0))))
    if not (0 <= column < cols and 0 <= row < rows):
        return None
    return row, column
