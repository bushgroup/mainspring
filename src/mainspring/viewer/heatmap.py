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
from contextlib import contextmanager

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import QGraphicsPixmapItem

from ..uimf import DisplayAxes
from . import fonts, labels, theme
from .controls import describe
from .workers import DEBOUNCE_MS

__all__ = [
    "AXIS_HEIGHT",
    "AXIS_PEN_WIDTH",
    "AXIS_WIDTH",
    "COLOUR_BAR_AXIS_WIDTH",
    "COLOUR_BAR_WIDTH",
    "LINE_WIDTH_LIMITS",
    "REFERENCE_VIEWPORT",
    "RIGHT_AXIS_WIDTH",
    "SIDE_PLOT_SIZE",
    "TICK_LENGTH",
    "TICK_PEN_WIDTH",
    "TOP_AXIS_HEIGHT",
    "HeatmapView",
    "UimfViewBox",
    "pixel_of",
    "scaled",
]


def scaled(image: np.ndarray, colour_scale: str) -> np.ndarray:
    """The display transform behind the log/sqrt colour toggle. `"linear"` is a no-op.

    Public because the colour levels are computed in this space and not in the
    intensities' own -- so anything that has to reproduce a level, `export.py` included,
    has to be able to get into the same space rather than approximate its way there."""
    if colour_scale == "log":
        return np.log1p(np.clip(image, 0.0, None))
    if colour_scale == "sqrt":
        return np.sqrt(np.clip(image, 0.0, None))
    return image

AXIS_WIDTH = 68
"""Pixels reserved for the heatmap's left axis **at 100 per cent text**. Fixed rather
than fitted so that the heatmap and the mass-spectrum plot above it, which are linked in
x, also line up in x on screen -- their plot areas start where their left margins end, and
pyqtgraph does not equalise those. The side plots mirror this number in their own layout
grids (`side_plots.py`), and both sides multiply it by `fonts.extent()` when they apply
it: a scale applied to one side alone would pull the two out of alignment."""

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

LOGO_RENDER_SIZE = 320
"""The mainspring mark is rendered once, at construction, at this many pixels square --
comfortably above the ~48 px turn-closure threshold `tools/make_icon.py` documents for
this same spiral -- and then scaled down by the view box to whatever cell (0, 1) is,
rather than re-rendered on every resize."""


def _resource(name: str) -> str:
    """A file in `resources/`, resolved the same way `app._icon_path` resolves the
    `.ico`: `os.path.dirname(__file__)` is right in a frozen build too, since
    `packaging/mainspring.spec` collects the package's data files beside it."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources", name)


def _render_svg_pixmap(path: str, size: int) -> QPixmap:
    """Rasterise the SVG at `path` into a `size`x`size` transparent `QPixmap`.

    The same recipe as `tools/make_icon.py`'s `_render`, aimed at a `QPixmap` instead of
    a `QImage` because a `QGraphicsPixmapItem` is what a `ViewBox` can hold.
    """
    renderer = QSvgRenderer(path)
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
    renderer.render(painter, QRectF(0, 0, size, size))
    painter.end()
    return pixmap

COLOUR_BAR_WIDTH = 80
"""Pixels for the colour bar's column **at 100 per cent text**: pyqtgraph's 25 px strip,
its `COLOUR_BAR_AXIS_WIDTH` value axis and the item's own margins."""

COLOUR_BAR_AXIS_WIDTH = 45
"""Of that, what `pg.ColorBarItem` gives its value axis. It *fixes* that width rather
than letting the axis expand into the space it needs, so a bigger tick font is clipped
rather than accommodated: at 150 per cent the bar showed one value and otherwise said
nothing about what a colour meant. `set_text_scale` sets both this and the column again,
and only this part of the column grows, since the strip beside it is not text."""

TICK_LENGTH = -8
"""Major tick length on the heatmap's axes. Negative points the ticks **into** the plot
(pyqtgraph's convention); minor levels are drawn at 1/1.5 and 1/2 of this. Inward and
long enough to read against the image, because a tick is how a zoomed view is read
against the axis values -- and on the top and right edges, which carry no values, the
ticks are the whole axis."""

TICK_PEN_WIDTH = 1.5
"""How thick a tick is drawn **at `REFERENCE_VIEWPORT`**, in the same "prominent enough
to read against the image" judgement as `TICK_LENGTH` (lab record, task 11). The width is
here and the colour is `theme.py`'s: how much a tick asserts itself is this module's
decision, and what colour it asserts itself in is the palette's."""

AXIS_PEN_WIDTH = 1.0
"""The same for the axis line itself, which is furniture rather than a reading aid and is
drawn lighter than the ticks that sit on it. The ratio between the two is what a resize
preserves; neither is scaled on its own."""

REFERENCE_VIEWPORT = 700
"""The viewport's smaller dimension, in pixels, that the two widths above are the widths
at. 700 is the height of the 1000x700 window the viewer was designed against, which is
what makes this a change of nothing at that size and a change of something on the 4K
panel where a 1.5 px tick is a hairline (lab record, task 24)."""

LINE_WIDTH_LIMITS = (1.0, 3.0)
"""The narrowest and widest either line is ever drawn, whatever the viewport. Below one
pixel a line stops being drawn reliably at all, and above three the furniture starts
competing with the data it is there to measure. Both widths are clamped to this, so on a
small window they meet at one pixel and the axis line stops being lighter than the ticks
on it -- which is what a floor means."""


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

    def zoom_axis(self, axis: int, lo: float, hi: float) -> None:
        """Set one axis's visible range and leave the other where it is.

        What a band drawn on a projection means (`side_plots.ProjectionViewBox`). It
        goes through `setRange` like every other gesture, so `set_extent`'s limits clamp
        it -- a band cannot leave the frame and cannot zoom in past the floor of two
        source elements.
        """
        if axis == 0:
            self.setRange(xRange=(lo, hi), padding=0.0)
        else:
            self.setRange(yRange=(lo, hi), padding=0.0)

    def reset_axis(self, axis: int) -> None:
        """Back to the frame's full range on one axis, leaving the other zoomed.

        The one-axis counterpart of `reset_range`, and the same one-step contract: a
        band is undone by a double-click on the plot that drew it, with no history in
        between.
        """
        if self._full_range is None:
            return
        (x0, x1), (y0, y1) = self._full_range
        if axis == 0:
            self.setRange(xRange=(x0, x1), padding=0.0)
        else:
            self.setRange(yRange=(y0, y1), padding=0.0)

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

    _line_width = TICK_PEN_WIDTH
    _axis_width = AXIS_PEN_WIDTH
    _built = False
    """Class-level defaults, because `pg.GraphicsView.__init__` calls `resizeEvent`
    before this class's own constructor has run a line. `_built` is what stops that
    first resize from repainting axes that do not exist yet; the widths are there so
    that `_fit_lines` has something to compare against when it does."""

    def __init__(self, colour_map: str = "viridis", parent: "object | None" = None) -> None:
        super().__init__(parent=parent)
        self._view_box = UimfViewBox()
        self._plot = self.addPlot(row=1, col=0, viewBox=self._view_box)
        self._plot.showGrid(x=False, y=False)
        self._plot.setMenuEnabled(False)
        # pyqtgraph's own "A" button, shown on hover in the corner, calls
        # `enableAutoRange` -- which this module's auto-range-is-off-and-stays-off
        # invariant forbids, and which Home and a double-click already do properly. It
        # is also a white pixmap that no palette can reach, so a light canvas would show
        # an invisible control that breaks the view if found.
        self._plot.hideButtons()
        # Extents, tick fonts and the label style all come from `set_text_scale` at the
        # end of this constructor, the one path they change by.
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
        # Their pens are not set here: every colour on this widget belongs to
        # `set_palette`, called at the end of this constructor and again on every
        # `View > Light mode` toggle.

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
        # `colorMapMenu=False`: pyqtgraph's own right-click menu offers every registered
        # colormap, and picking one there would neither tick the View menu's choice nor
        # reach `ViewerSettings` -- `set_colour_map` below is the one path that does both.
        self._colour_bar = pg.ColorBarItem(colorMap=colour_map, colorMapMenu=False)
        self._colour_bar.setImageItem(self._image_item)
        self._colour_bar.getAxis("top").setHeight(TOP_AXIS_HEIGHT)
        _BAR_TIP = (
            "The intensity each colour stands for. Drag an end to set the limits by hand,"
            " and tick Keep levels to hold them across frames."
        )
        describe(self._colour_bar, _BAR_TIP)
        for name in ("left", "bottom", "top", "right"):
            describe(self._colour_bar.getAxis(name), _BAR_TIP)
        self.ci.addItem(self._colour_bar, row=1, col=2)

        # The mainspring mark, in the cell above the arrival-time projection that
        # `side_plot_slots` never hands out (row 0, col 1 is otherwise empty). Purely
        # decorative: a `ViewBox` and a `QGraphicsPixmapItem` are neither in
        # `controls._TIPPED_ITEMS` nor in `theme._PAINTED`, so this needs no tooltip and
        # no palette-registered colour -- the full-colour mark is the same under both
        # themes, unlike the axes and curves `set_palette` repaints.
        self._logo_box = self.ci.addViewBox(row=0, col=1, lockAspect=True, enableMouse=False)
        self._logo_box.setMenuEnabled(False)
        self._logo_box.setBorder(None)
        self._logo_item = QGraphicsPixmapItem(
            _render_svg_pixmap(_resource("mainspring.svg"), LOGO_RENDER_SIZE)
        )
        self._logo_box.addItem(self._logo_item)

        # Row 0 and column 1 are the side plots' (`side_plot_slots`); the stretch factors
        # are what keep the image the large thing on screen once they are filled. No
        # spacing between cells: the projections should touch the image they project.
        self.ci.layout.setSpacing(0)
        self.ci.layout.setRowFixedHeight(0, SIDE_PLOT_SIZE)
        self.ci.layout.setColumnFixedWidth(1, SIDE_PLOT_SIZE)
        # Column 2 is the colour bar's, and its width follows the text in it
        # (`set_text_scale`, called at the end of this constructor).
        self.ci.layout.setColumnStretchFactor(0, 1)
        self.ci.layout.setRowStretchFactor(1, 1)

        self._debug = pg.TextItem(anchor=(0, 0))
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
        self._label_style: "dict[str, str]" = {}
        self._fit_lines()
        self.set_palette(theme.active())
        self.set_text_scale(fonts.active())
        # Both ink variants render at the same size, so the logo's bounding rect never
        # changes between them -- one fit, not one per theme toggle.
        self._logo_box.autoRange(padding=0.08)
        self._built = True

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

    def zoom_axis(self, axis: int, lo: float, hi: float) -> None:
        """Set one axis's visible range: what a band on a projection asks for."""
        self._view_box.zoom_axis(axis, lo, hi)

    def reset_axis(self, axis: int) -> None:
        """Back to the frame's full range on one axis alone."""
        self._view_box.reset_axis(axis)

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
        displayed = scaled(result.image, colour_scale)
        self._image_item.setImage(displayed, autoLevels=False)
        self._image_item.setRect(x0, y0, x1 - x0, y1 - y0)
        # The display spelling, not the rasteriser's own (`labels.py`): an axis label is
        # HTML, and `m/z` is italic in a figure and plain in the status bar.
        self._plot.setLabel("bottom", labels.html(axes.x_label), **self._label_style)
        self._plot.setLabel("left", labels.html(axes.y_label), **self._label_style)
        if not self._levels_held:
            low, high = float(displayed.min()), float(displayed.max())
            self._colour_bar.setLevels((low, high if high > low else low + 1.0))

    @contextmanager
    def hidden_debug(self):
        """Hide the render-time overlay for the duration, then put it back.

        What an export renders through (`export.py`). A developer's number is not part
        of a figure, and `MAINSPRING_DEBUG_RENDER` being set is not a reason to write it
        into one. A context manager and not two calls, because a render that raises
        between them would leave the overlay off for the rest of the session.

        The only thing left of the `substituted` this replaced. Until task 24 an export
        also stood a finer image in for the viewport-sized one, which is what moved the
        colour levels and made a figure a different picture from the one on screen.
        """
        previous = self._debug.isVisible()
        try:
            self._debug.setVisible(False)
            yield
        finally:
            self._debug.setVisible(previous)

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

    def set_colour_map(self, name: str) -> None:
        """Switch the colour bar (and the image it drives) to one of `COLOUR_MAPS`.

        Independent of the palette in both directions: the map is a statement about the
        data, and `View > Light mode` never moves it (`theme.py`).
        """
        self._colour_bar.setColorMap(name)

    def set_palette(self, palette: "theme.Palette") -> None:
        """Repaint every colour this widget owns, without rebuilding anything.

        Called once at construction and again on every `View > Light mode` toggle
        (`theme.apply`, the one path). Nothing here touches the image, the colour map or
        the levels, so a toggle costs the user nothing they had set up.

        The order inside the loop is load-bearing. `setTextPen` writes its colour into
        `labelStyle` in place and re-renders the label, so calling it **last** recolours
        a label that `set_image` has already set without disturbing its text or its bold
        weight -- and `_label_style` is refreshed first so that the next `set_image`
        writes the same colour rather than putting the old one back.
        """
        self.setBackground(palette.background)
        self._label_style = theme.label_style(palette)
        for axis in self._axes():
            axis.setPen(pg.mkPen(palette.foreground, width=self._axis_width))
            axis.setTickPen(pg.mkPen(palette.foreground, width=self._line_width))
            axis.setTextPen(pg.mkPen(palette.foreground))
        self._debug.setColor(palette.debug)

    def set_text_scale(self, scale: float) -> None:
        """Redraw every piece of text on this widget at `scale`, and reserve room for it.

        Called once at construction and again on every `View > Text size` change
        (`fonts.apply`, the one path). Nothing here touches the image, the colour map or
        the levels, so a change costs the user nothing they had set up.

        Three separate things, because Qt treats them as three:

        * **Tick values** go through `setTickFont`, which also drops the axis's cached
          `QPicture`. Without that the axis would keep drawing the old size from the
          cache, and an export would replay it.
        * **Axis labels** are HTML in a `QGraphicsTextItem`, which does not follow an
          application font change, so the size travels in `label_style` and the label is
          set again. The text is read back off the axis rather than passed in: nothing
          here knows what the axes are called, and `setLabel(text=None)` would blank them.
        * **The extents**: the two the projections are aligned against, on this side of
          that agreement (`side_plots.py` holds the other side), and the colour bar's
          own column, which is mostly the value axis a reader gets a level off.
        """
        font = fonts.scaled_font(scale)
        self._plot.getAxis("left").setWidth(round(AXIS_WIDTH * fonts.extent()))
        self._plot.getAxis("bottom").setHeight(round(AXIS_HEIGHT * fonts.extent()))
        self._colour_bar.getAxis("bottom").setHeight(round(AXIS_HEIGHT * fonts.extent()))
        bar_axis = round(COLOUR_BAR_AXIS_WIDTH * fonts.extent())
        self._colour_bar.getAxis("right").setWidth(bar_axis)
        self.ci.layout.setColumnFixedWidth(
            2, COLOUR_BAR_WIDTH - COLOUR_BAR_AXIS_WIDTH + bar_axis
        )
        self._label_style = theme.label_style(theme.active())
        for axis in self._axes():
            axis.setTickFont(font)
            axis.setLabel(axis.labelText, **self._label_style)
        self._debug.setFont(font)

    def _axes(self) -> "list[pg.AxisItem]":
        """The heatmap's four axes and the colour bar's four.

        The colour bar's are included because they are furniture on the same canvas: its
        value axis is the one a user reads a level off, and only its gradient belongs to
        the colour map.
        """
        return [
            plot.getAxis(name)
            for plot in (self._plot, self._colour_bar)
            for name in ("left", "bottom", "top", "right")
        ]

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
        # Widths first, and only when they have actually moved: `set_palette` rebuilds
        # eight axes' worth of pens, and a drag of the window edge is a hundred resize
        # events. Going through `set_palette` rather than setting the pens here is what
        # keeps the two independent -- a theme toggle cannot reset the thickness, because
        # it reads `_line_width`, and a resize cannot reset the colour, because it asks
        # for the active palette.
        if self._built and self._fit_lines():
            self.set_palette(theme.active())
        width, height = self.pixel_size()
        self.view_resized.emit(width, height)

    def _fit_lines(self) -> bool:
        """Recompute the line widths for this viewport, and say whether they moved.

        The viewport's **smaller** dimension: a window dragged wide and short is not a
        window whose furniture should get heavier, and the smaller dimension is what
        limits how much of the plot a reader is taking in at once. Both widths are
        clamped by `LINE_WIDTH_LIMITS`, and the tick is what the comparison is made on,
        at one decimal place -- that is the resolution a painter draws at, and every
        finer difference would be a rebuild nobody could see.
        """
        width, height = self.pixel_size()
        scale = min(width, height) / REFERENCE_VIEWPORT
        low, high = LINE_WIDTH_LIMITS
        fitted = min(high, max(low, TICK_PEN_WIDTH * scale))
        if round(fitted, 1) == round(self._line_width, 1):
            return False
        self._line_width = fitted
        self._axis_width = min(high, max(low, AXIS_PEN_WIDTH * scale))
        return True


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
