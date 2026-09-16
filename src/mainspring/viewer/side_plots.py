"""The two side plots: the mass spectrum and the arrival-time distribution in view.

Both are projections of **what is on screen**, not of the whole frame: the mass
spectrum sums the points inside the current arrival-time range and the arrival-time
plot sums those inside the current m/z range, so narrowing one axis sharpens the other
plot rather than leaving it describing a window the user has left. That linkage is the
reason the two plots earn their space; a static full-frame spectrum beside a zoomed
heatmap tells you nothing you did not already know.

Axes are shared with the heatmap in the obvious way -- the mass spectrum's x is the
heatmap's x, the arrival-time plot's y is the heatmap's y -- and they stay shared when
the axes are swapped, which is why the plots take their orientation from `DisplayAxes`
rather than from their own idea of which is which. Everything here is named for a role,
`x_plot` and `y_plot`, and not for a quantity: the swap-axes toggle
(`main_window._rebuild_axes`) then changes what the roles mean and nothing else. The
plot above the image is always the projection onto the horizontal axis and the plot
beside it always the projection onto the vertical one, whatever those axes are.

**The projections are bare curves.** No axis, no label, no tick value: the heatmap's own
axes, which the plots are aligned to, already say what the shared axis is, and the
intensity axis of a projection of *what is on screen* is a number that changes with
every gesture and means little on its own. What the curve's shape says -- where the
peaks are, and how they sharpen as the other axis narrows -- is the whole point.

Their projections come from `mainspring.uimf.raster.profile`, on the render worker with
the image, so a gesture produces one consistent set of three pictures rather than three
that arrive at different times.

**Native resolution, not the pixel grid.** `profile` gives one value per source element
in view, so a spectrum's peaks are the file's peaks and not a resampling of them, and
pyqtgraph's peak-preserving downsampler is what reduces 114688 points to a drawable
curve without flattening a one-bin spike. Doing that in the plot rather than in the
projection is what makes zooming in *reveal* structure: the data are already there, and
the downsampler simply stops discarding it.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QPointF, QRectF, Qt

from . import fonts, theme
from .controls import describe
from .heatmap import AXIS_HEIGHT, AXIS_WIDTH, RIGHT_AXIS_WIDTH, TOP_AXIS_HEIGHT

__all__ = ["ProjectionViewBox", "SidePlots"]

MIN_BAND_PIXELS = 2.0
"""How far a drag has to travel along the shared axis before it is a band rather than a
click that wobbled. The same number `UimfViewBox._rect_zoom` uses on the heatmap, so one
hand produces one result wherever it presses."""

CURVE_WIDTH = 1
"""How thick a projection is drawn. One pixel, so that a spectrum reduced by the
peak-preserving downsampler still shows a single-bin spike as a spike rather than as a
blob; the colour is the palette's (`theme.py`)."""

# A `PlotItem`'s own grid: the axes and the view box sit in fixed cells (title row 0;
# top axis (1, 1); left axis (2, 0); view box (2, 1); right axis (2, 2); bottom axis
# (3, 1)). A hidden axis is zero-sized, so aligning a bare plot with the heatmap means
# fixing these rows and columns to the heatmap's axis extents directly.
_LEFT_AXIS_COLUMN = 0
_RIGHT_AXIS_COLUMN = 2
_TOP_AXIS_ROW = 1
_BOTTOM_AXIS_ROW = 3


class ProjectionViewBox(pg.ViewBox):
    """The box under one projection, where a drag zooms the axis it shares.

    A projection is where the peak you want is *visible*: the spectrum resolves an
    isotope pattern that the heatmap draws as one column, and picking the pattern off
    the curve is the natural gesture. Until task 24 these boxes had the mouse disabled
    entirely, so the pointer over a projection did nothing at all -- the right button was
    free, and this is what it now means.

    Same contract as `UimfViewBox`, so there is one gesture to learn and not two: a
    right-drag, or Shift and a left-drag, sets a range; a double-click resets. What
    differs is the dimension. The drag is read on the **shared** axis alone and the band
    is drawn across the whole plot on the other, because the other one is intensity,
    which the projection auto-ranges and which nobody means to zoom. A plain left drag
    and the wheel stay inert, since panning a curve away from the image it belongs to
    would only break the alignment the two are built around.

    The box does not act on itself. `axis` is the heatmap's axis index (0 for the
    horizontal, 1 for the vertical) and the two callbacks reach `HeatmapView`, which owns
    the range the projections are linked to -- so a band drawn here travels the same path
    to the render worker as a gesture on the image, and `setLimits` clamps it the same
    way (lab record, task 24).
    """

    def __init__(self, axis: int, zoom, reset, parent: "object | None" = None) -> None:
        super().__init__(parent=parent, enableMenu=False, defaultPadding=0.0)
        self._axis = int(axis)
        self._zoom = zoom
        self._reset = reset

    # --- gestures ---------------------------------------------------------------------

    def mouseDragEvent(self, ev, axis=None) -> None:
        """Right drag, or Shift and a left drag, bands the shared axis. Nothing else."""
        button = ev.button()
        shift = bool(ev.modifiers() & Qt.KeyboardModifier.ShiftModifier)
        if button == Qt.MouseButton.RightButton or (
            button == Qt.MouseButton.LeftButton and shift
        ):
            self._band_zoom(ev)
            return
        ev.ignore()

    def mouseClickEvent(self, ev) -> None:
        """A double-click resets this axis alone; a single click does nothing."""
        if ev.double():
            ev.accept()
            self._reset(self._axis)
            return
        ev.ignore()

    def wheelEvent(self, ev, axis=None) -> None:
        """Inert, explicitly. The heatmap is where a wheel zooms, and a projection that
        scrolled its own intensity axis would leave a curve that no longer describes the
        image beside it."""
        ev.ignore()

    def _band_zoom(self, ev) -> None:
        """The band while the button is down, and the range it means once it comes up."""
        ev.accept()
        start = ev.buttonDownPos(ev.button())
        corners = self._corners(start, ev.pos())
        if not ev.isFinish():
            self.updateScaleBox(*corners)
            return
        self.rbScaleBox.hide()
        travelled = (
            abs(ev.pos().x() - start.x()) if self._axis == 0 else abs(ev.pos().y() - start.y())
        )
        if travelled <= MIN_BAND_PIXELS:
            return  # a click that wobbled, not a band; leave the view alone
        band = self.childGroup.mapRectFromParent(QRectF(*corners).normalized())
        lo, hi = (band.left(), band.right()) if self._axis == 0 else (band.top(), band.bottom())
        self._zoom(self._axis, min(lo, hi), max(lo, hi))

    def _corners(self, start: QPointF, end: QPointF) -> "tuple[QPointF, QPointF]":
        """The drag as a band: bounded by the pointer on the shared axis, and by the
        box's own extent on the other, so what is drawn is what will be applied."""
        rect = self.boundingRect()
        if self._axis == 0:
            return QPointF(start.x(), rect.top()), QPointF(end.x(), rect.bottom())
        return QPointF(rect.left(), start.y()), QPointF(rect.right(), end.y())


class SidePlots:
    """The mass spectrum above and the arrival-time distribution beside the heatmap.

    Not a widget. The two `PlotItem`s live in the heatmap's own `GraphicsLayout`, in the
    cells `HeatmapView.side_plot_slots()` reserves, because a linked axis that is not
    also *aligned* on screen looks like a bug -- and alignment is a property of one
    layout, not of two widgets side by side.
    """

    def __init__(self, heatmap: "object") -> None:
        layout, y_cell, x_cell = heatmap.side_plot_slots()
        plot_item = heatmap.plot_item

        # Each projection drives the heatmap axis it is linked to, and nothing else:
        # `y_plot` beside the image shares the vertical axis (1), `x_plot` above it the
        # horizontal (0). The heatmap owns the ranges, so the boxes are handed its two
        # methods rather than a reference to it.
        self.y_plot = layout.addPlot(
            row=y_cell[0],
            col=y_cell[1],
            viewBox=ProjectionViewBox(1, heatmap.zoom_axis, heatmap.reset_axis),
        )
        self.x_plot = layout.addPlot(
            row=x_cell[0],
            col=x_cell[1],
            viewBox=ProjectionViewBox(0, heatmap.zoom_axis, heatmap.reset_axis),
        )
        for plot in (self.y_plot, self.x_plot):
            plot.setMenuEnabled(False)
            # Pan and wheel off: `ProjectionViewBox` handles the two gestures it does
            # answer before pyqtgraph's own handling is ever reached.
            plot.setMouseEnabled(x=False, y=False)
            plot.showGrid(x=False, y=False)
            plot.hideButtons()
            for name in ("left", "right", "top", "bottom"):
                plot.hideAxis(name)

        # Linked on the shared axis, auto-ranged on the other: linking already turns
        # auto-range off for the shared axis, and the intensity axis must follow the
        # data since a projection of the visible window rescales with every gesture.
        self.x_plot.setXLink(plot_item)
        self.x_plot.enableAutoRange(x=False, y=True)
        self.y_plot.setYLink(plot_item)
        self.y_plot.enableAutoRange(x=True, y=False)

        # The heatmap fixes its own axis extents for exactly this: a linked axis that
        # is not also aligned in pixels reads as a broken layout rather than a link.
        # `x_plot` shares the heatmap's column, so its left and right margins mirror
        # the heatmap's left and right axes; `y_plot` shares its row, so its top and
        # bottom margins mirror the heatmap's top and bottom axes.
        self.x_plot.layout.setColumnFixedWidth(_RIGHT_AXIS_COLUMN, RIGHT_AXIS_WIDTH)
        self.y_plot.layout.setRowFixedHeight(_TOP_AXIS_ROW, TOP_AXIS_HEIGHT)
        # The two that are not zero follow the text size, on both sides of the agreement
        # (`set_text_scale`, called at the end of this constructor).

        # Named by role, tipped by role: the swap-axes toggle changes what each one
        # projects onto, so neither tooltip may name a quantity.
        _ZOOM = (
            " Right-drag, or Shift and a left-drag, to zoom that axis alone, and"
            " double-click to reset it."
        )
        describe(
            self.x_plot,
            "Total intensity along the horizontal axis, over the vertical range in view."
            + _ZOOM,
        )
        describe(
            self.y_plot,
            "Total intensity along the vertical axis, over the horizontal range in view."
            + _ZOOM,
        )

        self._x_curve = self.x_plot.plot()
        # Peak-preserving downsampling, and clipping to the visible window, both work
        # along a curve's x axis. That is the display axis for the mass spectrum, where a
        # full-range view is 114688 points and a one-bin spike must survive being drawn --
        # and it is the *intensity* axis for the arrival-time plot, where neither would
        # mean anything and where there are only ever a few thousand points anyway.
        self._x_curve.setDownsampling(auto=True, method="peak")
        self._x_curve.setClipToView(True)
        self._y_curve = self.y_plot.plot()
        self.set_palette(theme.active())
        self.set_text_scale(fonts.active())

    def set_text_scale(self, scale: float) -> None:
        """Follow `View > Text size`: the reserved margins, and the hidden axes' fonts.

        The projections carry no text of their own, so what a scale changes here is the
        space they reserve to stay aligned with the heatmap, which does. The two numbers
        are the heatmap's, multiplied by the same `fonts.extent()` the heatmap applies,
        because alignment is an agreement between the two and half of an agreement is a
        layout bug (`heatmap.AXIS_WIDTH`).

        The eight hidden axes are given the font as well, for the reason `set_palette`
        gives them the palette: each was born with whatever was active when its plot was
        built, and a size that is only wrong while it cannot be seen is the kind
        `fonts.sized` exists to refuse to let through.
        """
        self.x_plot.layout.setColumnFixedWidth(
            _LEFT_AXIS_COLUMN, round(AXIS_WIDTH * fonts.extent())
        )
        self.y_plot.layout.setRowFixedHeight(
            _BOTTOM_AXIS_ROW, round(AXIS_HEIGHT * fonts.extent())
        )
        font = fonts.scaled_font(scale)
        for plot in (self.x_plot, self.y_plot):
            for name in ("left", "right", "top", "bottom"):
                plot.getAxis(name).setTickFont(font)

    def set_palette(self, palette: "theme.Palette") -> None:
        """Repaint both curves in `palette`, live.

        Called at construction and again on every `View > Light mode` toggle
        (`theme.apply`, the one path). `PlotDataItem.setPen` pushes the pen straight
        through to the curve item that draws it, so a projection already on screen
        changes colour without being asked for its data again.

        The eight hidden axes are set as well. Nothing draws them today, but each was
        born holding the palette that was active when its plot was built, and a colour
        that is only wrong while it cannot be seen is the kind `theme.themed` exists to
        refuse to let through.
        """
        pen = pg.mkPen(palette.foreground)
        for plot in (self.x_plot, self.y_plot):
            for name in ("left", "right", "top", "bottom"):
                axis = plot.getAxis(name)
                axis.setPen(pen)
                axis.setTickPen(pen)
                axis.setTextPen(pen)
        for curve in self.curves:
            curve.setPen(pg.mkPen(palette.curve, width=CURVE_WIDTH))

    @property
    def curves(self) -> "tuple[pg.PlotDataItem, pg.PlotDataItem]":
        """`(x_curve, y_curve)`, for a caller that wants to read back what is drawn."""
        return self._x_curve, self._y_curve

    def set_profiles(
        self,
        x_profile: "tuple[np.ndarray, np.ndarray]",
        y_profile: "tuple[np.ndarray, np.ndarray]",
    ) -> None:
        """Show a matched pair of projections, each as `(edges, values)`.

        Drawn on element centres rather than as a step: a step needs one more x than y,
        which pyqtgraph supports along x and not along y, and drawing the two plots by
        two different rules to save a half-element offset would be the wrong trade.
        """
        x_edges, x_values = x_profile
        y_edges, y_values = y_profile
        self._x_curve.setData(_centres(x_edges), np.asarray(x_values, dtype=np.float64))
        # x and y swap for the arrival-time plot: intensity runs across, the display
        # axis down, so that its vertical axis is the heatmap's and the link is honest.
        self._y_curve.setData(np.asarray(y_values, dtype=np.float64), _centres(y_edges))

    def clear(self) -> None:
        """Empty both plots, for a file being closed."""
        empty = np.empty(0, dtype=np.float64)
        self._x_curve.setData(empty, empty)
        self._y_curve.setData(empty, empty)


def _centres(edges: "np.ndarray") -> "np.ndarray":
    """Element centres from an `n + 1` edge table, or an empty array from an empty one."""
    edges = np.asarray(edges, dtype=np.float64)
    if edges.size < 2:
        return np.empty(0, dtype=np.float64)
    return 0.5 * (edges[:-1] + edges[1:])
