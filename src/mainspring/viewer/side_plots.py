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
`x_plot` and `y_plot`, and not for a quantity: the swap toggle (task 06) then changes
what the roles mean and nothing else.

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

from .heatmap import AXIS_HEIGHT, AXIS_WIDTH

__all__ = ["SidePlots"]

_PEN = pg.mkPen(color=(190, 210, 255), width=1)


class SidePlots:
    """The mass spectrum below and the arrival-time distribution beside the heatmap.

    Not a widget. The two `PlotItem`s live in the heatmap's own `GraphicsLayout`, in the
    cells `HeatmapView.side_plot_slots()` reserves, because a linked axis that is not
    also *aligned* on screen looks like a bug -- and alignment is a property of one
    layout, not of two widgets side by side.
    """

    def __init__(self, heatmap: "object") -> None:
        layout, y_cell, x_cell = heatmap.side_plot_slots()
        plot_item = heatmap.plot_item

        self.y_plot = layout.addPlot(row=y_cell[0], col=y_cell[1])
        self.x_plot = layout.addPlot(row=x_cell[0], col=x_cell[1])
        for plot in (self.y_plot, self.x_plot):
            plot.setMenuEnabled(False)
            plot.setMouseEnabled(x=False, y=False)  # the heatmap is what a gesture drives
            plot.showGrid(x=False, y=False)
            plot.hideButtons()

        # Each plot shows only the axis the heatmap does not already show for it: the
        # heatmap's own left axis sits between `y_plot` and the image and reads for both,
        # and its bottom axis sits between the image and `x_plot`.
        self.y_plot.hideAxis("left")
        self.y_plot.setYLink(plot_item)
        self.x_plot.hideAxis("bottom")
        self.x_plot.setXLink(plot_item)
        # The heatmap fixes its own axis extents for exactly this: a linked axis that
        # is not also aligned in pixels reads as a broken layout rather than a link.
        self.x_plot.getAxis("left").setWidth(AXIS_WIDTH)
        self.y_plot.getAxis("bottom").setHeight(AXIS_HEIGHT)

        self._x_curve = self.x_plot.plot(pen=_PEN)
        # Peak-preserving downsampling, and clipping to the visible window, both work
        # along a curve's x axis. That is the display axis for the mass spectrum, where a
        # full-range view is 114688 points and a one-bin spike must survive being drawn --
        # and it is the *intensity* axis for the arrival-time plot, where neither would
        # mean anything and where there are only ever a few thousand points anyway.
        self._x_curve.setDownsampling(auto=True, method="peak")
        self._x_curve.setClipToView(True)
        self._y_curve = self.y_plot.plot(pen=_PEN)

    @property
    def curves(self) -> "tuple[pg.PlotDataItem, pg.PlotDataItem]":
        """`(x_curve, y_curve)`, for a caller that wants to read back what is drawn."""
        return self._x_curve, self._y_curve

    def set_axes(self, axes: "object") -> None:
        """Label the two plots for the roles the current `DisplayAxes` gives them."""
        self.x_plot.setLabel("left", "Intensity")
        self.y_plot.setLabel("bottom", "Intensity")
        self.x_plot.setLabel("bottom", axes.x_label)
        self.y_plot.setLabel("left", axes.y_label)

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
