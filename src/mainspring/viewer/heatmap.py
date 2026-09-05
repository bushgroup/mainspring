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
be a single key, always available, and never lose the frame you were on.

View changes are debounced and sent to the render worker's mailbox, so a continuous
gesture repaints at the rate the rasteriser can sustain and always with the newest view
(`workers.py`). The axis items show display units, and the colour bar's levels are
either auto-scaled per image or held, depending on the keep-levels setting.
"""

from __future__ import annotations

import pyqtgraph as pg
from PySide6.QtCore import Signal

__all__ = ["HeatmapView", "UimfViewBox"]


class UimfViewBox:
    """A `pyqtgraph.ViewBox` with the one-gesture zoom, pan and reset above.

    Arrives with the lab record's task 05; `heatmap.py` gets a plain box in task 04 so
    that a frame can be seen before the interaction is built.
    """

    def __init__(self) -> None:
        raise NotImplementedError("the custom view box arrives with the lab record's task 05")

    def reset_range(self) -> None:
        """Back to the frame's full m/z and arrival-time range. Arrives with task 05."""
        raise NotImplementedError("the custom view box arrives with the lab record's task 05")


class HeatmapView(pg.GraphicsLayoutWidget):
    """The image, its axes, and its colour bar, as one widget.

    Plain `pyqtgraph` defaults: a `PlotItem` with its stock `ViewBox`, an `ImageItem`
    placed by `setRect` and never resampled, and a `ColorBarItem` tracking it. Task 05
    swaps the `ViewBox` for `UimfViewBox`; nothing else here changes.
    """

    view_resized = Signal(int, int)
    """Emitted with the viewport's pixel size on every resize, so the owner can
    re-rasterise at the new size (`MainWindow`, which holds the frame this needs)."""

    def __init__(self, colour_map: str = "viridis", parent: "object | None" = None) -> None:
        super().__init__(parent=parent)
        self._plot = self.addPlot()
        self._plot.showGrid(x=False, y=False)
        self._plot.setMenuEnabled(False)
        self._image_item = pg.ImageItem()
        self._plot.addItem(self._image_item)
        self._colour_bar = pg.ColorBarItem(colorMap=colour_map)
        self._colour_bar.setImageItem(self._image_item, insert_in=self._plot)

    @property
    def image_item(self) -> pg.ImageItem:
        """The underlying `ImageItem`, mostly so a test can read back its state."""
        return self._image_item

    def pixel_size(self) -> "tuple[int, int]":
        """`(width, height)` of the drawing area, in device pixels, at least 1x1.

        What `rasterise` should be asked for: the viewport rather than the whole widget,
        since the axes and the colour bar take some of it (lab record, task 04).
        """
        size = self.viewport().size()
        return max(1, size.width()), max(1, size.height())

    def set_image(self, result: object) -> None:
        """Show a `RasterResult`, placing it by its display ranges.

        Levels go through the colour bar's own `setLevels`, computed from this image,
        rather than through `ImageItem`'s `autoLevels` -- once a `ColorBarItem` is
        attached it owns the applied levels and does not follow an image's own
        auto-scaling (no `sigLevelsChanged` on plain `ImageItem` in this pyqtgraph
        version), so `autoLevels=True` here would silently keep showing whatever the
        colour bar's levels happened to be, which is 0-1 until told otherwise.
        """
        axes = result.axes
        x0, x1 = result.x_range
        y0, y1 = result.y_range
        self._image_item.setImage(result.image, autoLevels=False)
        self._image_item.setRect(x0, y0, x1 - x0, y1 - y0)
        self._plot.setLabel("bottom", axes.x_label)
        self._plot.setLabel("left", axes.y_label)
        low, high = float(result.image.min()), float(result.image.max())
        self._colour_bar.setLevels((low, high if high > low else low + 1.0))

    def set_levels(self, low: float, high: float) -> None:
        """Pin the colour levels, for the keep-levels setting. Arrives with task 06."""
        raise NotImplementedError("held colour levels arrive with the lab record's task 06")

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        width, height = self.pixel_size()
        self.view_resized.emit(width, height)
