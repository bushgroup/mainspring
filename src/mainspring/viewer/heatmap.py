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


class HeatmapView:
    """The image, its axes, and its colour bar, as one widget.

    A `pyqtgraph.GraphicsLayoutWidget` once it exists. Arrives with the lab record's
    task 04.
    """

    def __init__(self) -> None:
        raise NotImplementedError("the heatmap arrives with the lab record's task 04")

    def set_image(self, result: object) -> None:
        """Show a `RasterResult`, placing it by its display ranges. Arrives with task 04."""
        raise NotImplementedError("the heatmap arrives with the lab record's task 04")

    def set_levels(self, low: float, high: float) -> None:
        """Pin the colour levels, for the keep-levels setting. Arrives with task 06."""
        raise NotImplementedError("held colour levels arrive with the lab record's task 06")
