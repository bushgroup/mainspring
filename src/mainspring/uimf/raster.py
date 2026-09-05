"""Sparse frame to screen-sized image, in display coordinates.

The heatmap is never the frame; it is the frame reduced to one value per pixel of the
viewport, at most a few hundred thousand of them however large the file is. That
reduction happens **here, in the data layer**, because it is numpy work with no Qt in
it and because the viewer must be able to ask for it off the GUI thread.

**Rasterise in display coordinates.** A pixel column is a half-open m/z interval and a
pixel row a half-open arrival-time interval, so the mapping from a point's `(scan, bin)`
to its pixel is a `searchsorted` into the axis tables `Calibration.mz_axis` and
`scan_axis_ms`. Three things follow, and they are the reason the viewer stays simple:

* the image is **linear in display space by construction**, so pyqtgraph's
  `ImageItem.setRect` places it exactly, with no per-pixel transform and no warping of
  a bin-space image;
* **swapping the axes or switching to raw bin/scan units is a re-assignment of the two
  axis tables**, not a different code path -- `DisplayAxes` is the only thing that
  changes;
* an m/z axis that is quadratic in bin is handled once, in the table, rather than in
  every consumer.

Aggregation is `sum` by default and `max` on a toggle. `sum` is the honest default for
a zoomed-out view -- it conserves the total, so a full-range image's total equals the
sum of the file's stored `TIC` -- while `max` is what finds a single sharp feature that
one bin wide.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .calib import Calibration
from .frame import SparseFrame

__all__ = ["AGGREGATES", "DisplayAxes", "RasterResult", "profile", "rasterise"]

AGGREGATES = ("sum", "max")
"""How points falling in one pixel combine. Default `sum`; `max` on a toggle."""


@dataclass(frozen=True)
class DisplayAxes:
    """What the two screen axes mean for one frame: the tables, their units, their order.

    `x_edges` and `y_edges` are monotonic increasing edge tables, `n + 1` long for `n`
    source elements: m/z over bins and milliseconds over scans in the default view, bin
    and scan index when raw units are on. `swapped` records that x is the scan axis, so
    that a readout can name the axes without re-deriving which is which.

    Built once per (frame, unit, orientation) and cached with the calibration; a view
    change re-uses it, and only a file or a toggle rebuilds it.
    """

    x_edges: np.ndarray
    y_edges: np.ndarray
    x_label: str
    y_label: str
    swapped: bool = False

    @classmethod
    def build(
        cls,
        frame: SparseFrame,
        calibration: Calibration,
        average_tof_length_ns: float,
        raw_units: bool = False,
        swapped: bool = False,
    ) -> "DisplayAxes":
        """The default m/z-vs-arrival-time axes, or their raw and swapped variants.

        Arrives with the lab record's task 03.
        """
        raise NotImplementedError("display axes arrive with the lab record's task 03")


@dataclass(frozen=True)
class RasterResult:
    """One rendered image and everything a viewer must say about it.

    `image` is `(height, width)` float32 in display orientation, row 0 at `y_range[0]`.
    The readouts travel with the image rather than being recomputed from the view,
    because the panel must describe *the image on screen*, not a newer view the render
    has not caught up with. `max_intensity` is what the per-push readout divides by
    `Accumulations`, and it is a raw stored intensity even when the aggregate is `sum`,
    so the readout stays meaningful (lab record, task 01).
    """

    image: np.ndarray
    x_range: tuple[float, float]
    y_range: tuple[float, float]
    axes: DisplayAxes
    aggregate: str
    points_in_view: int
    tic_in_view: float
    max_intensity: float


def rasterise(
    frame: SparseFrame,
    axes: DisplayAxes,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    width: int,
    height: int,
    aggregate: str = "sum",
) -> RasterResult:
    """Reduce a frame to a `height x width` image over the given display window.

    One pass over the points inside the window; nothing dense is allocated beyond the
    image itself. `width` and `height` are the widget's pixel size, so the cost is set
    by the viewport rather than by the file.

    Arrives with the lab record's task 03.
    """
    raise NotImplementedError("rasterisation arrives with the lab record's task 03")


def profile(
    frame: SparseFrame,
    axes: DisplayAxes,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    along: str = "x",
    bins: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """The side plots: the in-view points summed onto one axis, as `(edges, values)`.

    `along="x"` gives the mass spectrum of the window and `along="y"` its arrival-time
    distribution, each restricted to the other axis's range so that the side plots
    always describe what is on screen. Arrives with the lab record's task 05.
    """
    raise NotImplementedError("axis profiles arrive with the lab record's task 05")
