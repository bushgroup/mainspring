"""Sparse frame to screen-sized image, in display coordinates.

The heatmap is never the frame; it is the frame reduced to one value per pixel of the
viewport, at most a few hundred thousand of them however large the file is. That
reduction happens **here, in the data layer**, because it is numpy work with no Qt in
it and because the viewer must be able to ask for it off the GUI thread.

**Rasterise in display coordinates.** A pixel column is a half-open m/z interval and a
pixel row a half-open arrival-time interval, and a point's pixel is the display
coordinate of its bin or scan -- read out of the axis tables `Calibration.mz_axis` and
`scan_axis_ms` -- divided by the pixel width. (A uniform pixel grid is what makes that
a division rather than the `searchsorted` the design sketched: the search into a table
of equally spaced edges *is* the division, and the table would be 114689 entries long.)
Three things follow, and they are the reason the viewer stays simple:

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

from .calib import Calibration, scan_axis_ms
from .frame import SparseFrame

__all__ = [
    "AGGREGATES", "DisplayAxes", "RasterResult", "profile", "rasterise", "render_view",
]

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

    @property
    def bin_edges(self) -> np.ndarray:
        """Whichever of the two tables runs over TOF bins."""
        return self.y_edges if self.swapped else self.x_edges

    @property
    def scan_edges(self) -> np.ndarray:
        """Whichever of the two tables runs over scans."""
        return self.x_edges if self.swapped else self.y_edges

    @property
    def full_range(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """`(x_range, y_range)` covering the whole frame: what a reset zooms back to."""
        return (
            (float(self.x_edges[0]), float(self.x_edges[-1])),
            (float(self.y_edges[0]), float(self.y_edges[-1])),
        )

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

        Raw units fall back to bin and scan index. So does a calibration that cannot
        produce an axis -- a frame the writer never calibrated, or one whose slope is
        missing -- because a plausible-looking m/z axis over uncalibrated data is worse
        than an honest bin axis, and `Calibration.done` is what the info panel reports.
        The same applies to arrival time when `AverageTOFLength` is absent.
        """
        if raw_units or not calibration.usable:
            bin_edges = np.arange(frame.bins + 1, dtype=np.float64)
            bin_label = "TOF bin"
        else:
            bin_edges = calibration.mz_axis(frame.bins)
            bin_label = "m/z"
        if raw_units or average_tof_length_ns <= 0.0:
            scan_edges = np.arange(frame.scans + 1, dtype=np.float64)
            scan_label = "Scan"
        else:
            scan_edges = scan_axis_ms(frame.scans, float(average_tof_length_ns))
            scan_label = "Arrival time (ms)"
        if swapped:
            return cls(scan_edges, bin_edges, scan_label, bin_label, True)
        return cls(bin_edges, scan_edges, bin_label, scan_label, False)


@dataclass(frozen=True)
class RasterResult:
    """One rendered image and everything a viewer must say about it.

    `image` is `(height, width)` float32 in display orientation, row 0 at `y_range[0]`.
    The readouts travel with the image rather than being recomputed from the view,
    because the panel must describe *the image on screen*, not a newer view the render
    has not caught up with. `max_intensity` is what the per-push readout divides by
    `Accumulations`, and it is a raw stored intensity even when the aggregate is `sum`,
    so the readout stays meaningful (lab record, task 01).

    `image` is float32 because it is a display product and Qt wants it that way; the
    numbers anything is quoted from are `tic_in_view` and `max_intensity`, which are
    float64 and exact. Summing the image instead will agree to about seven digits and
    no further.
    """

    image: np.ndarray
    x_range: tuple[float, float]
    y_range: tuple[float, float]
    axes: DisplayAxes
    aggregate: str
    points_in_view: int
    tic_in_view: float
    max_intensity: float


def _axis_cells(
    edges: np.ndarray, low: float, high: float, max_cells: int
) -> tuple[int, int, np.ndarray, int]:
    """Which source elements fall in `[low, high)`, and the cell each of them lands in.

    Returns `(index_lo, index_hi, cell_of_index, cells)` with `cell_of_index` parallel to
    `range(index_lo, index_hi)`. An element is placed by the **centre** of its interval,
    so that every element of a full-range view lands inside it and the total is conserved.

    The cell is a floor division and not a `searchsorted`, because a pixel grid is
    uniform in display units by definition: the search into a uniform table *is* the
    division, and the table is 114689 entries long on a SLIMPHONY file.

    `cells` is `min(max_cells, elements in view)`: zoomed in past one element per pixel
    there is nothing to gain from a wider image, and the viewer stretches what it gets.
    """
    if not high > low:
        raise ValueError(f"empty display range [{low}, {high})")
    centres = 0.5 * (edges[:-1] + edges[1:])
    # A binary search and not a mask: an edge table is monotonic by construction (the
    # calibration is clamped so that it never decreases -- lab record, task 03), so the
    # elements in view are a contiguous run, and finding its ends by comparing all
    # 148000 of them costs more than the render it is part of.
    index_lo = int(np.searchsorted(centres, low, side="left"))
    index_hi = int(np.searchsorted(centres, high, side="left"))
    if index_hi <= index_lo:
        return 0, 0, np.empty(0, dtype=np.int32), max(1, int(max_cells))
    cells = max(1, min(int(max_cells), index_hi - index_lo))
    cell = np.floor((centres[index_lo:index_hi] - low) * (cells / (high - low)))
    return index_lo, index_hi, np.clip(cell, 0, cells - 1).astype(np.int32), cells


def _points_in_view(
    frame: SparseFrame, scan_lo: int, scan_hi: int, bin_lo: int, bin_hi: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`(scan_index, bin_index, intensity)` of the points inside the window.

    The scan window is a CSR slice -- one pair of row pointers -- and the bin window is
    a mask over what that slice holds. Nothing dense is built and nothing is copied
    until the mask is applied.
    """
    start = int(frame.scan_start[scan_lo])
    stop = int(frame.scan_start[scan_hi])
    bin_index = frame.bin_index[start:stop]
    intensity = frame.intensity[start:stop]
    scan_index = np.repeat(
        np.arange(scan_lo, scan_hi, dtype=np.int32),
        np.diff(frame.scan_start[scan_lo:scan_hi + 1]),
    )
    keep = (bin_index >= bin_lo) & (bin_index < bin_hi)
    if keep.all():
        return scan_index, bin_index, intensity
    return scan_index[keep], bin_index[keep], intensity[keep]


def _max_into(flat: np.ndarray, values: np.ndarray, size: int) -> np.ndarray:
    """Per-cell maximum of `values` at indices `flat`, as a float64 array of `size`.

    Sorted and reduced rather than `np.maximum.at`, which is an unbuffered ufunc loop
    and roughly two orders of magnitude slower on the point counts a frame carries.
    """
    out = np.zeros(size, dtype=np.float64)
    if flat.size == 0:
        return out
    order = np.argsort(flat, kind="stable")
    sorted_flat = flat[order]
    group_start = np.concatenate(
        (np.zeros(1, dtype=np.int64), np.flatnonzero(np.diff(sorted_flat)) + 1)
    )
    out[sorted_flat[group_start]] = np.maximum.reduceat(values[order], group_start)
    return out


@dataclass(frozen=True)
class _View:
    """The points in a display window, and where each of them lands on each axis.

    Everything the image and the two profiles need, and the only expensive part of
    producing any of them: two `_axis_cells` and one pass over the frame's points.
    Building it once is what `render_view` is for.

    `x_source` and `y_source` are the source element indices along each display axis --
    bins along x and scans along y in the default orientation, the other way round when
    the axes are swapped -- so nothing downstream has to ask which is which again.
    """

    x_lo: int
    x_hi: int
    x_cell: np.ndarray
    cols: int
    y_lo: int
    y_hi: int
    y_cell: np.ndarray
    rows: int
    x_source: np.ndarray
    y_source: np.ndarray
    intensity: np.ndarray

    @property
    def empty(self) -> bool:
        """No points in the window -- a legitimate view, not an error."""
        return self.intensity.size == 0


def _view(
    frame: SparseFrame,
    axes: DisplayAxes,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    width: int,
    height: int,
) -> "_View":
    """Select the window's points once, in display-axis terms."""
    x_lo, x_hi, x_cell, cols = _axis_cells(axes.x_edges, x0, x1, width)
    y_lo, y_hi, y_cell, rows = _axis_cells(axes.y_edges, y0, y1, height)
    if x_hi == x_lo or y_hi == y_lo:
        none = np.empty(0, dtype=np.int64)
        return _View(x_lo, x_hi, x_cell, cols, y_lo, y_hi, y_cell, rows,
                     none, none, np.empty(0, dtype=frame.intensity.dtype))
    scan_lo, scan_hi = (x_lo, x_hi) if axes.swapped else (y_lo, y_hi)
    bin_lo, bin_hi = (y_lo, y_hi) if axes.swapped else (x_lo, x_hi)
    scan_index, bin_index, intensity = _points_in_view(
        frame, scan_lo, scan_hi, bin_lo, bin_hi
    )
    x_source, y_source = (scan_index, bin_index) if axes.swapped else (bin_index, scan_index)
    return _View(x_lo, x_hi, x_cell, cols, y_lo, y_hi, y_cell, rows,
                 x_source, y_source, intensity)


def _image_of(view: "_View", aggregate: str) -> np.ndarray:
    """The `rows x cols` image, float32 and in display orientation."""
    if view.empty:
        return np.zeros((view.rows, view.cols), dtype=np.float32)
    column = view.x_cell[view.x_source - view.x_lo]
    row = view.y_cell[view.y_source - view.y_lo]
    flat = row.astype(np.int64) * view.cols + column
    size = view.rows * view.cols
    if aggregate == "sum":
        image = np.bincount(flat, weights=view.intensity, minlength=size)
    else:
        image = _max_into(flat, view.intensity.astype(np.float64), size)
    return image.reshape(view.rows, view.cols).astype(np.float32)


def _result_of(
    view: "_View",
    axes: DisplayAxes,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    aggregate: str,
) -> RasterResult:
    return RasterResult(
        image=_image_of(view, aggregate),
        x_range=x_range,
        y_range=y_range,
        axes=axes,
        aggregate=aggregate,
        points_in_view=int(view.intensity.size),
        tic_in_view=float(np.sum(view.intensity, dtype=np.float64)),
        max_intensity=0.0 if view.empty else float(view.intensity.max()),
    )


def _native_profile(
    view: "_View", axes: DisplayAxes, along: str
) -> tuple[np.ndarray, np.ndarray]:
    """One value per source element in view, on that element's own edges."""
    edges_table = axes.x_edges if along == "x" else axes.y_edges
    lo, hi = (view.x_lo, view.x_hi) if along == "x" else (view.y_lo, view.y_hi)
    source = view.x_source if along == "x" else view.y_source
    edges = np.asarray(edges_table[lo:hi + 1], dtype=np.float64)
    cells = max(0, hi - lo)
    if cells == 0 or view.empty:
        return edges, np.zeros(max(cells, 0), dtype=np.float64)
    return edges, np.bincount(
        (source - lo).astype(np.int64), weights=view.intensity, minlength=cells
    )


def render_view(
    frame: SparseFrame,
    axes: DisplayAxes,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    width: int,
    height: int,
    aggregate: str = "sum",
) -> tuple[RasterResult, tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """The image and both native-resolution profiles of one window, from one selection.

    Returns `(result, x_profile, y_profile)`, each profile the `(edges, values)` pair
    `profile` returns. Identical to calling `rasterise` and `profile` twice, and several
    times faster on a full-range view of a dense frame: the expensive part of all three
    is selecting the points and deriving the axis cells, and that happens once here
    rather than three times.

    This is what an interactive viewer's render path should call. The three pictures are
    one window's, they are wanted together, and computing them together is both cheaper
    and the only way they cannot disagree.
    """
    if aggregate not in AGGREGATES:
        raise ValueError(f"unknown aggregate {aggregate!r}; one of {AGGREGATES}")
    x0, x1 = float(x_range[0]), float(x_range[1])
    y0, y1 = float(y_range[0]), float(y_range[1])
    view = _view(frame, axes, x0, x1, y0, y1, width, height)
    return (
        _result_of(view, axes, (x0, x1), (y0, y1), aggregate),
        _native_profile(view, axes, "x"),
        _native_profile(view, axes, "y"),
    )


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

    The returned image spans exactly `x_range` by `y_range`, which is what lets the
    viewer place it with a single `setRect` and never resample it.
    """
    if aggregate not in AGGREGATES:
        raise ValueError(f"unknown aggregate {aggregate!r}; one of {AGGREGATES}")
    x0, x1 = float(x_range[0]), float(x_range[1])
    y0, y1 = float(y_range[0]), float(y_range[1])
    view = _view(frame, axes, x0, x1, y0, y1, width, height)
    return _result_of(view, axes, (x0, x1), (y0, y1), aggregate)


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
    always describe what is on screen. `len(edges) == len(values) + 1`.

    `bins=0`, the default, is **native resolution**: one value per source element in
    view, on that element's own edges. That is what a mass spectrum has to be -- a
    spectrum resampled onto a pixel grid has had its peaks widened before anyone looks
    at them -- and it costs nothing, because it is the same `bincount` over a slice of
    the axis table. A positive `bins` divides the display range uniformly instead, for
    a caller that wants a fixed-width plot.

    Summing this against `RasterResult.tic_in_view` for the same window is an equality,
    not an approximation: both are the same points added up.
    """
    if along not in ("x", "y"):
        raise ValueError(f"along must be 'x' or 'y', got {along!r}")
    x0, x1 = float(x_range[0]), float(x_range[1])
    y0, y1 = float(y_range[0]), float(y_range[1])

    view = _view(frame, axes, x0, x1, y0, y1, 1, 1)
    if bins <= 0:
        return _native_profile(view, axes, along)

    edges_table = axes.x_edges if along == "x" else axes.y_edges
    lo, hi = (view.x_lo, view.x_hi) if along == "x" else (view.y_lo, view.y_hi)
    low, high = (x0, x1) if along == "x" else (y0, y1)
    source = view.x_source if along == "x" else view.y_source
    edges = np.linspace(low, high, int(bins) + 1)
    cells = int(bins)
    if view.empty or hi == lo:
        return edges, np.zeros(cells, dtype=np.float64)

    centres = 0.5 * (edges_table[lo:hi] + edges_table[lo + 1:hi + 1])
    cell_of = np.clip(
        np.floor((centres - low) * (cells / (high - low))), 0, cells - 1
    ).astype(np.int64)
    cell = cell_of[source - lo]
    return edges, np.bincount(cell, weights=view.intensity, minlength=cells)
