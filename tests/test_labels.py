"""Task 24: the display spelling of an axis name, against the rasteriser's own.

The table in `viewer/labels.py` is keyed by the strings `DisplayAxes.build` produces, and
the two lists are held together here rather than by hope: an edit to `raster.py` that
renamed one of them would otherwise leave the axis silently falling back to the raw
string, which is the failure this file exists to make loud.
"""

from __future__ import annotations

import numpy as np
import pytest

from mainspring.uimf import Calibration, SparseFrame
from mainspring.uimf.raster import DisplayAxes
from mainspring.viewer import labels


def _frame(bins: int = 64, scans: int = 8) -> SparseFrame:
    return SparseFrame(
        frame=1,
        scans=scans,
        bins=bins,
        scan_start=np.zeros(scans + 1, dtype=np.int64),
        bin_index=np.empty(0, dtype=np.int32),
        intensity=np.empty(0, dtype=np.int32),
    )


def test_every_name_the_rasteriser_produces_has_a_display_spelling():
    """All four, taken from `DisplayAxes.build` itself rather than typed out again."""
    frame = _frame()
    calibrated = Calibration(slope=0.738123, intercept=0.0769, bin_width_ns=1.0)
    produced = set()
    for raw_units in (False, True):
        axes = DisplayAxes.build(
            frame, calibrated, average_tof_length_ns=129003.6, raw_units=raw_units
        )
        produced |= {axes.x_label, axes.y_label}

    assert produced == set(labels.DISPLAY_NAMES)


@pytest.mark.parametrize("name", ["m/z", "TOF bin", "Arrival time (ms)", "Scan"])
def test_neither_spelling_is_empty(name):
    assert labels.plain(name) and labels.html(name)


def test_mz_is_italic_on_an_axis_and_plain_in_the_status_bar():
    """A quantity symbol is italic wherever it is printed properly, and a lone italic
    `m` beside a value in a line of numbers reads as an error rather than as typesetting."""
    assert labels.html("m/z") == "<i>m</i>/<i>z</i>"
    assert labels.plain("m/z") == "m/z"


def test_arrival_time_reads_as_quantity_solidus_unit():
    """How an axis is labelled in a paper, which is where a figure from this viewer goes.
    `raster.py` keeps the parenthesised form, which is how a variable is named in code."""
    assert labels.plain("Arrival time (ms)") == "Arrival Time / ms"
    assert labels.html("Arrival time (ms)") == "Arrival Time / ms"


def test_an_unknown_name_passes_through_escaped():
    """A writer whose frames produce an axis nobody anticipated gets an honest label."""
    assert labels.plain("Drift <voltage>") == "Drift <voltage>"
    assert labels.html("Drift <voltage>") == "Drift &lt;voltage&gt;"


def test_the_heatmap_sets_the_html_form_on_its_axes(qtbot):
    from mainspring.viewer.heatmap import HeatmapView
    from mainspring.uimf.raster import RasterResult

    view = HeatmapView()
    qtbot.addWidget(view)
    axes = DisplayAxes(
        x_edges=np.array([0.0, 1.0]), y_edges=np.array([0.0, 1.0]),
        x_label="m/z", y_label="Arrival time (ms)",
    )
    view.set_image(RasterResult(
        image=np.array([[1.0]]), x_range=(0.0, 1.0), y_range=(0.0, 1.0), axes=axes,
        aggregate="sum", points_in_view=1, tic_in_view=1.0, max_intensity=1.0,
    ))

    assert view.plot_item.getAxis("bottom").labelText == "<i>m</i>/<i>z</i>"
    assert view.plot_item.getAxis("left").labelText == "Arrival Time / ms"
