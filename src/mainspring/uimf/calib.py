"""Bins to m/z, scans to arrival time: the two axis transforms and their lookup tables.

m/z is UIMF-Library's `MzCalibrator` and nothing more:

    t  = bin * BinWidth_ns / 1000            microseconds
    mz = (K * (t - T0)) ** 2                 K = CalibrationSlope, T0 = CalibrationIntercept

with the inverse `bin = (sqrt(mz) / K + T0) * 1000 / BinWidth_ns`. **`TimeOffset` is not
applied**, despite what its description in the file says, and the residual polynomial
`a2..f2` is zero in every file seen and unapplied by the library. Both were checked
against the writers' own `BPI_MZ` on four files (lab record, task 01): on a 2026
SLIMPHONY acquisition the implied bin comes out with a zero fractional part, which is
what settles the units and the constants. Deliberate divergence from the library:
none. See the lab record before adding a term here.

Arrival time is `t_ms = scan * AverageTOFLength_ns * 1e-6`, from the frame parameters.

Both transforms are monotonic and cheap, and the viewer needs them as **lookup tables**
rather than as scalar functions: rasterisation works in display coordinates, so it
wants `bin -> m/z` over the whole axis once and then indexes it. A `Calibration` is
frozen and hashable so those tables cache on `(K, T0, BinWidth)`, which is what makes a
per-frame calibration cost nothing in the common case where every frame shares one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["Calibration", "arrival_time_ms", "scan_axis_ms"]


@dataclass(frozen=True)
class Calibration:
    """One frame's m/z calibration: slope `K`, intercept `T0` in microseconds, bin width.

    Frozen and hashable on purpose -- see the module docstring on LUT caching. `done`
    carries the file's `CalibrationDone` flag so that a viewer can say "uncalibrated"
    rather than draw a plausible axis over a file the writer never calibrated.
    """

    slope: float
    intercept: float
    bin_width_ns: float
    done: bool = True

    def mz(self, bin_index: np.ndarray | float) -> np.ndarray | float:
        """m/z at a bin index, fractional bins included. Arrives with task 03."""
        raise NotImplementedError("m/z calibration arrives with the lab record's task 03")

    def bin_of(self, mz: np.ndarray | float) -> np.ndarray | float:
        """The fractional bin index at an m/z: the inverse of `mz`. Arrives with task 03."""
        raise NotImplementedError("m/z calibration arrives with the lab record's task 03")

    def mz_axis(self, bins: int) -> np.ndarray:
        """`bin -> m/z` for bin edges `0 .. bins`, float64, length `bins + 1`.

        Edges rather than centres: the heatmap maps a pixel to a half-open bin range,
        and an edge table makes that a search rather than an off-by-half. Arrives with
        the lab record's task 03.
        """
        raise NotImplementedError("m/z calibration arrives with the lab record's task 03")


def arrival_time_ms(scan: np.ndarray | float, average_tof_length_ns: float) -> np.ndarray | float:
    """Arrival time of a scan, in milliseconds. Arrives with the lab record's task 03."""
    raise NotImplementedError("the arrival-time axis arrives with the lab record's task 03")


def scan_axis_ms(scans: int, average_tof_length_ns: float) -> np.ndarray:
    """`scan -> ms` for scan edges `0 .. scans`, the arrival-time twin of `mz_axis`.

    Arrives with the lab record's task 03.
    """
    raise NotImplementedError("the arrival-time axis arrives with the lab record's task 03")
