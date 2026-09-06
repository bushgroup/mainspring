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
from functools import lru_cache

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


    @property
    def usable(self) -> bool:
        """Whether this calibration can produce an m/z axis at all.

        A zero or negative slope -- an uncalibrated frame, or one whose writer left the
        parameters empty -- would map every bin to the same m/z, which is not an axis.
        Callers switch to raw bin units rather than drawing a plausible one; `done` is
        the file's own opinion, this is whether the arithmetic works.
        """
        return self.slope > 0.0 and self.bin_width_ns > 0.0

    def mz(self, bin_index: np.ndarray | float) -> np.ndarray | float:
        """m/z at a bin index, fractional bins included.

        Bins below `T0` are before the calibration's own time origin -- the first 77 of
        them on our sample -- where the parabola turns back up and m/z would decrease
        with increasing bin. Time is clamped at `T0` there, so those bins come out at
        m/z 0 rather than at a mirror image of the low mass range. Nothing is measured
        that early; what matters is that the axis never decreases (lab record, task 01).
        """
        time_us = np.asarray(bin_index, dtype=np.float64) * self.bin_width_ns / 1000.0
        return (self.slope * np.maximum(time_us - self.intercept, 0.0)) ** 2

    def bin_of(self, mz: np.ndarray | float) -> np.ndarray | float:
        """The fractional bin index at an m/z: the inverse of `mz`.

        Exact only above the m/z the intercept sits at, since `mz` is flat below it.
        """
        if not self.usable:
            raise ValueError(f"cannot invert an unusable calibration: {self!r}")
        mz = np.asarray(mz, dtype=np.float64)
        return (np.sqrt(np.maximum(mz, 0.0)) / self.slope + self.intercept) * 1000.0 / self.bin_width_ns

    def mz_axis(self, bins: int) -> np.ndarray:
        """`bin -> m/z` for bin edges `0 .. bins`, float64, length `bins + 1`.

        Edges rather than centres: the heatmap maps a pixel to a half-open bin range,
        and an edge table makes that a search rather than an off-by-half.

        The result is cached on `(K, T0, BinWidth, bins)` and handed out read-only. It
        is a 900 KB array on a SLIMPHONY file and every render wants the same one, so a
        caller that mutated it would corrupt every later frame.
        """
        return _mz_axis(self.slope, self.intercept, self.bin_width_ns, int(bins))


@lru_cache(maxsize=8)
def _mz_axis(slope: float, intercept: float, bin_width_ns: float, bins: int) -> np.ndarray:
    if bins < 0:
        raise ValueError(f"bins must not be negative, got {bins}")
    edges = Calibration(slope, intercept, bin_width_ns).mz(np.arange(bins + 1, dtype=np.float64))
    edges.flags.writeable = False
    return edges


def arrival_time_ms(scan: np.ndarray | float, average_tof_length_ns: float) -> np.ndarray | float:
    """Arrival time of a scan, in milliseconds.

    `t_ms = scan * AverageTOFLength_ns * 1e-6`, straight from the frame parameters: one
    scan is one TOF trigger interval, and the drift time of a scan is how many of them
    have gone by (lab record, task 01).
    """
    return np.asarray(scan, dtype=np.float64) * average_tof_length_ns * 1e-6


@lru_cache(maxsize=8)
def scan_axis_ms(scans: int, average_tof_length_ns: float, t0_offset_ms: float = 0.0) -> np.ndarray:
    """`scan -> ms` for scan edges `0 .. scans`, the arrival-time twin of `mz_axis`.

    `t0_offset_ms` is the viewer's own "Arrival offset" control, subtracted from every
    edge -- not a file-native calibration term, and not the same `T0` as `Calibration`'s
    (that one is the m/z axis's `CalibrationIntercept`, and this module does not even
    apply it; see the module docstring). Unlike `mz_axis`, there is no physical floor to
    clamp to, so a large offset is free to carry the axis negative.

    Cached and read-only for the same reason, though this one is small: the pair of
    tables is what a `DisplayAxes` is made of, and they should behave alike.
    """
    if scans < 0:
        raise ValueError(f"scans must not be negative, got {scans}")
    edges = np.asarray(arrival_time_ms(np.arange(scans + 1, dtype=np.float64),
                                       average_tof_length_ns)) - t0_offset_ms
    edges.flags.writeable = False
    return edges
