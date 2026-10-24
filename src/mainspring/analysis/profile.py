"""The extracted-ion arrival-time profile: an m/z window's intensity per scan.

An ion's arrival-time distribution is the intensity of every stored point whose m/z lies
in a window, summed per mobility scan; over a run of frames it is the sum of each
frame's. Every pipeline the lab has written on top of the reader spells this out for
itself (lab record, task 42), and this is the one definition:

* **The window is a half-open bin range**, `ceil(bin_of(lo))` to `floor(bin_of(hi)) + 1`,
  through the frame's own `Calibration`. For a monotonic calibration that is the closed
  interval `lo <= mz(bin) <= hi`, except where an edge lands within float rounding of
  a whole bin.
* **Each frame through its own calibration**, read with `bin_range=` so only the
  window's points are kept, one `read_frame` per frame, nothing dense and nothing
  cached.
* **The frames summed are the frames given.** Which frames make a run -- leaving out a
  first repetition, a method frame from `FrameGrouping`, the frames after a release
  mark -- is the caller's convention, not this module's.

Not to be confused with `mainspring.uimf.raster.profile`, the viewer's side plots, which
sum what is on screen onto a display axis. This one is shaped for a pipeline: m/z in,
one value per scan out.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from mainspring.uimf.calib import Calibration
from mainspring.uimf.frame import SparseFrame
from mainspring.uimf.reader import UimfFile

__all__ = ["ScanProfile", "bin_range", "scan_profile", "summed_profile"]


def _window(mz_window) -> tuple[float, float]:
    lo, hi = (float(v) for v in mz_window)
    if not (math.isfinite(lo) and math.isfinite(hi)) or lo > hi:
        raise ValueError(f"mz_window must be a finite (lo, hi) with lo <= hi, not {mz_window!r}")
    return lo, hi


def bin_range(calibration: Calibration, mz_window) -> tuple[int, int]:
    """The half-open bin range `(lo, hi)` whose m/z lies within `mz_window`, `(lo, hi)`.

    `ceil(bin_of(lo))` to `floor(bin_of(hi)) + 1`: the bins of the closed interval
    `lo <= mz(bin) <= hi`, up to rounding where an edge falls on a whole bin. Not
    clipped to the file's bins; `read_frame` and `SparseFrame.slice` take any range.
    Raises `ValueError` for a calibration that is not `usable` and for a window that is
    not a finite pair in ascending order.
    """
    lo, hi = _window(mz_window)
    if not calibration.usable:
        raise ValueError(f"an m/z window cannot be placed on an unusable calibration: {calibration!r}")
    return (int(math.ceil(float(calibration.bin_of(lo)))),
            int(math.floor(float(calibration.bin_of(hi)))) + 1)


def scan_profile(frame: SparseFrame, calibration: Calibration, mz_window) -> np.ndarray:
    """The intensity in `mz_window` per scan of `frame`: float64, length `frame.scans`.

    `calibration` is the frame's own (`FrameParams.calibration`). The frame may be whole
    or already cut to the window by `read_frame(..., bin_range=)`; the points outside
    the window's bin range are left out either way.
    """
    lo, hi = bin_range(calibration, mz_window)
    inside = (frame.bin_index >= lo) & (frame.bin_index < hi)
    return np.bincount(frame.scan_of()[inside],
                       weights=frame.intensity[inside].astype(np.float64),
                       minlength=frame.scans).astype(np.float64, copy=False)


@dataclass(frozen=True, eq=False)
class ScanProfile:
    """A profile summed over frames: `values` per scan, the scan tick, the frames summed.

    `values` is float64 and read-only. `tick_ms` is one scan in ms, the first frame's
    `AverageTOFLength`; `frames` are the frame numbers summed, in the order given.
    """

    values: np.ndarray
    tick_ms: float
    frames: tuple[int, ...]

    def ms(self, scan):
        """A scan, or an array of them, in ms; `None` stays `None`, so a statistic of an
        empty profile goes through unchanged."""
        if scan is None:
            return None
        if np.ndim(scan) == 0:
            return float(scan) * self.tick_ms
        return np.asarray(scan, dtype=np.float64) * self.tick_ms

    @property
    def axis_ms(self) -> np.ndarray:
        """Every scan's arrival time in ms, the same length as `values`."""
        return np.arange(self.values.size, dtype=np.float64) * self.tick_ms


def summed_profile(uimf: UimfFile, frames: Iterable[int], mz_window) -> ScanProfile:
    """`mz_window`'s profile summed over `frames` of `uimf`, as a `ScanProfile`.

    Each frame is read with `bin_range=` and placed through its own calibration, so a
    file whose calibration changes from frame to frame is summed correctly. A frame with
    fewer scans than another adds onto the start of the longest axis. The tick is the
    first frame's.

    Raises `ValueError` for no frames, for a window that is not a finite ascending pair,
    and for a frame whose calibration is not `usable`, which has a bin axis and no m/z
    axis to place the window on. `FrameParams.calibration_done`, the file's own opinion,
    is not consulted; a caller that wants it checks it first.
    """
    window = _window(mz_window)
    numbers = tuple(int(f) for f in frames)
    if not numbers:
        raise ValueError("summed_profile needs at least one frame")
    bin_width_ns = uimf.global_params().bin_width_ns
    total = np.zeros(0, dtype=np.float64)
    tick_ms = None
    for number in numbers:
        params = uimf.frame_params(number)
        calibration = params.calibration(bin_width_ns)
        if not calibration.usable:
            raise ValueError(
                f"{uimf.path}: frame {number}'s calibration (slope {calibration.slope}, "
                f"intercept {calibration.intercept}) gives no m/z axis to place a window on")
        if tick_ms is None:
            tick_ms = float(params.average_tof_length_ns) * 1e-6
        frame = uimf.read_frame(number, bin_range=bin_range(calibration, window))
        values = scan_profile(frame, calibration, window)
        if values.size > total.size:
            total = np.concatenate((total, np.zeros(values.size - total.size)))
        total[:values.size] += values
    total.flags.writeable = False
    return ScanProfile(values=total, tick_ms=tick_ms, frames=numbers)
