"""`SparseFrame`: a whole frame in memory, and never a dense one.

A frame is scans x TOF bins -- 5000 x 114688 on SLIMPHONY, 2.3 GB as int32 -- of which
a fraction of a percent is non-zero. Nothing in mainspring ever builds that array, not
even for one frame, so a frame in memory is the points that exist and an index into
them.

The layout is CSR by scan: `scan_start` has `scans + 1` entries, and the points of scan
`s` are `bin_index[scan_start[s]:scan_start[s + 1]]` with the parallel `intensity`. Two
consequences worth knowing before using it:

* **A scan with no row in the file is an empty slice, not a missing key.** Writers
  differ on this -- the SLIMPHONY sample stores only the 1656 scans that had signal,
  the 2011 files store all of them with empty blobs (lab record, task 01) -- and CSR
  makes the difference invisible to everything downstream.
* **The axis extent is `scans` and `bins` from the frame parameters, never the data.**
  A frame whose signal stops at scan 4992 still has an arrival-time axis that runs to
  `Scans - 1`.

Points within a scan are sorted by bin, which is what the writer's own stream gives us,
so a bin range within a scan is a pair of `searchsorted` calls and rasterisation is a
single pass in file order.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["SparseFrame"]


@dataclass(frozen=True)
class SparseFrame:
    """One frame's non-zero points, CSR by scan. See the module docstring for the layout.

    `intensity` keeps the file's own element type (int32, int16 or float32) rather than
    being widened on read: it is the largest array here and the viewer holds several.
    `provisional` marks a frame read from a file that is still being written, which the
    frame cache must refuse to keep (lab record, task 08).
    """

    frame: int
    scans: int
    bins: int
    scan_start: np.ndarray
    bin_index: np.ndarray
    intensity: np.ndarray
    provisional: bool = False

    def __len__(self) -> int:
        """The number of stored points, not the number of scans."""
        return int(self.bin_index.size)

    @property
    def nbytes(self) -> int:
        """What this frame costs the cache, its index included."""
        return int(self.scan_start.nbytes + self.bin_index.nbytes + self.intensity.nbytes)

    def scan(self, scan: int) -> tuple[np.ndarray, np.ndarray]:
        """`(bin_index, intensity)` of one scan, as views. Arrives with task 03."""
        raise NotImplementedError("SparseFrame accessors arrive with the lab record's task 03")

    def scan_of(self) -> np.ndarray:
        """The scan number of every point, expanded from `scan_start`. Arrives with task 03."""
        raise NotImplementedError("SparseFrame accessors arrive with the lab record's task 03")

    def tic(self) -> np.ndarray:
        """Total ion current per scan, length `scans`. Arrives with the lab record's task 03."""
        raise NotImplementedError("SparseFrame accessors arrive with the lab record's task 03")

    def bpi(self) -> np.ndarray:
        """Base peak intensity per scan, length `scans`. Arrives with task 03."""
        raise NotImplementedError("SparseFrame accessors arrive with the lab record's task 03")

    def slice(
        self,
        scan_range: tuple[int, int] | None = None,
        bin_range: tuple[int, int] | None = None,
    ) -> "SparseFrame":
        """The sub-frame inside a half-open scan and bin window, axes kept.

        Arrives with the lab record's task 03.
        """
        raise NotImplementedError("SparseFrame accessors arrive with the lab record's task 03")

    @classmethod
    def from_scans(
        cls,
        frame: int,
        scans: int,
        bins: int,
        points: "dict[int, tuple[np.ndarray, np.ndarray]]",
        provisional: bool = False,
    ) -> "SparseFrame":
        """Build the CSR arrays from per-scan `(bin_index, intensity)` pairs.

        The reader's assembly step, factored out here so that a test can build a frame
        without a file. Arrives with the lab record's task 03.
        """
        raise NotImplementedError("SparseFrame assembly arrives with the lab record's task 03")
