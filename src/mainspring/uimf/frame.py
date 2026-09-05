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
from typing import Callable, Iterable

import numpy as np

__all__ = ["SparseFrame", "sum_frames"]


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


    @property
    def extent(self) -> tuple[int, int]:
        """The axis shape this frame sits on, `(scans, bins)` -- never `max` of the data."""
        return (int(self.scans), int(self.bins))

    def scan(self, scan: int) -> tuple[np.ndarray, np.ndarray]:
        """`(bin_index, intensity)` of one scan, as views into the frame's own arrays.

        Views, not copies: the caller gets the same memory the cache is holding, so a
        side plot that reads a thousand scans allocates nothing. A scan the writer never
        stored is a pair of empty views rather than a `KeyError`.
        """
        if not 0 <= scan < self.scans:
            raise IndexError(f"scan {scan} outside 0..{self.scans - 1}")
        lo = int(self.scan_start[scan])
        hi = int(self.scan_start[scan + 1])
        return self.bin_index[lo:hi], self.intensity[lo:hi]

    def scan_of(self) -> np.ndarray:
        """The scan number of every point, expanded from `scan_start`.

        The one array CSR does not already have, and rasterisation's other coordinate.
        Built on demand rather than stored, because it is as large as `bin_index` and
        only the render path wants it.
        """
        counts = np.diff(self.scan_start)
        return np.repeat(np.arange(self.scans, dtype=np.int32), counts)

    def tic(self) -> np.ndarray:
        """Total ion current per scan, length `scans`, empty scans included as zero.

        A cumulative sum differenced at the row pointer: one pass, and exact on integer
        intensities rather than drifting the way a float accumulation over millions of
        points would.
        """
        accumulator = np.int64 if self.intensity.dtype.kind in "iu" else np.float64
        running = np.cumsum(self.intensity, dtype=accumulator)
        running = np.concatenate((np.zeros(1, dtype=accumulator), running))
        return running[self.scan_start[1:]] - running[self.scan_start[:-1]]

    def bpi(self) -> np.ndarray:
        """Base peak intensity per scan, length `scans`, zero where a scan has no points.

        `np.maximum.reduceat` cannot express an empty group, so it is handed only the
        non-empty ones. That is exact rather than lucky: empty scans contribute no
        points, so each non-empty group's start is the previous one's end.
        """
        out = np.zeros(self.scans, dtype=self.intensity.dtype)
        starts = self.scan_start[:-1]
        occupied = starts < self.scan_start[1:]
        if occupied.any():
            out[occupied] = np.maximum.reduceat(self.intensity, starts[occupied])
        return out

    def bpi_bin(self) -> np.ndarray:
        """The bin of each scan's base peak, length `scans`, -1 where a scan is empty.

        What `BPI_MZ` claims to be, computed from the points instead. Every writer we
        have seen disagrees with its own data here by up to three bins, which is why
        `uimf-info --verify` compares this as a tolerance and `TIC`/`BPI` as equalities
        (lab record, task 01).
        """
        out = np.full(self.scans, -1, dtype=np.int64)
        starts = self.scan_start[:-1]
        occupied = np.flatnonzero(starts < self.scan_start[1:])
        for scan in occupied.tolist():
            lo = int(self.scan_start[scan])
            hi = int(self.scan_start[scan + 1])
            out[scan] = self.bin_index[lo + int(np.argmax(self.intensity[lo:hi]))]
        return out

    def slice(
        self,
        scan_range: tuple[int, int] | None = None,
        bin_range: tuple[int, int] | None = None,
    ) -> "SparseFrame":
        """The sub-frame inside a half-open scan and bin window, axes kept.

        The result still carries the whole frame's `scans` and `bins`, so a slice knows
        where it sits and can be rasterised against the same axis tables; scans outside
        the window become empty rather than disappearing. Points within a scan are
        sorted by bin, so the bin window is two `searchsorted` calls per scan rather
        than a mask over every point.
        """
        scan_lo, scan_hi = (0, self.scans) if scan_range is None else scan_range
        scan_lo = max(0, int(scan_lo))
        scan_hi = min(self.scans, int(scan_hi))
        if scan_hi <= scan_lo:
            return SparseFrame(
                frame=self.frame, scans=self.scans, bins=self.bins,
                scan_start=np.zeros(self.scans + 1, dtype=np.int64),
                bin_index=self.bin_index[:0], intensity=self.intensity[:0],
                provisional=self.provisional,
            )

        if bin_range is None:
            lo = int(self.scan_start[scan_lo])
            hi = int(self.scan_start[scan_hi])
            counts = np.diff(self.scan_start)
            kept = np.zeros(self.scans, dtype=np.int64)
            kept[scan_lo:scan_hi] = counts[scan_lo:scan_hi]
            bin_index = self.bin_index[lo:hi]
            intensity = self.intensity[lo:hi]
        else:
            bin_lo, bin_hi = int(bin_range[0]), int(bin_range[1])
            kept = np.zeros(self.scans, dtype=np.int64)
            pieces_bin, pieces_intensity = [], []
            for scan in range(scan_lo, scan_hi):
                lo = int(self.scan_start[scan])
                hi = int(self.scan_start[scan + 1])
                if hi == lo:
                    continue
                window = self.bin_index[lo:hi]
                start = lo + int(np.searchsorted(window, bin_lo, side="left"))
                stop = lo + int(np.searchsorted(window, bin_hi, side="left"))
                if stop > start:
                    kept[scan] = stop - start
                    pieces_bin.append(self.bin_index[start:stop])
                    pieces_intensity.append(self.intensity[start:stop])
            bin_index = (np.concatenate(pieces_bin) if pieces_bin
                         else self.bin_index[:0])
            intensity = (np.concatenate(pieces_intensity) if pieces_intensity
                         else self.intensity[:0])

        scan_start = np.zeros(self.scans + 1, dtype=np.int64)
        np.cumsum(kept, out=scan_start[1:])
        return SparseFrame(
            frame=self.frame, scans=self.scans, bins=self.bins, scan_start=scan_start,
            bin_index=bin_index, intensity=intensity, provisional=self.provisional,
        )

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
        without a file. Scans absent from `points` are empty; a scan number at or past
        `scans` is an error rather than a silently widened axis, because it means the
        caller and the frame parameters disagree about the frame.
        """
        counts = np.zeros(int(scans), dtype=np.int64)
        for scan, (bin_index, _) in points.items():
            if not 0 <= scan < scans:
                raise ValueError(f"scan {scan} outside the frame's 0..{scans - 1}")
            counts[scan] = np.asarray(bin_index).size
        scan_start = np.zeros(int(scans) + 1, dtype=np.int64)
        np.cumsum(counts, out=scan_start[1:])

        ordered = sorted(points)
        if ordered:
            bin_index = np.concatenate(
                [np.asarray(points[s][0], dtype=np.int32) for s in ordered]
            )
            intensity = np.concatenate([np.asarray(points[s][1]) for s in ordered])
        else:
            bin_index = np.empty(0, dtype=np.int32)
            intensity = np.empty(0, dtype=np.int32)
        return cls(
            frame=int(frame), scans=int(scans), bins=int(bins), scan_start=scan_start,
            bin_index=bin_index, intensity=intensity, provisional=bool(provisional),
        )


def sum_frames(
    frames: "Iterable[SparseFrame]",
    should_cancel: "Callable[[], bool] | None" = None,
) -> SparseFrame | None:
    """Add frames point-wise into one frame, or None if `should_cancel` said to stop.

    A `SparseFrame` *is* a CSR matrix of scans by bins, so this is scipy's sparse
    addition with the arrays handed over as they are -- no dense intermediate, and the
    duplicate-summing and re-sorting that adding two sparse frames needs are already
    written and tested there. The result's `frame` is 0, meaning "not a frame of the
    file", and it is **provisional if any of its inputs was**: a total that included a
    frame the instrument may still be growing is itself only as final as that frame,
    and the flag is what stops the cache keeping it (lab record, task 08). Propagating
    beats refusing, which would make "sum all" fail on a live file, and beats ignoring,
    which would quietly leave a frame out of a total.

    `should_cancel` is polled between frames. Summing every frame of a large file is
    minutes of work behind a progress bar, and the user must be able to stop it.
    """
    from scipy import sparse

    total = None
    scans = bins = 0
    provisional = False
    for frame in frames:
        if should_cancel is not None and should_cancel():
            return None
        provisional |= frame.provisional
        scans = max(scans, frame.scans)
        bins = max(bins, frame.bins)
        matrix = sparse.csr_matrix(
            (frame.intensity, frame.bin_index, frame.scan_start),
            shape=(frame.scans, frame.bins),
        )
        total = matrix if total is None else total + matrix
    if total is None:
        return SparseFrame(
            frame=0, scans=0, bins=0, scan_start=np.zeros(1, dtype=np.int64),
            bin_index=np.empty(0, dtype=np.int32), intensity=np.empty(0, dtype=np.int32),
        )
    total.sort_indices()
    total.sum_duplicates()
    return SparseFrame(
        frame=0,
        scans=scans,
        bins=bins,
        scan_start=np.asarray(total.indptr, dtype=np.int64),
        bin_index=np.asarray(total.indices, dtype=np.int32),
        intensity=total.data,
        provisional=provisional,
    )
