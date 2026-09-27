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
from collections.abc import Sequence
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

    **Two paths, one result.** Where numba is installed, frames of one shape and one
    element type are added by a compiled kernel instead: per scan, every input's points
    are scatter-added into one accumulator row *in frame order* and read back off a
    bitmap in bin order, so each point is visited once rather than once per addition
    that follows it -- 3.9 s against 23 s for a hundred-repetition beam-on fold (lab
    record, task 34). Frame order is the left fold scipy computes, so the result is the
    same arrays to the bit, float32 included, and zeros are dropped the way scipy's
    addition drops them. A sequence is added in one pass, since its caller holds it
    already; any other iterable is merged into the running total whenever the frames
    gathered since take as many bytes (and at least 64 MiB), so a generator over a large
    file is never held whole. One frame on its own, frames that differ in shape or type, or a scan whose
    bins are not strictly ascending go the scipy way, which is also where a total that
    has already been merged carries on from, being exactly what scipy had by then.
    """
    from scipy import sparse

    kernel = _sum_kernel()
    hold_all = isinstance(frames, Sequence)
    total = None
    merged: SparseFrame | None = None
    pending: list[SparseFrame] = []
    pending_bytes = 0
    scans = bins = 0
    provisional = False

    def fold_pending() -> bool:
        nonlocal merged, pending, pending_bytes
        result = _merge(kernel, ([merged] if merged is not None else []) + pending)
        if result is None:
            return False
        merged, pending, pending_bytes = result, [], 0
        return True

    def to_scipy() -> None:
        nonlocal total, kernel, merged, pending, pending_bytes
        kernel = None
        for part in ([merged] if merged is not None else []) + pending:
            matrix = _as_csr(sparse, part)
            total = matrix if total is None else total + matrix
        merged, pending, pending_bytes = None, [], 0

    for frame in frames:
        if should_cancel is not None and should_cancel():
            return None
        provisional |= frame.provisional
        scans = max(scans, frame.scans)
        bins = max(bins, frame.bins)
        if kernel is not None:
            first = merged if merged is not None else (pending[0] if pending else frame)
            if _kernel_takes(frame, first):
                pending.append(frame)
                pending_bytes += frame.nbytes
                if (not hold_all and len(pending) + (merged is not None) >= 2
                        and pending_bytes >= max(_MERGE_FLOOR_BYTES,
                                                 0 if merged is None else merged.nbytes)):
                    if not fold_pending():
                        to_scipy()
                continue
            to_scipy()
        matrix = _as_csr(sparse, frame)
        total = matrix if total is None else total + matrix
    if kernel is not None and len(pending) + (merged is not None) >= 2:
        if fold_pending():
            return SparseFrame(
                frame=0, scans=scans, bins=bins, scan_start=merged.scan_start,
                bin_index=merged.bin_index, intensity=merged.intensity,
                provisional=provisional,
            )
    if kernel is not None:
        to_scipy()
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


# --- the compiled sum -----------------------------------------------------------------

# Below this many bytes gathered, a stream is not merged yet: a merge walks every scan
# of the frame, which is worth doing a few times per sum rather than once per frame.
# Bytes rather than points, because a sparse raw frame is mostly its scan index.
_MERGE_FLOOR_BYTES = 64 << 20
_SUM_KERNEL = None
_SUM_KERNEL_LOOKED = False
# `_CTZ[((w & -w) * _DEBRUIJN) >> 58]` is the index of the lowest set bit of a uint64.
_DEBRUIJN = 0x03F79D71B4CB0A89
_CTZ = np.zeros(64, dtype=np.int64)
for _bit in range(64):
    _CTZ[(((1 << _bit) * _DEBRUIJN) & 0xFFFFFFFFFFFFFFFF) >> 58] = _bit
del _bit


def _sum_kernel():
    """The compiled sum, or None without numba; compiled on first use, looked up once."""
    global _SUM_KERNEL, _SUM_KERNEL_LOOKED
    if _SUM_KERNEL_LOOKED:
        return _SUM_KERNEL
    _SUM_KERNEL_LOOKED = True
    try:
        from numba import njit
    except Exception:  # noqa: BLE001 -- absent, broken and incompatible all mean "scipy"
        return None
    from .decode import install_frozen_cache_locator

    install_frozen_cache_locator()
    _SUM_KERNEL = njit(cache=True, nogil=True)(_k_sum_rows)
    return _SUM_KERNEL


def _as_csr(sparse, frame: SparseFrame):
    return sparse.csr_matrix(
        (frame.intensity, frame.bin_index, frame.scan_start),
        shape=(frame.scans, frame.bins),
    )


def _kernel_takes(frame: SparseFrame, first: SparseFrame) -> bool:
    """Whether `frame` can join `first` in the kernel: one shape, one plain numeric type."""
    dtype = frame.intensity.dtype
    return (frame.scans == first.scans and frame.bins == first.bins
            and dtype == first.intensity.dtype and dtype.kind in "iuf" and dtype.isnative
            and frame.intensity.ndim == frame.bin_index.ndim == frame.scan_start.ndim == 1
            and frame.scan_start.size == frame.scans + 1)


def _merge(kernel, parts: "list[SparseFrame]") -> "SparseFrame | None":
    """The kernel's frame-order sum of `parts`, or None if a scan was not canonical."""
    from numba.typed import List

    first = parts[0]
    starts, indices, values = List(), List(), List()
    for part in parts:
        starts.append(np.ascontiguousarray(part.scan_start, dtype=np.int64))
        indices.append(np.ascontiguousarray(part.bin_index, dtype=np.int32))
        values.append(np.ascontiguousarray(part.intensity))
    accumulator = np.zeros(first.bins, dtype=first.intensity.dtype)
    scan_start, bin_index, intensity, n, ok = kernel(
        starts, indices, values, first.scans, first.bins, accumulator, _CTZ)
    if not ok:
        return None
    return SparseFrame(frame=0, scans=first.scans, bins=first.bins, scan_start=scan_start,
                       bin_index=bin_index[:n].copy(), intensity=intensity[:n].copy())


def _k_sum_rows(starts, indices, values, scans, bins, acc, ctz):
    """Frame-order sum of CSR inputs of one shape: (scan_start, bin_index, intensity, n, ok).

    Per scan: scatter-add every input's points into `acc` in input order, setting a bit
    per bin touched, then walk the bitmap's words from the lowest touched to the highest,
    emitting each non-zero sum and clearing what it read. `ok` is False, and nothing is
    to be trusted, if any input's bins in a scan are not strictly ascending and inside
    `bins`; the caller then goes the scipy way.
    """
    words = (bins + 63) >> 6
    bitmap = np.zeros(words, dtype=np.uint64)
    one = np.uint64(1)
    low6 = np.uint64(63)
    debruijn = np.uint64(0x03F79D71B4CB0A89)
    shift = np.uint64(58)
    cap = 1 << 20
    for k in range(len(indices)):
        if indices[k].size > cap:
            cap = indices[k].size
    cap = cap + (cap >> 1)
    out_idx = np.empty(cap, dtype=np.int32)
    out_val = np.empty(cap, dtype=acc.dtype)
    out_ptr = np.zeros(scans + 1, dtype=np.int64)
    n = 0
    for r in range(scans):
        bound = 0
        lo = words
        hi = -1
        for k in range(len(indices)):
            ip = starts[k]
            ix = indices[k]
            dv = values[k]
            begin = ip[r]
            end = ip[r + 1]
            if end <= begin:
                continue
            bound += end - begin
            prev = -1
            for j in range(begin, end):
                b = ix[j]
                if b <= prev or b >= bins:
                    return out_ptr, out_idx, out_val, 0, False
                prev = b
                bitmap[b >> 6] |= one << (np.uint64(b) & low6)
                acc[b] += dv[j]
            if (ix[begin] >> 6) < lo:
                lo = ix[begin] >> 6
            if (ix[end - 1] >> 6) > hi:
                hi = ix[end - 1] >> 6
        if n + bound > cap:
            cap = max(n + bound, cap + (cap >> 1))
            grown_idx = np.empty(cap, dtype=np.int32)
            grown_val = np.empty(cap, dtype=acc.dtype)
            grown_idx[:n] = out_idx[:n]
            grown_val[:n] = out_val[:n]
            out_idx = grown_idx
            out_val = grown_val
        for w in range(lo, hi + 1):
            x = bitmap[w]
            if x == 0:
                continue
            bitmap[w] = 0
            base = w << 6
            while x != 0:
                low = x & (~x + one)
                b = base + ctz[(low * debruijn) >> shift]
                v = acc[b]
                if v != 0:
                    out_idx[n] = b
                    out_val[n] = v
                    n += 1
                acc[b] = 0
                x ^= low
        out_ptr[r + 1] = n
    return out_ptr, out_idx, out_val, n, True
