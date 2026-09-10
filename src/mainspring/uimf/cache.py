"""`FrameCache`: keep recently read frames, bounded by bytes rather than by count.

Frames differ in size by orders of magnitude -- a sparse SLIMPHONY frame is a few
hundred kilobytes, a dense one from a busy LC run tens of megabytes -- so a cache that
counts frames either wastes memory or thrashes. This one counts `SparseFrame.nbytes`
against a budget and evicts least-recently-used until it fits, which makes stepping
through frames cheap and makes the worst case a number the viewer can put in a setting.

**A provisional frame is never cached.** The last frame of a file the instrument is
still writing may be incomplete, and a cached copy of it would be wrong for as long as
the cache held it; `put` refuses one rather than making the caller remember (lab
record, task 08).

Not thread-safe by construction. The viewer's decode worker owns one and the GUI
thread reaches it only through the worker's mailbox, which is the same discipline that
keeps the render path single-writer (lab record, task 05).
"""

from __future__ import annotations

from .frame import SparseFrame

__all__ = ["DEFAULT_BUDGET_BYTES", "FrameCache"]

DEFAULT_BUDGET_BYTES = 512 * 1024 * 1024
"""Half a gigabyte: tens of frames of anything we have seen, small beside a 190 GB
workstation and still safe on an 8 GB instrument PC. A viewer setting overrides it."""


class FrameCache:
    """Least-recently-used cache of `SparseFrame` under a byte budget.

    Keys are `(path, frame)` so that one cache can serve a session that opens several
    files, and so that a stale entry cannot survive a reopen of a different file at the
    same frame number.
    """

    def __init__(self, budget_bytes: int = DEFAULT_BUDGET_BYTES) -> None:
        self.budget_bytes = int(budget_bytes)
        self._entries: dict[tuple[str, int], SparseFrame] = {}
        self._nbytes = 0

    @property
    def nbytes(self) -> int:
        """What the cache is holding right now.

        A running total kept by `put` and by `_discard`, not a sum over the entries. The
        sum was what `put` asked before deciding to evict, so caching a frame cost a
        pass over everything already cached and paging through a file got steadily
        slower the further in it went: 0.036 ms per put at a hundred entries against
        0.58 ms at two thousand, on a default budget that holds thirty-seven thousand
        per-repetition frames (lab record, task 17).
        """
        return self._nbytes

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, key: object) -> bool:
        return key in self._entries

    def get(self, path: str, frame: int) -> SparseFrame | None:
        """The cached frame, marking it most recently used, or None.

        Recency is the insertion order of the backing dict: a hit is popped and put
        back, so the least recently used entry is always the first one.
        """
        key = (path, int(frame))
        hit = self._entries.pop(key, None)
        if hit is not None:
            self._entries[key] = hit  # same frame, new place: the byte total is unchanged
        return hit

    def put(self, path: str, frame: SparseFrame) -> bool:
        """Cache a frame, evicting to fit; False if it was refused or is too large.

        Refused means provisional: a frame read from a file the instrument may still be
        writing is incomplete, and a cached copy of it would stay wrong for as long as
        the cache held it (lab record, task 08). A frame larger than the whole budget is
        not cached either, rather than evicting everything else to fail anyway.
        """
        if frame.provisional:
            return False
        if frame.nbytes > self.budget_bytes:
            return False
        key = (path, int(frame.frame))
        self._discard(key)
        self._entries[key] = frame
        self._nbytes += frame.nbytes
        while self._nbytes > self.budget_bytes and len(self._entries) > 1:
            self._discard(next(iter(self._entries)))
        return True

    def _discard(self, key: "tuple[str, int]") -> None:
        """Drop one entry and take its bytes off the running total. The only route out
        of `_entries` other than `clear`, so that the total cannot drift from the truth."""
        gone = self._entries.pop(key, None)
        if gone is not None:
            self._nbytes -= gone.nbytes

    def clear(self) -> None:
        """Drop everything. Cheap, and the right response to a file changing under us."""
        self._entries.clear()
        self._nbytes = 0
