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

    @property
    def nbytes(self) -> int:
        """What the cache is holding right now."""
        return sum(frame.nbytes for frame in self._entries.values())

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, key: object) -> bool:
        return key in self._entries

    def get(self, path: str, frame: int) -> SparseFrame | None:
        """The cached frame, marking it most recently used, or None. Arrives with task 03."""
        raise NotImplementedError("the frame cache arrives with the lab record's task 03")

    def put(self, path: str, frame: SparseFrame) -> bool:
        """Cache a frame, evicting to fit; False if it was refused or too large for the budget.

        Arrives with the lab record's task 03.
        """
        raise NotImplementedError("the frame cache arrives with the lab record's task 03")

    def clear(self) -> None:
        """Drop everything. Cheap, and the right response to a file changing under us."""
        self._entries.clear()
