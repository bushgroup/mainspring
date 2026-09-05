"""Threads and the render mailbox: how the GUI stays responsive while a frame is drawn.

Two pieces of work must leave the GUI thread. Reading and decoding a frame takes tens
to hundreds of milliseconds and happens once per frame; rasterising takes a few
milliseconds and happens on every wheel tick, every drag, every resize.

The rasteriser is fast but it is asked for far more often than it can answer during a
gesture, and every stale answer is wasted work drawn over. So the render path is a
**single-slot mailbox**, not a queue: a new request overwrites whatever is waiting, the
worker takes the latest and only the latest, and the GUI thread never blocks on it.
Combined with a short debounce on the view-changed signal, a continuous zoom costs one
render per frame of animation instead of one per event, and the image that lands is
always the newest view rather than the oldest queued one. A queue here would render a
gesture's whole history, arriving progressively later, which is exactly the lag PNNL's
viewer has.

`LoadWorker` is the other side: it owns the `UimfFile` and the `FrameCache`, so the
short-lived-connection rule and the never-cache-a-provisional-frame rule live on one
thread and nothing else touches SQLite (lab record, tasks 03 and 08).

Qt lives here, and only here on the data side of the viewer: `mainspring.uimf` knows
nothing about threads, and everything these workers call is plain numpy.
"""

from __future__ import annotations

import queue
from dataclasses import dataclass

from PySide6.QtCore import QThread, Signal

from ..uimf import DisplayAxes, RasterResult, SparseFrame, UimfFile
from ..uimf.cache import DEFAULT_BUDGET_BYTES, FrameCache
from ..uimf.decode import numba_available

__all__ = ["DEBOUNCE_MS", "LoadWorker", "RenderMailbox", "RenderRequest", "RenderWorker"]

DEBOUNCE_MS = 30
"""How long a view change waits for the next one before a render is requested. Long
enough to coalesce a wheel burst, short enough that a single tick feels immediate."""


@dataclass(frozen=True)
class RenderRequest:
    """One "draw this window at this size" ask. Immutable: it may cross a thread."""

    frame: SparseFrame
    axes: DisplayAxes
    x_range: tuple[float, float]
    y_range: tuple[float, float]
    width: int
    height: int
    aggregate: str = "sum"
    serial: int = 0


class RenderMailbox:
    """A one-slot handoff: put overwrites, take blocks until something is there.

    Deliberately not a queue -- see the module docstring. Thread-safe; the only shared
    state between the GUI thread and the render worker.
    """

    def put(self, request: RenderRequest) -> None:
        """Replace whatever is waiting. Never blocks. Arrives with the lab record's task 05."""
        raise NotImplementedError("the render mailbox arrives with the lab record's task 05")

    def take(self, timeout_s: float | None = None) -> RenderRequest | None:
        """The latest request, or None on timeout or shutdown. Arrives with task 05."""
        raise NotImplementedError("the render mailbox arrives with the lab record's task 05")

    def close(self) -> None:
        """Wake a waiting `take` with None so the worker can exit. Arrives with task 05."""
        raise NotImplementedError("the render mailbox arrives with the lab record's task 05")


class RenderWorker:
    """The thread that turns `RenderRequest` into `RasterResult` and signals the window.

    A `QThread` subclass once it exists; declared here so that the seam is visible
    before it does. Arrives with the lab record's task 05.
    """

    def __init__(self, mailbox: RenderMailbox) -> None:
        raise NotImplementedError("the render worker arrives with the lab record's task 05")

    def rendered(self) -> RasterResult:
        """The Qt signal carrying a finished image. Arrives with the lab record's task 05."""
        raise NotImplementedError("the render worker arrives with the lab record's task 05")


class LoadWorker(QThread):
    """The thread that owns the file and the frame cache and decodes frames.

    A plain `queue.Queue` rather than the render mailbox above: every open and every
    frame request must be honoured, not just the latest, so this is a work queue and not
    a single slot. It costs nothing here because loads are rare (once per file, once per
    frame navigation) rather than once per animation frame.

    Started on construction and left running for the window's lifetime; `stop()` and
    `wait()` on close is what lets it exit instead of blocking the process (lab record,
    task 04).

    **Warms numba on construction, before any file is asked for.** Importing numba and
    compiling the decode kernels the first time costs the better part of a second on its
    own (`mainspring.uimf.decode`), on top of whatever the first decode itself takes --
    which is what "first paint under 1 s including numba warm-up" actually asks for: that
    cost paid here, off the GUI thread, while the window is still empty and the user is
    still choosing a file, rather than inside the interactive `open_file` it would
    otherwise land in.
    """

    opened = Signal(object, object)
    """`GlobalParams, list[int]` -- a file's parameters and its frame numbers."""
    frame_loaded = Signal(int, object, object)
    """`frame number, SparseFrame, FrameParams` -- a decoded frame and its parameters."""
    failed = Signal(str)
    """An open or a decode raised; the message, never the exception object itself."""

    def __init__(self, cache_budget_bytes: int = 0) -> None:
        super().__init__()
        self._cache = FrameCache(cache_budget_bytes or DEFAULT_BUDGET_BYTES)
        self._file: UimfFile | None = None
        self._queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self.start()
        self._queue.put(("warm", None))

    def open(self, path: str) -> None:
        """Open a file and report its parameters; the `opened` signal carries them."""
        self._queue.put(("open", path))

    def request_frame(self, frame: int) -> None:
        """Ask for a frame; the `frame_loaded` signal carries it."""
        self._queue.put(("frame", int(frame)))

    def stop(self) -> None:
        """Ask the loop to exit at its next turn. Follow with `wait()`."""
        self._queue.put(("stop", None))

    def run(self) -> None:
        while True:
            kind, payload = self._queue.get()
            if kind == "stop":
                return
            try:
                if kind == "open":
                    self._open(str(payload))
                elif kind == "frame":
                    self._request_frame(int(payload))
                elif kind == "warm":
                    numba_available()  # compiles the kernels; return value unneeded here
            except Exception as exc:  # noqa: BLE001 -- reported to the window, not raised here
                self.failed.emit(str(exc))

    def _open(self, path: str) -> None:
        file = UimfFile(path)
        numbers = file.frame_numbers()
        globals_ = file.global_params()
        self._file = file
        self._cache.clear()
        self.opened.emit(globals_, numbers)

    def _request_frame(self, frame: int) -> None:
        if self._file is None:
            raise RuntimeError("request_frame before a file is open")
        sparse = self._cache.get(self._file.path, frame)
        if sparse is None:
            sparse = self._file.read_frame(frame)
            self._cache.put(self._file.path, sparse)
        self.frame_loaded.emit(frame, sparse, self._file.frame_params(frame))
