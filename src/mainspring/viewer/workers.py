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

**The image and the two profiles are computed together, in one request**, by
`mainspring.uimf.raster.render_view`. They are three views of the same window; computing
the side plots on the GUI thread after the image arrived would put a second cost on the
interactive path and let the three pictures disagree for a frame while a gesture is in
flight. It is also the cheaper way round -- the expensive part of all three is selecting
the window's points, and `render_view` does that once (lab record, task 05). `RenderResult`
carries all three plus the time they took, so the window has one thing to draw and one
number to report.

`LoadWorker` is the other side: it owns the `UimfFile` and the `FrameCache`, so the
short-lived-connection rule and the never-cache-a-provisional-frame rule live on one
thread and nothing else touches SQLite (lab record, tasks 03 and 08).

**The follow poll is that thread's idle time, not a timer.** `LoadWorker.run` already
blocks on its work queue; following makes that block time out after `POLL_INTERVAL_S`
and asks the file what has changed. Three things fall out of putting it there rather
than on a `QTimer` in the window. The poll's SQLite work is on the thread that is
allowed to do SQLite work, by construction rather than by care. A poll can never
interleave with a frame read or a sum, because the queue is served first and the poll
only runs when the queue is empty -- so an acquisition being followed while a
twenty-second sum-all runs is not also being polled twenty times. And a slow poll
delays only the next poll, never a gesture.

Qt lives here, and only here on the data side of the viewer: `mainspring.uimf` knows
nothing about threads, and everything these workers call is plain numpy. What crosses
the seam are frozen dataclasses over read-only numpy arrays -- a `SparseFrame` and a
`DisplayAxes` are never mutated after they are built, so passing them to the render
thread needs no copy and no lock.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass

import numpy as np
from PySide6.QtCore import QThread, Signal

from ..uimf import (
    DisplayAxes,
    FrameGrouping,
    LiveState,
    RasterResult,
    SparseFrame,
    UimfFile,
    sum_frames,
)
from ..uimf.cache import DEFAULT_BUDGET_BYTES, FrameCache
from ..uimf.decode import numba_available
from ..uimf.raster import render_view

__all__ = [
    "DEBOUNCE_MS",
    "POLL_INTERVAL_S",
    "LoadWorker",
    "RenderMailbox",
    "RenderRequest",
    "RenderResult",
    "RenderWorker",
    "SumRequest",
]

DEBOUNCE_MS = 30
"""How long a view change waits for the next one before a render is requested. Long
enough to coalesce a wheel burst, short enough that a single tick feels immediate."""

POLL_INTERVAL_S = 1.0
"""How often a followed file is asked what has changed. One console frame is one ion
mobility experiment, about a second long at the SLIMPHONY pusher rate, so a second is
the rate at which there is anything new to find; the poll itself costs 11 ms of query
on a 5,000-frame file (lab record, tasks 08 and 17)."""


@dataclass(frozen=True)
class SumRequest:
    """One "add these frames up" ask, echoed back with the result.

    The window has two callers for one sum -- `Sum all` and `Sum method frame`, plus the
    running total a followed acquisition recomputes on its own -- and they want different
    things when the answer arrives: a dialog closed, a status line worded, or nothing at
    all. Echoing the ask beside the answer is what lets `_on_summed` tell them apart
    without the window holding a flag that a second sum arriving first would falsify.
    """

    frames: tuple[int, ...] = ()
    what: str = ""
    live: str = ""
    """Which `Show` mode asked for this total, empty when the user did.

    Truthy exactly when the follow poll asked: no progress dialog, and the view must not
    move if a newer one is already on screen. The mode's own name rather than a flag
    because there are two of them and they are worded differently when they land -- one
    grows repetition by repetition until the method frame ends, the other is a window of
    fixed length that moves. Nothing here reads the string; it is carried."""


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


@dataclass(frozen=True)
class RenderResult:
    """What one `RenderRequest` produced: the image, the two profiles, and the cost.

    `x_profile` and `y_profile` are `(edges, values)` pairs from
    `mainspring.uimf.raster.profile`, at native resolution -- one value per source
    element in view, on that element's own edges, `len(edges) == len(values) + 1`.

    `serial` is the request's, so the window can drop a result that a newer frame or a
    newer file has already overtaken. `elapsed_ms` is the whole request -- image and
    both profiles -- which is the number the 100 ms render budget is about, not the
    rasteriser's share of it alone.
    """

    result: RasterResult
    x_profile: tuple[np.ndarray, np.ndarray]
    y_profile: tuple[np.ndarray, np.ndarray]
    serial: int
    elapsed_ms: float


class RenderMailbox:
    """A one-slot handoff: put overwrites, take blocks until something is there.

    Deliberately not a queue -- see the module docstring. Thread-safe; the only shared
    state between the GUI thread and the render worker.

    Closing is part of the protocol rather than an afterthought: a worker blocked in
    `take()` has to be woken to exit, and a `put` racing a close must be dropped rather
    than left in a slot nobody will empty.
    """

    def __init__(self) -> None:
        self._ready = threading.Condition()
        self._pending: RenderRequest | None = None
        self._closed = False

    @property
    def closed(self) -> bool:
        """Whether `close()` has been called; what tells a `take()` of None from a timeout."""
        with self._ready:
            return self._closed

    def put(self, request: RenderRequest) -> None:
        """Replace whatever is waiting. Never blocks."""
        with self._ready:
            if self._closed:
                return
            self._pending = request
            self._ready.notify()

    def take(self, timeout_s: float | None = None) -> RenderRequest | None:
        """The latest request, or None on timeout or shutdown.

        Waits only while the slot is empty, so a request put while the worker was busy
        is taken without a further wait -- which is the case that matters during a
        gesture, when a put lands for every debounce interval the last render outlived.
        """
        with self._ready:
            if self._pending is None and not self._closed:
                self._ready.wait(timeout_s)
            request, self._pending = self._pending, None
            return None if self._closed else request

    def close(self) -> None:
        """Wake a waiting `take` with None so the worker can exit. Idempotent."""
        with self._ready:
            self._closed = True
            self._pending = None
            self._ready.notify_all()


class RenderWorker(QThread):
    """The thread that turns `RenderRequest` into `RenderResult` and signals the window.

    All it does is loop on the mailbox and call `mainspring.uimf.raster`. It holds no
    state of its own -- everything a render needs is in the request -- which is what
    lets a frame change, an axis swap and a resize all be "the next request" rather than
    three kinds of invalidation.
    """

    rendered = Signal(object)
    """`RenderResult` -- a finished image and its two profiles."""
    failed = Signal(str)
    """A render raised; the message, never the exception object itself. A degenerate
    window is the realistic case, and it must not take the thread down with it."""

    def __init__(self, mailbox: RenderMailbox, parent: "object | None" = None) -> None:
        super().__init__(parent)
        self._mailbox = mailbox

    def run(self) -> None:
        while True:
            request = self._mailbox.take()
            if request is None:
                if self._mailbox.closed:
                    return
                continue
            try:
                self.rendered.emit(self._render(request))
            except Exception as exc:  # noqa: BLE001 -- reported to the window, not raised here
                self.failed.emit(str(exc))

    @staticmethod
    def _render(request: RenderRequest) -> RenderResult:
        started = time.perf_counter()
        result, x_profile, y_profile = render_view(
            request.frame, request.axes, request.x_range, request.y_range,
            request.width, request.height, aggregate=request.aggregate,
        )
        return RenderResult(
            result=result,
            x_profile=x_profile,
            y_profile=y_profile,
            serial=request.serial,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
        )


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

    opened = Signal(object, object, object, object)
    """`GlobalParams, list[int], dict[int, int], FrameGrouping` -- a file's parameters,
    its frame numbers, each frame's `FrameType`, and how the frames group into method
    frames. The last two are each **one query for the whole file** rather than one
    `frame_params` call per frame, which is the difference between 43 ms and 6.9 s on a
    clockwork raw file of 5,000 frames (lab record, task 17)."""
    frame_loaded = Signal(int, object, object)
    """`frame number, SparseFrame, FrameParams` -- a decoded frame and its parameters."""
    summing_progress = Signal(int, int)
    """`(frames done, frames total)` -- one tick per frame `sum_all` has read, for a
    progress dialog. Emitted from this thread; the window marshals it onto the GUI
    thread the way every other signal here does."""
    summed = Signal(object, object, object)
    """`SparseFrame, FrameParams | None, SumRequest` -- the sum of the requested frames,
    its `frame` number 0 (`mainspring.uimf.frame.sum_frames`), paired with the first
    summed frame's parameters so the window has a calibration to build axes from, and
    with the ask that produced it. `None` for the first two if the sum was cancelled or
    the frame list was empty -- nothing to show."""
    live_update = Signal(object, object, object)
    """`LiveState, dict[int, int] | None, FrameGrouping | None` -- what one poll of a
    followed file found: the frame list and which frames may still be growing, and the
    same two whole-file answers `opened` carries, each frame's `FrameType` and the
    method-frame grouping.

    Emitted only when there is something to act on: the frame list changed, a frame
    became final, or a frame is still growing and so may have more in it than the last
    look found. The types and the grouping are `None` unless the frame list actually
    *grew*, because neither can change while it does not and re-reading the grouping is
    43 ms against the poll's own 11 ms (lab record, task 17)."""
    follow_stopped = Signal(str)
    """A poll raised, and following has been switched off; the message.

    Its own signal rather than `failed`, because a failing poll is the one failure that
    repeats: the file has been moved, deleted or unmounted, and a viewer that reported
    that once a second until someone noticed would be worse than one that stops and says
    so. The window puts its own toggle back with this."""
    failed = Signal(str)
    """An open or a decode raised; the message, never the exception object itself."""

    def __init__(self, cache_budget_bytes: int = 0) -> None:
        super().__init__()
        self._cache = FrameCache(cache_budget_bytes or DEFAULT_BUDGET_BYTES)
        self._file: UimfFile | None = None
        self._queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self._cancel_sum = threading.Event()
        self._poll_interval: float | None = None
        self._live: LiveState | None = None
        self.start()
        self._queue.put(("warm", None))

    def open(self, path: str) -> None:
        """Open a file and report its parameters; the `opened` signal carries them."""
        self._queue.put(("open", path))

    def request_frame(self, frame: int) -> None:
        """Ask for a frame; the `frame_loaded` signal carries it."""
        self._queue.put(("frame", int(frame)))

    def sum_all(self, frames: "list[int]", what: str = "", *, live: str = "") -> None:
        """Sum several frames into one; the `summed` signal carries the result.

        Progress is one `summing_progress` tick per frame *read*, not per frame added --
        reading (a query plus a decode) is the part of this that takes real time, the
        sparse addition itself is fast, so a tick per read is what makes the bar move at
        the rate the user is actually waiting on.

        `what` and `live` are not used here at all: they are carried through to the
        `summed` signal in a `SumRequest`, so that the window knows which of its several
        reasons for asking this answer belongs to.
        """
        self._queue.put(("sum", SumRequest(tuple(int(f) for f in frames), what, str(live))))

    def set_follow(self, following: bool) -> None:
        """Start or stop polling the open file for what the instrument has written.

        Goes through the queue rather than setting a flag, because the loop spends its
        life blocked in `get()` and a flag set from the GUI thread would not be looked
        at until the next piece of real work arrived -- which, on an idle viewer
        watching an acquisition, is never.
        """
        self._queue.put(("follow", bool(following)))

    def cancel_sum(self) -> None:
        """Ask an in-progress `sum_all` to stop at its next frame. Idempotent."""
        self._cancel_sum.set()

    def stop(self) -> None:
        """Ask the loop to exit at its next turn. Follow with `wait()`."""
        self._queue.put(("stop", None))

    def run(self) -> None:
        while True:
            try:
                kind, payload = self._queue.get(timeout=self._poll_interval)
            except queue.Empty:
                # Nothing asked of us for a whole interval, and we are following: this
                # is the poll. Work always wins -- a frame request, a sum or a close
                # sitting in the queue is served before the file is asked anything.
                try:
                    self._poll()
                except Exception as exc:  # noqa: BLE001 -- a poll must not kill the thread
                    self._poll_interval = None
                    self.follow_stopped.emit(str(exc))
                continue
            if kind == "stop":
                return
            try:
                if kind == "open":
                    self._open(str(payload))
                elif kind == "frame":
                    self._request_frame(int(payload))
                elif kind == "sum":
                    self._sum(payload)
                elif kind == "follow":
                    self._set_follow(bool(payload))
                elif kind == "warm":
                    numba_available()  # compiles the kernels; return value unneeded here
            except Exception as exc:  # noqa: BLE001 -- reported to the window, not raised here
                self.failed.emit(str(exc))

    def _open(self, path: str) -> None:
        file = UimfFile(path)
        numbers = file.frame_numbers()
        globals_ = file.global_params()
        # Skipped rather than queried on an empty result: there is nothing to filter by
        # type, and a minimal or malformed file with no frames need not carry a
        # `FrameType` column either (`frame_types()` assumes one on a legacy table).
        # The grouping is skipped on the same condition and returns empty on any file
        # that does not carry it, which is most of them.
        types = file.frame_types() if numbers else {}
        grouping = file.frame_grouping() if numbers else FrameGrouping()
        self._file = file
        self._cache.clear()
        # A new file is a new acquisition: stop following the old one and forget what
        # the last poll of it found, so the window's first poll of this file reports
        # everything rather than a difference against something else's frame list.
        self._poll_interval = None
        self._live = None
        self.opened.emit(globals_, numbers, types, grouping)

    def _read_frame(self, frame: int) -> SparseFrame:
        assert self._file is not None
        sparse = self._cache.get(self._file.path, frame)
        if sparse is None:
            sparse = self._file.read_frame(frame)
            self._cache.put(self._file.path, sparse)
        return sparse

    def _request_frame(self, frame: int) -> None:
        if self._file is None:
            raise RuntimeError("request_frame before a file is open")
        sparse = self._read_frame(frame)
        self.frame_loaded.emit(frame, sparse, self._file.frame_params(frame))

    def _set_follow(self, following: bool) -> None:
        """Turn the idle-time poll on or off, and poll once immediately when turning on.

        The immediate poll is what makes the toggle feel like a switch rather than a
        subscription: a user who turns Follow on while the instrument is between frames
        would otherwise wait a whole interval to be told what is already there.
        """
        self._poll_interval = POLL_INTERVAL_S if following else None
        self._live = None
        if following:
            self._poll()

    def _poll(self) -> None:
        """Ask the file what has changed, and report it if anything has.

        Emits on three conditions, and the third is the one that is easy to miss: the
        frame list changed; a frame became final; or some frame is *still* provisional,
        in which case the file may have grown inside a frame the list cannot see. A
        frame is one ion mobility experiment and its scans arrive in batches, so a view
        of the newest frame that only repainted when the frame count changed would show
        the experiment in one jump at the end rather than filling.
        """
        if self._file is None or self._poll_interval is None:
            return
        state = self._file.refresh()
        if state == self._live and not state.provisional:
            return
        grew = self._live is None or len(state.frames) != len(self._live.frames)
        types = self._file.frame_types() if grew and state.frames else None
        grouping = self._file.frame_grouping() if grew and state.frames else None
        self._live = state
        self.live_update.emit(state, types, grouping)

    def _sum(self, request: SumRequest) -> None:
        if self._file is None:
            raise RuntimeError("sum_all before a file is open")
        frames = list(request.frames)
        self._cancel_sum.clear()
        total = len(frames)

        def _read_each() -> "object":
            # No cancel check of its own: `sum_frames` below is the sole authority on
            # stopping early, and it must be -- it checks *before* folding a frame into
            # the running total and returns None outright, discarding everything summed
            # so far. A check here too would sometimes win the race and end the
            # iterable instead, which looks to `sum_frames` like "no more frames" and
            # returns the partial total as if it were the whole answer.
            for done, frame in enumerate(frames, start=1):
                yield self._read_frame(frame)
                self.summing_progress.emit(done, total)

        combined = sum_frames(_read_each(), should_cancel=self._cancel_sum.is_set)
        if combined is None:
            # Cancelled: `sum_frames` discarded the partial total rather than returning
            # it, so there is nothing to show and the request is echoed empty-handed.
            self.summed.emit(None, None, request)
            return
        params = self._file.frame_params(frames[0]) if frames else None
        self.summed.emit(combined, params, request)
