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

**The chromatogram's expensive source runs in that same queue, in chunks.** The file's
own per-frame totals are one query and need nothing special; totalling the heatmap's
*current view* in every frame is a read and a selection each, seconds to a minute over a
whole run. A job that long cannot be one queue item, because the queue is FIFO and the
frame a user asks for next would wait behind all of it -- so `_walk_restricted` works
for `CHUNK_BUDGET_S` and puts itself back at the end of the queue, carrying a serial a
newer request supersedes and keeping what it has already computed under the window it
computed it for.

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
from dataclasses import dataclass, replace

import numpy as np
from PySide6.QtCore import QThread, Signal

from ..uimf import (
    DisplayAxes,
    FileGone,
    FrameGrouping,
    LiveState,
    RasterResult,
    SparseFrame,
    UimfFile,
    sum_frames,
)
from ..uimf.cache import DEFAULT_BUDGET_BYTES, FrameCache
from ..uimf.decode import numba_available
from ..uimf.raster import render_view, tic_in_view

__all__ = [
    "CHUNK_BUDGET_S",
    "DEBOUNCE_MS",
    "POLL_INTERVAL_S",
    "LoadWorker",
    "RenderMailbox",
    "RenderRequest",
    "RenderResult",
    "RenderWorker",
    "RestrictedRequest",
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

CHUNK_BUDGET_S = 0.1
"""How long the chromatogram's view-restricted walk works before re-queueing itself.

**A time budget rather than a frame count**, and measured rather than chosen. The walk
reads and selects every frame of the file -- 16 s over the 5,000-frame synthetic and
about a minute over a real run of the same length -- and the queue it runs in is FIFO,
so what has to be bounded is how long a frame request can sit behind it. A fixed count
bounds that only if a frame costs the same everywhere, and it does not: 32 frames is
104 ms on the synthetic raw file, 431 ms on a real clockwork run and 1.15 s on the
densest frame PNNL publishes (lab record, task 31). A budget bounds it at one frame plus
this number whatever the file holds, and one frame is the part nothing can avoid."""


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
class RestrictedRequest:
    """One "total these frames over this window" ask, and where the next chunk resumes.

    Immutable and re-queued with a new `start` after each chunk, rather than a mutable
    cursor the worker keeps: the queue already carries the work, and a request that
    describes the whole job is also the thing a newer one supersedes.

    `serial` is the window's, and a chunk whose serial is not the current one is
    dropped. The axis options travel with the ranges because a window is only a window
    given the axes it was drawn in -- a swap or a units toggle means the same two pairs
    of numbers name a different set of points, and the answer has to be recomputed
    rather than served out of the cache.
    """

    frames: tuple[int, ...] = ()
    x_range: tuple[float, float] = (0.0, 0.0)
    y_range: tuple[float, float] = (0.0, 0.0)
    raw_units: bool = False
    swapped: bool = False
    t0_offset_ms: float = 0.0
    serial: int = 0
    start: int = 0

    @property
    def view(self) -> tuple:
        """What makes two asks the same question, for the cache to key totals by.

        The frame list is deliberately **not** in it: a live run appends frames to the
        same window, and a key that included the list would throw away every total
        already computed each time one arrived (lab record, task 31).
        """
        return (self.x_range, self.y_range, self.raw_units, self.swapped,
                self.t0_offset_ms)


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
    chromatogram = Signal(object, object, bool)
    """`dict[int, float], dict[int, float], bool` -- each frame's stored `TIC` total,
    each frame's `StartTimeMinutes`, and whether this replaces the series or adds to it.

    Two whole-file queries, like `frame_types` and `frame_grouping` beside them: 79 ms
    over a real 45 MB clockwork run and 114 ms over a 5,000-frame one, which is why the
    window asks for it *after* the first frame and only when the panel is actually on
    screen. `replace` is False on the poll's form, where both are narrowed to the frames
    that can have changed and arrive one or two at a time (lab record, task 31)."""
    restricted = Signal(object, object, int, int)
    """`serial, dict[int, float], done, total` -- the view-restricted walk so far.

    Emitted once per chunk rather than once at the end, because a walk over a whole run
    is seconds and a trace that filled in is worth more than one that appeared. The
    dictionary is the accumulated answer and not the chunk, so a receiver draws what it
    is handed and keeps nothing. `serial` is the request's, for the same reason a
    `RenderResult` carries one: the window drops a chunk its view has moved past."""
    follow_stopped = Signal(str, bool)
    """A poll raised, and following has been switched off: the message, and whether the
    file itself has gone.

    Its own signal rather than `failed`, because a failing poll is the one failure that
    repeats: the file has been moved, deleted or unmounted, and a viewer that reported
    that once a second until someone noticed would be worse than one that stops and says
    so. The window puts its own toggle back with this.

    The flag is `FileGone` and nothing else, because a file that is not there is the one
    end a *successful* run has: a run that does not keep its raw file deletes it at the
    close, and the operator should read that as the run ending rather than as SQLite
    failing to open something. What to say about it, and what to offer instead, is the
    window's (lab record, task 28); what cannot be worked out there is which of the two
    happened, since by then there is nothing left to ask."""
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
        self._chromatogram = False
        # `view signature -> {frame: total}`. Per frame and not per array, so that a
        # live run appends to a walk instead of invalidating it, and keyed by the view
        # so that going back to a window already walked is free. Never evicted and
        # cleared on open: a completed walk is 5,000 floats, and each distinct one costs
        # the user seconds of waiting, so nobody can accumulate enough of them to matter.
        self._restricted_totals: dict[tuple, dict[int, float]] = {}
        self._restricted_serial = 0
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

    def request_chromatogram(self) -> None:
        """Read the whole file's per-frame totals and start times; `chromatogram` carries them.

        Asked for by the window rather than done on every open, and asked for *after*
        the first frame, so that a panel nobody has opened costs nothing and an open
        that does have it open still paints the frame first.
        """
        self._queue.put(("chromatogram", None))

    def set_chromatogram(self, wanted: bool) -> None:
        """Say whether a follow poll should also report what the newest frames total.

        Through the queue for the reason `set_follow` is: the loop spends its life
        blocked in `get()`. A poll that finds nothing new sends nothing here either --
        the extra query is inside the branch that has already decided to emit, and is
        narrowed to the frames whose totals can have changed, so it is 0.3 to 2.2 ms
        (lab record, task 31).
        """
        self._queue.put(("want_chromatogram", bool(wanted)))

    def request_restricted(self, request: RestrictedRequest) -> None:
        """Start (or resume) the view-restricted walk; `restricted` reports each chunk.

        Supersedes whatever was walking: the serial moves, and the chunk in flight is
        dropped when it comes back round. Anything already computed for the same window
        is kept, because it is keyed by the window and not by the request.
        """
        self._restricted_serial = int(request.serial)
        self._queue.put(("restricted", request))

    def cancel_restricted(self) -> None:
        """Abandon the walk at its next chunk. Idempotent, and keeps what it has."""
        self._restricted_serial += 1

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
                    self.follow_stopped.emit(str(exc), isinstance(exc, FileGone))
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
                elif kind == "chromatogram":
                    self._read_chromatogram()
                elif kind == "want_chromatogram":
                    self._chromatogram = bool(payload)
                elif kind == "restricted":
                    self._walk_restricted(payload)
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
        # Every restricted total belongs to the file it was walked over, and a serial
        # moved here is what makes a chunk still in the queue from the last file drop
        # itself rather than read frames out of this one.
        self._restricted_totals.clear()
        self._restricted_serial += 1
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

    def _read_chromatogram(self, since: "int | None" = None) -> None:
        """Both of the chromatogram's whole-file answers, and emit them together.

        Together rather than as two signals, because they are one picture: a trace drawn
        against frame number for a moment and then re-drawn against time would read as a
        glitch, and the panel's own decision about whether the start times mean anything
        needs both in hand (`chromatogram.elapsed_minutes`).
        """
        if self._file is None:
            return
        self.chromatogram.emit(
            self._file.frame_totals(since),
            self._file.frame_start_times(since),
            since is None,
        )

    def _walk_restricted(self, request: RestrictedRequest) -> None:
        """One chunk of the view-restricted walk, then re-queue or stop.

        Three things are load-bearing here and none of them is the arithmetic.

        **It re-queues rather than looping**, so a frame request or a close that arrives
        mid-walk is served at the end of this chunk instead of at the end of the run --
        the queue is FIFO and the walk puts itself at the back of it.

        **The budget is time, not frames** (`CHUNK_BUDGET_S`): what has to be bounded is
        the wait, and a frame costs 3 ms on one file and 14 ms on another. At least one
        frame is always done, so a file whose every frame is over budget still finishes.

        **The totals are kept per frame under the view's own key**, so a walk resumed
        after a superseding request, or extended by a frame a live run has just written,
        adds to what is there instead of starting again.
        """
        if self._file is None or request.serial != self._restricted_serial:
            return
        totals = self._restricted_totals.setdefault(request.view, {})
        globals_ = self._file.global_params()
        deadline = time.perf_counter() + CHUNK_BUDGET_S
        index = int(request.start)
        frames = request.frames
        while index < len(frames):
            number = int(frames[index])
            index += 1
            if number not in totals:
                params = self._file.frame_params(number)
                sparse = self._read_frame(number)
                axes = DisplayAxes.build(
                    sparse, params.calibration(globals_.bin_width_ns),
                    params.average_tof_length_ns,
                    raw_units=request.raw_units, swapped=request.swapped,
                    t0_offset_ms=request.t0_offset_ms,
                )
                totals[number] = tic_in_view(
                    sparse, axes, request.x_range, request.y_range
                )
            if time.perf_counter() >= deadline:
                break
        done = {n: totals[n] for n in frames if n in totals}
        self.restricted.emit(request.serial, done, index, len(frames))
        if index < len(frames):
            self._queue.put(("restricted", replace(request, start=index)))

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
        previous = self._live
        self._live = state
        self.live_update.emit(state, types, grouping)
        if self._chromatogram:
            self._extend_chromatogram(previous, state)

    def _extend_chromatogram(
        self, previous: "LiveState | None", state: LiveState
    ) -> None:
        """The third query of a poll, and only about the frames that can have moved.

        A finished frame's total never changes and a frame's start time is written once,
        before any of its scans, so what is worth re-reading is exactly the frames that
        were provisional at the last look plus the ones that were not there at all. Both
        sets are at the tail, so one `FrameNum >= <the lowest of them>` covers them, and
        narrowed to one or two frames the pair costs 0.3 to 2.2 ms against the poll's own
        11 ms (lab record, task 31).

        `refresh()` itself is untouched by any of this, which is what keeps the counted
        "a poll is two queries" check reading two: this is a third query made by the
        poll's caller, on a condition the poll has already decided is worth reporting.
        """
        if self._file is None or not state.frames:
            return
        if previous is None:
            self._read_chromatogram()
            return
        moved = set(state.frames) - set(previous.frames) | set(previous.provisional)
        if not moved:
            return
        self._read_chromatogram(since=min(moved))

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
