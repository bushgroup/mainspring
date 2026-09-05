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

from dataclasses import dataclass

from ..uimf import DisplayAxes, RasterResult, SparseFrame

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


class LoadWorker:
    """The thread that owns the file and the frame cache and decodes frames.

    Arrives with the lab record's task 04.
    """

    def __init__(self, cache_budget_bytes: int = 0) -> None:
        raise NotImplementedError("the load worker arrives with the lab record's task 04")

    def open(self, path: str) -> None:
        """Open a file and report its parameters. Arrives with the lab record's task 04."""
        raise NotImplementedError("the load worker arrives with the lab record's task 04")

    def request_frame(self, frame: int) -> None:
        """Ask for a frame; the loaded signal carries it. Arrives with task 04."""
        raise NotImplementedError("the load worker arrives with the lab record's task 04")
