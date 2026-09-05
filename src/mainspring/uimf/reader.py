"""`UimfFile`: the SQLite side. Parameters, frame lists, and frames read as points.

A UIMF file is a SQLite database with four tables that matter -- `Global_Params`,
`Frame_Param_Keys` + `Frame_Params`, and `Frame_Scans` -- and legacy fixed-column twins
of the first two, `Global_Parameters` and `Frame_Parameters`, which are all a 2011 file
has. Modern tables win where both exist, which is what UIMF-Library does and what our
own sample needs, since its two tables disagree about frame type (lab record, task 01).

**Every read opens its own read-only connection and closes it.** This is not tidiness;
it is the one rule that lets the viewer open a file the instrument is still writing.
UIMF writers use `journal_mode=delete`, under which a reader's shared lock blocks the
writer's commit for as long as it is held -- a long-lived connection or a cursor left
open across a user's coffee break can stall an acquisition. So: `mode=ro` through a URI,
a short `busy_timeout`, one query per connection, and no caching of the last frame
(lab record, task 08). Copying a file mid-write is not an alternative; the copy is
corrupt.

Parameter values are TEXT in the modern tables and typed by `ParamDataType`, so
`GlobalParams` and `FrameParams` name the handful the viewer needs, coerced, and keep
everything else as raw strings in `extra` -- the info panel shows whatever a file
happens to carry, and a key nobody anticipated is worth displaying rather than dropping.
"""

from __future__ import annotations

import contextlib
import os
import sqlite3
from dataclasses import dataclass, field
from typing import Iterator, Mapping

import numpy as np

from .calib import Calibration
from .frame import SparseFrame

__all__ = ["BUSY_TIMEOUT_MS", "FrameParams", "GlobalParams", "UimfFile", "connect"]

BUSY_TIMEOUT_MS = 250
"""How long a read waits for the writer's lock before giving up. Short on purpose: a
viewer that is a quarter of a second stale is fine, a viewer that blocks an acquisition
is not."""


@contextlib.contextmanager
def connect(path: str | os.PathLike[str], busy_timeout_ms: int = BUSY_TIMEOUT_MS) -> Iterator[sqlite3.Connection]:
    """A short-lived read-only connection to a UIMF file, closed on the way out.

    Read-only through a `file:...?mode=ro` URI rather than by good intentions: an
    accidental write to an acquisition in progress is unrecoverable. The caller is
    expected to run one query and let go -- see the module docstring on why.
    """
    uri = "file:" + os.fspath(os.path.abspath(path)).replace("?", "%3f").replace("#", "%23") + "?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=busy_timeout_ms / 1000.0)
    try:
        conn.execute(f"PRAGMA busy_timeout = {int(busy_timeout_ms)}")
        yield conn
    finally:
        conn.close()


@dataclass(frozen=True)
class GlobalParams:
    """The dataset-wide scalars, from `Global_Params` or its legacy twin.

    `bins` and `bin_width_ns` set the TOF axis, `tof_intensity_type` the blob element
    type. `time_offset_ns` is read and reported but **not applied to the calibration**;
    it is here because the info panel shows it and because seeing it is how a reader of
    this code finds out that it is deliberately unused (lab record, task 01).
    """

    instrument_name: str = ""
    date_started: str = ""
    num_frames: int = 0
    bins: int = 0
    bin_width_ns: float = 1.0
    tof_intensity_type: str = "ADC"
    time_offset_ns: float = 0.0
    dataset_type: str = ""
    extra: Mapping[str, str] = field(default_factory=dict)

    @property
    def dtype(self) -> np.dtype:
        """The intensity element type this file's blobs are packed in."""
        from .decode import dtype_for

        return dtype_for(self.tof_intensity_type)


@dataclass(frozen=True)
class FrameParams:
    """One frame's scalars, from `Frame_Params` or the legacy `Frame_Parameters` row.

    `frame_type` is normalised so that a caller can filter MS1 from MS2 without knowing
    which table it came from: the sample's two tables say 0 and 1 for the same frame and
    both mean MS1 (lab record, task 01).
    """

    frame_number: int = 0
    frame_type: int = 0
    scans: int = 0
    accumulations: int = 1
    average_tof_length_ns: float = 0.0
    calibration_slope: float = 0.0
    calibration_intercept: float = 0.0
    calibration_done: bool = False
    extra: Mapping[str, str] = field(default_factory=dict)

    def calibration(self, bin_width_ns: float) -> Calibration:
        """This frame's `Calibration`, given the file's bin width."""
        return Calibration(
            slope=self.calibration_slope,
            intercept=self.calibration_intercept,
            bin_width_ns=bin_width_ns,
            done=self.calibration_done,
        )

    @property
    def is_ms1(self) -> bool:
        """Whether this is an MS1 frame; 0 and 1 both mean MS1 across the two tables."""
        return self.frame_type in (0, 1)

    @property
    def duration_ms(self) -> float:
        """Arrival-time extent of the frame: `Scans * AverageTOFLength`, in milliseconds."""
        return self.scans * self.average_tof_length_ns * 1e-6


class UimfFile:
    """A UIMF file, opened lazily and never held open. See the module docstring.

    Construction only records the path and checks that it is a file; every method opens
    its own connection. Parameters are cached in the instance because they are small and
    a live file's parameters do not change, but **frames are not** -- see `read_frame`.
    """

    def __init__(self, path: str | os.PathLike[str], busy_timeout_ms: int = BUSY_TIMEOUT_MS) -> None:
        self.path = os.path.abspath(os.fspath(path))
        self.busy_timeout_ms = int(busy_timeout_ms)
        if not os.path.isfile(self.path):
            raise FileNotFoundError(self.path)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.path!r})"

    @property
    def tables(self) -> frozenset[str]:
        """The table and view names present, which is how the legacy fallback decides.

        Arrives with the lab record's task 03.
        """
        raise NotImplementedError("the reader arrives with the lab record's task 03")

    @property
    def is_legacy_only(self) -> bool:
        """True for a file with only `Global_Parameters`/`Frame_Parameters` (2011 writers).

        Arrives with the lab record's task 03.
        """
        raise NotImplementedError("the reader arrives with the lab record's task 03")

    def global_params(self) -> GlobalParams:
        """The dataset-wide parameters, modern table preferred. Arrives with task 03."""
        raise NotImplementedError("the reader arrives with the lab record's task 03")

    def frame_numbers(self) -> list[int]:
        """The frames that actually exist, ascending.

        Derived from the parameter tables rather than from `NumFrames`, which is only
        what the writer had said by the time we looked (lab record, task 01).

        Arrives with the lab record's task 03.
        """
        raise NotImplementedError("the reader arrives with the lab record's task 03")

    def frame_params(self, frame: int) -> FrameParams:
        """One frame's parameters, modern table preferred. Arrives with task 03."""
        raise NotImplementedError("the reader arrives with the lab record's task 03")

    def scan_summary(self, frame: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """`(scan, non_zero_count, bpi, tic)` straight from `Frame_Scans`, nothing decoded.

        The file's own summary columns: exact ground truth for `TIC` and `BPI`, an upper
        bound for `NonZeroCount` (lab record, task 01). Cheap enough to drive a frame
        list or a total-ion chromatogram without touching a blob.

        Arrives with the lab record's task 03.
        """
        raise NotImplementedError("the reader arrives with the lab record's task 03")

    def read_frame(
        self,
        frame: int,
        scan_range: tuple[int, int] | None = None,
        bin_range: tuple[int, int] | None = None,
    ) -> SparseFrame:
        """Decode a frame into a `SparseFrame`, optionally only part of it.

        The result carries the frame's `Scans` and the file's `Bins` as its axis extent
        whatever the ranges were, so that a partial read still knows where it sits.

        Arrives with the lab record's task 03.
        """
        raise NotImplementedError("the reader arrives with the lab record's task 03")
