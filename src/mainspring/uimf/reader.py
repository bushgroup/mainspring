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
import time
from dataclasses import dataclass, field
from typing import Iterator, Mapping

import numpy as np

from .calib import Calibration
from .decode import decode_frame_blobs
from .frame import SparseFrame

__all__ = [
    "BUSY_TIMEOUT_MS",
    "PROVISIONAL_WINDOW_S",
    "FrameParams",
    "GlobalParams",
    "UimfFile",
    "connect",
]

BUSY_TIMEOUT_MS = 250
"""How long a read waits for the writer's lock before giving up. Short on purpose: a
viewer that is a quarter of a second stale is fine, a viewer that blocks an acquisition
is not."""

PROVISIONAL_WINDOW_S = 5.0
"""How recently a file must have been written to for its last frame to count as still
being acquired. A placeholder until the writer is measured (lab record, task 08); it is
deliberately generous, because the cost of being wrong is one uncached frame."""


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
        self._tables: frozenset[str] | None = None
        self._global: GlobalParams | None = None
        self._frames: dict[int, FrameParams] = {}

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.path!r})"

    # --- schema ---------------------------------------------------------------------

    @property
    def tables(self) -> frozenset[str]:
        """The table and view names present, which is how the legacy fallback decides.

        Read once and remembered: a file does not grow tables while it is being
        acquired, and this is on the path of every other method.
        """
        if self._tables is None:
            with connect(self.path, self.busy_timeout_ms) as conn:
                self._tables = frozenset(
                    row[0] for row in conn.execute(
                        "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')"
                    )
                )
        return self._tables

    @property
    def is_legacy_only(self) -> bool:
        """True for a file with only `Global_Parameters`/`Frame_Parameters` (2011 writers)."""
        return "Frame_Params" not in self.tables or "Frame_Param_Keys" not in self.tables

    # --- parameters -----------------------------------------------------------------

    def global_params(self) -> GlobalParams:
        """The dataset-wide parameters, modern table preferred.

        Cached: the values are a dozen short strings and they do not change while a file
        is being written, so re-reading them per frame would be lock traffic for nothing.
        """
        if self._global is None:
            with connect(self.path, self.busy_timeout_ms) as conn:
                raw = self._read_global(conn)
            self._global = _global_params_from(raw)
        return self._global

    def _read_global(self, conn: sqlite3.Connection) -> dict[str, str]:
        if "Global_Params" in self.tables:
            return {
                str(name): _text(value)
                for name, value in conn.execute(
                    "SELECT ParamName, ParamValue FROM Global_Params"
                )
            }
        if "Global_Parameters" in self.tables:
            cursor = conn.execute("SELECT * FROM Global_Parameters LIMIT 1")
            row = cursor.fetchone()
            if row is None:
                return {}
            return {
                _LEGACY_GLOBAL_NAMES.get(column[0].lower(), column[0]): _text(value)
                for column, value in zip(cursor.description, row)
            }
        raise ValueError(f"{self.path}: no Global_Params or Global_Parameters table")

    def frame_numbers(self) -> list[int]:
        """The frames that actually exist, ascending.

        Derived from the parameter tables rather than from `NumFrames`, which is only
        what the writer had said by the time we looked (lab record, task 01). Not cached
        for the same reason: on a live file this is the number that changes.
        """
        query = (
            "SELECT DISTINCT FrameNum FROM Frame_Params ORDER BY FrameNum"
            if not self.is_legacy_only
            else "SELECT FrameNum FROM Frame_Parameters ORDER BY FrameNum"
        )
        with connect(self.path, self.busy_timeout_ms) as conn:
            return [int(row[0]) for row in conn.execute(query)]

    def frame_types(self) -> dict[int, int]:
        """`frame -> FrameType` for every frame, for the viewer's MS1/MS2 filter.

        One query rather than a `frame_params` call per frame: a 25-frame file makes
        that difference invisible and a 2000-frame LC run does not.
        """
        params = {}
        with connect(self.path, self.busy_timeout_ms) as conn:
            if not self.is_legacy_only:
                rows = conn.execute(
                    "SELECT FP.FrameNum, FP.ParamValue FROM Frame_Params FP"
                    " JOIN Frame_Param_Keys K ON FP.ParamID = K.ParamID"
                    " WHERE K.ParamName = 'FrameType'"
                )
            else:
                rows = conn.execute("SELECT FrameNum, FrameType FROM Frame_Parameters")
            for frame, value in rows:
                params[int(frame)] = _as_int(value, 0)
        return params

    def frame_params(self, frame: int) -> FrameParams:
        """One frame's parameters, modern table preferred.

        Cached per frame: a frame's parameters are written once, before its scans, and
        the render path asks for the calibration on every view change.
        """
        frame = int(frame)
        if frame not in self._frames:
            with connect(self.path, self.busy_timeout_ms) as conn:
                raw = self._read_frame_params(conn, frame)
            if not raw:
                raise KeyError(f"{self.path}: no parameters for frame {frame}")
            self._frames[frame] = _frame_params_from(frame, raw)
        return self._frames[frame]

    def _read_frame_params(self, conn: sqlite3.Connection, frame: int) -> dict[str, str]:
        if not self.is_legacy_only:
            rows = conn.execute(
                "SELECT K.ParamName, FP.ParamValue FROM Frame_Params FP"
                " JOIN Frame_Param_Keys K ON FP.ParamID = K.ParamID WHERE FP.FrameNum = ?",
                (frame,),
            ).fetchall()
            if rows:
                return {str(name): _text(value) for name, value in rows}
        if "Frame_Parameters" in self.tables:
            cursor = conn.execute("SELECT * FROM Frame_Parameters WHERE FrameNum = ?", (frame,))
            row = cursor.fetchone()
            if row is not None:
                return {
                    _LEGACY_FRAME_NAMES.get(column[0].lower(), column[0]): _text(value)
                    for column, value in zip(cursor.description, row)
                }
        return {}

    # --- scans and frames -------------------------------------------------------------

    def scan_summary(self, frame: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """`(scan, non_zero_count, bpi, tic)` straight from `Frame_Scans`, nothing decoded.

        The file's own summary columns: exact ground truth for `TIC` and `BPI`, an upper
        bound for `NonZeroCount` (lab record, task 01). Cheap enough to drive a frame
        list or a total-ion chromatogram without touching a blob. The arrays are as long
        as the rows the writer stored, which on a SLIMPHONY file is a third of `Scans`.
        """
        with connect(self.path, self.busy_timeout_ms) as conn:
            rows = conn.execute(
                "SELECT ScanNum, NonZeroCount, BPI, TIC FROM Frame_Scans"
                " WHERE FrameNum = ? ORDER BY ScanNum",
                (int(frame),),
            ).fetchall()
        if not rows:
            empty = np.empty(0, dtype=np.int64)
            return empty, empty, np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
        columns = list(zip(*rows))
        return (
            np.asarray(columns[0], dtype=np.int64),
            np.asarray([0 if v is None else v for v in columns[1]], dtype=np.int64),
            np.asarray([0 if v is None else v for v in columns[2]], dtype=np.float64),
            np.asarray([0 if v is None else v for v in columns[3]], dtype=np.float64),
        )

    def is_provisional(self, frame: int) -> bool:
        """Whether this frame may still be growing under us, and so must not be cached.

        Conservative and deliberately crude until the writer's behaviour is measured
        (lab record, task 08): the last frame of a file that has been written to within
        the last few seconds. Over-reporting costs a cache entry; under-reporting would
        hand the viewer half a frame and let it keep it.
        """
        try:
            age = time.time() - os.path.getmtime(self.path)
        except OSError:
            return False
        if age > PROVISIONAL_WINDOW_S:
            return False
        numbers = self.frame_numbers()
        return bool(numbers) and int(frame) == numbers[-1]

    def read_frame(
        self,
        frame: int,
        scan_range: tuple[int, int] | None = None,
        bin_range: tuple[int, int] | None = None,
    ) -> SparseFrame:
        """Decode a frame into a `SparseFrame`, optionally only part of it.

        The result carries the frame's `Scans` and the file's `Bins` as its axis extent
        whatever the ranges were, so that a partial read still knows where it sits, and
        points outside that axis are dropped rather than widening it -- the axis is what
        the parameters say, never what the rows happen to reach (lab record, task 01).
        A dropped point would show up immediately as a `TIC` mismatch under
        `uimf-info --verify`, which is the check that would catch a file needing
        otherwise.

        One connection, one query, and the blobs decoded after it is closed: the lock is
        held for the read and not for the arithmetic.
        """
        frame = int(frame)
        params = self.frame_params(frame)
        globals_ = self.global_params()
        scans = int(params.scans)
        bins = int(globals_.bins)

        query = ("SELECT ScanNum, Intensities FROM Frame_Scans WHERE FrameNum = ?"
                 " ORDER BY ScanNum")
        arguments: tuple = (frame,)
        if scan_range is not None:
            query = ("SELECT ScanNum, Intensities FROM Frame_Scans WHERE FrameNum = ?"
                     " AND ScanNum >= ? AND ScanNum < ? ORDER BY ScanNum")
            arguments = (frame, int(scan_range[0]), int(scan_range[1]))
        with connect(self.path, self.busy_timeout_ms) as conn:
            rows = conn.execute(query, arguments).fetchall()

        provisional = self.is_provisional(frame)
        keep = [(int(scan), blob) for scan, blob in rows if 0 <= int(scan) < scans]
        counts, bin_index, intensity = decode_frame_blobs(
            [blob for _, blob in keep], globals_.dtype, bins
        )

        per_scan = np.zeros(scans, dtype=np.int64)
        if keep:
            per_scan[np.asarray([scan for scan, _ in keep], dtype=np.int64)] = counts
        scan_start = np.zeros(scans + 1, dtype=np.int64)
        np.cumsum(per_scan, out=scan_start[1:])
        built = SparseFrame(
            frame=frame, scans=scans, bins=bins, scan_start=scan_start,
            bin_index=bin_index, intensity=intensity, provisional=provisional,
        )
        return built if bin_range is None else built.slice(bin_range=bin_range)


# --- parameter names and coercion ----------------------------------------------------

# The legacy fixed columns under the names the modern tables use, so that everything
# above this line reads one vocabulary. Keyed lower case: the same column is
# `CalibrationDone` in a 2026 file and `CALIBRATIONDONE` in a 2011 one.
_LEGACY_GLOBAL_NAMES = {
    "instrument_name": "InstrumentName",
    "prescan_tofpulses": "PrescanTOFPulses",
    "prescan_accumulations": "PrescanAccumulations",
    "prescan_ticthreshold": "PrescanTICThreshold",
    "prescan_continuous": "PrescanContinuous",
    "prescan_profile": "PrescanProfile",
}
_LEGACY_FRAME_NAMES = {
    "calibrationdone": "CalibrationDone",
    "temperature": "AmbientTemperature",
    "duration": "DurationSeconds",
}


def _text(value: object) -> str:
    """A stored parameter as a string. Blobs become their length, not their bytes: the
    only one any file carries is `FragmentationProfile`, which nothing displays."""
    if value is None:
        return ""
    if isinstance(value, (bytes, bytearray, memoryview)):
        return f"<{len(bytes(value))} bytes>"
    return str(value)


def _as_int(value: object, default: int = 0) -> int:
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return default


def _as_float(value: object, default: float = 0.0) -> float:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default


def _global_params_from(raw: Mapping[str, str]) -> GlobalParams:
    return GlobalParams(
        instrument_name=raw.get("InstrumentName", ""),
        date_started=raw.get("DateStarted", ""),
        num_frames=_as_int(raw.get("NumFrames"), 0),
        bins=_as_int(raw.get("Bins"), 0),
        bin_width_ns=_as_float(raw.get("BinWidth"), 1.0) or 1.0,
        tof_intensity_type=(raw.get("TOFIntensityType") or "ADC").strip() or "ADC",
        time_offset_ns=_as_float(raw.get("TimeOffset"), 0.0),
        dataset_type=raw.get("DatasetType", ""),
        extra=dict(raw),
    )


def _frame_params_from(frame: int, raw: Mapping[str, str]) -> FrameParams:
    return FrameParams(
        frame_number=frame,
        frame_type=_as_int(raw.get("FrameType"), 0),
        scans=_as_int(raw.get("Scans"), 0),
        accumulations=_as_int(raw.get("Accumulations"), 1) or 1,
        average_tof_length_ns=_as_float(raw.get("AverageTOFLength"), 0.0),
        calibration_slope=_as_float(raw.get("CalibrationSlope"), 0.0),
        calibration_intercept=_as_float(raw.get("CalibrationIntercept"), 0.0),
        calibration_done=bool(_as_int(raw.get("CalibrationDone"), 0)),
        extra=dict(raw),
    )
